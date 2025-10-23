import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, LlamaForCausalLM, LlamaTokenizer
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
import numpy as np
from tqdm import tqdm
import wandb
from types import SimpleNamespace
import pdb

class SSL_LLM(sb.Brain):
    def __init__(self, *args, patience=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.no_improve_epochs = 0
        self.best_per_list = []  # List of (PER, epoch, ckpt_name)
        self.best_mpd_f1_list = []  # List of (mpd_f1, epoch, ckpt_name)
        self.best_per = float('inf')
        self.best_mpd_f1 = float('-inf')
        self.last_improved_epoch = 0
        
        # 初始化设备（必须先于依赖device的模块创建）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建 LayerNorm 作为模型的一部分（而不是在 forward 中每次创建）
        # 将在 on_fit_start 后初始化（因为此时 modules.LLM 还未创建）
        self.inputs_embeds_norm = None
        
        # Prompt 相关初始化
        self.prompt_embeddings = None
        self.use_prompt = getattr(self.hparams, "use_prompt", False)
        if self.use_prompt:
            print(f"[Model] 启用 Prompt Tuning 模式")
        
        # 将SSL模型移至正确设备
        if getattr(self.modules, "perceived_ssl", None) is not None:
            self.modules.perceived_ssl.to(self.device)
        
        # 训练追踪
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}
        
        # 创建phoneme token掩码
        # self.setup_phoneme_mask()

    def setup_prompt_embeddings(self):
        """初始化 soft prompt embeddings"""
        if not self.use_prompt or self.prompt_embeddings is not None:
            return
        
        prompt_len = getattr(self.hparams, "prompt_len", 8)
        prompt_init = getattr(self.hparams, "prompt_init", "normal")
        prompt_dropout = getattr(self.hparams, "prompt_dropout", 0.0)
        
        # 获取 LLM 的 embedding 维度
        H = self.modules.LLM.get_input_embeddings().weight.shape[1]
        
        # 创建 prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.zeros(prompt_len, H))
        
        # 初始化
        if prompt_init == "normal":
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        elif prompt_init == "xavier":
            nn.init.xavier_uniform_(self.prompt_embeddings)
        elif prompt_init == "zeros":
            pass  # 已经是 zeros
        else:
            raise ValueError(f"Unknown prompt_init: {prompt_init}")
        
        # 添加 dropout（如果需要）
        if prompt_dropout > 0:
            self.prompt_dropout = nn.Dropout(prompt_dropout)
        else:
            self.prompt_dropout = None
        
        # 移到正确的设备
        self.prompt_embeddings = self.prompt_embeddings.to(self.device)
        
        print(f"[Model] 创建 Prompt Embeddings: {prompt_len} × {H}")
        print(f"[Model] Prompt 初始化方式: {prompt_init}")
        print(f"[Model] Prompt dropout: {prompt_dropout}")

    def setup_phoneme_mask(self):
        """创建一个掩码，只允许生成音素相关的token"""
        vocab_size = self.modules.LLM.get_input_embeddings().weight.shape[0]
        # 创建一个全为 -inf 的掩码
        self.phoneme_bias = torch.full(
            (vocab_size,), float('-10e9'), device=self.device
        )
        # 将音素token的位置设为0（允许生成）
        valid_tokens = list(range(44))  # 0-43 是音素相关的token（包括blank, bos, eos）
        self.phoneme_bias[valid_tokens] = 0
        
    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model forward pass.
        Returns:
            - p_ctc: [B, T, C]
            - ce_logits: [B, L+1, C_small] or None
            - ce_targets: [B, L+1] with ignore_index padded or None
            - wav_lens: [B]
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_emb, phn_emb_lens = batch.phn_encoded_target

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # SSL encoder
        wav_feats = self.modules.perceived_ssl(wavs)
        # Project SSL features to LLaMA dimension (time length不变)
        
        Z = self.modules.enc(wav_feats.transpose(-2, -1))  # [B, T, H]
        Z = Z.transpose(-2, -1) 
        
        # CTC分支
        if self.hparams.ctc_head_input == "ssl":
            ctc_logits = self.modules.ctc_lin(wav_feats)
        elif getattr(self.modules, "enc_ctc") and self.hparams.ctc_head_input == "enc_ctc":
            ctc_in = self.modules.enc_ctc(wav_feats)
            ctc_logits = self.modules.ctc_lin(ctc_in)
        elif self.hparams.ctc_head_input == "enc_llm":
            # use LLM encoder's input, good for ASR but not proper for Phoneme recognition
            ctc_logits = self.modules.ctc_lin(Z)
        
        p_ctc = self.hparams.log_softmax(ctc_logits)

        # 仅训练CTC则跳过LLM
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        use_small_head = getattr(self.hparams, "use_small_vocab_head", True)
        if ctc_weight >= 1.0 - 1e-8 or not use_small_head:
            return p_ctc, None, None, wav_lens

        # 方案A：小词表头，构建teacher-forced文本嵌入
        B, L_max = phns_emb.size()
        H = Z.size(-1)
        bos_idx = getattr(self.hparams, "ph_bos_index", 42)
        eos_idx = getattr(self.hparams, "ph_eos_index", 43)
        ignore_index = getattr(self.hparams, "ce_label_ignore_index", -100)
        
        # 计算每条样本的真实长度（从ratio转为绝对token数）
        if phn_emb_lens.dtype.is_floating_point:
            abs_lens = (phn_emb_lens.float() * L_max).round().clamp(min=0, max=L_max).long()
        else:
            abs_lens = phn_emb_lens.clamp(min=0, max=L_max).long()
        
        # 构造输入ids: [BOS, y0..y_{L-1}] -> [B, L_max+1]
        bos_col = torch.full((B, 1), bos_idx, device=self.device, dtype=phns_emb.dtype)
        inp_ids = torch.cat([bos_col, phns_emb], dim=1)
        # 构造目标序列: [y0..y_{L-1}, EOS] 并用ignore填充
        ce_targets = torch.full((B, L_max + 1), ignore_index, device=self.device, dtype=phns_emb.dtype)
        # 逐样本填充有效目标与EOS
        arange_b = torch.arange(B, device=self.device)
        for b in range(B):
            Lb = int(abs_lens[b].item())
            if Lb > 0:
                ce_targets[b, :Lb] = phns_emb[b, :Lb]
            # EOS位置
            if Lb <= L_max:
                ce_targets[b, Lb] = eos_idx
        
        # 文本嵌入
        txt_emb = self.modules.phn_embed(inp_ids)  # [B, L_max+1, H]
        
        # ============ Prompt Tuning 逻辑 ============
        # 如果启用 prompt，在音频和文本之间插入 prompt embeddings
        if self.use_prompt:
            # 初始化 prompt（延迟初始化）
            if self.prompt_embeddings is None:
                self.setup_prompt_embeddings()
            
            # 获取 prompt embeddings 并扩展到 batch
            prompt_emb = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, P, H]
            
            # 应用 dropout（训练时）
            if stage == sb.Stage.TRAIN and self.prompt_dropout is not None:
                prompt_emb = self.prompt_dropout(prompt_emb)
            
            # 拼接顺序: [Audio] [Prompt] [Text]
            inputs_embeds = torch.cat([Z, prompt_emb, txt_emb], dim=1)  # [B, T+P+L_max+1, H]
            
            # 更新 attention_mask（prompt 部分全为 1）
            P = self.prompt_embeddings.size(0)
            prefix_mask = torch.ones(B, Z.size(1), dtype=torch.long, device=self.device)
            prompt_mask = torch.ones(B, P, dtype=torch.long, device=self.device)
            text_mask = torch.zeros(B, L_max + 1, dtype=torch.long, device=self.device)
            
            for b in range(B):
                Lb = int(abs_lens[b].item())
                text_mask[b, : Lb + 1] = 1
            attention_mask = torch.cat([prefix_mask, prompt_mask, text_mask], dim=1)  # [B, T+P+L_max+1]
        else:
            # 原始逻辑：直接拼接音频和文本
            inputs_embeds = torch.cat([Z, txt_emb], dim=1)  # [B, T + L_max+1, H]
            
            # 构造attention_mask：前缀全1；文本部分仅保留有效(BOS到L_b位置)为1，其余0
            prefix_mask = torch.ones(B, Z.size(1), dtype=torch.long, device=self.device)
            text_mask = torch.zeros(B, L_max + 1, dtype=torch.long, device=self.device)
            
            for b in range(B):
                Lb = int(abs_lens[b].item())
                text_mask[b, : Lb + 1] = 1
            attention_mask = torch.cat([prefix_mask, text_mask], dim=1)  # [B, T+L_max+1]
        # ============================================
        
        # 使用持久化的 LayerNorm（会被优化）
        if self.inputs_embeds_norm is None:
            # 延迟初始化（第一次forward时创建）
            self.inputs_embeds_norm = nn.LayerNorm(H).to(self.device)
            print(f"[Model] 创建 inputs_embeds_norm: {H} 维")
        
        inputs_embeds = self.inputs_embeds_norm(inputs_embeds)
        
        # 将dtype对齐到LLM权重dtype，避免float/half不一致
        llm_dtype = self.modules.LLM.get_input_embeddings().weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # 运行LLM获得hidden states
        llm_out = self.modules.LLM(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden = llm_out.hidden_states[-1]  # [B, T + (P) + L_max+1, H]
        
        # 取用于预测的隐状态：跳过音频(T)和prompt(P)，用剩余部分预测 [y0..y_{L-1}, EOS]
        T = Z.size(1)
        if self.use_prompt and self.prompt_embeddings is not None:
            P = self.prompt_embeddings.size(0)
            pred_h = hidden[:, T + P : T + P + (L_max + 1), :]  # [B, L_max+1, H]
        else:
            pred_h = hidden[:, T : T + (L_max + 1), :]  # [B, L_max+1, H]
        
        # 小词表分类头
        # pdb.set_trace()
        # ce_logits = self.modules.phn_head(pred_h)  # [B, L_max+1, 44]
        ce_logits = self.modules.phn_head(pred_h.to(torch.float32))  # [B, L_max+1, 44]
        
        return p_ctc, ce_logits, ce_targets, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """计算训练目标：CTC损失和CE损失（小词表头）"""
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target
        targets_eos, target_eos_lens = batch.phn_encoded_target_eos
        # 解包
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, lens_for_ctc = predictions
        else:
            # 兼容旧版
            out, p_ctc, lens_for_ctc = predictions
            ce_logits, ce_targets = None, None
        
        # CTC损失（fp32更稳）
        # 防止目标长度超过可用时间步
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # CE损失（小词表头）
        loss_ce = torch.tensor(0.0, device=self.device)
        
        if ce_logits is not None and ce_targets is not None:
            B, Lp1, Csm = ce_logits.size()
            ignore_index = getattr(self.hparams, "ce_label_ignore_index", -100)
            ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss_ce = ce_loss_fn(ce_logits.reshape(-1, Csm), ce_targets.reshape(-1))
        # 总损失
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.0)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_ce
        
        # 记录
        if stage == sb.Stage.TRAIN:
            self.train_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.train_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.train_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
            # logging
            # print(f"CE Loss: {loss_CE.item():.4f}, CTC Loss: {loss_ctc.item():.4f}", end=", ")
        else:
            self.valid_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.valid_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.valid_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        
        # 更新CTC指标
        # per_metrics append
        if stage != sb.Stage.TRAIN:
            ctc_sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            
            self.ctc_metrics.append(ids, p_ctc, targets, lens_for_ctc, clipped_target_lens)
            self.per_metrics.append(
                ids=ids,
                predict=ctc_sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            
            p_llm = F.log_softmax(ce_logits, dim=-1)
            llm_sequence = p_llm.argmax(dim=-1) if ce_logits is not None else None
            # 观察是不是每次都一样？
            self.llm_metrics.append( 
                                    ids,
                                    log_probabilities=p_llm,
                                    targets=targets_eos,
                                    length=target_eos_lens)
            self.llm_per_metrics.append(
                ids=ids,
                predict=llm_sequence,
                target=targets_eos,
                predict_len=None,
                target_len=target_eos_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
        
        return loss
    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        self.llm_metrics = self.hparams.llm_stats()  # 添加LLM损失统计
        if hasattr(self.modules, "ctc_lin"):
            self.ctc_metrics = self.hparams.ctc_stats()
            
        if hasattr(self.hparams, "augmentation"):
            self.modules.perceived_ssl.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()
            self.llm_per_metrics = self.hparams.per_stats()# 添加LLM PER统计
            
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage to summarize and log."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats["epoch"] = epoch
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")  # 添加LLM PER计算
            stage_stats["ctc_per"] = per_ctc
            
            stage_stats["llm_per"] = per_llm  # 添加LLM PER到统计
            # Summarize and log metrics
            llm_loss = self.llm_metrics.summarize("average")
            stage_stats["llm_loss"] = llm_loss
        
            if hasattr(self.modules, "ctc_lin"):
                ctc_loss = self.ctc_metrics.summarize("average")
                stage_stats["ctc_loss"] = ctc_loss
                
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats,
            )
            # You can add your custom checkpointing logic here
                # e.g., self.checkpointer.save_and_keep_only(meta={"PER": stage_stats['ctc_per']}, min_keys=["PER"])
            if epoch % self.hparams.valid_search_interval == 0:

                improved = False
                ckpt_name = f"{epoch:03d}_CTC_PER_{per_ctc:.4f}_LLM_PER_{per_llm:.4f}.ckpt"
                self.checkpointer.save_and_keep_only(meta={"CTC_PER": per_ctc, "LLM_PER": per_llm},
                                                    name=ckpt_name,
                                                    num_to_keep=2,
                                                    importance_keys=[
                                                        lambda ckpt: (
                                                            -ckpt.meta["CTC_PER"],  
                                                            -ckpt.meta["LLM_PER"],  
                                                        )
                                                    ]
                                                )
                if stage_loss < self.best_valid_loss or len(self.best_valid_loss_list) < 10:
                    ckpt_name = f"best_valid_loss_{epoch:03d}_{stage_loss:.4f}.ckpt"
                    # Do NOT save checkpoint for valid loss (just update stats)
                    self.best_valid_loss_list.append((stage_loss, epoch, ckpt_name))
                    self.best_valid_loss_list = sorted(self.best_valid_loss_list, key=lambda x: x[0])[:10]
                    self.best_valid_loss = self.best_valid_loss_list[0][0]
                    improved = True

                if improved:
                    self.no_improve_epochs = 0
                    self.last_improved_epoch = epoch
                else:
                    self.no_improve_epochs += 1
            
            wandb.log({
                f"{stage.name.lower()}_loss": stage_loss,
                f"{stage.name.lower()}_ctc_per": per_ctc,
                f"{stage.name.lower()}_llm_per": per_llm,
                f"{stage.name.lower()}_llm_loss": llm_loss,
                f"{stage.name.lower()}_ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None,
            }, step=epoch)
            if self.no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                raise StopIteration
        
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta=stage_stats, message="Test results"
            )
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")  # 添加LLM PER计算
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per_ctc, 
                            "PER_seq": per_llm, 
                            "llm_loss": llm_loss,
                            "ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None},
                message="Final test results",
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nCTC PER stats:\n")
                self.per_metrics.write_stats(w)
            if not hasattr(self.hparams, 'per_seq_file'):
                self.hparams.per_seq_file = self.hparams.per_file.replace('.txt', '_llm.txt')
            with open(self.hparams.per_seq_file, "w") as w:
                w.write("LLM loss stats:\n")
                self.llm_metrics.write_stats(w)
                w.write("\nLLM PER stats:\n")
                self.llm_per_metrics.write_stats(w)
            
            
    def check_gradients(self, loss):
        """Check if gradients are finite"""
        if not torch.isfinite(loss):
            print("Warning: loss is not finite, skipping step")
            return False
        return True

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """

        if self.hparams.auto_mix_prec:
            self.pretrained_opt_class.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation and scale for mixed precision
            self.scaler.scale(loss / self.hparams.gradient_accumulation).backward()
            self.scaler.unscale_(self.pretrained_opt_class)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                # 梯度裁剪
                if hasattr(self.hparams, 'grad_clip') and self.hparams.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.pretrained_opt_class.param_groups[0]['params'] if p.requires_grad],
                        self.hparams.grad_clip
                    )
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.adam_optimizer.param_groups[0]['params'] if p.requires_grad],
                        self.hparams.grad_clip
                    )
                
                if any(p.requires_grad for p in self.pretrained_opt_class.param_groups[0]['params']):
                    self.scaler.step(self.pretrained_opt_class)
                if any(p.requires_grad for p in self.adam_optimizer.param_groups[0]['params']):
                    self.scaler.step(self.adam_optimizer)


            self.scaler.update()

        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not finite
                if self.check_gradients(loss):
                    # 梯度裁剪
                    if hasattr(self.hparams, 'grad_clip') and self.hparams.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.pretrained_opt_class.param_groups[0]['params'] if p.requires_grad],
                            self.hparams.grad_clip
                        )
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.adam_optimizer.param_groups[0]['params'] if p.requires_grad],
                            self.hparams.grad_clip
                        )
                    
                    self.pretrained_opt_class.step()
                    self.adam_optimizer.step()

                self.pretrained_opt_class.zero_grad()
                self.adam_optimizer.zero_grad()    

        return loss.detach().cpu()
    
    def generate(self, wavs, wav_lens=None, method="beam", num_beams=5, max_length=100, return_type="text"):
        """从音频生成音素序列。
        
        序列结构: [Speech] [BOS] [Text] [EOS]
        - Speech: 编码后的语音特征
        - BOS: 开始标记，同时作为语音和文本的分隔符
        - Text: 生成的音素序列
        - EOS: 结束标记
        
        Args:
            wavs (torch.Tensor): 输入音频 [B, T]
            wav_lens (torch.Tensor, optional): 音频长度 [B]
            method (str, optional): 解码方法 "greedy" 或 "beam". Defaults to "beam".
            num_beams (int, optional): 束搜索的束宽. Defaults to 5.
            max_length (int, optional): 最大生成长度. Defaults to 100.
            return_type (str, optional): 返回类型. 可选:
                - "text": 返回音素文本列表，如 ["sil ae p ax l sil", ...]
                - "list": 返回音素列表的列表，如 [["sil", "ae", "p", "ax", "l", "sil"], ...]
                - "both": 返回元组 (text_list, phoneme_lists)
            
        Returns:
            Union[List[str], List[List[str]], Tuple[List[str], List[List[str]]]]:
            根据return_type返回处理后的音素序列
            
        Example:
            >>> # 文本形式
            >>> texts = brain.generate(wavs, return_type="text")
            >>> print(texts[0])  # "sil ae p ax l sil"
            >>> 
            >>> # 列表形式
            >>> phonemes = brain.generate(wavs, return_type="list")
            >>> print(phonemes[0])  # ["sil", "ae", "p", "ax", "l", "sil"]
            >>> 
            >>> # 两种形式都要
            >>> texts, phonemes = brain.generate(wavs, return_type="both")
        """
        # 编码音频
        with torch.no_grad():
            # SSL编码并投影到LLaMA维度
            wav_feats = self.modules.perceived_ssl(wavs)
            Z = self.modules.enc(wav_feats.transpose(-2, -1))  # [B, T, H]
            Z = Z.transpose(-2, -1)
            
            # 添加BOS作为分隔符
            bos_embed = self.modules.LLM.get_input_embeddings()(
                torch.tensor([42], device=self.device)  # BOS token
            ).expand(Z.size(0), 1, -1)
            
            # ============ Prompt 支持 ============
            # 如果启用 prompt，在 Audio 和 BOS 之间插入 prompt
            if self.use_prompt and self.prompt_embeddings is not None:
                prompt_emb = self.prompt_embeddings.unsqueeze(0).expand(Z.size(0), -1, -1)
                # 构建输入序列：[Speech] [Prompt] [BOS]
                inputs_embeds = torch.cat([Z, prompt_emb, bos_embed], dim=1)
            else:
                # 原始逻辑：[Speech] [BOS]
                inputs_embeds = torch.cat([Z, bos_embed], dim=1)
            # ====================================
            
            # 对齐dtype到LLM
            llm_dtype = self.modules.LLM.get_input_embeddings().weight.dtype
            if inputs_embeds.dtype != llm_dtype:
                inputs_embeds = inputs_embeds.to(llm_dtype)
            
            # 设置生成参数
            gen_kwargs = {
                "max_length": max_length,
                "min_length": 1,
                "num_return_sequences": 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "logits_processor": [self._phoneme_logits_processor],
                "bos_token_id": 42,
                "eos_token_id": 43,
                "pad_token_id": 0,
            }
            
            if method == "beam":
                gen_kwargs.update({
                    "num_beams": num_beams,
                    "length_penalty": 1.0,
                    "early_stopping": True,
                })
            else:  # greedy
                gen_kwargs.update({
                    "do_sample": False,
                    "num_beams": 1,
                })
            
            # 生成序列
            outputs = self.modules.LLM.generate(
                inputs_embeds=inputs_embeds,
                **gen_kwargs
            )
            
            return self.process_generated_phonemes(outputs)
            
    def process_generated_phonemes(self, phoneme_ids, return_type="text"):
        """处理生成的音素ID序列。
        
        Args:
            phoneme_ids (torch.Tensor): 音素ID序列 [B, L]
            return_type (str, optional): 返回类型. 可选:
                - "text": 返回音素文本列表，如 ["sil ae p ax l sil", ...]
                - "list": 返回音素列表的列表，如 [["sil", "ae", "p", "ax", "l", "sil"], ...]
                - "both": 返回元组 (text_list, phoneme_lists)
                
        Returns:
            Union[List[str], List[List[str]], Tuple[List[str], List[List[str]]]:
            根据return_type返回处理后的音素序列
        """
        # 将tensor移到CPU并转换为numpy
        phoneme_ids = phoneme_ids.cpu().numpy()
        batch_size = phoneme_ids.shape[0]
        
        text_outputs = []
        phoneme_lists = []
        
        for i in range(batch_size):
            # 获取当前序列
            seq = phoneme_ids[i]
            
            # 移除特殊token（bos, eos, pad）并获取有效音素
            valid_phonemes = []
            for p_id in seq:
                # 跳过特殊token
                if p_id in [0, 42, 43]:  # blank, bos, eos
                    continue
                # 获取音素文本
                phoneme = self.hparams.tokenizer.id2lab[p_id]
                valid_phonemes.append(phoneme)
            
            # 保存音素列表
            phoneme_lists.append(valid_phonemes)
            # 生成音素文本（空格分隔）
            text_outputs.append(" ".join(valid_phonemes))
        
        # 根据返回类型返回结果
        if return_type == "text":
            return text_outputs
        elif return_type == "list":
            return phoneme_lists
        else:  # "both"
            return text_outputs, phoneme_lists
            
    def _phoneme_logits_processor(self, input_ids, scores):
        """处理生成的logits，只保留音素相关的token"""
        scores += self.phoneme_bias
        return scores
        
    # def on_stage_end(self, stage, stage_loss, epoch):
    #     """处理每个训练阶段结束时的操作"""
    #     if stage == sb.Stage.TRAIN:
    #         # 计算并记录训练统计数据
    #         self.train_stats = {
    #             "ctc_loss": np.mean(self.train_stats["ctc_loss"]),
    #             "ce_loss": np.mean(self.train_stats["ce_loss"]),
    #             "total_loss": np.mean(self.train_stats["total_loss"])
    #         }
            
    #         # 记录到wandb
    #         wandb.log({
    #             "train/ctc_loss": self.train_stats["ctc_loss"],
    #             "train/ce_loss": self.train_stats["ce_loss"],
    #             "train/total_loss": self.train_stats["total_loss"],
    #             "epoch": epoch
    #         })
            
    #     else:
    #         # 计算验证/测试指标
    #         ctc_metric = self.ctc_metrics.summarize()
    #         per_ctc = self.per_metrics.summarize("error_rate")
    #         per_llm = self.llm_per_metrics.summarize("error_rate")
            
    #         # 更新验证统计
    #         self.valid_stats = {
    #             "ctc_loss": np.mean(self.valid_stats["ctc_loss"]),
    #             "ce_loss": np.mean(self.valid_stats["ce_loss"]),
    #             "total_loss": np.mean(self.valid_stats["total_loss"]),
    #             "per": per_ctc
    #         }
            
    #         # 记录到wandb
    #         wandb.log({
    #             "valid/ctc_loss": self.valid_stats["ctc_loss"],
    #             "valid/ce_loss": self.valid_stats["ce_loss"],
    #             "valid/total_loss": self.valid_stats["total_loss"],
    #             "valid/per_ctc": per_ctc,
    #             "valid/per_llm": per_llm,
    #             "epoch": epoch
    #         })
            

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()
    
    def init_optimizers(self):
        # 收集需要用 adam 优化的参数（排除 SSL）
        adam_params = []
        prompt_params = []
        
        # 添加 enc, enc_ctc, ctc_lin, phn_embed, phn_head 的参数
        for module in [self.modules.enc, self.modules.enc_ctc, self.modules.ctc_lin,
                       self.modules.phn_embed, self.modules.phn_head]:
            for param in module.parameters():
                if param.requires_grad:
                    adam_params.append(param)
        
        # 添加 inputs_embeds_norm 的参数
        if self.inputs_embeds_norm is not None:
            for param in self.inputs_embeds_norm.parameters():
                if param.requires_grad:
                    adam_params.append(param)
        
        # 添加 prompt embeddings（如果启用）
        if self.use_prompt and self.prompt_embeddings is not None:
            prompt_params.append(self.prompt_embeddings)
            print(f"[Optimizer] Prompt embeddings: {self.prompt_embeddings.numel():,} 参数")
        
        # 添加 LLM 中可训练的参数（仅 LoRA）
        llm_trainable_params = []
        for name, param in self.modules.LLM.named_parameters():
            if param.requires_grad:
                llm_trainable_params.append(param)
        
        print(f"[Optimizer] Adam 基础模块: {len(adam_params)} 个参数组")
        print(f"[Optimizer] Adam 基础参数量: {sum(p.numel() for p in adam_params):,}")
        print(f"[Optimizer] LLM LoRA 参数量: {sum(p.numel() for p in llm_trainable_params):,}")
        
        # 根据是否有 prompt 使用不同的学习率
        if self.use_prompt and len(prompt_params) > 0:
            # Prompt 使用单独的学习率
            lr_prompt = getattr(self.hparams, "lr_prompt", self.hparams.lr * 10)
            self.adam_optimizer = self.hparams.adam_opt_class([
                {'params': adam_params, 'lr': self.hparams.lr},
                {'params': llm_trainable_params, 'lr': self.hparams.lr},
                {'params': prompt_params, 'lr': lr_prompt}
            ])
            print(f"[Optimizer] 使用分组学习率:")
            print(f"  - 基础模块 lr: {self.hparams.lr}")
            print(f"  - LLM LoRA lr: {self.hparams.lr}")
            print(f"  - Prompt lr: {lr_prompt}")
        else:
            # 原始逻辑
            all_params = adam_params + llm_trainable_params
            self.adam_optimizer = self.hparams.adam_opt_class(all_params)
            print(f"[Optimizer] 总参数量: {sum(p.numel() for p in all_params):,}")
        
        # SSL 模型使用单独的优化器
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)
            # 保存 prompt embeddings
            if self.use_prompt and self.prompt_embeddings is not None:
                self.checkpointer.add_recoverable("prompt_embeddings", self.prompt_embeddings)  
    