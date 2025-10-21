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

def phn_list_to_seq(batch):
    """
    Args:
        batch:[["sil", "aa", "x"], ["sil", "xa", "th"]]
    return
        batch ["sil aa x", "sil xa th"]
    """
    result = []
    for phn_list in batch:
        result.append(" ".join(x for x in phn_list))
    return result
    
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
        # 创建LayerNorm层用于特征归一化
        # self.embed_layer_norm = nn.LayerNorm(self.modules.LLM.config.hidden_size).to(self.device)
        
        # 将SSL模型移至正确设备
        if getattr(self.modules, "perceived_ssl", None) is not None:
            self.modules.perceived_ssl.to(self.device)
        
        # 训练追踪
        self.best_valid_loss = float('inf')
        self.best_valid_loss_list = []  # List of (valid_loss, epoch, ckpt_name)
        self.train_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": []}
        self.valid_stats = {"ctc_loss": [], "ce_loss": [], "total_loss": [], "per": []}
        
        # 创建phoneme token掩码
        self.phoneme_bias = None
        self.setup_phoneme_mask()

    def setup_phoneme_mask(self):
        """创建一个掩码，只允许生成音素相关的token"""
        if getattr(self, "phoneme_bias", None) is not None:
            return

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
        
        # 方案B 用 通用的 Language Tokenizer embed Phn
        phn_list = batch.phn_list_target
        perceived_list = batch.phn_list_perceived
        canonical_list = batch.phn_list_canonical
        phn_seq = phn_list_to_seq(phn_list)
        per_seq = phn_list_to_seq(perceived_list)
        can_seq = phn_list_to_seq(canonical_list)
        
        word_seq = batch.wrd
        # phn_seq = phn_list_to_seq(word_list)
        
        phn_seq_eos = [x + self.hparams.LLM_tokenizer.eos_token for x in phn_seq]
        
        target_tokens = self.hparams.LLM_tokenizer(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        
        target_tokens_eos = self.hparams.LLM_tokenizer(phn_seq_eos, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        
        word_tokens =  self.hparams.LLM_tokenizer(word_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        # per_tokens  = self.hparams.LLM_tokenizer(per_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        can_tokens  = self.hparams.LLM_tokenizer(can_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        # # ADD special token
        # if self.hparams.LLM_tokenizer.sep_token == None:
        #     # USE reserved special token
        #     self.hparams.LLM_tokenizer.sep_token = "<|reserved_special_token_0|>"
            
        # SEP_ID= self.hparams.LLM_tokenizer.convert_tokens_to_ids(self.hparams.LLM_tokenizer.sep_token)
        # BOS_ID= self.hparams.LLM_tokenizer.convert_tokens_to_ids(self.hparams.LLM_tokenizer.bos_token)
        # EOS_ID= self.hparams.LLM_tokenizer.convert_tokens_to_ids(self.hparams.LLM_tokenizer.eos_token)

        # SEP_emb = self.hparams.LLM.get_input_embeddings()(SEP_ID)
        # BOS_emb = self.hparams.LLM.get_input_embeddings()(BOS_ID)
        # EOS_emb = self.hparams.LLM.get_input_embeddings()(EOS_ID)
        
        target_embed = self.modules.LLM.get_input_embeddings()(target_tokens["input_ids"])
        # per_embed = self.modules.LLM.get_input_embeddings()(per_tokens["input_ids"])
        # can_embed = self.modules.LLM.get_input_embeddings()(can_tokens["input_ids"])
        
        word_embed =  self.modules.LLM.get_input_embeddings()(word_tokens["input_ids"])
        
        
        tok   = self.hparams.LLM_tokenizer
        model = self.modules.LLM                      # AutoModelForCausalLM
        device = self.device

        # 1) 准备 sep（若无则复用 reserved）
        if tok.sep_token is None:
            tok.sep_token = "<|reserved_special_token_0|>"

        SEP_ID = tok.sep_token_id
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        
        # 2) 文本转 ids（不自动加 special）
        #    target: 你的 phoneme 序列；per/can: 你需要的其他序列
        target = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        per    = tok(per_seq,  return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        can    = tok(can_seq,  return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        target_ids = target["input_ids"]    # [B, T_t]
        per_ids    = per["input_ids"]       # [B, T_p]
        can_ids    = can["input_ids"]       # [B, T_c]

        B = target_ids.size(0)

        # 3) 把特殊 token 作为一个 1-step 列向量，方便与 batch 拼接
        def col(tok_id):  # -> [B, 1]
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)

        SEP = col(SEP_ID)
        BOS = col(BOS_ID)
        EOS = col(EOS_ID)

        # 4) 直接在 ID 层面把整段拼好（示例：<text_prompt> <sep> <speech_embeds> <bos> <phoneme> <eos>）
        #    其中 speech 是连续特征，不在 ID 里拼；只把“需要 embedding 的离散部分”一起 embed。
        #    假设你还有一个 text_prompt_ids（可选），没有就去掉即可。
        # text_prompt_ids = tok(prompt_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)["input_ids"]

        # 5) 只调用一次嵌入层，把所有离散 token 变成 embedding
        embed_tokens = model.get_input_embeddings()   # same as model.model.embed_tokens
        # 例：把 per/can/target 分别 embed；或者先在 ID 层面 cat 再 embed，一次搞定
        target_emb = embed_tokens(target_ids)         # [B, T_t, H]
        per_emb    = embed_tokens(per_ids)            # [B, T_p, H]
        can_emb    = embed_tokens(can_ids)            # [B, T_c, H]
        SEP_emb    = embed_tokens(SEP)                # [B, 1,   H]
        BOS_emb    = embed_tokens(BOS)                # [B, 1,   H]
        EOS_emb    = embed_tokens(EOS)                # [B, 1,   H]

        # ce_targets = target_tokens
        ce_targets = target_tokens_eos
    
        # ce_targets = 
        # 文本嵌入
        if getattr(self.hparams, "use_seperate_phn_head", None) != None:
            txt_emb = self.modules.phn_embed(inp_ids)  # [B, L_max+1, H]
        else:
            txt_emb = target_embed
        
        # Prompt
        use_prompt = getattr(self.hparams, "use_prompt", False)
        use_word_prompt = getattr(self.hparams, "use_word", False)
        use_cano_prompt = getattr(self.hparams, "use_cano", False)
        prompt_tokens = None
        # <Text Prompt Emb> <sep> <Speech Embedding> <BOS> <Phoneme Emb> <EOS>
        if use_prompt:
            prompt_tokens = self.hparams.LLM_tokenizer.encode(prompt_2, add_special_tokens=False, return_tensors="pt").to(self.device)
            prompt_embed = self.modules.LLM.get_input_embeddings()(prompt_tokens)  # [1, P, H]
            # inputs_embeds = torch.cat([Z, prompt_embed.expand(B, -1, -1), txt_emb], dim=1)
            # inputs_embeds = torch.cat(prompt_embed.expand(B, -1, -1), Z, txt_emb, dim=1)
            prompt_embed_batch = prompt_embed.expand(B, -1, -1)
            if use_word_prompt:
                prompt_embed_batch = torch.cat([prompt_embed_batch, word_embed], dim=1)
            if use_cano_prompt:
                prompt_embed_batch = torch.cat([prompt_embed_batch, can_emb], dim=1)
                
            inputs_embeds = torch.cat(
                [
                    prompt_embed_batch,            # 如果有就放这（可选）
                    SEP_emb,
                    Z,          # 连续特征
                    BOS_emb,
                    target_emb,             # 教师强制时才拼；推理时只到 BOS
                    EOS_emb,                # 训练时 labels 需要覆盖 EOS
                ],
                dim=1,
            )
        else:
            inputs_embeds = torch.cat([Z, BOS_emb, txt_emb, EOS_emb], dim=1)  # [B, T + L_max+1, H]

        norm_layer = torch.nn.LayerNorm(H).to(self.device)
        inputs_embeds = norm_layer(inputs_embeds)
        # 将dtype对齐到LLM权重dtype，避免float/half不一致
        llm_dtype = self.modules.LLM.get_input_embeddings().weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # # 构造attention_mask：前缀全1；文本部分仅保留有效(BOS到L_b位置)为1，其余0
        # prefix_mask = torch.ones(B, Z.size(1), dtype=torch.long, device=self.device)
        # # text_mask = torch.zeros(B, L_max + 1, dtype=torch.long, device=self.device)
        
        # for b in range(B):
        #     Lb = int(abs_lens[b].item())
        #     # text_mask[b, : Lb + 1] = 1
        #     text_mask = target_tokens['attention_mask']

        # if use_prompt and prompt_tokens is not None:
        #     prompt_len = prompt_tokens.size(1)
        #     prompt_mask = torch.ones(B, prompt_len, dtype=torch.long, device=self.device)
        #     attention_mask = torch.cat([prefix_mask, prompt_mask, text_mask], dim=1)
        #     text_mask_inference = torch.zeros(text_mask.shape, device=self.device)
        #     # attention_mask_inference, mask the padded speech embeddings and the reference sequence.
        #     # attention_mask_inference = torch.cat([prefix_mask, prompt_mask, text_mask_inference], dim=1)
        #     attention_mask_inference = torch.cat([prompt_mask, prefix_mask, text_mask_inference], dim=1)
        # else:
        #     text_mask_inference = torch.zeros(text_mask.shape, device=self.device)
        #     attention_mask = torch.cat([prefix_mask, text_mask], dim=1)
        #     attention_mask_inference = torch.cat([prefix_mask, text_mask_inference], dim=1)
        tok = self.hparams.LLM_tokenizer
        device = self.device

        B = Z.size(0)
        Ts = Z.size(1)                        # speech 序列长度
        sep_len = 1                           # 你有 <SEP>
        prompt_len = 0
        if use_prompt and prompt_embed_batch is not None:
            prompt_len = prompt_embed_batch.size(1)
        

        # target_tokens['attention_mask'] 是 [B, L_max]，每行前 Lb 为1，其余0
        # 我们的文本侧真实输入是: <BOS> + target + <EOS>  → 长度 = 1 + L_max + 1
        L_max = target_tokens['attention_mask'].size(1)
        T_text = 1 + L_max + 1 

        # 计算每个样本的有效文本长度（含 BOS/EOS）
        # abs_lens: 每个样本有效 target 长度 Lb
        abs_lens = target_tokens['attention_mask'].sum(dim=1)                # [B]
        valid_text_len = 1 + abs_lens + 1                                     # [B] = BOS + Lb + EOS

        # ===== 训练用 attention_mask =====
        # 前缀 (prompt + <SEP> + speech) 全1
        prefix_len = prompt_len + sep_len + Ts
        prefix_mask = torch.ones(B, prefix_len, dtype=torch.long, device=device)

        # 文本段 mask：前 valid_text_len 为1，其余0（向量化比较，无需for）
        ar = torch.arange(T_text, device=device).unsqueeze(0).expand(B, -1)   # [B, T_text]: 0..T_text-1
        text_mask = (ar < valid_text_len.unsqueeze(1)).long()                 # [B, T_text]

        attention_mask = torch.cat([prefix_mask, text_mask], dim=1)           # [B, prefix_len + T_text]
        # import pdb; pdb.set_trace()
        # ===== 推理用 attention_mask_inference =====
        # 推理时只提供到 <BOS> 为止： [prompt? + <SEP> + speech + <BOS>]
        inference_len = prefix_len + 1   # +1 是 <BOS>
        attention_mask_inference = torch.ones(B, inference_len, dtype=torch.long, device=device)
        # pdb.set_trace()
        # 运行LLM获得hidden states
        if stage == sb.Stage.TRAIN:
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = llm_out.hidden_states[-1]  # [B, T + L_max+1, H]   
        else:
            # for 
            # inputs_embeds_inference = torch.cat([Z, prompt_embed.expand(B, -1, -1)], dim=1)
            prompt_embed_batch = prompt_embed.expand(B, -1, -1)
            if use_word_prompt:
                prompt_embed_batch = torch.cat([prompt_embed_batch, word_embed], dim=1)
            if use_cano_prompt:
                prompt_embed_batch = torch.cat([prompt_embed_batch, word_embed], dim=1)
            inputs_embeds_inference = torch.cat([prompt_embed_batch, Z, BOS_emb], dim=1)
            inputs_embeds_inference = norm_layer(inputs_embeds_inference).to(llm_dtype)
            if stage == sb.Stage.VALID:                
                llm_out = self.modules.LLM(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask_inference,
                    # attention_mas=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = llm_out.hidden_states[-1]  # [B, T + L_max+1, H]   
            elif stage == sb.Stage.TEST:
                llm_out = self.modules.LLM.generate(inputs_embeds=inputs_embeds_inference, 
                                                    attention_mask=attention_mask_inference,
                                                    max_new_tokens=text_mask.size(1),
                                                    min_new_tokens=text_mask.size(1),
                                                    num_return_sequences=1,
                                                    return_dict_in_generate=True,
                                                    output_hidden_states=True,
                                                    output_attentions=True,
                                                    pad_token_id=0,
                )
                
                steps = []
                # len(llm_out.hidden_states) == 11
                # len(llm_out.hidden_states[-1]) == 33
                for t in llm_out.hidden_states[1:]:
                    steps.append(t[-1])
                # import pdb; pdb.set_trace()
                hidden = torch.stack(steps, dim=1).squeeze(-2)
                # import pdb; pdb.set_trace()
            # duration teacher forcing
        
        # 小词表分类头
        # hidden = llm_out.hidden_states[-1]  # [B, T + L_max+1, H]   
        # import pdb; pdb.set_trace()
        # 取用于预测的隐状态：用位置 [T .. T+L_max] 来预测 [y0..y_{L-1}, EOS]
        
        T = Z.size(1)
        start_idx = T
        
        if use_prompt and prompt_embed_batch is not None:
            prompt_len = prompt_embed_batch.size(1)
            start_idx += prompt_len
        
        # ce_logits = self.modules.phn_head(pred_h.to(torch.float32))  # [B, L_max+1, 44]
        
        # 大词表分类头
        # import pdb; pdb.set_trace()
        L_mask = text_mask.size(1)
        if stage == sb.Stage.TRAIN or stage == sb.Stage.VALID:
            pred_h = hidden[:,  start_idx + 1 : start_idx + (L_max + 1), :]  # [B, L_max+1, H]
            # ce_logits = llm_out.logits[:, start_idx: start_idx + L_mask, :] # [B, <bos><text><eos>, LLM_Vocab]
            ce_logits =  llm_out.logits[:, start_idx + 1: start_idx + L_mask, :] # [B, <text><eos>, LLM_Vocab]
            # import pdb; pdb.set_trace()
            if stage == sb.Stage.VALID:
                print(self.hparams.LLM_tokenizer.decode(ce_targets['input_ids'][0]))
                print(self.hparams.LLM_tokenizer.decode(ce_logits.argmax(-1)[0]))
                # print(self.hparams.LLM_tokenizer.decode(F.log_softmax(ce_logits).argmax(-1)[0]))
            return p_ctc, ce_logits, ce_targets, wav_lens
        else:
            ce_logits = self.modules.LLM.lm_head(hidden)
            # hidden
            # Test
            print(self.hparams.LLM_tokenizer.decode(ce_targets['input_ids'][0]))
            print(self.hparams.LLM_tokenizer.batch_decode(ce_logits.argmax(-1)))
            
            hyps, _, _,_ = llm_out
            # ce_logits = llm_out
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
            try:
                loss_ce = ce_loss_fn(ce_logits.reshape(-1, Csm), ce_targets.reshape(-1))
            except:
                # pdb.set_trace()
                mask_flat = ce_targets["attention_mask"].reshape(-1).bool()
                
                loss_ce = ce_loss_fn(ce_logits.reshape(-1, Csm)[mask_flat], ce_targets["input_ids"].reshape(-1)[mask_flat])

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
            
            if ce_logits is not None:
                p_llm = F.log_softmax(ce_logits, dim=-1)
                llm_sequence = ce_logits.argmax(dim=-1)
                self.llm_metrics.append(
                    ids,
                    log_probabilities=p_llm,
                    targets=ce_targets["input_ids"],
                    length=target_lens,
                )
                
                self.llm_per_metrics.append(
                    ids=ids,
                    predict=llm_sequence,
                    target=ce_targets["input_ids"],
                    predict_len=None,
                    target_len=target_lens,
                    ind2lab=lambda ids: [s.split() for s in self.hparams.LLM_tokenizer.batch_decode(ids, skip_special_tokens=True)]
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
        elif stage == sb.Stage.VALID:
            stage_stats["epoch"] = epoch
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")  # 添加LLM PER计算
            stage_stats["ctc_per"] = per_ctc
            stage_stats["llm_per"] = per_llm  # 添加LLM PER到统计
            llm_loss = self.llm_metrics.summarize("average")
            # Summarize and log metrics
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
                                                            -ckpt.meta["LLM_PER"],  
                                                            -ckpt.meta["CTC_PER"],  
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
                stats_meta=stage_stats,
            )
            # import pdb; pdb.set_trace()
            per_ctc = self.per_metrics.summarize("error_rate")
            per_llm = self.llm_per_metrics.summarize("error_rate")  # 添加LLM PER计算
            llm_loss = self.llm_metrics.summarize("average")
            ctc_loss = self.ctc_metrics.summarize("average")
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per_ctc, 
                            "PER_seq": per_llm, 
                            "llm_loss": llm_loss,
                            "ctc_loss": ctc_loss if hasattr(self.modules, "ctc_lin") else None},
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
                # gradient clipping & early stop if loss is not fini
                
                if self.check_gradients(loss):
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
            enc_in = wav_feats.transpose(-2, -1)
            Z = self.modules.enc(enc_in)
            Z = Z.transpose(-2, -1)
            B = Z.size(0)
            H = Z.size(-1)
            # # 添加BOS作为分隔符
            # bos_embed = self.modules.LLM.get_input_embeddings()(
            #     torch.tensor([42], device=self.device)  # BOS token
            # ).expand(Z.size(0), 1, -1)
            
            # Append Text Prompt
                    # Prompt
            use_prompt = getattr(self.hparams, "use_prompt", False)
            prompt_tokens = None
            if use_prompt:
                prompt_tokens = self.hparams.LLM_tokenizer.encode(prompt_2, add_special_tokens=False, return_tensors="pt").to(self.device)
                prompt_embed = self.modules.LLM.get_input_embeddings()(prompt_tokens)  # [1, P, H]
                inputs_embeds = torch.cat([Z, prompt_embed.expand(B, -1, -1)], dim=1)
            else:
                inputs_embeds = torch.cat([Z, ], dim=1)  # [B, T + L_max+1, H]

            norm_layer = torch.nn.LayerNorm(H).to(self.device)
            inputs_embeds = norm_layer(inputs_embeds)
            
            # 对齐dtype到LLM
            llm_dtype = self.modules.LLM.get_input_embeddings().weight.dtype
            if inputs_embeds.dtype != llm_dtype:
                inputs_embeds = inputs_embeds.to(llm_dtype)
            
            # 设置生成参数
            max_length = 100
            gen_kwargs = {
                "max_length": max_length,
                "min_length": 1,
                "num_return_sequences": 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "pad_token_id": 0,
            }
                # "logits_processor": [self._phoneme_logits_processor],
            
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
            
    # def _phoneme_logits_processor(self, input_ids, scores):
    #     """处理生成的logits，只保留音素相关的token"""
    #     if getattr(self, "phoneme_bias", None) is None:
    #         self.setup_phoneme_mask()
    #     scores += self.phoneme_bias
    #     return scores
        
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
        
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                min_key="LLM_PER",
            )
    
    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters(), 
        )
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        if self.checkpointer is not None:
            # if self.hparams.perceived_ssl is not None and not self.hparams.perceived_ssl.freeze:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            # import pdb; pdb.set_trace()
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)  
    
    
    
prompt = """
You are a phoneme-ID transcriber.
You will be given an input speech segment. Please transcribe it into a sequence of phonemes. Output only the phoneme embedding only.
phoneme can only be generated from these given phoneme symbols:
sh 
ae 
l 
ay 
k 
r 
iy 
y
uw
sil 
aa 
eh 
d
w
ah
z
ch
ey
n
jh
aw
dh
s
hh
f
th
ih
m
ao
ow
v
er
g
t
uh
zh
ng
err
p
b
oy

"""

# prompt_2 = """
# You are a phoneme-ID transcriber.
# Given the preceding speech, produce a single line of integers that encodes the phoneme sequence.

# Constraints:
# - ID set = {0..43}; BOS=42; EOS=43; BLANK=0.
# - Start with 42 and end with 43.
# - Use BLANK only as an internal filler when necessary.
# - Output must be digits and spaces only (no letters, no punctuation, no extra text).
# - Exactly one space between integers.

# Return the line now:
# """


prompt_2 = """
You are a phoneme transcriber.
Given the preceding speech, produce a single line of CMUdict phoneme that encodes the phoneme sequence.
I will give you the reference word sequence.
"""

# prompt_2 = """
# You are a phoneme transcriber.
# Transcribe Speech to phonemes. Output the transcription directly without redundant content. 
# Ensure that the output is not duplicated.

# I will give you the reference word sequence and canonical phoneme sequence, you will be predicting the perceived (real) uttered phoeneme sequence.

# Example:
# WORD: Surely I will excuse you she cried.

# Now you will give us the perceived phoneme result.
# """
# canonical aligned: sil sh uh r l iy sil ay w ih l ih k s k y uw z y uw sil sh iy k r ay d sil sil