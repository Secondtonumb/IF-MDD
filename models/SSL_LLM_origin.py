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

import logging
logger = logging.getLogger(__name__)
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
    
class SSL_LLM_origin(sb.Brain):
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
        
    def _ensure_initialized(self):
        """Lazy initialization for components that need LLM to be loaded.
        This ensures inference works even without calling on_fit_start."""
        # import pdb; pdb.set_trace()
        # Initialize llm_norm if not exists
        if not hasattr(self, "llm_norm") or self.llm_norm is None:
            hidden_size = self.modules.LLM.config.hidden_size
            self.llm_norm = nn.LayerNorm(hidden_size).to(self.device)
            print(f"[Lazy Init] Created llm_norm with hidden_size={hidden_size}")
        
        # Initialize prompt_embed based on prompt_type if needed
        use_prompt = getattr(self.hparams, "use_prompt", False)
        if use_prompt and (not hasattr(self, "prompt_embed") or self.prompt_embed is None):
            prompt_type = getattr(self.hparams, "prompt_type", "soft")
            prompt_len = getattr(self.hparams, "prompt_len", 10)
            hidden_size = self.modules.LLM.config.hidden_size
            
            if prompt_type == "soft":
                # Learnable soft prompt
                init_method = getattr(self.hparams, "prompt_init", "xavier")
                self.prompt_embed = nn.Parameter(
                    torch.zeros(prompt_len, hidden_size, device=self.device)
                )
                if init_method == "xavier":
                    nn.init.xavier_uniform_(self.prompt_embed)
                elif init_method == "normal":
                    nn.init.normal_(self.prompt_embed, mean=0.0, std=0.02)
                elif init_method == "zeros":
                    pass  # Already zeros
                
                print(f"[Lazy Init] Created soft prompt: {self.prompt_embed.shape}, init={init_method}")
            
            elif prompt_type in ["text", "discrete"]:
                # Text prompt - frozen embeddings
                # User requested Llama 3 specific prompt structure
                prompt_llama3_prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a phoneme transcriber.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                prompt_llama3_suffix = "\nTranscribe the preceding speech into CMUdict phonemes.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                tok = self.hparams.LLM_tokenizer
                embed_fn = self.modules.LLM.get_input_embeddings()
                
                # Tokenize Prefix
                prefix_tokens = tok(prompt_llama3_prefix, return_tensors="pt", add_special_tokens=False).to(self.device)
                prefix_ids = prefix_tokens["input_ids"].squeeze(0)
                
                # Tokenize Suffix
                suffix_tokens = tok(prompt_llama3_suffix, return_tensors="pt", add_special_tokens=False).to(self.device)
                suffix_ids = suffix_tokens["input_ids"].squeeze(0)
                
                with torch.no_grad():
                    self.prompt_prefix_embed = embed_fn(prefix_ids)
                    self.prompt_suffix_embed = embed_fn(suffix_ids)
                
                # Keep prompt_embed not None to pass checks
                self.prompt_embed = torch.cat([self.prompt_prefix_embed, self.prompt_suffix_embed], dim=0)
                print(f"[Lazy Init] Created Llama 3 Prompts. Prefix: {self.prompt_prefix_embed.shape}, Suffix: {self.prompt_suffix_embed.shape}")
        
    def compute_forward(self, batch, stage):
        """Given an input batch it computes the model forward pass.
        
        Big-vocab causal LM approach using 128K LLaMA tokenizer.
        
        Returns:
            - p_ctc: [B, T, 41] - CTC logits 
            - ce_logits: [B, L, 128000] or None - LLM logits over big vocab
            - ce_targets: dict with target token IDs, or None
            - wav_lens: [B] - sequence lengths
        """
        # Ensure critical components are initialized (for inference compatibility)
        self._ensure_initialized()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs)

        # ===== CTC Branch =====
        wav_feats = self.modules.perceived_ssl(wavs)  # [B, T, 1024]
        
        if self.hparams.ctc_head_input == "ssl":
            ctc_logits = self.modules.ctc_lin(wav_feats)
        elif getattr(self.modules, "enc_ctc", None) and self.hparams.ctc_head_input == "enc_ctc":
            ctc_in = self.modules.enc_ctc(wav_feats)
            ctc_logits = self.modules.ctc_lin(ctc_in)
        elif self.hparams.ctc_head_input == "enc_llm":
            Z_tmp = self.modules.enc(wav_feats.transpose(-2, -1)).transpose(-2, -1)
            ctc_logits = self.modules.ctc_lin(Z_tmp)
        else:
            raise ValueError(f"Unknown ctc_head_input: {self.hparams.ctc_head_input}")
        
        p_ctc = self.hparams.log_softmax(ctc_logits)
        # If CTC-only training, skip LLM
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        if ctc_weight >= 1.0 - 1e-8:
            return p_ctc, None, None, wav_lens

        # ===== LLM Branch: Big-vocab (128K) Causal LM =====
        # Project SSL to LLM dimension
        Z = self.modules.enc(wav_feats.transpose(-2, -1))  # [B, T, H]
        Z = Z.transpose(-2, -1)  # [B, T, H]
        # import pdb; pdb.set_trace()
        # if Z.requires_grad:
        #      Z.register_hook(lambda grad: print(f"=== Z Gradient Mean: {grad.abs().mean().item():.6e} | Max: {grad.abs().max().item():.6e} ==="))
        
        B, Ts, H = Z.shape
        device = self.device
        tok = self.hparams.LLM_tokenizer
        embed_fn = self.modules.LLM.get_input_embeddings()
        
        # ===== Tokenize phoneme sequences =====
        phn_seq = phn_list_to_seq(batch.phn_list_target)
        phn_tokens = tok(phn_seq, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # phn_tokens["input_ids"]: [B, L_phn]
        # phn_tokens["attention_mask"]: [B, L_phn]
        # tok.batch_decode(phn_tokens["input_ids"], skip_special_tokens=False)
        
        phn_ids = phn_tokens["input_ids"]
        phn_mask = phn_tokens["attention_mask"]
        L_phn = phn_ids.size(1)
        
        # ===== Prepare special tokens =====
        BOS_ID = tok.bos_token_id
        EOS_ID = tok.eos_token_id
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
        
        # SEP token (use reserved special token if not defined)
        if tok.sep_token is None or tok.sep_token_id is None:
            tok.sep_token = "<|reserved_special_token_0|>"
        SEP_ID = tok.sep_token_id
        
        def col_tokens(tok_id):
            """Create [B, 1] tensor of token ID"""
            return torch.full((B, 1), tok_id, dtype=torch.long, device=device)
        
        SEP = col_tokens(SEP_ID)  # [B, 1]
        BOS = col_tokens(BOS_ID)  # [B, 1]
        EOS = col_tokens(EOS_ID)  # [B, 1]
        
        # ===== Optional prompt tuning =====
        use_prompt = getattr(self.hparams, "use_prompt", False)
        prompt_len = 0
        
        # Check if we have prompt embeddings (either soft or text)
        has_prompt = use_prompt and hasattr(self, "prompt_embed") and self.prompt_embed is not None
        
        if has_prompt:
            prompt_embed = self.prompt_embed  # [P, H] soft/text embedding
            prompt_len = prompt_embed.size(0)
            prompt_embed_batch = prompt_embed.unsqueeze(0).expand(B, -1, -1)  # [B, P, H]
        else:
            prompt_embed = None
            prompt_embed_batch = None
        
        # ===== Build input embedding sequence =====
        # Sequence structure: <prompt?> <SEP> <speech> <BOS> <phoneme_tokens> <EOS>
        # Note: we will construct labels such that speech/prompt section is masked (-100)
        
        # Embed phoneme tokens
        phn_embed = embed_fn(phn_ids)  # [B, L_phn, H]
        
        # Embed special tokens
        SEP_embed = embed_fn(SEP)  # [B, 1, H]
        BOS_embed = embed_fn(BOS)  # [B, 1, H]
        EOS_embed = embed_fn(EOS)  # [B, 1, H]
        
        # Concatenate everything
        has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")

        if has_split_prompt:
             prefix_embed = self.prompt_prefix_embed
             suffix_embed = self.prompt_suffix_embed
             inputs_embeds = torch.cat([
                prefix_embed.unsqueeze(0).expand(B, -1, -1),
                Z,
                suffix_embed.unsqueeze(0).expand(B, -1, -1),
                phn_embed,
                EOS_embed
            ], dim=1)
        elif prompt_embed is not None:
            inputs_embeds = torch.cat([
                prompt_embed_batch,  # [B, P, H]
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, P+1+Ts+1+L_phn+1, H]
        else:
            inputs_embeds = torch.cat([
                SEP_embed,          # [B, 1, H]
                Z,                   # [B, Ts, H]
                BOS_embed,          # [B, 1, H]
                phn_embed,          # [B, L_phn, H]
                EOS_embed           # [B, 1, H]
            ], dim=1)  # [B, 1+Ts+1+L_phn+1, H]
        
        # Persist LayerNorm (created in on_fit_start, not per-forward)
        # if hasattr(self, "llm_norm") and self.llm_norm is not None:
        #     inputs_embeds = self.llm_norm(inputs_embeds)
        
        # Align dtype with LLM weights
        llm_dtype = embed_fn.weight.dtype
        if inputs_embeds.dtype != llm_dtype:
            inputs_embeds = inputs_embeds.to(llm_dtype)
        
        # ===== Build attention mask =====
        seq_len = inputs_embeds.size(1)
        
        if stage == sb.Stage.TRAIN:
            # Build attention mask: 1 for valid positions, 0 for padding
            # Note: LLM internally applies causal mask, so we only mark valid/invalid positions
            attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
            
            if has_split_prompt:
                 text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            else:
                # Mask out padding in phoneme sequence
                # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                sep_pos = prompt_len
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end
                text_start = bos_pos + 1
            
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                # Mask positions after actual phonemes + EOS
                if num_phn < L_phn:
                    # text_start + num_phn is EOS position
                    # Everything after EOS should be masked (padding)
                    attention_mask[b, text_start + num_phn + 1:] = 0
                        # Ignore index for masked positions
            ignore_idx = -100
            labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            if has_split_prompt:
                 text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                 bos_pos = text_start - 1
            else:
                # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                sep_pos = prompt_len  # SEP位置
                speech_start = sep_pos + 1
                speech_end = speech_start + Ts
                bos_pos = speech_end  # BOS位置
                text_start = bos_pos + 1  # phoneme开始位置
            
            text_end = text_start + L_phn
            eos_pos = text_end  # EOS位置

            # Set phoneme labels (left-shifted for causal LM)
            # hidden[i] predicts token[i+1]
            # pdb.set_trace()
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())  # actual phoneme length
                if num_phn > 0:
                    # BOS predicts first phoneme, phonemes predict next phoneme
                    labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
                    # Last phoneme position predicts EOS
                    labels[b, text_start + num_phn - 1] = EOS_ID
            
            labels_noshift = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            for b in range(B):
                num_phn = int(phn_mask[b].sum().item())
                if num_phn > 0:
                    labels_noshift[b, bos_pos] = BOS_ID
                    labels_noshift[b, text_start : text_start + num_phn] = phn_ids[b, :num_phn]
                    labels_noshift[b, text_start + num_phn] = EOS_ID
            
            # Run forward pass
            # Note: LlamaForCausalLM internally applies causal attention mask
            # (position i can only attend to positions 0...i-1)
            # We only need to provide padding mask via attention_mask
            # mask
            # self.modules.LLM.loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)
            llm_out = self.modules.LLM(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_noshift,
                output_hidden_states=False,
                return_dict=True,
            )
            # llm_out.logits: [B, seq_len, 128000]
            # pdb.set_trace()
            loss = llm_out.loss  # CrossEntropyLoss if labels provided
            # print(f"[DEBUG] LLM Loss: {loss.item()}")
            # pdb.set_trace()
            ce_logits = llm_out.logits
            
            # Debug: verify causal masking (only run once)
            if not hasattr(self, '_causal_check_done'):
                print(f"[INFO] LLM type: {type(self.modules.LLM).__name__}")
                print(f"[INFO] LLM should use causal attention by default")
                print(f"[INFO] Logits shape: {ce_logits.shape}")
                self._causal_check_done = True
            
            # Build labels: mask out speech/prompt/SEP, keep phoneme tokens
            # Labels structure: <prompt?> <SEP> <speech> <BOS> <phoneme_tokens> <EOS>
            #                   (masked) (mask) (masked) (mask) (labels)         (label)
            
            # Ignore index for masked positions
            # ignore_idx = -100
            # labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
            
            # Calculate positions
            # if has_split_prompt:
            #      text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
            #      bos_pos = text_start - 1
            # else:
            #     # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
            #     sep_pos = prompt_len  # SEP位置
            #     speech_start = sep_pos + 1
            #     speech_end = speech_start + Ts
            #     bos_pos = speech_end  # BOS位置
            #     text_start = bos_pos + 1  # phoneme开始位置
            
            # text_end = text_start + L_phn
            # eos_pos = text_end  # EOS位置
            
            # Set phoneme labels (left-shifted for causal LM)
            # hidden[i] predicts token[i+1]
            # for b in range(B):
            #     num_phn = int(phn_mask[b].sum().item())  # actual phoneme length
            #     if num_phn > 0:
            #         # BOS predicts first phoneme, phonemes predict next phoneme
            #         labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
            #         # Last phoneme position predicts EOS
            #         labels[b, text_start + num_phn - 1] = EOS_ID
            # pdb.set_trace()

            return p_ctc, ce_logits, {"labels": labels}, wav_lens
        
        else:
            # ===== Validation/Test Stage =====
            if stage == sb.Stage.VALID:
                # Teacher-forcing (use full target): same as training
                # Build attention mask: 1 for valid positions, 0 for padding
                # Note: LLM internally applies causal mask, so we only mark valid/invalid positions
                
                attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                else:
                    # Mask out padding in phoneme sequence
                    # Sequence: [prompt(P)] [SEP(1)] [speech(Ts)] [BOS(1)] [phn(L_phn)] [EOS(1)]
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    # Mask positions after actual phonemes + EOS
                    if num_phn < L_phn:
                        # text_start + num_phn is EOS position
                        # Everything after EOS should be masked (padding)
                        attention_mask[b, text_start + num_phn + 1:] = 0
                
                llm_out = self.modules.LLM(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                ce_logits = llm_out.logits
                
                # Labels same as training
                ignore_idx = -100
                labels = torch.full((B, seq_len), ignore_idx, dtype=torch.long, device=device)
                
                # Calculate positions (same as training)
                if use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed"):
                    text_start = self.prompt_prefix_embed.size(0) + Ts + self.prompt_suffix_embed.size(0)
                    bos_pos = text_start - 1
                else:
                    sep_pos = prompt_len
                    speech_start = sep_pos + 1
                    speech_end = speech_start + Ts
                    bos_pos = speech_end
                    text_start = bos_pos + 1
                
                for b in range(B):
                    num_phn = int(phn_mask[b].sum().item())
                    if num_phn > 0:
                        labels[b, bos_pos : text_start + num_phn - 1] = phn_ids[b, : num_phn]
                        labels[b, text_start + num_phn - 1] = EOS_ID
                
                return p_ctc, ce_logits, {"labels": labels}, wav_lens
            
            elif stage == sb.Stage.TEST:
                # ===== Autoregressive generation (no teacher forcing) =====
                # Only provide: [prompt?] [SEP] [speech] [BOS]
                # Model generates: [phn[0]] [phn[1]] ... [EOS]
                has_split_prompt = use_prompt and hasattr(self, "prompt_prefix_embed") and hasattr(self, "prompt_suffix_embed")
                
                if has_split_prompt:
                     prefix_embed = self.prompt_prefix_embed
                     suffix_embed = self.prompt_suffix_embed
                     inputs_embeds_inference = torch.cat([
                        prefix_embed.unsqueeze(0).expand(B, -1, -1),
                        Z,
                        suffix_embed.unsqueeze(0).expand(B, -1, -1)
                    ], dim=1)
                
                elif prompt_embed_batch is not None:
                    inputs_embeds_inference = torch.cat([
                        prompt_embed_batch,  # [B, P, H]
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, P+1+Ts+1, H]
                else:
                    inputs_embeds_inference = torch.cat([
                        SEP_embed,          # [B, 1, H]
                        Z,                   # [B, Ts, H]
                        BOS_embed           # [B, 1, H]
                    ], dim=1)  # [B, 1+Ts+1, H]
                
                # if hasattr(self, "llm_norm") and self.llm_norm is not None:
                #     inputs_embeds_inference = self.llm_norm(inputs_embeds_inference)
                if inputs_embeds_inference.dtype != llm_dtype:
                    inputs_embeds_inference = inputs_embeds_inference.to(llm_dtype)
                
                attention_mask_inference = torch.ones(B, inputs_embeds_inference.size(1), dtype=torch.long, device=device)
                
                # print(f"[DEBUG] Generate input shape: {inputs_embeds_inference.shape}")
                # print(f"[DEBUG] max_new_tokens: {L_phn + 10}, BOS_ID: {BOS_ID}, EOS_ID: {EOS_ID}, PAD_ID: {PAD_ID}")
                # print(f"[DEBUG] Using full 128K vocab (no constraint)")
                
                # Generate phoneme tokens with full vocabulary
                # import pdb; pdb.set_trace()
                # pdb.set_trace()
                gen_out = self.modules.LLM.generate(
                    inputs_embeds=inputs_embeds_inference,
                    attention_mask=attention_mask_inference,
                    num_return_sequences=1,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    do_sample=True,
                    use_cache=True,
                    max_new_tokens=100, 
                    top_k = 100,
                    top_p = 0.9,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                )
                # from matplotlib import pyplot as plt
                # plt.imshow(inputs_embeds_inference[0].cpu().numpy(), aspect='auto')
                # plt.savefig('input_embed.png')
                # plt.imshow(attention_mask_inference.cpu().numpy(), aspect='auto')
                # plt.savefig('attention_mask.png')
                # max_new_tokens=L_phn + 10,  # 增加生成长度

                # gen_out: [B, generated_len] 
                # 注意：当使用 inputs_embeds 时，generate() 只返回新生成的 tokens，不包含 prompt
                # print(f"[DEBUG] Generate output shape: {gen_out.shape}")
                # print(f"[DEBUG] First sample tokens: {gen_out[0].tolist()}")
                
                # 使用 inputs_embeds 时，generate 已经只返回新生成的部分
                gen_tokens = gen_out  # [B, generated_len]
                # pdb.set_trace()
                
                # print(f"[DEBUG] Generated tokens shape: {gen_tokens.shape}")
                # print(f"[DEBUG] Generated tokens[0]: {gen_tokens[0].tolist()}")
                # if gen_tokens.size(1) > 0:
                #     print(f"[DEBUG] Decoded: {self.hparams.LLM_tokenizer.decode(gen_tokens[0], skip_special_tokens=False)}")
                # else:
                #     print(f"[DEBUG] Decoded: <empty>")
                
                # Return for metrics calculation
                return p_ctc, None, {"generated_ids": gen_tokens, "target_phonemes": batch.phn_list_target}, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute training objectives: CTC loss + LLM loss (big-vocab, 128K tokens)"""
        ids = batch.id
        wavs, wav_lens = batch.sig
        targets, target_lens = batch.phn_encoded_target  # CTC targets (phoneme indices)
        
        # Unpack predictions from compute_forward
        if len(predictions) == 4:
            p_ctc, ce_logits, ce_targets, lens_for_ctc = predictions
        else:
            raise ValueError(f"Expected 4 values from compute_forward, got {len(predictions)}")
        
        # ===== CTC Loss =====
        T = p_ctc.size(1)
        clipped_target_lens = torch.minimum(target_lens, torch.full_like(target_lens, T))
        # import pdb; pdb.set_trace()
        loss_ctc = self.hparams.ctc_cost(p_ctc.float(), targets, lens_for_ctc, clipped_target_lens)
        
        # ===== LLM Loss (big-vocab CrossEntropyLoss) =====
        loss_ce = torch.tensor(0.0, device=self.device)
        
         # --- 1. Compute CE Loss (TRAIN & VALID) ---
        if stage != sb.Stage.TEST:
            if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                labels = ce_targets["labels"]  # [B, seq_len]
                if labels is not None:
                    B, seq_len, vocab_size = ce_logits.shape
                    ignore_idx = -100
                    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx)
                    loss_ce = ce_loss_fn(
                        ce_logits.reshape(-1, vocab_size),
                        labels.reshape(-1)
                    )

        # --- 2. Compute Metrics (VALID & TEST) ---
        if stage != sb.Stage.TRAIN:
            # 2.1 CTC Metrics (Common)
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

            # 2.2 LLM Metrics (Stage Dependent)
            if stage == sb.Stage.VALID:
                # VALID: Teacher-Forcing Evaluation
                # Use logits to compute perplexity-like metrics and greedy decoding accuracy
                if ce_logits is not None and isinstance(ce_targets, dict) and "labels" in ce_targets:
                    labels = ce_targets["labels"]
                    
                    llm_predictions = ce_logits.argmax(dim=-1)
                    p_llm = F.log_softmax(ce_logits, dim=-1)
                    
                    valid_mask = (labels != -100)
                    B = labels.size(0)
                    for b in range(B):
                        valid_pos = valid_mask[b]
                        num_valid = valid_pos.sum().item()
                        if num_valid > 0:
                            # Log Probablity Metric
                            self.llm_metrics.append(
                                ids=[ids[b]],
                                log_probabilities=p_llm[b, valid_pos, :].unsqueeze(0),
                                targets=labels[b, valid_pos].unsqueeze(0),
                                length=torch.tensor([num_valid], device=labels.device)
                            )
                            # PER Metric (Teacher Forced)
                            pred_tokens = llm_predictions[b, valid_pos]
                            valid_labels = labels[b, valid_pos]
                            
                            pred_text = self.hparams.LLM_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                            target_text = self.hparams.LLM_tokenizer.decode(valid_labels, skip_special_tokens=True)
                            
                            self.llm_per_metrics.append(
                                ids=[ids[b]],
                                predict=[pred_text.split()],
                                target=[target_text.split()],
                                predict_len=None,
                                target_len=None,
                                ind2lab=lambda x: x 
                            )
                            
            elif stage == sb.Stage.TEST:
                # TEST: Autoregressive Generation Evaluation
                # Use generated IDs to compute real PER
                if isinstance(ce_targets, dict) and "generated_ids" in ce_targets:
                    gen_ids = ce_targets["generated_ids"]
                    
                    # 1. Decode Hypotheses (Generated)
                    hyps = self.hparams.LLM_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    
                    # 2. Decode References (Targets)
                    refs = phn_list_to_seq(batch.phn_list_target)
                    
                    # 3. Compute PER
                    self.llm_per_metrics.append(
                        ids=ids,
                        predict=[hyp.split() for hyp in hyps],
                        target=[ref.split() for ref in refs],
                        predict_len=None,
                        target_len=None,
                        ind2lab=lambda x: x
                    )

        # ===== Combined Loss =====
        ctc_weight = getattr(self.hparams, "ctc_weight", 0.3)
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_ce
        
        # ===== Logging =====
        if stage == sb.Stage.TRAIN:
            self.train_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.train_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.train_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        else:
            self.valid_stats.setdefault("ctc_loss", []).append(float(loss_ctc.detach().cpu()))
            self.valid_stats.setdefault("ce_loss", []).append(float(loss_ce.detach().cpu()))
            self.valid_stats.setdefault("total_loss", []).append(float(loss.detach().cpu()))
        
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
            
            # Check if LLM metrics have data before summarizing (避免空数据错误)
            per_llm = None
            llm_loss = None
            
            try:
                per_llm = self.llm_per_metrics.summarize("error_rate")
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] LLM PER metrics are empty: {type(e).__name__}: {e}")
            
            try:
                llm_loss = self.llm_metrics.summarize("average")
            except (ZeroDivisionError, ValueError, IndexError, RuntimeError) as e:
                print(f"[Warning] LLM loss metrics are empty: {type(e).__name__}: {e}")
            
            ctc_loss = self.ctc_metrics.summarize("average")
            
            test_stats = {"loss": stage_loss, "PER": per_ctc}
            if per_llm is not None:
                test_stats["PER_seq"] = per_llm
            if llm_loss is not None:
                test_stats["llm_loss"] = llm_loss
            if hasattr(self.modules, "ctc_lin"):
                test_stats["ctc_loss"] = ctc_loss
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )
            with open(self.hparams.per_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nCTC PER stats:\n")
                self.per_metrics.write_stats(w)
            
            # Only write LLM stats if data exists
            if llm_loss is not None or per_llm is not None:
                if not hasattr(self.hparams, 'per_seq_file'):
                    self.hparams.per_seq_file = self.hparams.per_file.replace('.txt', '_llm.txt')
                with open(self.hparams.per_seq_file, "w") as w:
                    if llm_loss is not None:
                        w.write("LLM loss stats:\n")
                        self.llm_metrics.write_stats(w)
                    if per_llm is not None:
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
    
    # def load_pretrained_components(self, checkpoint_path, components_to_load=None, freeze_loaded=True):
    #     """
    #     Load specific components from a pretrained model checkpoint
        
    #     Args:
    #         checkpoint_path (str): Path to the checkpoint directory or file
    #         components_to_load (list): List of components to load. 
    #                                  Options: ['ssl'] for SSL model only,
    #                                           ['ssl', 'ctc_branch'] for SSL + CTC branch, mainly for CTC per
    #                                  If None, loads ['ssl'] by default
    #         freeze_loaded (bool): Whether to freeze the loaded components
    #     """
    #     if components_to_load is None:
    #         components_to_load = ['ssl']  # Default: load SSL 
    #         logger.info("No components specified for loading, defaulting to ['ssl']")
        
    #     logger.info(f"\n🔄 Loading pretrained components from: {checkpoint_path}")
    #     logger.info(f"   Components to load: {components_to_load}")
        
    #     from speechbrain.utils.parameter_transfer import Pretrainer
        
    #     if "ctc_branch" in components_to_load:
    #         logger.info("⏳ Loading pretrained SSL model and CTC branch from %s", checkpoint_path)
    #         pretrainer = Pretrainer(
    #             collect_in=self.hparams.pretrained_model_path, 
    #             loadables={
    #                 "perceived_ssl":     self.modules.perceived_ssl,
    #                 "ctc_branch":     self.hparams.ctc_branch,
    #             },
    #             paths={
    #                 "perceived_ssl":     "perceived_ssl.ckpt",
    #                 "ctc_branch":   "model.ckpt",
    #             },
    #             custom_hooks={}
    #         )
    #     else:
    #         logger.info("⏳ Loading pretrained SSL model only from %s", checkpoint_path)
    #         pretrainer = Pretrainer(
    #             collect_in=self.hparams.pretrained_model_path, 
    #             loadables={
    #                 "perceived_ssl":     self.modules.perceived_ssl,
    #             },
    #             paths={
    #                 "perceived_ssl":     "perceived_ssl.ckpt",
    #             },
    #             custom_hooks={}
    #         )
    #     # DONE
    #     paths = pretrainer.collect_files(default_source=self.hparams.pretrained_model_path)

    #     # Check SSL
    #     # before = self.modules.perceived_ssl.state_dict()["model.encoder.layers.23.final_layer_norm.weight"]
    #     # Check CTC head
    #     # before = self.hparams.ctc_branch[1].state_dict()["w.weight"]
    #     # self.modules.ctc_lin.state_dict()['w.weight']
        
    #     pretrainer.load_collected()
        
    #     # Freeze loaded components if requested
    #     if freeze_loaded:
    #         for component in components_to_load:
    #             if component == 'ssl':
    #                 for param in self.modules.perceived_ssl.parameters():
    #                     param.requires_grad = False
    #                 self.ssl_frozen = True
    #                 logger.info("   🔒 SSL model frozen")
                    
    #             elif component == 'ctc_branch':
    #                 for param in self.hparams.ctc_branch.parameters():
    #                     param.requires_grad = False
    #                 self.ctc_branch_frozen = True
    #                 logger.info("   🔒 CTC branch frozen")
        
    #             elif component == 'encoder':
    #                 for param in self.modules.TransASR.encoder.parameters():
    #                     param.requires_grad = False
    #                 if hasattr(self.modules.TransASR, 'custom_src_module'):
    #                     for param in self.modules.TransASR.custom_src_module.parameters():
    #                         param.requires_grad = False
    #                 self.encoder_frozen = True
    #                 logger.info("   🔒 Encoder frozen")
                    
    #             elif component == 'enc':
    #                 if hasattr(self.modules, 'enc'):
    #                     for param in self.modules.enc.parameters():
    #                         param.requires_grad = False
    #                     print("   🔒 Encoder projection frozen")
                        
    #             elif component == 'ctc_head':
    #                 for param in self.modules.ctc_lin.parameters():
    #                     param.requires_grad = False
    #                 logger.info("   🔒 CTC head frozen")
    
    #     # print(f"   ✅ Successfully loaded components: {loaded_components}")
    #     # return loaded_components

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

        # ===== NOTE: Initialization moved to _ensure_initialized() =====
        # This allows the model to work in inference mode without calling fit()
        # The lazy initialization happens on first forward pass
        # 
        # Previously initialized here:
        # - self.llm_norm (LayerNorm for LLM hidden states)
        # - self.prompt_embed (soft or text prompt embeddings)
        # 
        # Now these are created in _ensure_initialized() which is called by compute_forward()
        
        # Initialize optimizers (happens during training only)
        self.init_optimizers()
        
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                min_key="LLM_PER",
            )
        if getattr(self.hparams, 'load_pretrained_components', False):
            pretrained_path = getattr(self.hparams, 'pretrained_model_path', '')
            components = getattr(self.hparams, 'components_to_load', ['ssl'])
            freeze_loaded = getattr(self.hparams, 'freeze_loaded_components', True)
        
        import os
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                self.load_pretrained_components(
                    checkpoint_path=pretrained_path,
                    components_to_load=components,
                    freeze_loaded=freeze_loaded
                )
            except Exception as e:
                print(f"❌ Failed to load pretrained components: {e}")
                print("   Continuing with random initialization...")
        else:
            print(f"⚠️  Pretrained model path not found: {pretrained_path}")
            print("   Continuing with random initialization...")

    
    def init_optimizers(self):
        # Collect all trainable parameters
        model_params = list(self.hparams.model.parameters())
        
        # Add prompt embeddings if they exist AND are learnable parameters
        # (Only soft prompts are nn.Parameter, text prompts are plain tensors)
        if hasattr(self, "prompt_embed") and self.prompt_embed is not None:
            if isinstance(self.prompt_embed, torch.nn.Parameter):
                model_params.append(self.prompt_embed)
                print(f"[Optimizer] Added soft prompt embeddings to optimizer")
            else:
                print(f"[Optimizer] Text prompt embeddings are frozen (not added to optimizer)")
        
        # Add LayerNorm parameters
        # if hasattr(self, "llm_norm") and self.llm_norm is not None:
        #     model_params.extend(self.llm_norm.parameters())
        
        # import pdb; pdb.set_trace()
        # Create optimizer with all model parameters
        self.adam_optimizer = self.hparams.adam_opt_class(
            model_params, 
        )
        
        # SSL encoder optimizer (separate)
        self.pretrained_opt_class = self.hparams.pretrained_opt_class(
            self.modules.perceived_ssl.parameters(), 
        )
        
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
            self.checkpointer.add_recoverable("pretrained_opt", self.pretrained_opt_class)
            self.checkpointer.add_recoverable("tokenizer", self.label_encoder)  
    
    


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