#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Embedding Debugger for SSL_LLM_MultiTarget_ver1
详细追踪 input embedding 构建的每一步，排查阿拉伯语和音素处理问题
"""

import torch
import json
from pathlib import Path

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

class InputEmbeddingDebugger:
    """详细追踪 input embedding 构建的每一步"""
    
    def __init__(self, brain, tokenizer, output_dir="./debug_outputs"):
        self.brain = brain
        self.tok = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.step_counter = 0
        
    def log(self, message, data=None):
        """打印并保存日志"""
        print(f"\n{'='*80}")
        print(f"[STEP {self.step_counter}] {message}")
        print(f"{'='*80}")
        
        if data is not None:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            
            # 保存到文件
            with open(self.output_dir / f"step_{self.step_counter:03d}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.step_counter += 1
    
    def check_prompt_template(self):
        """Step 1: 检查 Prompt Template"""
        self.log("检查 Prompt Template 构建")
        
        # 检查是否使用 prompt
        use_prompt = getattr(self.brain.hparams, "use_prompt", False)
        prompt_type = getattr(self.brain.hparams, "prompt_type", "none")
        
        info = {
            "use_prompt": use_prompt,
            "prompt_type": prompt_type,
        }
        
        if use_prompt and prompt_type in ["text", "discrete"]:
            # 检查 chat template
            if hasattr(self.brain, "prefix_ids") and hasattr(self.brain, "suffix_ids"):
                prefix_text = self.tok.decode(self.brain.prefix_ids, skip_special_tokens=False)
                suffix_text = self.tok.decode(self.brain.suffix_ids, skip_special_tokens=False)
                
                info["has_split_prompt"] = True
                info["prefix_token_count"] = len(self.brain.prefix_ids)
                info["suffix_token_count"] = len(self.brain.suffix_ids)
                info["prefix_text"] = prefix_text
                info["suffix_text"] = suffix_text
                info["prefix_tokens"] = self.brain.prefix_ids.tolist()
                info["suffix_tokens"] = self.brain.suffix_ids.tolist()
                
                # 重建完整 prompt
                full_prompt = prefix_text + "<<<SPEECH>>>" + suffix_text
                info["reconstructed_full_prompt"] = full_prompt
                
        self.log("Prompt Template Info", info)
        return info
    
    def check_arabic_tokenization(self, arabic_text):
        """Step 2: 检查阿拉伯语 tokenization"""
        self.log(f"检查阿拉伯语文本的 tokenization: {arabic_text}")
        
        # Tokenize
        tokens = self.tok(arabic_text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokens["input_ids"][0].tolist()
        
        # 逐 token 解码
        individual_tokens = []
        for tid in token_ids:
            token_str = self.tok.decode([tid], skip_special_tokens=False)
            individual_tokens.append({
                "id": tid,
                "text": token_str,
                "bytes": token_str.encode("utf-8").hex()
            })
        
        # 重新解码检查一致性
        decoded = self.tok.decode(token_ids, skip_special_tokens=False)
        
        info = {
            "original_text": arabic_text,
            "original_bytes": arabic_text.encode("utf-8").hex(),
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "individual_tokens": individual_tokens,
            "decoded_text": decoded,
            "decoded_bytes": decoded.encode("utf-8").hex(),
            "match": (arabic_text == decoded)
        }
        
        self.log("Arabic Tokenization", info)
        return info
    
    def check_phoneme_tokenization(self, phoneme_list):
        """Step 3: 检查音素序列 tokenization"""
        # from utils.functions import phn_list_to_seq
        
        phn_seq = phn_list_to_seq([phoneme_list])
        self.log(f"检查音素序列 tokenization: {phn_seq}")
        
        # Tokenize
        tokens = self.tok(phn_seq, return_tensors="pt", add_special_tokens=False)
        token_ids = tokens["input_ids"][0].tolist()
        
        # 逐 token 解码
        individual_tokens = []
        for tid in token_ids:
            token_str = self.tok.decode([tid], skip_special_tokens=False)
            individual_tokens.append({
                "id": tid,
                "text": token_str
            })
        
        # 重新解码
        decoded = self.tok.decode(token_ids, skip_special_tokens=False)
        
        info = {
            "original_phoneme_list": phoneme_list,
            "converted_sequence": phn_seq,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "individual_tokens": individual_tokens,
            "decoded_text": decoded,
            "decoded_split": decoded.split(),
            "match": (phoneme_list == decoded.split())
        }
        
        self.log("Phoneme Tokenization", info)
        return info
    
    def check_compact_ids_construction(self, wrd_ids, wrd_mask, phn_can_ids, phn_can_mask, 
                                       phn_tgt_ids, phn_tgt_mask, SEP_TGT_ID, EOS_ID):
        """Step 4: 检查 Compact IDs 构建"""
        self.log("检查 Compact Token IDs 构建")
        
        batch_idx = 0  # 检查第一个样本
        
        # 提取实际长度
        n_wrd = wrd_mask[batch_idx].sum().item()
        n_can = phn_can_mask[batch_idx].sum().item()
        n_tgt = phn_tgt_mask[batch_idx].sum().item()
        
        # 提取实际 tokens
        wrd_actual = wrd_ids[batch_idx, :n_wrd].tolist()
        can_actual = phn_can_ids[batch_idx, :n_can].tolist()
        tgt_actual = phn_tgt_ids[batch_idx, :n_tgt].tolist()
        
        # 预期的 compact sequence
        expected_compact = wrd_actual + [SEP_TGT_ID] + can_actual + [SEP_TGT_ID] + tgt_actual + [EOS_ID]
        
        # 解码每部分
        wrd_decoded = self.tok.decode(wrd_actual, skip_special_tokens=False)
        can_decoded = self.tok.decode(can_actual, skip_special_tokens=False)
        tgt_decoded = self.tok.decode(tgt_actual, skip_special_tokens=False)
        sep_decoded = self.tok.decode([SEP_TGT_ID], skip_special_tokens=False)
        eos_decoded = self.tok.decode([EOS_ID], skip_special_tokens=False)
        
        # 解码完整 compact sequence
        full_compact_decoded = self.tok.decode(expected_compact, skip_special_tokens=False)
        
        info = {
            "batch_idx": batch_idx,
            "lengths": {
                "word": n_wrd,
                "canonical": n_can,
                "target": n_tgt,
                "total_compact": len(expected_compact)
            },
            "word_tokens": {
                "ids": wrd_actual,
                "decoded": wrd_decoded
            },
            "canonical_tokens": {
                "ids": can_actual,
                "decoded": can_decoded
            },
            "target_tokens": {
                "ids": tgt_actual,
                "decoded": tgt_decoded
            },
            "special_tokens": {
                "SEP_TGT": {"id": SEP_TGT_ID, "decoded": sep_decoded},
                "EOS": {"id": EOS_ID, "decoded": eos_decoded}
            },
            "compact_sequence": {
                "ids": expected_compact,
                "decoded": full_compact_decoded,
                "visual_structure": f"[{wrd_decoded}] {sep_decoded} [{can_decoded}] {sep_decoded} [{tgt_decoded}] {eos_decoded}"
            }
        }
        
        self.log("Compact IDs Construction", info)
        return info
    
    def check_input_embeds_assembly(self, inputs_embeds, prompt_len, Ts, text_start, compact_ids, batch_idx=0):
        """Step 5: 检查 Input Embeddings 拼接"""
        self.log("检查 Input Embeddings 拼接")
        
        B, seq_len, H = inputs_embeds.shape
        
        # 计算各部分位置
        positions = {}
        if hasattr(self.brain, "prompt_prefix_embed"):
            # Split prompt 模式
            prefix_len = self.brain.prompt_prefix_embed.size(0)
            suffix_len = self.brain.prompt_suffix_embed.size(0)
            positions = {
                "prompt_prefix": (0, prefix_len),
                "speech": (prefix_len, prefix_len + Ts),
                "prompt_suffix": (prefix_len + Ts, prefix_len + Ts + suffix_len),
                "text": (prefix_len + Ts + suffix_len, seq_len)
            }
        else:
            # 普通模式
            sep_pos = prompt_len
            speech_start = sep_pos + 1
            speech_end = speech_start + Ts
            bos_pos = speech_end
            text_start_pos = bos_pos + 1
            
            positions = {
                "sep": (sep_pos, sep_pos + 1),
                "speech": (speech_start, speech_end),
                "bos": (bos_pos, bos_pos + 1),
                "text": (text_start_pos, seq_len)
            }
        
        # 提取 text region 对应的 token IDs（从 compact_ids）
        text_len = compact_ids.size(1)
        text_tokens = compact_ids[batch_idx, :text_len].tolist()
        text_decoded = self.tok.decode(text_tokens, skip_special_tokens=False)
        
        info = {
            "batch_idx": batch_idx,
            "shape": {
                "batch_size": B,
                "sequence_length": seq_len,
                "hidden_dim": H
            },
            "positions": positions,
            "text_region": {
                "token_count": len(text_tokens),
                "token_ids": text_tokens,
                "decoded": text_decoded
            },
            "embedding_dtype": str(inputs_embeds.dtype)
        }
        
        self.log("Input Embeddings Assembly", info)
        return info
    
    def check_generated_output(self, gen_tokens, batch_idx=0):
        """Step 6: 检查生成的输出"""
        self.log("检查生成的输出 (TEST stage)")
        
        gen_seq = gen_tokens[batch_idx].tolist()
        
        # 查找特殊 token 位置
        SEP_TGT_ID = self.brain.SEP_TGT_ID
        EOS_ID = self.brain.EOS_ID
        
        sep_positions = [i for i, tid in enumerate(gen_seq) if tid == SEP_TGT_ID]
        eos_positions = [i for i, tid in enumerate(gen_seq) if tid == EOS_ID]
        
        # 逐 token 解码
        individual_tokens = []
        for i, tid in enumerate(gen_seq[:100]):  # 只看前100个
            token_str = self.tok.decode([tid], skip_special_tokens=False)
            individual_tokens.append({
                "position": i,
                "id": tid,
                "text": token_str,
                "is_SEP_TGT": (tid == SEP_TGT_ID),
                "is_EOS": (tid == EOS_ID)
            })
        
        # 完整解码
        full_decoded = self.tok.decode(gen_seq, skip_special_tokens=False)
        
        # 尝试分割提取
        segments = {}
        if len(sep_positions) >= 2 and len(eos_positions) > 0:
            sep1, sep2, eos = sep_positions[0], sep_positions[1], eos_positions[0]
            
            wrd_tokens = gen_seq[:sep1]
            can_tokens = gen_seq[sep1+1:sep2]
            tgt_tokens = gen_seq[sep2+1:eos]
            
            segments = {
                "word": {
                    "position": (0, sep1),
                    "token_ids": wrd_tokens,
                    "decoded": self.tok.decode(wrd_tokens, skip_special_tokens=False)
                },
                "canonical": {
                    "position": (sep1+1, sep2),
                    "token_ids": can_tokens,
                    "decoded": self.tok.decode(can_tokens, skip_special_tokens=False)
                },
                "target": {
                    "position": (sep2+1, eos),
                    "token_ids": tgt_tokens,
                    "decoded": self.tok.decode(tgt_tokens, skip_special_tokens=False)
                }
            }
        
        info = {
            "batch_idx": batch_idx,
            "generated_length": len(gen_seq),
            "SEP_TGT_positions": sep_positions,
            "EOS_positions": eos_positions,
            "first_100_tokens": individual_tokens,
            "full_decoded": full_decoded,
            "full_decoded_length": len(full_decoded),
            "segments": segments
        }
        
        self.log("Generated Output", info)
        return info
    
    def full_diagnosis(self, batch, stage):
        """完整诊断流程"""
        self.log("="*50 + " 开始完整诊断 " + "="*50)
        
        # Step 1: Prompt Template
        self.check_prompt_template()
        
        # Step 2: 阿拉伯语 tokenization
        if hasattr(batch, "wrd") and len(batch.wrd) > 0:
            self.check_arabic_tokenization(batch.wrd[0])
        
        # Step 3: 音素 tokenization
        if hasattr(batch, "phn_list_canonical") and len(batch.phn_list_canonical) > 0:
            self.check_phoneme_tokenization(batch.phn_list_canonical[0])
        
        if hasattr(batch, "phn_list_target") and len(batch.phn_list_target) > 0:
            self.check_phoneme_tokenization(batch.phn_list_target[0])
        
        self.log("="*50 + " 基础诊断完成 " + "="*50)
