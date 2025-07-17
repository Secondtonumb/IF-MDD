import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import speechbrain as sb
import os
from peft import PeftModel

# --- Add SpeechBrain YAML/dataloader imports and dataio_prep from train.py ---
import sys
from hyperpyyaml import load_hyperpyyaml
# add path to your train.py if necessary
import os
sys.path.append(os.path.expanduser("~/MDD/mpl-mdd"))
from train import dataio_prep, dataio_prep_for_llm
from speechbrain.dataio.dataloader import SaveableDataLoader
from transformers.models.qwen2_audio import processing_qwen2_audio
# Load hyperparameters (pass path to your YAML as first argument)
hparams_file = sys.argv[1]
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin)
from torch.nn.utils.rnn import pad_sequence

import wandb

from speechbrain.utils.edit_distance import wer_details_for_batch

from tqdm import tqdm

# pad_collate_fn 只负责收集，不做 pad
def pad_collate_fn(batch):
    return {
        'canonical_prompt': [item['phn_list_canonical'] for item in batch],
        'perceived_prompt': [item['phn_list_perceived']   for item in batch],
        'word':             [item['wrd']                 for item in batch],
        'audio':            [item['sig'].cpu().numpy()   for item in batch],  # list of 1-D numpy
    }

# Prepare datasets using train.py’s pipeline
train_data, valid_data, test_data, label_encoder = dataio_prep_for_llm(hparams)

# train_data = [x for x in train_data][0: 100]  # For quick testing, limit to 100 samples
# valid_data = [x for x in valid_data][0: 20]  # For quick testing, limit to 100 samples
# test_data = [x for x in test_data][0: 10]    # For

# Create a SpeechBrain dataloader for training
# 然后在初始化 DataLoader 时传入这个 collate_fn：
train_dataloader = SaveableDataLoader(
    train_data,
    collate_fn=pad_collate_fn,
    **hparams["train_dataloader_opts"]
)

valid_dataloader = SaveableDataLoader(
    valid_data,
    collate_fn=pad_collate_fn,
    **hparams["valid_dataloader_opts"]
)

test_dataloader = SaveableDataLoader(
    test_data,
    collate_fn=pad_collate_fn,
    **hparams["test_dataloader_opts"]
)

# Number of epochs from config
num_epochs = hparams["epoch_counter"].limit


class AudioEncoder(nn.Module):
    def __init__(self, llm_dim=4096, ctc_vocab_size=50):
        super().__init__()
        # 主干使用预训练 Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        hidden_size = self.wav2vec.config.hidden_size  # 通常 768
        # 提示投影
        self.proj = nn.Linear(hidden_size, llm_dim)
        # 可选下采样
        self.downsample = nn.Conv1d(llm_dim, llm_dim, kernel_size=3, stride=2, padding=1)
        # CTC 分支
        self.ctc_proj = nn.Linear(hidden_size, ctc_vocab_size)

    def forward(self, input_values):
        # input_values: (B, T)
        outputs = self.wav2vec(input_values, return_dict=True)
        C = outputs.last_hidden_state  # (B, T', hidden_size)
        # CTC 分支
        ctc_logits = self.ctc_proj(C)  # (B, T', vocab)
        # 提示分支
        P = self.proj(C)               # (B, T', llm_dim)
        # 下采样
        P = P.transpose(1,2)           # -> (B, llm_dim, T')
        P = self.downsample(P)         # -> (B, llm_dim, T'' )
        P = P.transpose(1,2)           # -> (B, T'', llm_dim)
        return P, ctc_logits
    
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig

# Load Qwen2-Audio 7B multimodal model and processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
llm = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True, device_map="auto"
)

# 应用 LoRA 适配
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], 
    lora_dropout=0.05, bias="none"
)
llm = get_peft_model(llm, peft_config)

# Initialize Weights & Biases
wandb.init(project=hparams.get("project_name", "qwen2_audio_phoneme_ft"), config=hparams)
wandb.watch(llm, log="all")

# Directory to save and resume checkpoints
output_dir = hparams.get("qwen_output_dir", "./checkpoints")
os.makedirs(output_dir, exist_ok=True)
resume_checkpoint = hparams.get("resume_from_checkpoint", None)
if resume_checkpoint and os.path.isdir(resume_checkpoint):
    # Load LoRA adapters
    llm = PeftModel.from_pretrained(llm, resume_checkpoint)
    print(f"Resumed LoRA adapters from {resume_checkpoint}")
    # Load audio_encoder and optimizer if checkpoint contains them
    ckpt_audio = os.path.join(resume_checkpoint, "audio_encoder.pt")
    ckpt_opt   = os.path.join(resume_checkpoint, "optimizer.pt")
    if os.path.isfile(ckpt_audio):
        audio_encoder.load_state_dict(torch.load(ckpt_audio, map_location=llm.device))
        print("Loaded audio_encoder state from checkpoint.")
    if os.path.isfile(ckpt_opt):
        optimizer.load_state_dict(torch.load(ckpt_opt, map_location=llm.device))
        print("Loaded optimizer state from checkpoint.")

# Freeze all non-LoRA parameters
for name, param in llm.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False

import torch.nn.functional as F

audio_encoder = AudioEncoder(llm_dim=4096, ctc_vocab_size=len(label_encoder))


optimizer = torch.optim.AdamW([
    {"params": audio_encoder.parameters(), "lr": 1e-4},
    {"params": llm.parameters(), "lr": 1e-5}
])
lambda_ctc = 1.0

# pad_collate_fn 只负责收集，不做 pad
def pad_collate_fn(batch):
    return {
        'canonical_prompt': [item['phn_list_canonical'] for item in batch],
        'perceived_prompt': [item['phn_list_perceived']   for item in batch],
        'word':             [item['wrd']                 for item in batch],
        'audio':            [item['sig'].cpu().numpy()   for item in batch],  # list of 1-D numpy
    }
            
best_valid_per = float('inf')
for epoch in range(num_epochs):
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        # 1) 为每个样本构造 prompt 列表
        prompts = []
        for can, perc, wrd in zip(batch["canonical_prompt"],
                                  batch["perceived_prompt"],
                                  batch["word"]):
            # 仅提示“听音频输出音素”
            prompts.append("<|AUDIO|>, recognize this and return the phoneme sequence")

        # 2) 批量化地做动态 padding
        inputs = processor(
            text=prompts,            # list[str]
            audios=batch["audio"],   # list[np.ndarray]
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            is_split_into_words=True,
        )
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}

        # 3) 构造 labels_full：同样是 (B, seq_len) 大小，全初始化为 -100
        input_ids = inputs["input_ids"]          # (B, L)
        labels_full = torch.full_like(input_ids, -100)

        # 4) 为每条样本填入它自己的 perceived_phoneme
        #    tokenizer(add_special_tokens=False) 保证生成的 token 串没有多余 begin/end
        #    这里的 perceived_prompt 是一个 list[str]，每个元素是一个音素序列
        
        perceived_prompt = " ".join(x for x in batch["perceived_prompt"][0])  # str
        enc = processor.tokenizer(
            perceived_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        
        perc_ids = enc.input_ids.to(llm.device)  # (B, P)
        P = perc_ids.size(1)
        # 把每行最后 P 个位置替换成对应的 perceived_ids
        labels_full[:, -P:] = perc_ids

        inputs["labels"] = labels_full

        # 5) 前向并仅对 perceived 部分计算 loss
        outputs = llm(**inputs)
        loss = outputs.loss

        # audio show decoded and calculate per

        # get predicted phoneme sequences, skip the masked tokens based on the labels
        predicted_phonemes = processor.batch_decode(
            outputs.logits.argmax(dim=-1),
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # calcuate phoneme error rate:
        # outputs.logits.argmax(dim=-1) and inputs["labels"] to get the phoneme sequences
        hyp = outputs.logits.argmax(dim=-1)
        gt = inputs["labels"]
        # mask out the -100 labels
        mask = gt != -100
        hyp = hyp[mask]
        gt = gt[mask]   
        # convert to phoneme strings
        hyp_phonemes = processor.tokenizer.batch_decode(hyp, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
        gt_phonemes = processor.tokenizer.batch_decode(gt, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)      
        # calculate phoneme error rate
        from jiwer import wer
        per = wer(gt_phonemes, hyp_phonemes)    
        
        wer_details = wer_details_for_batch(
            ids = ["dummy"],
            refs = gt_phonemes,
            hyps = hyp_phonemes,
            compute_alignments=True
        )
        wandb.log({
            "batch_PER": per,
            "batch_substitutions": wer_details[0]["substitutions"],
            "batch_deletions": wer_details[0]["deletions"],
            "batch_insertions": wer_details[0]["insertions"],
            "train_loss": loss.item(),
        }, step=epoch)
        
        # 6) 反向 + 优化 + 日志
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # validation
    # Prepare for validation
    valid_refs = []
    valid_hyps = []
    llm.eval()
    valid_loss = 0.0
    valid_per = 0.0
    valid_count = 0
    with torch.no_grad():
        for valid_batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch}"):
            # Prepare prompts for validation
            prompts = ["<|AUDIO|>, recognize this and return the phoneme sequence"] * len(valid_batch["audio"])
            inputs = processor(
                text=prompts,
                audios=valid_batch["audio"],
                return_tensors="pt",
                sampling_rate=16000,
                padding=True,
                is_split_into_words=True,
            )
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}

            # Masked labels
            input_ids = inputs["input_ids"]
            labels_full = torch.full_like(input_ids, -100)
            perceived_prompt = " ".join(x for x in valid_batch["perceived_prompt"][0])  # str
            enc = processor.tokenizer(
                perceived_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False
            )
            perc_ids = enc.input_ids.to(llm.device)
            P = perc_ids.size(1)
            labels_full[:, -P:] = perc_ids
            inputs["labels"] = labels_full

            # Forward and loss
            outputs = llm(**inputs)
            loss = outputs.loss
            valid_loss += loss.item() * input_ids.size(0)

            # Decode per-sample hypotheses and references
            hyp_ids = outputs.logits.argmax(dim=-1)
            batch_refs = []
            batch_hyps = []
            for i in range(hyp_ids.size(0)):
                mask_i = labels_full[i] != -100
                hyp_seq = processor.tokenizer.decode(hyp_ids[i][mask_i], skip_special_tokens=True)
                gt_seq = processor.tokenizer.decode(labels_full[i][mask_i], skip_special_tokens=True)
                batch_hyps.append(hyp_seq)
                batch_refs.append(gt_seq)
            # Compute batch PER
            batch_per = wer(batch_refs, batch_hyps)
            valid_per += batch_per * input_ids.size(0)
            valid_refs.extend(batch_refs)
            valid_hyps.extend(batch_hyps)

            valid_count += input_ids.size(0)

        valid_loss_avg = valid_loss / valid_count
        valid_per_avg  = valid_per  / valid_count
        # Detailed validation error counts
        wer_details = wer_details_for_batch(
            ids = ["dummy"] * len(valid_refs),  # Dummy IDs since we don't need them for WER,
            refs=valid_refs, 
            hyps =valid_hyps,
            compute_alignments=True)
        measures = wer_details
        wandb.log({
            "valid_substitutions": measures[0]["substitutions"],
            "valid_deletions": measures[0]["deletions"],
            "valid_insertions": measures[0]["insertions"],
        }, step=epoch)
    wandb.log({
        "valid_loss": valid_loss_avg,
        "valid_PER": valid_per_avg,
    }, step=epoch)

    # test evaluation
    test_refs = []
    test_hyps = []
    test_loss = 0.0
    test_per = 0.0
    test_count = 0
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc="Testing"):
            prompts = ["<|AUDIO|>, recognize this and return the phoneme sequence"] * len(test_batch["audio"])
            inputs = processor(
                text=prompts,
                audios=test_batch["audio"],
                return_tensors="pt",
                sampling_rate=16000,
                padding=True,
                is_split_into_words=True,
            )
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            labels_full = torch.full_like(input_ids, -100)
            perceived_prompt = " ".join(x for x in test_batch["perceived_prompt"][0])  # str
            enc = processor.tokenizer(
                perceived_prompt,   
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False
            )
            perc_ids = enc.input_ids.to(llm.device)
            P = perc_ids.size(1)
            labels_full[:, -P:] = perc_ids
            inputs["labels"] = labels_full

            outputs = llm(**inputs)
            loss = outputs.loss
            test_loss += loss.item() * input_ids.size(0)

            hyp_ids = outputs.logits.argmax(dim=-1)
            batch_refs = []
            batch_hyps = []
            for i in range(hyp_ids.size(0)):
                mask_i = labels_full[i] != -100
                hyp_seq = processor.tokenizer.decode(hyp_ids[i][mask_i], skip_special_tokens=True)
                gt_seq = processor.tokenizer.decode(labels_full[i][mask_i], skip_special_tokens=True)
                batch_hyps.append(hyp_seq)
                batch_refs.append(gt_seq)
            batch_per = wer(batch_refs, batch_hyps)
            test_per += batch_per * input_ids.size(0)
            test_refs.extend(batch_refs)
            test_hyps.extend(batch_hyps)
            test_count += input_ids.size(0)

        test_loss_avg = test_loss / test_count
        test_per_avg  = test_per  / test_count
        wer_details = wer_details_for_batch(
            ids= ["dummy"] * len(test_refs),  # Dummy IDs since we don't need them for WER,
            refs=test_refs, 
            hyps=test_hyps,
            compute_alignments=True)
        measures_test = wer_details
        wandb.log({
            "test_substitutions": measures_test[0]["substitutions"],
            "test_deletions": measures_test[0]["deletions"],
            "test_insertions": measures_test[0]["insertions"],
        }, step=epoch)
    wandb.log({
        "test_loss": test_loss_avg,
        "test_PER": test_per_avg,
    }, step=epoch)

    # Save checkpoint for epoch
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save LoRA adapters
    llm.save_pretrained(ckpt_dir)
    # Save audio encoder and optimizer states
    torch.save(audio_encoder.state_dict(), os.path.join(ckpt_dir, "audio_encoder.pt"))
    torch.save(optimizer.state_dict(),   os.path.join(ckpt_dir, "optimizer.pt"))
    
    # Save only the best checkpoint based on validation PER
    if epoch == 0 or valid_per_avg < best_valid_per:
        best_valid_per = valid_per_avg
        best_ckpt_dir = os.path.join(output_dir, "best_checkpoint")
        os.makedirs(best_ckpt_dir, exist_ok=True)
        # Copy current epoch's checkpoint to best_checkpoint
        import shutil
        for filename in ["audio_encoder.pt", "optimizer.pt"]:
            shutil.copy(os.path.join(ckpt_dir, filename), os.path.join(best_ckpt_dir, filename))
        llm.save_pretrained(best_ckpt_dir)
        # Conditionally upload the best checkpoint to W&B only if enabled
        if hparams.get("upload_best", False):
            artifact = wandb.Artifact("best_checkpoint", type="model")
            artifact.add_dir(best_ckpt_dir)
            wandb.log_artifact(artifact)
    # print(f"Saved checkpoint to {ckpt_dir}")