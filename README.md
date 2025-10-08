# IF-MDD
Official implementation of the paper:  
[**IF-MDD: Indirect Fusion for Mispronunciation Detection and Diagnosis**](https://github.com/Secondtonumb/Secondtonumb.github.io/blob/main/docs/Geng_ICASSP_2026_final.pdf)
[Demo](https://secondtonumb.github.io/publication_demo/ICASSP_2026/index.html)

**Update (2025-10-08):**  
We released the pretrained CTC checkpoint and an inference example.  
The model is now available on Hugging Face and can be executed in **10 lines of code**.

---

## Installation
```bash
git clone https://github.com/Secondtonumb/IF-MDD.git
cd IF-MDD
conda create -n ifmdd python=3.10 -y
conda activate ifmdd
pip install -r requirements.txt
```

## Inference (Pretrained CTC Head)

```python
from huggingface_hub import hf_hub_download
import importlib.util

# Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Transcribe

asr_model = module.MyEncoderASR.from_hparams(source="Haopeng/CTC_for_IF-MDD", hparams_file="inference.yaml")
x = asr_model.transcribe_file("./examples/arctic_b0503.wav")
print(x)
```

## Training Steps
### **Step 0**: Data Preparation (TODO)

### **Step 1**: Initialize SSL model:
You have two options:

1. **Use the released CTC model**  
   You can directly initialize with our [released CTC model](https://huggingface.co/Haopeng/CTC_for_IF-MDD/tree/main).

2. **Pretrain your own monolingual SSL model with a CTC Head**  
   Example command:

```bash
python ver5_train.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc
```
### **step 2**: Train IF-MDD 
Initialize from a pretrained SSL model:

```bash
# Transformer MDD with error classifiation head
python train.py \
        hparams/transformer_mispro_cls.yaml \
        --prefix fuse_2_encdec_conf_RoPE_fromprectc_frz_errcls\
        --feature_fusion TransformerMDD_TP_encdec_errclass \
        --fuse_enc_or_dec encdec \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path "<pretrained_model_path>" \ 
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True \
        --plot_attention True \
        --plot_attention_interval 5
```
or training from scratch

```bash
# Transformer MDD
python train.py \
        hparams/transformer_mispro_cls.yaml \
        --prefix fuse_2_encdec_conf_RoPE_fromprectc_frz_errcls\
        --feature_fusion TransformerMDD_TP_encdec_errclass \
        --fuse_enc_or_dec encdec \
        --encoder_module conformer \
```

## Acknowledgements
This implementation is built upon the following repositories:
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [MPL-MDD](https://github.com/Mu-Y/mpl-mdd)
- [CTC-Attention-Mispronunciation](https://github.com/cageyoko/CTC-Attention-Mispronunciation)
