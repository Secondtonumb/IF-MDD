# IF-MDD

Official implementation of:

- [**IF-MDD: Indirect Fusion for Mispronunciation Detection and Diagnosis**](https://github.com/Secondtonumb/Secondtonumb.github.io/blob/main/docs/Geng_ICASSP_2026_final.pdf)
- [**Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for Mispronunciation Detection and Diagnosis**](https://arxiv.org/html/2604.22133v1)

For more details, check the demo:
[![Example](./fig/IF-MDD_example.png)](https://secondtonumb.github.io/publication_demo/ICASSP_2026/index.html)

## Updates

- `2026-06`: `main` now focuses on the public core research tracks from the new paper: `CROTTC`, `IF-MDD`, `IF + CROTTC`, and `LLM-MDD`.
- `2026-06`: `PPATP` is the primary documented `LLM-MDD` path in this repository.
- `2025-10`: added timestamp-aware CTC decoding in `inference.py`.
- `2025-10`: released the pretrained CTC checkpoint and inference example.

## Installation

```bash
git clone https://github.com/Secondtonumb/IF-MDD.git
cd IF-MDD
conda create -n ifmdd python=3.10 -y
conda activate ifmdd
pip install -r requirements.txt
```

Notes:

- `LLM-MDD` experiments additionally rely on `accelerate`, `bitsandbytes`, and `peft` via `requirements.txt`.
- The representative public configs use relative paths by default. For local datasets or pretrained checkpoints outside the repo, pass CLI overrides at launch time.

## Pretrained CTC Inference

Performance on L2-ARCTIC test:

| FRR  | FAR  | ER   | P     | R     | F1    | PER   |
|------|------|------|-------|-------|-------|-------|
| 6.07 | 45.08| 21.25| 60.38 | 54.92 | 57.52 | 14.30 |

```python
from huggingface_hub import hf_hub_download
import importlib.util

path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

asr_model = module.MyEncoderASR.from_hparams(
    source="Haopeng/CTC_for_IF-MDD",
    hparams_file="inference.yaml",
)
x = asr_model.transcribe_file("./examples/arctic_b0503.wav")
print(x)
```

<mark>For inference with timestamps, please refer [inference.py](./inference.py)</mark>

<details>
<summary>Check the CTC decode result with timestamps</summary>

![CTC Verbose Decoder Example](./fig/phoneme_wav.png)

</details>

## Public Research Tracks

### 1. Original IF-MDD baseline

- Baseline code: `models/phn_mono_ssl_model.py`
- Public baseline config: `hparams/phnmonossl.yaml`

```bash
python train.py hparams/phnmonossl.yaml --feature_fusion=PhnMonoSSL
```

### 2. CROTTC acoustic modeling

- Core model: `models/phn_mono_ssl_model_v3_refactored.py`
- Public config: `hparams/phnmonossl_crottc.yaml`

```bash
python train.py hparams/phnmonossl_crottc.yaml --feature_fusion=PhnMonoSSL
```

### 3. IF + CROTTC integration

- IF acoustic model: `models/phn_mono_ssl_model_v3_refactored_IF.py`
- IF sequence model: `models/Trans_IFMDD_ConPCO_ver2.py`
- Public configs:
  - `hparams_iqra/phnmonossl_crottc_confEnc_FT_IF.yaml`
  - `hparams_iqra/Trans_IFMDD_ConPCO_ver2.yaml`

```bash
python train.py hparams_iqra/phnmonossl_crottc_confEnc_FT_IF.yaml --feature_fusion=PhnMonoSSL_IF
python train.py hparams_iqra/Trans_IFMDD_ConPCO_ver2.yaml --feature_fusion=Trans_IFMDD_ConPCO_ver2
```

### 4. LLM-MDD with PPATP

- Primary public LLM model: `models/SSL_LLM_PPATP.py`
- Supporting projector: `models/projector.py`
- Public config: `hparams/SSL_LLM_Prompt_ver2_LLAMA3.2_PPATP.yaml`
- Secondary generic LLM path remains available through `models/SSL_LLM.py`

```bash
python train.py hparams/SSL_LLM_Prompt_ver2_LLAMA3.2_PPATP.yaml --feature_fusion=SSL_LLM_PPATP
```

## Public Smoke Matrix

Use the public smoke toolkit to validate the 5 public families:

```bash
bash run_scripts/public_smoke/submit_public_smoke_matrix.sh
```

Families:

- `ifmdd`
- `crottc`
- `phnmonossl_if`
- `trans_ifmdd`
- `ppatp`

## Acknowledgements

This implementation is built upon the following repositories:

- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [MPL-MDD](https://github.com/Mu-Y/mpl-mdd)
- [CTC-Attention-Mispronunciation](https://github.com/cageyoko/CTC-Attention-Mispronunciation)
