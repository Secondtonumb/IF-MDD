# IF-MDD

Official implementation of:

- [**IF-MDD: Indirect Fusion for Mispronunciation Detection and Diagnosis**](https://github.com/Secondtonumb/Secondtonumb.github.io/blob/main/docs/Geng_ICASSP_2026_final.pdf)
- [**Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for Mispronunciation Detection and Diagnosis**](https://arxiv.org/html/2604.22133v1)

For more details, check the demo:
[![Example](./fig/IF-MDD_example.png)](https://secondtonumb.github.io/publication_demo/ICASSP_2026/index.html)

## Updates

- `2026-06`: main branch now exposes the pre-FA core research tracks from the new paper: `CROTTC`, `IF-MDD`, and `LLM-MDD`.
- `2026-06`: recent forced-alignment analysis work is intentionally kept separate from `main`; use the `fa-research-integration` branch for that line of work.
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
- Some research configs still contain machine-specific dataset/checkpoint paths and should be adapted before training.

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
<summary> Check the CTC decode result with timestamps</summary>

![CTC Verbose Decoder Example](./fig/phoneme_wav.png)

</details>

## Research Tracks On `main`

### 1. Original IF-MDD release

- Baseline code: `models/phn_mono_ssl_model.py`
- Main training entry: `train.py`
- Original training examples: `run.sh`

### 2. CROTTC acoustic modeling

This track corresponds to the prompt-free acoustic modeling direction in the new paper.

- Core model: `models/phn_mono_ssl_model_v3_refactored.py`
- Main configs:
  - `hparams/phnmonossl_crottc.yaml`
  - `hparams/phnmonossl_crottc_confEnc.yaml`

Example:

```bash
python train.py hparams/phnmonossl_crottc.yaml
```

### 3. IF-MDD integration with CROTTC

This track adds indirect fusion / knowledge-transfer style supervision without using canonical prompts at inference time.

- IF acoustic model: `models/phn_mono_ssl_model_v3_refactored_IF.py`
- Encoder-decoder IF variant: `models/Trans_IFMDD_ConPCO_ver2.py`
- Supporting module: `trainer/ConPCO_TransASR.py`
- Example configs:
  - `hparams_iqra/phnmonossl_crottc_confEnc_FT_IF.yaml`
  - `hparams_iqra/Trans_IFMDD_ConPCO_ver2.yaml`

Examples:

```bash
python train.py hparams_iqra/phnmonossl_crottc_confEnc_FT_IF.yaml
python train.py hparams_iqra/Trans_IFMDD_ConPCO_ver2.yaml
```

### 4. LLM-MDD experiments

This track contains the investigation code for explicit canonical injection through LLM-based decoding/prompting.

- LLM model: `models/SSL_LLM.py`
- Loader: `trainer/AutoLLMLoader.py`
- Configs:
  - `hparams/SSL_LLM.yaml`
  - `hparams/SSL_LLM_NoPrompt.yaml`
  - `hparams/SSL_LLM_Prompt.yaml`

Example:

```bash
python train.py hparams/SSL_LLM_Prompt.yaml
```

## Scope Of This Branch

`main` is now focused on the core model families described above.

- Keep here: reusable model code, representative training configs, and README-level documentation for IF-MDD / CROTTC / IF / LLM-MDD.
- Keep out of `main`: the recent large forced-alignment analysis pipeline, bulk figures, and FA-only evaluation artifacts.

## Acknowledgements

This implementation is built upon the following repositories:

- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [MPL-MDD](https://github.com/Mu-Y/mpl-mdd)
- [CTC-Attention-Mispronunciation](https://github.com/cageyoko/CTC-Attention-Mispronunciation)
