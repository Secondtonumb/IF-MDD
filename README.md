# IF-MDD
Official implementation of the paper:
"IF-MDD: Indirect Fusion for Mispronunciation Detection and Diagnosis"

## Training Steps
1. **Step 0**: Prepare data (TODO)
2. **Step 1**: Pretrain SSL model (optional):
   - Pretrain a monolingual SSL model with CTC Head only

   ```bash
   python ver5_train.py \
           hparams/phnmonossl.yaml \
           --feature_fusion PhnMonoSSL \
           --perceived_ssl_model wavlm_large \
           --ENCODER_DIM 1024 \
           --prefix wavlm_ctc

```bash
python ver5_train.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc
```
+ step 2: Train IF-MDD model:
## Initialize from pretrained SSL model

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

## Training from scratch

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
- [SpeechBrain]
- [MPL-MDD](https://github.com/Mu-Y/mpl-mdd)
- [CTC-Attention-Mispronunciation](https://github.com/cageyoko/CTC-Attention-Mispronunciation)