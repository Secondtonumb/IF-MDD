# Mono_SSL_CTC model
python train.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc

# Evaluation
python evaluate.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc \
        --save_folder "<parent_of_save_ckpt_path>" 

# Transformer MDD with error classifiation head
python train.py \
        hparams/transformer_mispro_cls.yaml \
        --prefix fuse_2_encdec_conf_RoPE_fromprectc_frz_errcls\
        --feature_fusion TransformerMDD_TP_encdec_errclass \
        --fuse_enc_or_dec encdec \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path "<pretrained_ctc_model_path>" \
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True \
        --plot_attention True \
        --plot_attention_interval 5

# Evaluation
python evaluate.py \
        hparams/transformer_mispro_cls.yaml \
        --prefix fuse_2_encdec_conf_RoPE_fromprectc_frz_errcls\
        --feature_fusion TransformerMDD_TP_encdec_errclass \
        --fuse_enc_or_dec encdec \
        --encoder_module conformer \
        --save_folder "<parent_of_save_ckpt_path>"

python inference.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc