#!/bin/sh
#------ qsub option --------#
#PBS -q regular-mig
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=10:00:00
#PBS -W group_list=gm64
#PBS -j oe

# source ~/.bashrc
# cd /home/m64000/work/SSL_MDD

# conda activate sb
# nvidia-smi

# # Mono_SSL_CTC model
# python train.py \
#         hparams/phnmonossl.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc

# # Evaluation
# python evaluate.py \
#         hparams/phnmonossl.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc \
#         --save_folder "<parent_of_save_ckpt_path>" 

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

# # Evaluation
# python evaluate.py \
#         hparams/transformer_mispro_cls.yaml \
#         --prefix fuse_2_encdec_conf_RoPE_fromprectc_frz_errcls\
#         --feature_fusion TransformerMDD_TP_encdec_errclass \
#         --fuse_enc_or_dec encdec \
#         --encoder_module conformer \
#         --save_folder "<parent_of_save_ckpt_path>"


# # Mono_SSL_CTC model + llama
# python train.py \
#         hparams/SSL_LLM.yaml \
#         --feature_fusion SSL_LLM \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_CTC_LLM


# python train.py hparams/SSL_LLM_Prompt.yaml \
#         --feature_fusion SSL_LLM \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_CTC_LLM_Prompt


python train.py hparams/SSL_LLM_Prompt.yaml \
        --feature_fusion SSL_LLM_origin \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_CTC_LLM_No_Prompt \
        --use_prompt False


# feature projection for CLAP conPCO features (Working!!!! )
python train.py \
        /home/kevingenghaopeng/MDD/IF-MDD/hparams/transformer_TP_ver4_fuse_errclass_ConPCO_feat_proj.yaml \
        --prefix ConPCO_work \
        --feature_fusion TransformerMDD_TP_encdec_errclass_ConPCO \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path /home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD \
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True

# CLAP focus on conpco_lambda_t_phn (variance of audio feature's distribution) 0.1 -> 0.01
python train.py \
        /home/kevingenghaopeng/MDD/IF-MDD/hparams/transformer_TP_ver4_fuse_errclass_ConPCO_feat_proj.yaml \
        --prefix ConPCO_work \
        --feature_fusion TransformerMDD_TP_encdec_errclass_ConPCO \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path /home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD \
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True \
        --conpco_lambda_t_phn 0.01

# CLAP focus on conpco_lambda_clap_t2a focus on audio to phoneme representation (0.5 -> 1)
python train.py \
        /home/kevingenghaopeng/MDD/IF-MDD/hparams/transformer_TP_ver4_fuse_errclass_ConPCO_feat_proj.yaml \
        --prefix ConPCO_t2g_1.0 \
        --feature_fusion TransformerMDD_TP_encdec_errclass_ConPCO \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path /home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD \
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True \
        --conpco_lambda_clap_t2a 1.0

# CLAP focus on conpco_lambda_clap_t2a focus on audio to phoneme representation (0.5 -> 0.2)
python train.py \
        /home/kevingenghaopeng/MDD/IF-MDD/hparams/transformer_TP_ver4_fuse_errclass_ConPCO_feat_proj.yaml \
        --prefix ConPCO_work \
        --feature_fusion TransformerMDD_TP_encdec_errclass_ConPCO \
        --encoder_module conformer \
        --load_pretrained_components True \
        --pretrained_model_path /home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD \
        --components_to_load '["ssl", "enc"]' \
        --freeze_loaded_components True \
        --conpco_lambda_clap_t2a 0.2