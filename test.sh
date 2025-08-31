#!/bin/sh
#------ qsub option --------#
#PBS -q regular-mig
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=10:00:00
#PBS -W group_list=gm64
#PBS -j oe

source ~/.bashrc
cd /home/m64000/work/SSL_MDD

conda activate sb
nvidia-smi

# PBS_ARRAY_INDEX 环境变量在运行时会被替换成 0,1,…,9
# weight=0.${PBS_ARRAY_INDEX}
# evaluate_keys:
# PER PER_seq mpd_f1 mpd_f1_seq

# 1) evaluate_key 列表（按需要修改顺序/内容）
# EVAL_KEYS=(PER PER_seq mpd_f1 mpd_f1_seq)

# 用任务索引在数组中取值（如果索引超过列表长度，就取取模）
# ek_idx=$(( PBS_ARRAY_INDEX % ${#EVAL_KEYS[@]} ))
# EVALUATE_KEY=${EVAL_KEYS[$ek_idx]}

# confidence_thresholds=(-0.7 -0.8 -0.9 -1.0)
# ek_idx=$(( PBS_ARRAY_INDEX % ${#confidence_thresholds[@]} ))
# confidence_threshold=${confidence_thresholds[$ek_idx]}


# # Make a array with PBS_ARRAY_INDEX to assign to different evaluate keys

# python ver5_evaluate.py \
#         hparams/transformer_TP_ver4_fuse.yaml \
#         --feature_fusion TransformerMDD_TP_encdec \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_TP_fuse_2_encdec_conformer_RoPE_frzctc_new\
#         --fuse_enc_or_dec encdec \
#         --attention_type RoPEMHA \
#         --valid_search_interval 1 \
#         --encoder_module conformer \
#         --enable_metric_freezing True \
#         --enable_ctc_freezing False \
#         --plot_attention false \
#         --number_of_epochs 100 \
#         --allow_confidence_thresholding true \
#         --confidence_threshold ${confidence_threshold} \
#         --per_file transformer_2_TP_fuse_2_encdec_conformer_RoPE_frzctc_new/per_confidence_threshold_${confidence_threshold}.txt \
#         --mpd_file transformer_2_TP_fuse_2_encdec_conformer_RoPE_frzctc_new/mpd_confidence_threshold_${confidence_threshold}.txt \
#         --evaluate_key PER_seq


python ver5_evaluate.py \
        hparams/transformer_TP_ver4_fuse.yaml \
        --feature_fusion TransformerMDD_TP_encdec \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_TP_fuse_2_encdec_conformer_frzctc_dechead_perc \
        --fuse_enc_or_dec encdec \
        --attention_type RelPosMHAXL \
        --valid_search_interval 2 \
        --encoder_module conformer \
        --enable_ctc_freezing True \
        --decoder_target perceived \
        --eval_with_silence false \
        --per_file per_wo_sil.txt \
        --mpd_file mpd_wo_sil.txt


# python ver5_evaluate.py \
#     hparams/l2arctic/Trans.yaml \
#     --prefix Transformer_6_6 \
#     --perceived_ssl_model wav2vec2_base \
#     --feature_fusion SB \
#     --ctc_weight_decode ${weight} \
#     --per_seq_file per_fuse_ctc_weight_${weight}.txt \
#     --mpd_seq_file mpd_fuse_ctc_weight_${weight}.txt

# python ver5_evaluate.py \
#     hparams/l2arctic/Conformer.yaml \
#     --prefix branchformer_6_6 \
#     --perceived_ssl_model wav2vec2_base \
#     --feature_fusion SB \
#     --encoder_module branchformer 

# python ver5_evaluate.py \
#     hparams/l2arctic/Transformer.yaml \
#     --prefix transformer_6_6_8 \
#     --perceived_ssl_model wavlm_large \
#     --feature_fusion TransformerMDD \
#     --encoder_module transformer \
#     --num_encoder_layers 6 \
#     --num_decoder_layers 6 \
#     --nhead 8 \
#     --ctc_weight 0.3 \
#     --ENCODER_DIM 1024 

# python ver5_evaluate.py \
#     hparams/l2arctic/Transducer.yaml \
#     --prefix Transducer \
#     --perceived_ssl_model wavlm_large \
#     --feature_fusion TransducerMDD \
#     --ENCODER_DIM 1024 

# # # Light Transformer with MHA embedding
# python ver5_evaluate.py \
#        hparams/l2arctic/TransformerMHA.yaml \
#        --prefix  transformer_2_2_8_MHA \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDDMHA \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5 \
#        --evaluate_key PER

# dec_ga
# python ver5_evaluate.py  \
#         hparams/transformer_TP.yaml \
#         --feature_fusion TransformerMDD_TP \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_2_8_TP_fuse_2_dec_ga \
#         --num_encoder_layers 2 \
#         --num_decoder_layers 2 \
#         --nhead 8 \
#         --fuse_enc_or_dec dec \
#         --fuse_net_layers 2 \
#         --attention_type RelPosMHAXL \
#         --enable_ctc_freezing True \
#         --evaluate_key mpd_f1

# # frz ctc
# python ver5_evaluate.py  \
#         hparams/transformer_TP.yaml \
#         --feature_fusion TransformerMDD_TP \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_2_8_TP_fuse_2_dec_ga_conformer_causal_frz_ctc \
#         --num_encoder_layers 2 \
#         --num_decoder_layers 2 \
#         --nhead 8 \
#         --fuse_enc_or_dec dec \
#         --fuse_net_layers 2 \
#         --encoder_module conformer \
#         --attention_type RelPosMHAXL \
#         --causal true \
#         --enable_ctc_freezing True \
#         --evaluate_key ${EVALUATE_KEY}

# # python
# python ver5_evaluate.py  \
#         hparams/transformer_TP.yaml \
#         --feature_fusion TransformerMDD_TP \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_2_8_TP_fuse_2_dec_ga_conf_causal_frz_metric \
#         --num_encoder_layers 2 \
#         --num_decoder_layers 2 \
#         --nhead 8 \
#         --fuse_enc_or_dec dec \
#         --fuse_net_layers 2 \
#         --encoder_module conformer \
#         --attention_type RelPosMHAXL \
#         --causal true \
#         --enable_metric_freezing True \
#         --evaluate_key ${EVALUATE_KEY}

# python ver5_evaluate.py  \
#         hparams/transformer_TP.yaml \
#         --feature_fusion TransformerMDD_TP \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_2_8_TP_fuse_2_dec_ga_new_PoPE_conf_frz_metric \
#         --num_encoder_layers 2 \
#         --num_decoder_layers 2 \
#         --nhead 8 \
#         --fuse_enc_or_dec dec \
#         --fuse_net_layers 2 \
#         --valid_search_interval 5 \
#         --attention_type RoPEMHA \
#         --encoder_module conformer \
#         --enable_metric_freezing True \
#         --evaluate_key ${EVALUATE_KEY}
