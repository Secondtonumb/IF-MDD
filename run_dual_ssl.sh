#!/bin/sh
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=10:00:00
#PBS -W group_list=gm64
#PBS -j oe
source ~/.bashrc
cd /home/m64000/work/SSL_MDD

conda activate sb

# python ver4_train.py \
#         hparams/new_train_cano_perc_dual_enc_hybrid_attention_ctc_new.yaml \
#         --perceived_ssl_model wavlm_large \
#         --canonical_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --feature_fusion dual_ssl_enc_hybrid_ctc_attention \
#         --prefix "mispro"

# Using Multi-class classification for mispronunciation detection (ErrCls: Substitution, Deletion, Insertion, No-Error)
# python ver4_train.py \
#         hparams/new_train_cano_perc_dual_enc_hybrid_attention_ctc_new_errcls.yaml \
#         --perceived_ssl_model wavlm_large \
#         --canonical_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --feature_fusion dual_ssl_enc_hybrid_ctc_attention \
#         --prefix "mispro_errcls"

## Low Cano ratio
# python ver4_train.py \
#         hparams/new_train_cano_perc_dual_enc_hybrid_attention_ctc_new_errcls.yaml \
#         --perceived_ssl_model wavlm_large \
#         --canonical_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --feature_fusion dual_ssl_enc_hybrid_ctc_attention \
#         --prefix "mispro_errcls_alpha0.2" \
#         --blend_alpha 0.2 

## High Cano ratio
python ver4_train.py \
        hparams/new_train_cano_perc_dual_enc_hybrid_attention_ctc_new_errcls.yaml \
        --perceived_ssl_model wavlm_large \
        --canonical_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --feature_fusion dual_ssl_enc_hybrid_ctc_attention \
        --prefix "mispro_errcls_alpha0.8" \
        --blend_alpha 0.8