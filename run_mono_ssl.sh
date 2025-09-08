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

# python ver5_train.py \
#         hparams/phnmonossl_with_TransEnc.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_conformerEnc_unfrzfeatext \
#         --freeze_perceived_feature_extractor False \
#         --freeze_perceived_ssl False 

## CTC for Canonical only, mainly for RVQ
# ✅　%WER 4.94 [ 1664 / 33664, 442 ins, 663 del, 559 sub ]
# python ver5_train.py \
#         hparams/phnmonossl.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_Cano \
#         --training_target canonical

# ## Using Freeze SSL on Canonical but with RVQ to see how much distortion it can handle
# ✅　%WER 5.19 [ 1746 / 33664, 375 ins, 805 del, 566 sub ]
# python ver5_train.py \
#         hparams/phnmonossl_with_RVQ.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_Cano_RVQ \
#         --training_target canonical \
#         --load_pretrained_components True \
#         --freeze_loaded_components True \
#         --pretrained_model_path "/home/m64000/work/SSL_MDD/pretrained_models/Cano_CTC_best_per_043_4.8153.ckpt" \
#         --components_to_load '["ssl", "enc", "ctc_head"]' \
#         --freeze_perceived_ssl True 

## CTC for Perceived only, mainly for RVQ
# ✅　%WER 14.48 [ 4637 / 32026, 816 ins, 780 del, 3041 sub ]
# python ver5_train.py \
#         hparams/phnmonossl.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc \

## RVQ from scratch WavLM
## 🔺　%WER 9.18 [ 3090 / 33664, 686 ins, 1282 del, 1122 sub ]
# python ver5_train.py \
#         hparams/phnmonossl_with_RVQ.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_Cano_RVQ_from_scratch \
#         --training_target canonical \
#         --freeze_perceived_ssl True 

# RVQ Perceived 
## ✅　%WER 14.91 [ 4776 / 32026, 703 ins, 1048 del, 3025 sub ]
# python ver5_train.py \
#         hparams/phnmonossl_with_RVQ.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_RVQforPerc\
#         --training_target target \
#         --load_pretrained_components True \
#         --freeze_loaded_components True \
#         --pretrained_model_path "/home/m64000/work/SSL_MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_wavlm_ctc/save/CKPT+best_mpdf1_047_0.6546.ckpt" \
#         --components_to_load '["ssl", "enc", "ctc_head"]' \
#         --freeze_perceived_ssl True 

## RVQ for Cano with target perceived
## Continuous embedding -> CTC -> perceived
##                      -> RVQ -> CTC -> Canonical 
##                      -> Linear -> Mispro Head (Optional)
## 🔺　%WER 14.27 [ 4569 / 32026, 702 ins, 886 del, 2981 sub ]
# python ver5_train.py \
#         hparams/phnmonossl_with_RVQforCano.yaml \
#         --feature_fusion PhnMonoSSL_RVQforCano \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_CanoforRVQ \
#         --load_pretrained_components True \
#         --freeze_loaded_components True \
#         --pretrained_model_path "/home/m64000/work/SSL_MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_wavlm_ctc/save/CKPT+best_mpdf1_047_0.6546.ckpt" \
#         --components_to_load '["ssl", "enc"]' \
#         --freeze_perceived_ssl True

## Dual CTC Head for both Cano and Perc
##        Perc -> Shallow Layer  (-1)
##        Cano -> Deep Layer (-1)
## ✅ WER 13.91 [ 4456 / 32026, 716 ins, 777 del, 2963 sub ]
## ✅ WER 5.43 [ 1829 / 33664, 438 ins, 742 del, 649 sub ]

# python ver5_train.py \
#         hparams/phnmonossl_with_DualCTCHead.yaml \
#         --feature_fusion PhnMonoSSL_DualCTCHead \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_DualCTCHead \
#         --preceived_ssl_emb_layer 2 \
#         --canonical_ssl_emb_layer -1 

## Share Encoder for both Cano and Perc 
# WORKS! Only distorted a little bit
## ✅
## ✅ 
# python ver5_train.py \
#         hparams/phnmonossl_with_DualCTCHead_ShareEnc.yaml \
#         --feature_fusion PhnMonoSSL_DualCTCHead \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_DualCTCHead_ShareEnc_LastLayer \
#         --shareenc True \
#         --preceived_ssl_emb_layer -1 \
#         --canonical_ssl_emb_layer -1 

## Using SSL optimized botht on Cano and Perc to learn a better VQ for Canonical
## ✅ 
python ver5_train.py \
        hparams/phnmonossl_with_RVQforCano.yaml \
        --feature_fusion PhnMonoSSL_RVQforCano \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc_CanoforRVQ_BOTHOPT_SSL \
        --load_pretrained_components True \
        --freeze_loaded_components True \
        --pretrained_model_path "/home/m64000/work/SSL_MDD/pretrained_models/ShareEnc_PER_12.9250_PER_Cano_5.2734_F1_0.6509.ckpt" \
        --components_to_load '["ssl", "enc"]' \
        --freeze_perceived_ssl True

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
# python ver4_train.py \
#         hparams/new_train_cano_perc_dual_enc_hybrid_attention_ctc_new_errcls.yaml \
#         --perceived_ssl_model wavlm_large \
#         --canonical_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --feature_fusion dual_ssl_enc_hybrid_ctc_attention \
#         --prefix "mispro_errcls_alpha0.8" \
#         --blend_alpha 0.8