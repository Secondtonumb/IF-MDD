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
## PERC:🔺　%WER 14.27 [ 4569 / 32026, 702 ins, 886 del, 2981 sub ]
## Cano:🔺
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
##        Perc -> Shallow Layer
##        Cano -> Deep Layer

PERC_SSL_KEY=(6 12 18 24)
ek_idx=$(( PBS_ARRAY_INDEX % ${#PERC_SSL_KEY[@]} ))
EVALUATE_KEY=${PERC_SSL_KEY[$ek_idx]}

python ver5_train.py \
        hparams/phnmonossl_with_DualCTCHead.yaml \
        --feature_fusion PhnMonoSSL_DualCTCHead \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc_DualCTCHead_Perc_SSL_layer_${EVALUATE_KEY} \
        --preceived_ssl_emb_layer ${EVALUATE_KEY} \
        --canonical_ssl_emb_layer -1 
