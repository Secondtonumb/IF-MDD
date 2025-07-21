#!/bin/sh
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -W group_list=gm64
#PBS -j oe
source ~/.bashrc
cd /home/m64000/work/SSL_MDD

conda activate sb

python ver4_train.py \
        hparams/train_l2_arctic_cano_perc_dual_enc.yaml \
        --perceived_ssl_model wavlm_large \
        --canonical_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --feature_fusion dual_ssl_enc \
        --prefix ""