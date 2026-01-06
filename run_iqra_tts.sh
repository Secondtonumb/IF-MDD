#!/bin/sh
#------ qsub option --------#
#PBS -q regular-g 
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=02:00:00
#PBS -W group_list=gm64
#PBS -j oe

source ~/.bashrc
cd /home/m64000/work/IF-MDD

conda activate sb_k2
# nvidia-smi

# # Mono_SSL_CTC model
# python train.py \
#         hparams_iqra/phnmonossl.yaml \
#         --feature_fusion PhnMonoSSL \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix wavlm_ctc_iqra

# run_iqra_extra.sh
# OTTC with conformer
python train.py \
        ./hparams_iqra/TTS_phnmonossl_ottc_confEnc.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix ottc_confEnc