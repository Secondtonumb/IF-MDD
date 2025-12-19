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
python train.py \
        hparams_iqra/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc_iqra
