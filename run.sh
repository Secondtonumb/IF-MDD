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

# IS26 extra (real data, split into train/dev/test)
# speechbrain.utils.train_logger - Epoch loaded: 61 - test loss: 3.13e-01, test PER: 7.77, test mpd_f1: 4.24e-01
# CTC and PER stats written to file exp_iqra/wavlm_large_None_PhnMonoSSL_is26_extra_split/per.txt
# MPD results and stats written to file exp_iqra/wavlm_large_None_PhnMonoSSL_is26_extra_split/mpd.txt

# TTS test: 
python train.py \
        hparams_iqra/phnmonossl_is26_extra.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix is26_extra_split

# TTS data (split into train/dev/test)
# TTS set:
# extra test: %WER 29.58 
# New Precision: 0.08405545927209705, New Recall: 0.8220338983050848, New F1: 0.15251572327044027
python train.py \
        hparams_iqra/phnmonossl_is26_extra.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc_iqra_all # used for canonical coleection
