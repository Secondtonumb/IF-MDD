#!/bin/bash

conda activate sb
base_ssl_models=("wav2vec2_base" "wav2vec2_base_jp" "hubert_base" "wavlm_base" "wavlm_base_plus" "hubert_multilingual" "clap" "data2vec_base")
large_ssl_models=("wav2vec2_large" "hubert_large" "wavlm_large" "data2vec_large")

for base_ssl_model in ${base_ssl_models[@]}; do
    python ver3_train.py hparams/new_train_l2_arctic.yaml --perceived_ssl_model $base_ssl_model 
done

for large_ssl_model in ${large_ssl_models[@]}; do
    python ver3_train.py hparams/new_train_l2_arctic.yaml --perceived_ssl_model $large_ssl_model --ENCODER_DIM 1024
done
