#!/bin/bash
# 2 sets: tr (train), te (test)
sets="tr"

# ssl_models="wavlm_large wav2vec_large_xlsr_53 hubert_large_ll60k"
# ssl_models="wavlm_large"
ssl_models="EMA"
# wavlm_large 
# mimi (https://speechbrain.readthedocs.io/en/stable/API/speechbrain.lobes.models.huggingface_transformers.mimi.html#module-speechbrain.lobes.models.huggingface_transformers.mimi)

data_root="/home/kevingenghaopeng/MDD/IF-MDD/data/speechocean762_with_word_scores/"
ctm_root="/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/"
output_dir="/home/kevingenghaopeng/MDD/IF-MDD/data_so762/phoneme_ssl_features"
pooling="mean"
target_phn_frames=50
pretrained_model_path="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/"
GOPT_mode_pooling=False  # 是否使用 GOPT 的音素池化实现（t_e - t_s + 1）
ema_extraction_mode="all"  # EMA 特征提取模式 (mean, std, delta, all)
extraction_method="clip_first" # 特征提取方法 (full_then_clip, clip_first)

for set in $sets; do
    for ssl_model in $ssl_models; do
        mkdir -p ${output_dir}/${ssl_model}/${set}

        echo "Extracting features for set: $set, SSL model: $ssl_model"
        
        python3 extract_phoneme_ssl_features.py \
            --json-file ${data_root}/${set}.json \
            --ctm-file ${ctm_root}/${set}_phones_nosil.ctm \
            --output-dir ${output_dir}/${ssl_model}/${set} \
            --ssl-model ${ssl_model} \
            --pooling ${pooling} \
            --target-phn-frames ${target_phn_frames} \
            --pretrained-model-path ${pretrained_model_path} \
            --device "cuda" \
            --GOPT_mode_pooling ${GOPT_mode_pooling} \
            --ema_extraction_mode "all" \
            --extraction_method ${extraction_method} \
            --min_phoneme_duration 0.025
    done
done