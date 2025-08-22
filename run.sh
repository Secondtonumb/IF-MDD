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

# ctc only
python ver5_train.py \
        hparams/phnmonossl.yaml \
        --feature_fusion PhnMonoSSL \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix wavlm_ctc

python ver4_train.py \
        hparams/train_l2_arctic_cano_perc_dual_enc.yaml \
        --perceived_ssl_model wavlm_large \
        --canonical_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --feature_fusion dual_ssl_enc \
        --prefix ""

# Light transformer
python ver5_train.py \
        hparams/transformer.yaml \
        --feature_fusion TransformerMDD \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 

# very light transformer
python ver5_train.py \
        hparams/transformer.yaml \
        --feature_fusion TransformerMDD \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_1_2_8 \
        --num_encoder_layers 1 \
        --num_decoder_layers 2 \
        --nhead 8 

# transformer with mispro
## Fuse enc
python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_enc \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec enc

python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_2_enc_ga \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec enc \
        --fuse_net_layers 2 \
        --attention_type regularMHA \
        --valid_search_interval 1

# Fuse dec
python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_dec \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec dec

python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_2_dec_ga \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec dec \
        --fuse_net_layers 2 \
        --attention_type regularMHA

# Conformer Causal Training
python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_2_dec_ga_conformer_causal \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec dec \
        --fuse_net_layers 2 \
        --encoder_module conformer \
        --attention_type RelPosMHAXL \
        --causal true \
        --valid_search_interval 2


# RoPE Conformer 
python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_2_dec_ga_new_PoPE_conf \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec dec \
        --fuse_net_layers 2 \
        --valid_search_interval 5 \
        --attention_type RoPEMHA \
        --encoder_module conformer

# VER2: decoder_d_out: aligned perceived 
python ver5_train.py \
        hparams/transformer_TP.yaml \
        --feature_fusion TransformerMDD_TP_ver2 \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix 2_2_8_TP_fuse_2_dec_RoPE_conf_tgt_aln_perc \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec dec \
        --fuse_net_layers 2 \
        --encoder_module conformer \
        --attention_type RoPEMHA \
        --causal true

# Transformer init with pretrained_ssl, enc, TransASR_encoder,
# python ver5_train.py \
#         hparams/transformer.yaml \
#         --feature_fusion TransformerMDD \
#         --perceived_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --prefix transformer_2_2_8_from_pretrained_ctc \
#         --num_encoder_layers 2 \
#         --num_decoder_layers 2 \
#         --nhead 8 \
#         --load_pretrained_components true \
#         --pretrained_model_path "/home/kevingenghaopeng/MDD/MDD_ver3/exp_l2arctic/wavlm_large_None_TransformerMDD_transformer_2_6_8/save/CKPT+best_per_seq_500_13.8651.ckpt" \
#         --components_to_load '["ssl", "enc", "encoder", "ctc_head"]' \
#         --valid_search_interval 1

# Dual ctc loss, ctc on pre net only
python ver5_train.py \
        hparams/transformer_dual_ctc.yaml \
        --feature_fusion TransformerMDD_dual_ctc \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_dual_ctc_1 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --dual_ctc_loss_weight 1

# Dual ctc loss, ctc on post enc only
python ver5_train.py \
        hparams/transformer_dual_ctc.yaml \
        --feature_fusion TransformerMDD_dual_ctc \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_dual_ctc_0 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --dual_ctc_loss_weight 0
 
# Dual ctc loss, ctc on both
python ver5_train.py \
        hparams/transformer_dual_ctc.yaml \
        --feature_fusion TransformerMDD_dual_ctc \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_dual_ctc_0.5 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --dual_ctc_loss_weight 0.5


# reduction factor == 4 
python ver5_train.py \
        hparams/transformer_TP_ver3.yaml \
        --feature_fusion TransformerMDD_TP \
        --perceived_ssl_model wavlm_large \
        --ENCODER_DIM 1024 \
        --prefix transformer_2_2_8_TP_fuse_2_enc_reduce_cnn_4 \
        --num_encoder_layers 2 \
        --num_decoder_layers 2 \
        --nhead 8 \
        --fuse_enc_or_dec enc \
        --fuse_net_layers 2 \
        --attention_type RelPosMHAXL \
        --valid_search_interval 1 \
        --post_encoder_reduction_factor 4