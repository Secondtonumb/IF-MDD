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
nvidia-smi

<<<<<<< HEAD
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


=======
# MHA for Canonical Phn + Acoustic 
# python ver4_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix FUSE_NET_guided_attn \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion HMA_attn_ctc_to_mispro_ver2_1 

# Transformer 
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_6_6_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 6 \
#        --num_decoder_layers 6 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer

# # Transformer with new MPD metrics
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_6_6_8_new_mpd \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 6 \
#        --num_decoder_layers 6 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --valid_search_interval 5 \
#        --number_of_epochs 600

# # Light Transformer 
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_2_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer 

# # Light Transformer with new MPD metrics
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_2_2_8_new_mpd \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer 

# # Light Transformer with wav2vec2_large_xlsr_53
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_2_2_8_new_mpd \
#        --perceived_ssl_model wav2vec_large_xlsr_53 \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer 


# # # # Very Light Transformer 
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_1_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 1 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer 

# # (Decoder only) Light Transformer
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_0_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 0 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer 

## Light Transformer with ssl middle layer 22
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_2_2_8_midssl_embed \
#        --perceived_ssl_model wavlm_large \
#        --preceived_ssl_emb_layer 22 \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \

## Light Transformer with ssl middle layer 10

# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_2_2_8_midssl_embed_10 \
#        --perceived_ssl_model wavlm_large \
#        --preceived_ssl_emb_layer 10 \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \

# Light Transformer with extra loss for mispro detection 
## <TODOs>
# python ver5_train.py \
#        hparams/l2arctic/Transformer_with_extra_loss.yaml \
#        --prefix  transformer_el_2_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD_with_extra_loss \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 

# Transformer with dual path
# python ver5_train.py \
#        hparams/l2arctic/Transformer_dualSSL.yaml \
#        --prefix  transformer_6_6_8_dual \
#        --perceived_ssl_model wavlm_large \
#        --canonical_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD_dual_path \
#        --num_encoder_layers 6 \
#        --num_decoder_layers 6 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5

# # # # Light Transformer with dual path
# python ver5_train.py \
#        hparams/l2arctic/Transformer_dualSSL.yaml \
#        --prefix  transformer_2_2_8_dual \
#        --perceived_ssl_model wavlm_large \
#        --canonical_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD_dual_path \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5

# # # Light Conformer with dual path
# python ver5_train.py \
#        hparams/l2arctic/Transformer_dualSSL.yaml \
#        --prefix  conformer_2_2_8_dual \
#        --perceived_ssl_model wavlm_large \
#        --canonical_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD_dual_path \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module conformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5

# # # # Light Transformer with MHA embedding
#  python ver5_train.py \
#        hparams/l2arctic/TransformerMHA.yaml \
#        --prefix  transformer_2_2_8_MHA \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDDMHA \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5 \
#        --evaluate_key mpd_f1_seq \

# # # Light Transformer with MHA embedding, Phn Forward
python ver5_train.py \
       hparams/l2arctic/Transformer_PhnForward.yaml \
       --prefix  transformer_2_2_8_PhnForward \
       --perceived_ssl_model wavlm_large \
       --feature_fusion TransformerMDD_PhnForward \
       --num_encoder_layers 2 \
       --num_decoder_layers 2 \
       --nhead 8 \
       --ctc_weight 0.3 \
       --ENCODER_DIM 1024 \
       --encoder_module transformer \
       --number_of_epochs 600 \
       --valid_search_interval 1

# Heavy Transformer
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_8_8_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 8 \
#        --num_decoder_layers 8 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer

# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  transformer_4_4_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 4 \
#        --num_decoder_layers 4 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer

# Light Conformer
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  conformer_2_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module conformer \
#        --number_of_epochs 600 

# Heavy Conform

# Light Conformer
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  conformer_2_6_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 6 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module conformer

# # Transducer
# python ver5_train.py \
#        hparams/l2arctic/Transducer.yaml \
#        --prefix  Transducer \
#        --perceived_ssl_model wavlm_large \
#        --ENCODER_DIM 1024 \
#        --feature_fusion TransducerMDD


## Transducer with Conformer Encoder
# python ver5_train.py \
#        hparams/l2arctic/TransducerConformerEnc.yaml \
#        --prefix  TransducerConformerEnc \
#        --perceived_ssl_model wavlm_large \
#        --ENCODER_DIM 1024 \
#        --feature_fusion TransducerMDDConformerEnc \
#        --number_of_epochs 100 

# Ligher Encoder as SSL is already large, dont want to distory the features.
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  branchformer_6_6_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 6 \
#        --num_decoder_layers 6 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module branchformer

# Ligher Encoder as SSL is already large, dont want to distory the features.
# python ver5_train.py \
#        hparams/l2arctic/Transformer.yaml \
#        --prefix  branchformer_2_2_8 \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module branchformer

#### Transducer
# python ver5_train.py \
#        hparams/l2arctic/Transducer.yaml \
#        --prefix  Transducer \
#        --perceived_ssl_model wav2vec2_large \
#        --feature_fusion TransducerMDD 

# python ver4_train.py \
#         hparams/train_l2_arctic_cano_perc_dual_enc.yaml \
#         --perceived_ssl_model wavlm_large \
#         --canonical_ssl_model wavlm_large \
#         --ENCODER_DIM 1024 \
#         --feature_fusion dual_ssl_enc \
#         --prefix ""

# python ver4_train.py \
#         hparams/erj/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC.yaml \
#         --prefix MHA_guided_attn_loss_lam_1.0_new \
#         --perceived_ssl_model wav2vec2_base \
#         --feature_fusion mono_att_MHA_guided_attn \
#         --loss_lambda 1


# python ver4_train.py \
#         hparams/erj/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps.yaml \
#         --prefix MHA_guided_attn_loss_attn_ctc_to_cano_lam_0.5 \
#         --perceived_ssl_model wav2vec2_base \
#         --feature_fusion mono_att_HMA_ctc_to_canonical \
#         --loss_lambda 0.5

# python ver4_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver2.yaml \
#        --prefix MHA_guided_attn_loss_attn_ctc_to_cano_lam_0.5_ver2 \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion HMA_attn_ctc_to_mispro \
#        --loss_lambda 0.5

# MHA for Canonical Phn + Acoustic 
# python ver4_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix FUSE_NET_guided_attn \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion HMA_attn_ctc_to_mispro_ver2_1 
       
# python ver4_train.py \
#        hparams/erj/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix FUSE_NET_guided_attn_perceived_emb \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion HMA_attn_ctc_to_mispro_ver2_1_perceived

# python ver4_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix Mono_Hybrid_CTC \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion hybrid_ctc_attention

# python ver5_train.py \
#        hparams/erj/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix Mono_Hybrid_CTC_new \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion hybrid_ctc_attention_ver2

# python ver5_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix Hybrid_CTC \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion hybrid_ctc_attention_ver2

# python ver5_train.py \
#        hparams/l2arctic/PhnMonoSSLModel_withcanoPhnEmb_HMA_CTC_timestamps_ver3.yaml \
#        --prefix con \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion hybrid_ctc_attention_ver2

# python train_trans_asr_ver2.py \
#        hparams/conformer_large.yaml \
#        --prefix con \
#        --perceived_ssl_model wav2vec2_base 

# python train_trans_asr_ver2.py \
#        hparams/conformer_large.yaml \
#        --prefix Trans_all \
#        --perceived_ssl_model wav2vec2_base

# python ver5_train.py \
#        hparams/l2arctic/Conformer.yaml \
#        --prefix  branchformer_6_6 \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion SB \
#        --num_encoder_layers 6 \
#        --ctc_weight 0.3 \
#        --encoder_module branchformer 

# python ver5_train.py \
#        hparams/l2arctic/Conformer.yaml \
#        --prefix  transformer_6_3_12 \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion SB \
#        --num_encoder_layers 6 \
#        --num_decoder_layers 3 \
#        --nhead 12 \
#        --ctc_weight 0.3 \
#        --encoder_module transformer

# python ver5_train.py \
#         hparams/l2arctic/Trans.yaml\
#        --prefix  Transformer_6_6 \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion SB

# python ver5_evaluate.py \
#         hparams/l2arctic/Trans.yaml\
#        --prefix  Transformer_6_6 \
#        --perceived_ssl_model wav2vec2_base \
#        --feature_fusion SB --ctc_weight_decode 0.7
>>>>>>> 3e364161d58af691d4dc5019a2189b73343b80c0
