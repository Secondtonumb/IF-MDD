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

# ### Testing Error Label pos weights
# weight=${PBS_ARRAY_INDEX}
# python ver5_train.py \
#        hparams/l2arctic/Transformer_with_extra_loss.yaml \
#        --prefix  transformer_el_2_2_8_el_pos_weight_${weight} \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDD_with_extra_loss \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 300 \
#        --valid_search_interval 5 \
#        --mispro_pos_weight ${weight} 

weight=${PBS_ARRAY_INDEX}
python ver5_train.py \
       hparams/l2arctic/Transformer_with_extra_loss.yaml \
       --prefix  transformer_el_2_2_8_el_pos_weight_${weight} \
       --perceived_ssl_model wavlm_large \
       --feature_fusion TransformerMDD_with_extra_loss \
       --num_encoder_layers 2 \
       --num_decoder_layers 2 \
       --nhead 8 \
       --ctc_weight 0.3 \
       --ENCODER_DIM 1024 \
       --encoder_module transformer \
       --number_of_epochs 300 \
       --valid_search_interval 5 \
       --mispro_pos_weight ${weight} 


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

# # # Transformer with new MPD metrics
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

# Light Transformer with new MPD metrics
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

# # # Light Transformer with MHA embedding
# python ver5_train.py \
#        hparams/l2arctic/TransformerMHA.yaml \
#        --prefix  conformer_2_2_8_dual \
#        --perceived_ssl_model wavlm_large \
#        --feature_fusion TransformerMDDMHA \
#        --num_encoder_layers 2 \
#        --num_decoder_layers 2 \
#        --nhead 8 \
#        --ctc_weight 0.3 \
#        --ENCODER_DIM 1024 \
#        --encoder_module transformer \
#        --number_of_epochs 600 \
#        --valid_search_interval 5

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