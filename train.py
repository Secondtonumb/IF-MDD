"""
MDD (Mispronunciation Detection and Diagnosis) System - Main Training Script

Author: Haopeng (Kevin) Geng
Institution: The University of Tokyo
Year: 2025

This code is provided for non-commercial use only.
For commercial use, please contact the author.

This script implements the main training pipeline for the MDD system using
various SSL models for speech recognition and pronunciation assessment.
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# from mpd_eval_v3 import MpdStats
from mpd_eval_v4 import MpdStats
import librosa
import json
import wandb
import time
import torchaudio

from models.phn_mono_ssl_model import PhnMonoSSLModel, PhnMonoSSLModel_DualCTCHead, PhnMonoSSLModel_RVQforBoth
from models.Transformer import TransformerMDD
from models.Transformer_TP import TransformerMDD_TP
from models.Transformer_TP_fuse_errclass import TransformerMDD_TP_encdec_errclass
from models.SSL_LLM import SSL_LLM

from utils.DataPrepIO import LLMDataIOPrep, LLMDataIOPrep_ver2, LLMDataIOPrep_ver3
sys.path.append("./trainer")
logger = logging.getLogger(__name__)

# Define training procedure
# Mono ASR model

if __name__ == "__main__":
    # main()
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # log the running sys.argv[0: ] to logger
    logger.info(f"# " + " ".join([sys.executable] + sys.argv))
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)
    # Create experiment directory
    
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # DataPrep
    # DataPrep = TimestampDataIOPrepforHybridCTCAttn(hparams)
    # DataPrep = LLMDataIOPrep(hparams)
    # DataPrep = LLMDataIOPrep(hparams)
    
    # Model Selection
    if hparams["feature_fusion"] == "PhnMonoSSL":
        asr_brain_class = PhnMonoSSLModel
    elif hparams["feature_fusion"] == "TransformerMDD":
        asr_brain_class = TransformerMDD
    elif hparams["feature_fusion"] == "TransformerMDD_TP":
        asr_brain_class = TransformerMDD_TP
    elif hparams["feature_fusion"] == "TransformerMDD_TP_encdec_errclass":
        asr_brain_class = TransformerMDD_TP_encdec_errclass
    elif hparams["feature_fusion"] == "SSL_LLM":
        asr_brain_class = SSL_LLM
    
    if asr_brain_class == SSL_LLM:
        DataPrep  = LLMDataIOPrep_ver3(hparams)
    if asr_brain_class == TransformerMDD_TP_encdec_errclass:
        DataPrep  = LLMDataIOPrep_ver2(hparams)
    else:
        DataPrep  = LLMDataIOPrep(hparams)
    train_data, valid_data, test_data, label_encoder = DataPrep.prepare()

    logger.info(f"Using ASR brain class: {asr_brain_class.__name__}")
    
    asr_brain = asr_brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    
    # 
    from pathlib import Path
    # Wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    prefix = hparams.get("prefix", "Null")
    perceived_ssl_model = hparams.get("perceived_ssl_model", "Null")
    canonical_ssl_model = hparams.get("canonical_ssl_model", "Null")    
    feature_fusion = hparams.get("feature_fusion", "Null")
    prefix = hparams.get("prefix", None)
    model_type = type(asr_brain).__name__  # e.g., ASR_with_misproBCE_proj
    model_stem = Path(model_type).stem 
    
    run_id = time.strftime("%Y%m%d-%H%M%S") 
    run_name = f"{prefix}_{perceived_ssl_model}_{canonical_ssl_model}_{feature_fusion}"
    run_id = f"{run_name}_{run_id}"
    
    # wandb init group by hparams perceived_ssl_model, canonical_ssl_model, feature_fusion
    
    wandb.init(
        project=hparams.get("wandb_project", model_type), 
        name=run_name,
        id=run_id,
        resume="allow"
    )
    # limit train_data for quick debugging

    # train_record = train_data.data_ids[:512]  # Select first 1024 for debugging
    # valid_record = valid_data.data_ids[:128]  # Select first 128 for debugging
    # train_data = train_data.filtered_sorted(key_test={"id": lambda x: x in train_record},)
    # valid_data = valid_data.filtered_sorted(key_test={"id": lambda x: x in valid_record},)
    
    # test_record = test_data.data_ids[:128]
    # test_data = test_data.filtered_sorted(key_test={"id": lambda x: x in test_record},)
    
    
    # Training/validation loop
    try:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )
    except StopIteration:
        print("Training stopped early due to no improvement.")
    
    # Test
    if hparams.get("evaluate_key", True):
        key = hparams["evaluate_key"]
        if key == "mpd_f1" or key == "mpd_f1_seq":
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=hparams["test_dataloader_opts"],
                max_key=key
            )
        elif key == "PER" or key == "PER_seq" or key == "CTC_PER":
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=hparams["test_dataloader_opts"],
                min_key=key,
            )
