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
import argparse
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
import threading
from datetime import datetime

# from models.phn_mono_ssl_model import PhnMonoSSLModel, PhnMonoSSLModel_DualCTCHead, PhnMonoSSLModel_RVQforBoth
# from models.phn_mono_ssl_model import PhnMonoSSLModel_CRCTC
from models.phn_mono_ssl_model_v3_refactored import *
from models.phn_mono_ssl_model_v3_refactored_IF import PhnMonoSSLModel_IF

from models.Transformer import TransformerMDD
from models.Transformer_TP import TransformerMDD_TP
from models.Transformer_TP_fuse_errclass import TransformerMDD_TP_encdec_errclass
from models.Transformer_TP_fuse_errclass_ConPCO import TransformerMDD_TP_encdec_errclass_ConPCO
from models.Trans_IFMDD_ConPCO import Trans_IFMDD_ConPCO
from models.Trans_IFMDD_ConPCO_ver2 import Trans_IFMDD_ConPCO_ver2
from models.Trans_IFMDD_ConPCO_ver2_proj import Trans_IFMDD_ConPCO_ver2_proj

# CFMDD
from models.CFMDD import CFMDD
# from models.SSL_LLM import SSL_LLM
from models.SSL_LLM_origin import SSL_LLM_origin
from models.SSL_LLM_origin_ver2 import SSL_LLM_origin_ver2
from models.SSL_LLM_origin_ver2_with_cano import SSL_LLM_origin_ver2_with_cano
from models.SSL_LLM_MultiTarget_ver1 import SSL_LLM_MultiTarget_ver1
from models.SSL_LLM_MultiTarget_ver2 import SSL_LLM_MultiTarget_ver2

from models.SSL_LLM_PPATP import SSL_LLM_PPATP

# from models.SSL_LLM_origin_ver2_expand_tok import SSL_LLM_origin_ver2_expand_tok

from utils.DataPrepIO import LLMDataIOPrep, LLMDataIOPrep_ver2, LLMDataIOPrep_ver3, InferDataIOPrep

sys.path.append("/work/gm64/m64000/IF-MDD")

sys.path.append("./trainer")
logger = logging.getLogger(__name__)


# Define training procedure
# Mono ASR model

if __name__ == "__main__":
    # Add custom argument parser for mode selection FIRST
    parser = argparse.ArgumentParser(description='MDD Training/Evaluation Script', add_help=False)
    parser.add_argument('--mode', type=str, default='train+eval', 
                       choices=['train', 'eval', 'train+eval', 'infer', 'valid'],
                       help='Execution mode: train only, eval only, both, infer, or valid only (default: train+eval)')
    
    # Parse only the --mode argument, leave the rest for speechbrain
    args, remaining_argv = parser.parse_known_args()
    
    # Now parse speechbrain arguments from remaining_argv
    hparams_file, run_opts, overrides = sb.parse_arguments(remaining_argv)
    
    # log the running sys.argv to logger
    logger.info(f"# " + " ".join([sys.executable] + sys.argv))
    logger.info(f"Execution mode: {args.mode}")
    
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
    elif hparams["feature_fusion"] == "PhnMonoSSL_IF":
        asr_brain_class = PhnMonoSSLModel_IF
    elif hparams["feature_fusion"] == "PhnMonoSSL_TextGate":
        asr_brain_class = PhnMonoSSLModel_TextGate
    elif hparams["feature_fusion"] == "TransformerMDD":
        asr_brain_class = TransformerMDD
    elif hparams["feature_fusion"] == "TransformerMDD_TP":
        asr_brain_class = TransformerMDD_TP
    elif hparams["feature_fusion"] == "TransformerMDD_TP_encdec_errclass":
        asr_brain_class = TransformerMDD_TP_encdec_errclass
    elif hparams["feature_fusion"] == "TransformerMDD_TP_encdec_errclass_ConPCO":
        asr_brain_class = TransformerMDD_TP_encdec_errclass_ConPCO
    elif hparams["feature_fusion"] == "Trans_IFMDD_ConPCO":
        asr_brain_class = Trans_IFMDD_ConPCO
    elif hparams["feature_fusion"] == "Trans_IFMDD_ConPCO_ver2":
        asr_brain_class = Trans_IFMDD_ConPCO_ver2
    elif hparams["feature_fusion"] == "Trans_IFMDD_ConPCO_ver2_proj":
        asr_brain_class = Trans_IFMDD_ConPCO_ver2_proj
    elif hparams["feature_fusion"] == "SSL_LLM_MultiTarget_ver1":
        asr_brain_class = SSL_LLM_MultiTarget_ver1
    elif hparams["feature_fusion"] == "SSL_LLM_MultiTarget_ver2":
        asr_brain_class = SSL_LLM_MultiTarget_ver2
    elif hparams["feature_fusion"] == "CFMDD":
        asr_brain_class = CFMDD
        
    # elif hparams["feature_fusion"] == "SSL_LLM":
    #     asr_brain_class = SSL_LLM
    elif hparams["feature_fusion"] == "SSL_LLM_origin":
        asr_brain_class = SSL_LLM_origin
    elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2":
        asr_brain_class = SSL_LLM_origin_ver2
    elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2_with_cano":
        asr_brain_class = SSL_LLM_origin_ver2_with_cano
    elif hparams["feature_fusion"] == "SSL_LLM_PPATP":
        asr_brain_class = SSL_LLM_PPATP
    # elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2_expand_tok":
    #     asr_brain_class = SSL_LLM_origin_ver2_expand_tok
    # if asr_brain_class == SSL_LLM:
#         DataPrep  = LLMDataIOPrep_ver3(hparams)
    if asr_brain_class == TransformerMDD_TP_encdec_errclass or asr_brain_class == PhnMonoSSLModel_IF:
        DataPrep  = LLMDataIOPrep_ver2(hparams)
    else:
        DataPrep  = LLMDataIOPrep(hparams)
    
    train_data, valid_data, test_data, label_encoder = DataPrep.prepare()
    
    # Infer Data prep, return id, sig only
    if args.mode == "infer" and hparams.get("infer_annotation", False):
        if hparams.get("inference_prompt_mode") == "canonical_only":
            # canonical aware inference
            from utils.DataPrepIO import InferDataIOPrep_with_cano
            InferDataIOPrep_cano = InferDataIOPrep_with_cano(hparams)
            infer_data = InferDataIOPrep_cano.prepare()
        else:
            # default inference
            InferDataPrep  = InferDataIOPrep(hparams)
            infer_data = InferDataPrep.prepare()
        

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
    
    # load wandb_tags if exists
    wandb_tags = hparams.get("tags", [])
    if wandb_tags is None:
        wandb_tags = []
    
    wandb.init(
        project=hparams.get("wandb_project", model_type), 
        name=run_name,
        id=run_id,
        resume="allow",
        tags=wandb_tags,
    )
    
    # limit train_data for quick debugging
    # train_record = train_data.data_ids[:128]  # Select first 2048 for debugging
    # valid_record = valid_data.data_ids[:16]  # Select first 128 for debugging
    # train_data = train_data.filtered_sorted(key_test={"id": lambda x: x in train_record},)
    # valid_data = valid_data.filtered_sorted(key_test={"id": lambda x: x in valid_record},)
    # test_record = test_data.data_ids[:16]
    # test_data = test_data.filtered_sorted(key_test={"id": lambda x: x in test_record},)  
    
    # Debug
    
    # Training/validation loop
    if args.mode in ['train', 'train+eval']:
        try:
            asr_brain.fit(
                asr_brain.hparams.epoch_counter,
                train_data,
                valid_data,
                train_loader_kwargs=hparams["train_dataloader_opts"],
                valid_loader_kwargs=hparams["valid_dataloader_opts"],
            )
        except StopIteration:
            logger.info("Training stopped early due to no improvement.")
            # Don't return here, continue to evaluation if mode includes eval
    
    # Validation only - run on valid_data
    if args.mode == 'valid':
        if hparams.get("evaluate_key", True):
            key = hparams["evaluate_key"]
            logger.info(f"Starting validation-only mode with key: {key}")
            
            if key == "mpd_f1" or key == "mpd_f1_seq":
                asr_brain.evaluate(
                    valid_data,
                    test_loader_kwargs=hparams["valid_dataloader_opts"],
                    max_key=key
                )
            elif key == "PER" or key == "PER_seq" or key == "CTC_PER" or key == "LLM_PER":
                asr_brain.evaluate(
                    valid_data,
                    test_loader_kwargs=hparams["valid_dataloader_opts"],
                    min_key=key,
                )
        else:
            logger.warning("evaluate_key not set in hparams, skipping validation")
    
    # Test - run evaluation based on mode
    if args.mode in ['eval', 'train+eval']:
        if hparams.get("evaluate_key", True):
            key = hparams["evaluate_key"]
            logger.info(f"Starting evaluation with key: {key}")
            
            if key == "mpd_f1" or key == "mpd_f1_seq":
                asr_brain.evaluate(
                    test_data,
                    test_loader_kwargs=hparams["test_dataloader_opts"],
                    max_key=key
                )
            elif key == "PER" or key == "PER_seq" or key == "CTC_PER" or key == "LLM_PER":
                asr_brain.evaluate(
                    test_data,
                    test_loader_kwargs=hparams["test_dataloader_opts"],
                    min_key=key,
                )
        else:
            logger.warning("evaluate_key not set in hparams, skipping evaluation")
    
    if args.mode in ["infer"]:
        if hparams.get("evaluate_key", True):
            key = hparams["evaluate_key"]
            logger.info(f"Starting inference with key: {key}")
            # import pdb; pdb.set_trace()
            if key == "mpd_f1" or key == "mpd_f1_seq":
                asr_brain.inference(
                    infer_data,
                    test_loader_kwargs=hparams["test_dataloader_opts"],
                    max_key=key,
                    output_file=hparams.get("inference_output_file", None)
                )
            elif key == "PER" or key == "PER_seq" or key == "CTC_PER" or key == "LLM_PER":
                asr_brain.inference(
                    infer_data,
                    test_loader_kwargs=hparams["test_dataloader_opts"],
                    min_key=key,
                    output_file=hparams.get("inference_output_file", None)
                )
    
    else:
        logger.info(f"Skipping evaluation (mode={args.mode})")