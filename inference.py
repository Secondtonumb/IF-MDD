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
from utils.MyEncoderASR import MyEncoderASR
from utils.DataPrepIO import LLMDataIOPrep, LLMDataIOPrep_ver2

sys.path.append("./trainer")
logger = logging.getLogger(__name__)

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
    # Model Selection
    if hparams["feature_fusion"] == "PhnMonoSSL":
        asr_brain_class = PhnMonoSSLModel
    elif hparams["feature_fusion"] == "TransformerMDD":
        asr_brain_class = TransformerMDD
    elif hparams["feature_fusion"] == "TransformerMDD_TP":
        asr_brain_class = TransformerMDD_TP
    elif hparams["feature_fusion"] == "TransformerMDD_TP_encdec_errclass":
        asr_brain_class = TransformerMDD_TP_encdec_errclass
    
    lab_enc_file = "utils/label_encoder.txt"

    # save label_encoder as ckpt 
    logger.info(f"Using ASR brain class: {asr_brain_class.__name__}")
    
    asr_brain = asr_brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    ckpt = asr_brain.checkpointer.find_checkpoint(min_key="PER")
    hparams["pretrainer"].collect_files(default_source=ckpt.path)
    hparams["pretrainer"].load_collected()
    
    # from speechbrain.inference.ASR import EncoderASR
    from utils.MyEncoderASR import MyEncoderASR
    # pdb.set_trace()
    asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CKPT+best_mpdf1_044_0.6495.ckpt",
                                        hparams_file="inference.yaml",)
    # import pdb; pdb.set_trace()
    x = asr_model.transcribe_file("/common/db/TIMIT/timit/train/dr6/msmr0/si1405.wav")
    
    
