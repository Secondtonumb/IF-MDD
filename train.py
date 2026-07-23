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
import yaml

from models.phn_mono_ssl_model import (
    PhnMonoSSLModel,
    PhnMonoSSLModel_DualCTCHead,
    PhnMonoSSLModel_RVQforBoth,
)
from models.phn_mono_ssl_model_v3_refactored import (
    PhnMonoSSLModel as RefactoredPhnMonoSSLModel,
)
from models.phn_mono_ssl_model_v3_refactored_IF import PhnMonoSSLModel_IF
from models.Trans_IFMDD_ConPCO_ver2 import Trans_IFMDD_ConPCO_ver2
from models.Transformer import TransformerMDD
from models.Transformer_TP import TransformerMDD_TP
from models.Transformer_TP_fuse_errclass import TransformerMDD_TP_encdec_errclass
from models.SSL_LLM import SSL_LLM
from models.SSL_LLM_origin_ver2 import SSL_LLM_origin_ver2

from utils.DataPrepIO import LLMDataIOPrep, LLMDataIOPrep_ver2, LLMDataIOPrep_ver3
from utils.release_bundle import (
    apply_bundle_defaults,
    install_strict_inference_checkpointer,
    resolve_inference_bundle,
    validate_label_encoder,
    validate_model_signature,
)
sys.path.append("./trainer")
logger = logging.getLogger(__name__)


def _parse_release_arguments(argv):
    """Strip release-only options before passing the remainder to SpeechBrain."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "valid", "infer"),
        default="train",
    )
    parser.add_argument("--inference_ckpt")
    args, remaining = parser.parse_known_args(argv)
    if args.inference_ckpt and args.mode == "train":
        parser.error("--inference_ckpt is only supported with eval, valid, or infer")
    return args, remaining


def _metric_selection(hparams):
    key = hparams.get("evaluate_key")
    if key in {"mpd_f1", "mpd_f1_seq"}:
        return {"max_key": key}
    if key in {"PER", "PER_seq", "CTC_PER", "LLM_PER"}:
        return {"min_key": key}
    return {}


def _cli_override_keys(overrides):
    if isinstance(overrides, dict):
        return set(overrides)
    if isinstance(overrides, str) and overrides.strip():
        parsed = yaml.safe_load(overrides)
        if isinstance(parsed, dict):
            return set(parsed)
    return set()


def _inject_bundle_root(overrides, bundle_root):
    """Add the resolved bundle root before HyperPyYAML instantiates modules."""

    root_override = f"bundle_root: {json.dumps(str(bundle_root))}"
    if isinstance(overrides, str):
        return "\n".join(part for part in (overrides.strip(), root_override) if part)
    merged = dict(overrides)
    merged["bundle_root"] = str(bundle_root)
    return merged


# Define training procedure
# Mono ASR model

if __name__ == "__main__":
    # main()
    # CLI:
    release_args, speechbrain_argv = _parse_release_arguments(sys.argv[1:])
    hparams_file, run_opts, overrides = sb.parse_arguments(speechbrain_argv)
    inference_bundle = (
        resolve_inference_bundle(release_args.inference_ckpt)
        if release_args.inference_ckpt
        else None
    )
    effective_hparams_file = hparams_file
    load_overrides = overrides
    if inference_bundle is not None and inference_bundle.hyperparams is not None:
        effective_hparams_file = str(inference_bundle.hyperparams)
        load_overrides = _inject_bundle_root(overrides, inference_bundle.root)

    # log the running sys.argv[0: ] to logger
    logger.info(f"# " + " ".join([sys.executable] + sys.argv))
    # Load hyperparameters file with command-line overrides
    with open(effective_hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, load_overrides)
    if inference_bundle is not None:
        validate_model_signature(hparams, inference_bundle.manifest)
        apply_bundle_defaults(
            hparams,
            inference_bundle.manifest,
            cli_override_keys=_cli_override_keys(overrides),
        )
    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)
    # Create experiment directory
    
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=effective_hparams_file,
        overrides=load_overrides,
    )

    # DataPrep
    # DataPrep = TimestampDataIOPrepforHybridCTCAttn(hparams)
    # DataPrep = LLMDataIOPrep(hparams)
    # DataPrep = LLMDataIOPrep(hparams)
    
    # Model Selection
    if hparams["feature_fusion"] == "PhnMonoSSL":
        if "ctc_loss_type" in hparams or "encoder_type" in hparams:
            asr_brain_class = RefactoredPhnMonoSSLModel
        else:
            asr_brain_class = PhnMonoSSLModel
    elif hparams["feature_fusion"] == "PhnMonoSSL_IF":
        asr_brain_class = PhnMonoSSLModel_IF
    elif hparams["feature_fusion"] == "Trans_IFMDD_ConPCO_ver2":
        asr_brain_class = Trans_IFMDD_ConPCO_ver2
    elif hparams["feature_fusion"] == "TransformerMDD":
        asr_brain_class = TransformerMDD
    elif hparams["feature_fusion"] == "TransformerMDD_TP":
        asr_brain_class = TransformerMDD_TP
    elif hparams["feature_fusion"] == "TransformerMDD_TP_encdec_errclass":
        asr_brain_class = TransformerMDD_TP_encdec_errclass
    elif hparams["feature_fusion"] == "SSL_LLM":
        asr_brain_class = SSL_LLM
    elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2":
        asr_brain_class = SSL_LLM_origin_ver2
    else:
        raise ValueError(f"Unsupported feature_fusion: {hparams['feature_fusion']}")

    if asr_brain_class in (SSL_LLM, SSL_LLM_origin_ver2):
        DataPrep = LLMDataIOPrep_ver3(hparams)
    elif asr_brain_class in (TransformerMDD_TP_encdec_errclass, PhnMonoSSLModel_IF):
        DataPrep  = LLMDataIOPrep_ver2(hparams)
    else:
        DataPrep  = LLMDataIOPrep(hparams)
    train_data, valid_data, test_data, label_encoder = DataPrep.prepare()
    validate_label_encoder(
        hparams["lab_enc_file"],
        bundle_path=(
            inference_bundle.label_encoder if inference_bundle is not None else None
        ),
    )

    logger.info(f"Using ASR brain class: {asr_brain_class.__name__}")
    
    asr_brain = asr_brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    if inference_bundle is not None:
        install_strict_inference_checkpointer(asr_brain, inference_bundle)
    
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
    
    if release_args.mode == "train":
        wandb.init(
            project=hparams.get("wandb_project", model_type),
            name=run_name,
            id=run_id,
            resume="allow",
        )

    # Training/validation loop
    if release_args.mode == "train":
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
    if release_args.mode == "infer":
        asr_brain.inference(
            test_data,
            test_loader_kwargs=hparams["test_dataloader_opts"],
            output_file=hparams.get("inference_output_file"),
        )
    elif release_args.mode in {"eval", "valid"}:
        evaluation_data = valid_data if release_args.mode == "valid" else test_data
        loader_key = (
            "valid_dataloader_opts"
            if release_args.mode == "valid"
            else "test_dataloader_opts"
        )
        asr_brain.evaluate(
            evaluation_data,
            test_loader_kwargs=hparams[loader_key],
            **_metric_selection(hparams),
        )
    elif hparams.get("evaluate_key"):
        asr_brain.evaluate(
            test_data,
            test_loader_kwargs=hparams["test_dataloader_opts"],
            **_metric_selection(hparams),
        )
