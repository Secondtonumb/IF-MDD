"""
Decode All Checkpoints Script

This script decodes all checkpoint files from a specified directory
and generates per.txt and mpd.txt outputs for each checkpoint.

Author: Haopeng (Kevin) Geng
Institution: The University of Tokyo
Year: 2025
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v4 import MpdStats
import json
from pathlib import Path
import re
from tqdm import tqdm

# Import all model classes
from models.phn_mono_ssl_model_v3_refactored import *
from models.phn_mono_ssl_model_v3_refactored_IF import PhnMonoSSLModel_IF
from models.Transformer import TransformerMDD
from models.Transformer_TP import TransformerMDD_TP
from models.Transformer_TP_fuse_errclass import TransformerMDD_TP_encdec_errclass
from models.Transformer_TP_fuse_errclass_ConPCO import TransformerMDD_TP_encdec_errclass_ConPCO
from models.Trans_IFMDD_ConPCO import Trans_IFMDD_ConPCO
from models.Trans_IFMDD_ConPCO_ver2 import Trans_IFMDD_ConPCO_ver2
from models.Trans_IFMDD_ConPCO_ver2_proj import Trans_IFMDD_ConPCO_ver2_proj
from models.CFMDD import CFMDD
from models.SSL_LLM_origin import SSL_LLM_origin
from models.SSL_LLM_origin_ver2 import SSL_LLM_origin_ver2
from models.SSL_LLM_origin_ver2_with_cano import SSL_LLM_origin_ver2_with_cano
from models.SSL_LLM_MultiTarget_ver1 import SSL_LLM_MultiTarget_ver1
from models.SSL_LLM_MultiTarget_ver2 import SSL_LLM_MultiTarget_ver2
from models.SSL_LLM_PPATP import SSL_LLM_PPATP

from utils.DataPrepIO import LLMDataIOPrep, LLMDataIOPrep_ver2, LLMDataIOPrep_ver3

sys.path.append("/work/gm64/m64000/IF-MDD")
sys.path.append("./trainer")

logger = logging.getLogger(__name__)


def extract_epoch_from_ckpt(ckpt_path):
    """Extract epoch number from checkpoint filename."""
    filename = Path(ckpt_path).name
    # Format: CKPT+{epoch}_PER_{per}_F1_{f1}.ckpt
    match = re.search(r'CKPT\+(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def decode_checkpoint(asr_brain, test_data, ckpt_path, output_dir, hparams, test_loader_kwargs, save_dir, temp_dir):
    """Decode a single checkpoint and save results using move strategy."""
    import shutil
    
    epoch = extract_epoch_from_ckpt(ckpt_path)
    if epoch is None:
        logger.warning(f"Could not extract epoch from {ckpt_path}")
        return None
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Decoding checkpoint: {Path(ckpt_path).name}")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"{'='*80}\n")
    
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return None
    
    ckpt_name = checkpoint_path.name
    save_location = Path(save_dir) / ckpt_name
    decoded_location = Path(temp_dir) / "decoded_checkpoints" / ckpt_name
    
    # Step 1: Move checkpoint to save directory
    try:
        logger.info(f"Moving checkpoint to save directory: {save_location}")
        shutil.move(str(checkpoint_path), str(save_location))
    except Exception as e:
        logger.error(f"Failed to move checkpoint to save: {e}")
        return None
    
    # Create output directory for this epoch
    epoch_output_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    epoch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update output file paths while preserving the test annotation identifier
    # from hparams, e.g. test_clean_per.txt / test_other_mpd.txt.
    per_file = epoch_output_dir / Path(hparams["per_file"]).name
    mpd_file = epoch_output_dir / Path(hparams["mpd_file"]).name
    
    # Temporarily override hparams for output files
    original_per_file = hparams["per_file"]
    original_mpd_file = hparams["mpd_file"]
    
    hparams["per_file"] = str(per_file)
    hparams["mpd_file"] = str(mpd_file)
    
    result = {"epoch": epoch, "checkpoint": str(checkpoint_path.name)}
    
    try:
        # Run evaluation on test set (same as train.py)
        # Check evaluate_key from hparams
        key = hparams.get("evaluate_key", "PER")
        
        if key == "mpd_f1" or key == "mpd_f1_seq":
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=test_loader_kwargs,
                max_key=key
            )
        elif key in ["PER", "PER_seq", "CTC_PER", "LLM_PER"]:
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=test_loader_kwargs,
                min_key=key,
            )
        else:
            # Default to PER
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=test_loader_kwargs,
                min_key="PER",
            )
        
        logger.info(f"✓ Results saved to {epoch_output_dir}")
        logger.info(f"  - PER: {per_file}")
        logger.info(f"  - MPD: {mpd_file}")
        
        result["status"] = "success"
        
    except Exception as e:
        logger.error(f"Error evaluating checkpoint {ckpt_path}: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
    
    finally:
        # Restore original paths
        hparams["per_file"] = original_per_file
        hparams["mpd_file"] = original_mpd_file
    
    return result
    
    return result


def main():
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python decode_all_checkpoints.py <hparams_file> <checkpoint_dir> [output_dir]")
        print("\nExample:")
        print("  python decode_all_checkpoints.py hparams/phnmonossl_ottc.yaml \\")
        print("         /home/m64000/work/IF-MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_ottc_frz_lm/save \\")
        print("         ./decode_results")
        sys.exit(1)
    
    hparams_file = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./decode_all_results"
    
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, {})
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Hyperparameters: {hparams_file}")
    logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("")
    
    # Prepare data
    logger.info("Preparing test data...")
    
    # Model Selection (same as train.py)
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
    elif hparams["feature_fusion"] == "SSL_LLM_origin":
        asr_brain_class = SSL_LLM_origin
    elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2":
        asr_brain_class = SSL_LLM_origin_ver2
    elif hparams["feature_fusion"] == "SSL_LLM_origin_ver2_with_cano":
        asr_brain_class = SSL_LLM_origin_ver2_with_cano
    elif hparams["feature_fusion"] == "SSL_LLM_PPATP":
        asr_brain_class = SSL_LLM_PPATP
    else:
        raise ValueError(f"Unknown feature_fusion: {hparams['feature_fusion']}")
    
    # Data preparation (same as train.py)
    if asr_brain_class == TransformerMDD_TP_encdec_errclass or asr_brain_class == PhnMonoSSLModel_IF:
        DataPrep = LLMDataIOPrep_ver2(hparams)
    else:
        DataPrep = LLMDataIOPrep(hparams)
    
    _, _, test_data, label_encoder = DataPrep.prepare()
    logger.info(f"✓ Test data prepared: {len(test_data)} samples")
    
    # Initialize model
    logger.info(f"Initializing model: {asr_brain_class.__name__}")
    
    # Update checkpointer to point to checkpoint directory
    original_save_folder = hparams["save_folder"]
    hparams["save_folder"] = checkpoint_dir
    hparams["checkpointer"].checkpoints_dir = checkpoint_dir
    
    asr_brain = asr_brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    
    # Find all checkpoint directories (SpeechBrain checkpoints are directories)
    checkpoint_dir_path = Path(checkpoint_dir)
    # Filter for directories with CKPT+ pattern
    ckpt_files = sorted([p for p in checkpoint_dir_path.glob("CKPT+*.ckpt") if p.is_dir()])
    
    if not ckpt_files:
        logger.error(f"No checkpoint directories found in {checkpoint_dir}")
        sys.exit(1)
    
    logger.info(f"\nFound {len(ckpt_files)} checkpoint directories")
    logger.info("="*80)
    
    # Decode each checkpoint
    results_summary = []
    
    for ckpt_path in tqdm(ckpt_files, desc="Decoding checkpoints"):
        result = decode_checkpoint(
            asr_brain, 
            test_data, 
            ckpt_path, 
            output_dir, 
            hparams,
            hparams["test_dataloader_opts"]
        )
        
        if result is not None:
            results_summary.append(result)
    
    # Save summary
    summary_file = output_dir / "decode_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("DECODING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total checkpoints processed: {len(ckpt_files)}")
    logger.info(f"Successful: {sum(1 for r in results_summary if r['status'] == 'success')}")
    logger.info(f"Failed: {sum(1 for r in results_summary if r['status'] == 'failed')}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_file}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()
