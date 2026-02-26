"""
Decode All Checkpoints Script (Version 2 - Using train.py --mode eval)

This script manages checkpoint rotation and calls train.py for evaluation.
Strategy:
1. Move one checkpoint at a time into save directory
2. Run train.py --mode eval (which auto-loads the checkpoint)
3. Save results with epoch naming
4. Move checkpoint to decoded_checkpoints folder
5. Repeat for all checkpoints

Author: Haopeng (Kevin) Geng
Institution: The University of Tokyo
Year: 2026
"""

import os
import sys
import shutil
import subprocess
import logging
import json
from pathlib import Path
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_epoch_from_ckpt(ckpt_path):
    """Extract epoch number from checkpoint filename."""
    filename = Path(ckpt_path).name
    # Format: CKPT+{epoch}_PER_{per}_F1_{f1}.ckpt
    match = re.search(r'CKPT\+(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def copy_result_files(exp_output_folder, epoch_output_dir):
    """Copy annotation-specific PER/MPD files while preserving their names."""
    exp_output_folder = Path(exp_output_folder)
    epoch_output_dir = Path(epoch_output_dir)
    copied = []

    result_files = sorted(exp_output_folder.glob("*_per.txt")) + sorted(
        exp_output_folder.glob("*_mpd.txt")
    )

    # Backward-compatible fallback for older hparams that still write per.txt/mpd.txt.
    if not result_files:
        result_files = [p for p in [exp_output_folder / "per.txt", exp_output_folder / "mpd.txt"] if p.exists()]

    for result_file in result_files:
        dst = epoch_output_dir / result_file.name
        shutil.copy(result_file, dst)
        copied.append(dst)
        logger.info(f"✓ Copied {result_file} -> {dst}")

    if not copied:
        logger.warning(f"No PER/MPD result files found in {exp_output_folder}")

    return copied


def decode_checkpoint(ckpt_path, save_dir, output_base_dir, hparams_file, prefix, decoded_dir):
    """
    Decode a single checkpoint by:
    1. Moving it to save_dir
    2. Running train.py --mode eval
    3. Moving results to epoch-specific folder
    4. Moving checkpoint to decoded_dir
    """
    epoch = extract_epoch_from_ckpt(ckpt_path)
    if epoch is None:
        logger.warning(f"Could not extract epoch from {ckpt_path}")
        return None
    
    checkpoint_path = Path(ckpt_path)
    ckpt_name = checkpoint_path.name
    save_location = Path(save_dir) / ckpt_name
    decoded_location = Path(decoded_dir) / ckpt_name
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Decoding checkpoint: {ckpt_name}")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"{'='*80}\n")
    
    result = {"epoch": epoch, "checkpoint": ckpt_name}
    
    try:
        # Step 1: Move checkpoint to save directory
        logger.info(f"[1/4] Moving checkpoint to save directory...")
        if save_location.exists():
            logger.warning(f"Checkpoint already exists in save, removing old one")
            shutil.rmtree(save_location)
        shutil.move(str(checkpoint_path), str(save_location))
        logger.info(f"✓ Moved to {save_location}")
        
        # Step 2: Run train.py --mode eval
        logger.info(f"[2/4] Running evaluation...")
        cmd = [
            "python", "train.py",
            hparams_file,
            "--prefix", prefix,
            "--mode", "eval"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        result_code = subprocess.run(cmd, check=True, capture_output=False)
        
        if result_code.returncode == 0:
            logger.info(f"✓ Evaluation completed successfully")
            result["status"] = "success"
        else:
            logger.error(f"✗ Evaluation failed with code {result_code.returncode}")
            result["status"] = "failed"
            result["error"] = f"Return code {result_code.returncode}"
        
        # Step 3: Copy results to epoch-specific folder
        logger.info(f"[3/4] Organizing results...")
        epoch_output_dir = Path(output_base_dir) / f"epoch_{epoch:03d}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy annotation-specific PER/MPD files from experiment output folder.
        exp_output_folder = Path(save_dir).parent  # e.g., exp_l2arctic/wavlm_large_None_PhnMonoSSL_ottc_frz_lm
        copied_files = copy_result_files(exp_output_folder, epoch_output_dir)
        result["result_files"] = [str(path) for path in copied_files]
        
        # Step 4: Move checkpoint to decoded directory
        logger.info(f"[4/4] Moving checkpoint to decoded directory...")
        decoded_location.parent.mkdir(parents=True, exist_ok=True)
        if decoded_location.exists():
            shutil.rmtree(decoded_location)
        shutil.move(str(save_location), str(decoded_location))
        logger.info(f"✓ Moved to {decoded_location}")
        
        logger.info(f"\n✓ Successfully decoded epoch {epoch}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Evaluation command failed: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        
        # Still move checkpoint out even if failed
        try:
            if save_location.exists():
                decoded_location.parent.mkdir(parents=True, exist_ok=True)
                if decoded_location.exists():
                    shutil.rmtree(decoded_location)
                shutil.move(str(save_location), str(decoded_location))
        except Exception as move_err:
            logger.error(f"Failed to move checkpoint after error: {move_err}")
    
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
        
        # Try to restore checkpoint
        try:
            if save_location.exists():
                if not checkpoint_path.exists():
                    shutil.move(str(save_location), str(checkpoint_path))
        except Exception as restore_err:
            logger.error(f"Failed to restore checkpoint: {restore_err}")
    
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python decode_all_checkpoints_v2.py <hparams_file> <temp_checkpoint_dir> <output_dir> [prefix]")
        print("\nExample:")
        print("  python decode_all_checkpoints_v2.py \\")
        print("         hparams/phnmonossl_ottc.yaml \\")
        print("         /home/m64000/work/IF-MDD/exp_l2arctic/wavlm_large_None_PhnMonoSSL_ottc_frz_lm/temp_checkpoints \\")
        print("         ./decode_all_results_ottc_frz_lm \\")
        print("         ottc_frz_lm")
        sys.exit(1)
    
    hparams_file = sys.argv[1]
    temp_checkpoint_dir = sys.argv[2]
    output_dir = sys.argv[3]
    prefix = sys.argv[4] if len(sys.argv) > 4 else "ottc_frz_lm"
    
    # Infer save_dir from temp_checkpoint_dir (parent/save)
    save_dir = Path(temp_checkpoint_dir).parent / "save"
    decoded_dir = Path(temp_checkpoint_dir).parent / "decoded_checkpoints"
    
    logger.info(f"Configuration:")
    logger.info(f"  Hyperparameters:     {hparams_file}")
    logger.info(f"  Temp checkpoint dir: {temp_checkpoint_dir}")
    logger.info(f"  Save directory:      {save_dir}")
    logger.info(f"  Decoded directory:   {decoded_dir}")
    logger.info(f"  Output directory:    {output_dir}")
    logger.info(f"  Prefix:              {prefix}")
    logger.info("")
    
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all checkpoint directories in temp_checkpoint_dir
    temp_dir_path = Path(temp_checkpoint_dir)
    if not temp_dir_path.exists():
        logger.error(f"Temp checkpoint directory not found: {temp_checkpoint_dir}")
        sys.exit(1)
    
    ckpt_dirs = sorted([p for p in temp_dir_path.glob("CKPT+*.ckpt") if p.is_dir()])
    
    if not ckpt_dirs:
        logger.error(f"No checkpoint directories found in {temp_checkpoint_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(ckpt_dirs)} checkpoint directories")
    logger.info("="*80)
    
    # Decode each checkpoint
    results_summary = []
    
    for ckpt_path in tqdm(ckpt_dirs, desc="Decoding checkpoints"):
        result = decode_checkpoint(
            ckpt_path,
            save_dir,
            output_dir,
            hparams_file,
            prefix,
            decoded_dir
        )
        
        if result is not None:
            results_summary.append(result)
    
    # Save summary
    summary_file = Path(output_dir) / "decode_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("DECODING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total checkpoints processed: {len(ckpt_dirs)}")
    logger.info(f"Successful: {sum(1 for r in results_summary if r.get('status') == 'success')}")
    logger.info(f"Failed: {sum(1 for r in results_summary if r.get('status') == 'failed')}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Decoded checkpoints moved to: {decoded_dir}")
    logger.info(f"Summary: {summary_file}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    main()
