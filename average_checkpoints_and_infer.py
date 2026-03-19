#!/usr/bin/env python3
"""
Average multiple checkpoints using SpeechBrain's official API and run inference with the averaged model.

This script:
1. Uses SpeechBrain's average_state_dicts() function to load and average checkpoint files
2. Saves the averaged model state
3. Runs inference using the averaged model
"""

import sys
import os
from pathlib import Path
import csv
import torch
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import SpeechBrain utilities
try:
    from speechbrain.utils.checkpoints import average_state_dicts
    logger.info("✓ Successfully imported average_state_dicts from SpeechBrain")
except ImportError as e:
    logger.error(f"Failed to import average_state_dicts: {e}")
    logger.info("Please install speechbrain: pip install speechbrain")
    sys.exit(1)

# Helper to import MyEncoderASR
try:
    from trainer.MyEncoderASR import MyEncoderASR
except ImportError:
    logger.info("Adding current directory to sys.path for module imports...")
    sys.path.append(str(Path.cwd()))
    try:
        from trainer.MyEncoderASR import MyEncoderASR
    except ImportError as e:
        logger.error(f"Failed to import MyEncoderASR: {e}")
        sys.exit(1)


def load_and_average_checkpoints(checkpoint_paths, output_path=None):
    """
    Load and average multiple checkpoints using SpeechBrain's official average_state_dicts function.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        output_path: Path to save the averaged checkpoint (optional)
    
    Returns:
        Averaged state dict
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"AVERAGING {len(checkpoint_paths)} CHECKPOINTS")
    logger.info(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # Load all state dicts
    logger.info("Step 1: Loading individual checkpoints...")
    logger.info(f"{'-'*70}")
    
    def load_state_dict(ckpt_path):
        """Load a single checkpoint file and extract state dict."""
        logger.info(f"  Loading: {Path(ckpt_path).name}")
        try:
            ckpt_path_obj = Path(ckpt_path)
            
            # Handle both directory and file formats
            if ckpt_path_obj.is_dir():
                # SpeechBrain directory format
                logger.info(f"    (Directory format detected)")
                model_ckpt_path = ckpt_path_obj / 'model.ckpt'
                if not model_ckpt_path.exists():
                    raise FileNotFoundError(f"model.ckpt not found in {ckpt_path}")
                ckpt = torch.load(str(model_ckpt_path), map_location=device)
            else:
                # Single file format
                ckpt = torch.load(ckpt_path, map_location=device)
            
            # Handle both {'model': state_dict} and direct state_dict formats
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            logger.info(f"    ✓ Loaded successfully ({len(state_dict)} parameters)")
            return state_dict
            
        except Exception as e:
            logger.error(f"    ✗ Failed to load: {e}")
            raise
    
    # Load all state dicts
    state_dicts = []
    for i, path in enumerate(checkpoint_paths, 1):
        state_dict = load_state_dict(path)
        state_dicts.append(state_dict)
    
    # Average using SpeechBrain's official function
    logger.info(f"\nStep 2: Computing average of {len(state_dicts)} state dicts...")
    logger.info(f"{'-'*70}")
    
    try:
        averaged_state_dict = average_state_dicts(state_dicts)
        logger.info(f"✓ Successfully averaged {len(averaged_state_dict)} parameters")
    except Exception as e:
        logger.error(f"Failed to average state dicts: {e}")
        raise
    
    # Save averaged checkpoint if output path is provided
    if output_path:
        logger.info(f"\nStep 3: Saving averaged checkpoint...")
        logger.info(f"{'-'*70}")
        logger.info(f"Output path: {output_path}")
        
        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Save as SpeechBrain checkpoint format
            averaged_ckpt = {'model': averaged_state_dict}
            torch.save(averaged_ckpt, output_path)
            
            file_size = os.path.getsize(output_path) / (1024**2)  # Convert to MB
            logger.info(f"✓ Checkpoint saved successfully ({file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ Checkpoint averaging completed successfully!")
    logger.info(f"{'='*70}\n")
    
    return averaged_state_dict


def run_inference_with_averaged_model(
    averaged_state_dict,
    checkpoint_template_path,
    hparams_filename,
    output_csv_path,
    wav_dir=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run inference using the averaged model weights.
    
    Args:
        averaged_state_dict: The averaged state dict from averaging checkpoints
        checkpoint_template_path: Path to template checkpoint for model initialization
        hparams_filename: Path to hyperparameters YAML file
        output_csv_path: Path to save inference results CSV
        wav_dir: Directory containing WAV files to transcribe
        device: Device to run inference on ('cuda' or 'cpu')
    """
    logger.info(f"\n{'='*70}")
    logger.info("RUNNING INFERENCE WITH AVERAGED MODEL")
    logger.info(f"{'='*70}")
    
    device = torch.device(device)
    logger.info(f"Device: {device}\n")
    
    # Step 1: Initialize model from template checkpoint
    logger.info("Step 1: Initializing model...")
    logger.info(f"{'-'*70}")
    
    try:
        logger.info(f"  Template checkpoint: {Path(checkpoint_template_path).name}")
        logger.info(f"  Hyperparameters file: {hparams_filename}")
        
        asr_model = MyEncoderASR.from_hparams(
            source=checkpoint_template_path,
            hparams_file=hparams_filename,
            run_opts={"device": device}
        )
        logger.info("  ✓ Model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    # Step 2: Replace model weights with averaged weights
    logger.info(f"\nStep 2: Loading averaged model weights...")
    logger.info(f"{'-'*70}")
    
    try:
        # Replace with averaged state dict
        if hasattr(asr_model, 'modules'):
            asr_model.modules.model.load_state_dict(averaged_state_dict, strict=False)
        else:
            asr_model.load_state_dict(averaged_state_dict, strict=False)
        
        logger.info("  ✓ Model weights replaced with averaged state dict")
        
    except Exception as e:
        logger.error(f"Failed to load averaged weights: {e}")
        raise
    
    # Set model to evaluation mode
    asr_model.eval()
    
    # Step 3: Find WAV files to transcribe
    logger.info(f"\nStep 3: Locating WAV files...")
    logger.info(f"{'-'*70}")
    
    if wav_dir is None:
        wav_dir = Path("/home/m64000/work/dataset/data_iqra/test/wav")
    else:
        wav_dir = Path(wav_dir)
    
    logger.info(f"  WAV directory: {wav_dir}")
    
    if not wav_dir.exists():
        logger.error(f"WAV directory not found: {wav_dir}")
        logger.info("Skipping inference step")
        return
    
    # Get all .wav files
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    if not wav_files:
        logger.error(f"No .wav files found in {wav_dir}")
        return
    
    logger.info(f"  ✓ Found {len(wav_files)} WAV files")
    
    # Step 4: Run inference
    logger.info(f"\nStep 4: Running transcription with averaged model...")
    logger.info(f"{'-'*70}\n")
    
    results = []
    error_count = 0
    
    with torch.no_grad():
        for wav_path in tqdm(wav_files, desc="Transcribing"):
            try:
                file_id = wav_path.stem
                
                # Run transcription
                prediction = asr_model.transcribe_file(str(wav_path))
                
                results.append({
                    "ID": file_id,
                    "Labels": prediction
                })
                
            except Exception as e:
                logger.warning(f"Error processing {wav_path.name}: {str(e)}")
                results.append({
                    "ID": wav_path.stem,
                    "Labels": "ERROR"
                })
                error_count += 1
    
    # Step 5: Save results to CSV
    logger.info(f"\nStep 5: Saving results...")
    logger.info(f"{'-'*70}")
    logger.info(f"  Output file: {output_csv_path}")
    
    try:
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        
        fieldnames = ["ID", "Labels"]
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"  ✓ Results saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ Inference completed successfully!")
    logger.info(f"{'='*70}")
    logger.info(f"Processed: {len(wav_files)} files")
    logger.info(f"Successful: {len(results) - error_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output: {output_csv_path}\n")


def main():
    """
    Main function to coordinate checkpoint averaging and inference.
    """
    logger.info(f"\n{'#'*70}")
    logger.info("# CHECKPOINT AVERAGING AND ENSEMBLE INFERENCE")
    logger.info(f"{'#'*70}\n")
    
    # Configuration
    exp_dir = Path('/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3/save')
    
    # 4 checkpoints to average
    checkpoint_names = [
        'CKPT+180_PER_3.6162_F1_0.9137.ckpt',
        'CKPT+200_PER_4.2786_F1_0.9000.ckpt',
        'CKPT+201_PER_4.2248_F1_0.8811.ckpt',
        'CKPT+194_PER_3.6520_F1_0.8881.ckpt',
    ]
    
    checkpoint_paths = [
        str(exp_dir / name) for name in checkpoint_names
    ]
    
    # Verify all checkpoints exist
    logger.info("Verifying checkpoint files...")
    logger.info(f"{'-'*70}")
    
    for ckpt_path in checkpoint_paths:
        if not Path(ckpt_path).exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        logger.info(f"  ✓ {Path(ckpt_path).name}")
    
    # Output paths
    averaged_ckpt_path = exp_dir / 'CKPT_AVERAGED_180_200_201_194.ckpt'
    output_csv = Path('/home/m64000/work/IF-MDD/CROTTC_TTSbasedFT_k3_averaged_predictions.csv')
    yaml_file = Path('/home/m64000/work/IF-MDD/phnmonossl_crottc_confEnc_FT.yaml')
    
    # Step 1: Average checkpoints
    logger.info("")
    averaged_state_dict = load_and_average_checkpoints(
        checkpoint_paths,
        output_path=str(averaged_ckpt_path)
    )
    
    # Step 2: Run inference with averaged model
    if yaml_file.exists():
        template_checkpoint = checkpoint_paths[0]
        run_inference_with_averaged_model(
            averaged_state_dict=averaged_state_dict,
            checkpoint_template_path=template_checkpoint,
            hparams_filename=str(yaml_file),
            output_csv_path=str(output_csv)
        )
    else:
        logger.warning(f"\nYAML file not found: {yaml_file}")
        logger.info("Skipping inference step. Checkpoint averaging still completed successfully.")
    
    logger.info("\n✓ Pipeline completed successfully!")


if __name__ == '__main__':
    main()
