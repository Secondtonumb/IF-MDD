"""
Inference script for SSL_LLM with sequential interleaved sequences.

Loads a trained model and performs inference on test data,
parsing the interleaved output back to structured format.

Usage:
    python infer_interleaved.py hparams/SSL_LLM_Interleaved.yaml --checkpoint <path_to_checkpoint>
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import pandas as pd
import logging

from utils.DataPrepIO_Interleaved import LLMDataIOPrep_Interleaved
from models.SSL_LLM_Sequential_Interleaved import SSL_LLM_Sequential_Interleaved


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with SSL_LLM Interleaved model"
    )
    parser.add_argument(
        "hparams_file",
        type=str,
        help="Path to hyperparameter YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (if None, uses latest from save_folder)"
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default=None,
        help="Path to test data JSON file (if None, uses test_annotation from hparams)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (if None, uses save_folder)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("SSL_LLM Interleaved - Inference")
    logger.info("=" * 80)
    
    # Load hyperparameters
    with open(args.hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    output_dir = Path(args.output_dir or hparams['save_folder'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    DataPrep = LLMDataIOPrep_Interleaved(hparams)
    _, _, test_data = DataPrep.prepare()
    
    logger.info(f"Test set size: {len(test_data)}")
    
    # Create brain
    logger.info("Loading model...")
    brain = SSL_LLM_Sequential_Interleaved(
        modules=hparams['modules'],
        opt_class=hparams['adam_opt_class'],
        hparams=hparams,
        run_opts={
            'device': str(device),
            'debug': False,
        },
    )
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or (Path(hparams['save_folder']) / 'CKPT')
    if checkpoint_path:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        brain.load_checkpoint(checkpoint_path)
    else:
        logger.warning("No checkpoint specified or found!")
    
    # Setup label encoder
    brain.label_encoder = hparams['tokenizer']
    
    # Run inference
    logger.info("Running inference...")
    results = {
        'ids': [],
        'canonical': [],
        'perceived': [],
        'errors': [],
        'ctc_predictions': [],
    }
    
    test_loader = brain.make_dataloader(
        test_data,
        stage=sb.Stage.TEST,
        **hparams.get('test_dataloader_opts', {})
    )
    
    brain.modules.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Run inference
            inf_output = brain.inference_batch(
                batch,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )
            
            # Collect results
            for i, sample_id in enumerate(inf_output['ids']):
                file_stem = Path(sample_id).stem
                
                result = inf_output['results'][i]
                ctc_pred = inf_output['ctc_predictions'][i] if inf_output['ctc_predictions'] else ""
                
                results['ids'].append(file_stem)
                results['canonical'].append(result['canonical'])
                results['perceived'].append(result['perceived'])
                results['errors'].append(result['errors'])
                results['ctc_predictions'].append(ctc_pred)
    
    # Save results
    logger.info("Saving results...")
    
    # CSV format
    df_results = pd.DataFrame({
        'ID': results['ids'],
        'Canonical': results['canonical'],
        'Perceived': results['perceived'],
        'Errors': results['errors'],
    })
    
    csv_path = output_dir / 'infer_interleaved.csv'
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved interleaved results to: {csv_path}")
    
    # Also save with CTC predictions for comparison
    df_with_ctc = pd.DataFrame({
        'ID': results['ids'],
        'Canonical': results['canonical'],
        'Perceived': results['perceived'],
        'Errors': results['errors'],
        'CTC_Predictions': results['ctc_predictions'],
    })
    
    csv_with_ctc = output_dir / 'infer_interleaved_with_ctc.csv'
    df_with_ctc.to_csv(csv_with_ctc, index=False)
    logger.info(f"Saved results with CTC to: {csv_with_ctc}")
    
    # JSONL format
    jsonl_path = output_dir / 'infer_interleaved.jsonl'
    with open(jsonl_path, 'w') as f:
        for i in range(len(results['ids'])):
            record = {
                'id': results['ids'][i],
                'canonical': results['canonical'][i],
                'perceived': results['perceived'][i],
                'errors': results['errors'][i],
                'ctc_prediction': results['ctc_predictions'][i],
            }
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved results to JSONL: {jsonl_path}")
    
    logger.info("Inference completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
