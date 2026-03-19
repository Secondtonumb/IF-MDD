"""
Training script for SSL_LLM with sequential interleaved canonical-perceived-error prediction.

Usage:
    python train_interleaved.py hparams/SSL_LLM_Interleaved.yaml
"""

import sys
import logging
from pathlib import Path
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import wandb
from utils.DataPrepIO_Interleaved import LLMDataIOPrep_Interleaved
# from models.SSL_LLM_Sequential_Interleaved import SSL_LLM_Sequential_Interleaved
from models.SSL_LLM_Interleaved_ver2 import SSL_LLM_Interleaved_ver2


# Setup logger
logger = logging.getLogger(__name__)


def main(hparams_file):
    """Main training function."""
    
    # Load hyperparameters
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    
    # Create output folder
    hparams['save_folder'] = Path(hparams['save_folder'])
    hparams['save_folder'].mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(hparams['train_log']),
            logging.StreamHandler()
        ]
    )
    
    logger.info("=" * 80)
    logger.info("Training SSL_LLM with Sequential Interleaved Sequences")
    logger.info("=" * 80)
    logger.info(f"Hyperparameters: {hparams_file}")
    logger.info(f"Output folder: {hparams['save_folder']}")
    
    # Initialize wandb for tracking
    wandb.init(
        project="ssl-llm-interleaved",
        name=hparams.get('prefix', 'interleaved_seq'),
        config={
            'lr': hparams['lr'],
            'batch_size': hparams['batch_size'],
            'ctc_weight': hparams['ctc_weight'],
            'model': hparams['LLM_model'],
            'use_lora': hparams['use_lora'],
        }
    )
    
    # Prepare data
    logger.info("Preparing data with LLMDataIOPrep_Interleaved...")
    DataPrep = LLMDataIOPrep_Interleaved(hparams)
    train_data, valid_data, test_data, label_encoder = DataPrep.prepare()
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(valid_data)}")
    logger.info(f"Test set size: {len(test_data)}")
    
    # Create brain instance
    logger.info("Initializing SSL_LLM_Interleaved_ver2 brain...")
    brain = SSL_LLM_Interleaved_ver2(
        modules=hparams['modules'],
        opt_class=hparams['adam_opt_class'],
        hparams=hparams,
        run_opts={
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'debug': False,
            'debug_batches': 2,
            'auto_mix_prec': hparams.get('auto_mix_prec', False),
        },
        checkpointer=hparams.get('checkpointer', None),
    )
    
    # Setup label encoder
    brain.label_encoder = hparams['tokenizer']
    
    # Training loop
    logger.info("Starting training...")
    brain.fit(
        epoch_counter=hparams['epoch_counter'],
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams.get('train_dataloader_opts', {}),
        valid_loader_kwargs=hparams.get('valid_dataloader_opts', {}),
    )
    
    logger.info("Training completed!")
    
    # Test evaluation
    logger.info("Running test evaluation...")
    brain.evaluate(
        test_data,
        max_key='PER',
        test_loader_kwargs=hparams.get('test_dataloader_opts', {}),
    )
    
    wandb.finish()
    logger.info("All done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_interleaved.py <hparams_file>")
        print("Example: python train_interleaved.py hparams/SSL_LLM_Interleaved.yaml")
        sys.exit(1)
    
    hparams_file = sys.argv[1]
    
    if not Path(hparams_file).exists():
        print(f"Error: Hyperparameter file not found: {hparams_file}")
        sys.exit(1)
    
    main(hparams_file)
