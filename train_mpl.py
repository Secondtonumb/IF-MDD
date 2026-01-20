"""
Training script for Momentum Pseudo Label (MPL) training.

This script demonstrates how to use PhnMonoSSLModel_MPL for semi-supervised
training with labeled and unlabeled data.

Usage:
    python train_mpl.py hparams/mpl_training.yaml --data_folder /path/to/data
    
Key concepts:
- Student model learns from both labeled data and pseudo-labeled (unlabeled) data
- Teacher model generates pseudo labels and is updated via EMA
- Teacher sees clean audio, student sees augmented audio (consistency regularization)
"""

import os
import sys
import time
import logging
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from speechbrain.dataio.batch import PaddedBatch
import wandb

from models.phn_mono_ssl_model_mpl import (
    PhnMonoSSLModel_MPL,
    PhnMonoSSLModel_MPL_Interleaved,
    setup_mpl_dataloaders,
)

logger = logging.getLogger(__name__)


def dataio_prep(hparams):
    """
    Prepare datasets for MPL training.
    
    Creates:
    - Labeled training data (with targets)
    - Unlabeled training data (audio only)
    - Validation data
    - Test data
    """
    data_folder = hparams.get("data_folder_save", hparams.get("data_folder", "."))
    
    # =========================================================================
    # 1. Load datasets from JSON
    # =========================================================================
    
    # Labeled training data
    train_data_l = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    
    # Sort if specified
    sorting = hparams.get("sorting", "random")
    if sorting == "ascending":
        train_data_l = train_data_l.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif sorting == "descending":
        train_data_l = train_data_l.filtered_sorted(sort_key="duration", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False
    
    # Unlabeled training data
    train_data_u = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["unlabeled_annotation"],
        replacements={"data_root": data_folder},
    )
    
    # Validation data
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    
    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")
    
    datasets = [train_data_l, train_data_u, valid_data, test_data]
    
    # =========================================================================
    # 2. Label encoder
    # =========================================================================
    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    
    # =========================================================================
    # 3. Audio pipeline
    # =========================================================================
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        import torchaudio
        import librosa
        
        # Try torchaudio first, fallback to librosa
        try:
            sig, sr = torchaudio.load(wav)
            target_sr = hparams.get("sample_rate", 16000)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                sig = resampler(sig)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
            sig = sig.squeeze(0)
        except:
            target_sr = hparams.get("sample_rate", 16000)
            sig, _ = librosa.load(wav, sr=target_sr)
            sig = torch.tensor(sig)
        
        return sig
    
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    
    # =========================================================================
    # 4. Text pipeline for labeled data
    # =========================================================================
    
    # Training target field name (same as your existing configs)
    target_field = hparams.get("target_field", "perceived_train_target")
    canonical_field = hparams.get("canonical_field", "canonical_aligned")
    perceived_field = hparams.get("perceived_field", "perceived_aligned")
    
    @sb.utils.data_pipeline.takes(target_field)
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
    )
    def text_pipeline_train(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
    
    @sb.utils.data_pipeline.takes(target_field, canonical_field, perceived_field)
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
    )
    def text_pipeline_eval(target, canonical, perceived):
        # Target
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        
        # Canonical
        phn_list_canonical = canonical.strip().split()
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        
        # Perceived
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived
    
    # Apply different pipelines
    sb.dataio.dataset.add_dynamic_item([train_data_l], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_eval)
    # Note: train_data_u doesn't need text pipeline (unlabeled)
    
    # =========================================================================
    # 5. Load or create label encoder
    # =========================================================================

    save_folder = hparams["save_folder"]
    os.makedirs(save_folder, exist_ok=True)
    if hparams['lab_enc_file'] is None:
        # raise ValueError("Label encoder must support insert_bos_eos method")
        lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
        # for L2_arctic only
        label_encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=42,
            eos_index=43,
        )
    else:
        lab_enc_file = hparams['lab_enc_file']
        # copy lab_enc_file to current exp folder
        import shutil
        try:
            shutil.copy(lab_enc_file, os.path.join(save_folder, "label_encoder.txt"))
            lab_enc_file = os.path.join(save_folder, "label_encoder.txt")
        except Exception as e:
            print(f"Error copying label encoder file: {e}")
            import pdb; pdb.set_trace()
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data_l],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )
    
    # =========================================================================
    # 6. Set output keys
    # =========================================================================
    sb.dataio.dataset.set_output_keys(
        [train_data_l],
        ["id", "sig", "phn_encoded_target"],
    )
    sb.dataio.dataset.set_output_keys(
        [train_data_u],
        ["id", "sig"],  # Unlabeled: audio only
    )
    sb.dataio.dataset.set_output_keys(
        [valid_data, test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
    )
    
    return train_data_l, train_data_u, valid_data, test_data, label_encoder


def main():
    # =========================================================================
    # Parse arguments
    # =========================================================================
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # =========================================================================
    # Initialize distributed training
    # =========================================================================
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # =========================================================================
    # Create experiment directory
    # =========================================================================
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # =========================================================================
    # Prepare data
    # =========================================================================
    train_data_l, train_data_u, valid_data, test_data, label_encoder = dataio_prep(hparams)
    
    # limit train_data for quick debugging
    # train_record_l = train_data_l.data_ids[:512]  # Select first 1024 for debugging
    # train_record_u = train_data_u.data_ids[:512]  # Select first 1024 for debugging
    # train_data_l = train_data_l.filtered_sorted(key_test={"id": lambda x: x in train_record_l},)
    # valid_record = valid_data.data_ids[:128]  # Select first 128 for debugging
    # train_data_u = train_data_u.filtered_sorted(key_test={"id": lambda x: x in train_record_u},)
    # valid_data = valid_data.filtered_sorted(key_test={"id": lambda x: x in valid_record},)
    # test_record = test_data.data_ids[:10]
    # test_data = test_data.filtered_sorted(key_test={"id": lambda x: x in test_record},)
    
    
    # =========================================================================
    # Create data loaders
    # =========================================================================
    batch_size_l = hparams.get("batch_size_labeled", hparams.get("batch_size", 8))
    batch_size_u = hparams.get("batch_size_unlabeled", batch_size_l)
    num_workers = hparams.get("num_workers", 4)
    
    train_loader_l = DataLoader(
        train_data_l,
        batch_size=batch_size_l,
        drop_last=False,
        shuffle=True,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    train_loader_u = DataLoader(
        train_data_u,
        batch_size=batch_size_u,
        drop_last=False,
        shuffle=True,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size_l,
        drop_last=False,
        shuffle=False,
        collate_fn=PaddedBatch,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        collate_fn=PaddedBatch,
        num_workers=1
    )
    
    # =========================================================================
    # Initialize wandb
    # =========================================================================
    run_name = hparams.get("run_name", f"mpl_{time.strftime('%Y%m%d-%H%M%S')}")
    wandb.init(
        project=hparams.get("wandb_project", "IF-MDD-MPL"),
        name=run_name,
        config=hparams,
    )
    
    # =========================================================================
    # Initialize model
    # =========================================================================
    mpl_variant = hparams.get("mpl_variant", "chain")  # "chain" or "interleaved"
    
    if mpl_variant == "interleaved":
        ModelClass = PhnMonoSSLModel_MPL_Interleaved
    else:
        ModelClass = PhnMonoSSLModel_MPL
    
    asr_brain = ModelClass(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    asr_brain.label_encoder = label_encoder
    
    # Initialize teacher modules if defined in hparams
    if "modules_teacher" in hparams:
        asr_brain.modules_teacher = torch.nn.ModuleDict(
            hparams["modules_teacher"]
        ).to(asr_brain.device)
    
    # =========================================================================
    # Limit datasets for quick debugging (if enabled)
    # =========================================================================
    use_debug_mode = hparams.get("use_debug_mode", False)
    debug_unlabeled_ratio = hparams.get("debug_unlabeled_ratio", None)  # 单个比例，如 1, 5, 或 "full"
    debug_seed = hparams.get("debug_seed", 42)
    
    if use_debug_mode and debug_unlabeled_ratio is not None:
        logger.info(f"🔍 Debug mode enabled")
        logger.info(f"  📊 Labeled samples (fixed): {len(train_data_l)}")
        logger.info(f"  🔄 Unlabeled ratio: {debug_unlabeled_ratio}")
        logger.info(f"  🎲 Random seed: {debug_seed}")
        
        # 固定 labeled data（不变）
        num_labeled = len(train_data_l)
        
        # 固定随机种子
        torch.manual_seed(debug_seed)
        import numpy as np
        np.random.seed(debug_seed)
        
        all_unlabeled_ids = train_data_u.data_ids.copy()
        np.random.shuffle(all_unlabeled_ids)
        
        # 根据指定的比例获取样本
        if debug_unlabeled_ratio == "full":
            num_unlabeled = len(all_unlabeled_ids)
            selected_ids = all_unlabeled_ids
            ratio_name = "full"
        else:
            # 数值比例，如 1, 5
            ratio_value = int(debug_unlabeled_ratio)
            num_unlabeled = num_labeled * ratio_value
            selected_ids = all_unlabeled_ids[:num_unlabeled]
            ratio_name = f"{ratio_value}:1"
        
        logger.info(f"  ✓ {ratio_name} ratio → {num_unlabeled} unlabeled samples")
        logger.info(f"  📈 Valid samples (unchanged): {len(valid_data)}")
        logger.info(f"  📉 Test samples (unchanged): {len(test_data)}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Training with Unlabeled {ratio_name} ({num_unlabeled} samples)")
        logger.info(f"{'='*60}")
        
        # 过滤 unlabeled data
        train_data_u_subset = train_data_u.filtered_sorted(
            key_test={"id": lambda x: x in selected_ids}
        )
        
        # 创建 DataLoader
        train_loader_u_subset = DataLoader(
            train_data_u_subset,
            batch_size=batch_size_u,
            drop_last=False,
            shuffle=True,
            collate_fn=PaddedBatch,
            num_workers=num_workers
        )
        
        # 训练
        asr_brain.fit_mpl(
            epoch_counter=hparams["epoch_counter"],
            train_data_l=train_loader_l,
            train_data_u=train_loader_u_subset,
            valid_set=valid_loader,
        )
        
        logger.info(f"✅ Training completed for {ratio_name} ratio")
    else:
        # 正常训练：使用全部数据
        asr_brain.fit_mpl(
            epoch_counter=hparams["epoch_counter"],
            train_data_l=train_loader_l,
            train_data_u=train_loader_u,
            valid_set=valid_loader,
        )
    
    # =========================================================================
    # Test
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Test Evaluation")
    logger.info("=" * 60)
    
    asr_brain.evaluate(
        test_loader,
        min_key="PER",
        test_loader_kwargs=None,
    )
    
    wandb.finish()
    logger.info("✅ MPL Training completed!")


if __name__ == "__main__":
    main()
