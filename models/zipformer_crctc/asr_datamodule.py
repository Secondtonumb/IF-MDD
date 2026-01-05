# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#                2023  John Hopkins University  (author: Dongji Gao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from glob import glob

import torch
from lhotse import CutSet, load_manifest, load_manifest_lazy, Fbank
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
    OnTheFlyFeatures,
    PerturbVolume,
    PerturbSpeed
)
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from lhotse.dataset.input_strategies import AudioSamples  # noqa F401 For AudioSamples
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from lhotse.shar.readers import LazySharIterator

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class MultilingualAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--subset",
            type=int,
            default=25,
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=600.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=10,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=3.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="AudioSamples",
            help="AudioSamples or PrecomputedFeatures",
        )

        group.add_argument(
            "--train-manifest",
            type=str,
            default="librispeech_cuts_train-clean-100.jsonl.gz",
            help="Train manifest file.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = [PerturbVolume(scale_low=0.125, scale_high=2.0, p=0.5),]
                      #PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)]
        
        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms
        
        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                    max_frames_mask_fraction=0.15
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank()),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 50,
                shuffle_buffer_size=self.args.num_buckets * 100,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_iter_dataset = IterableDatasetWrapper(
                                dataset=train,
                                sampler=train_sampler,
                             )

        train_dl = DataLoader(
            train_iter_dataset,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")

        validate = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_strategy=OnTheFlyFeatures(Fbank()),
            return_cuts=self.args.return_cuts,
        )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        valid_iter_dataset = IterableDatasetWrapper(
                                dataset=validate,
                                sampler=valid_sampler,
                             )

        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            valid_iter_dataset,
            batch_size=None,
            num_workers=1,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank()),
            return_cuts=self.args.return_cuts,
        )

        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        test_iter_dataset = IterableDatasetWrapper(
                                dataset=test,
                                sampler=sampler,
                             )

        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test_iter_dataset,
            batch_size=None,
            num_workers=self.args.num_workers,
        )
        return test_dl


    '''
    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train cuts"
        )
        
        return CutSet.from_shar(
            fields={
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000000.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000001.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000002.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000003.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000004.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/cuts.000005.jsonl.gz",],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000000.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000001.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000002.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000003.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000004.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/librispeech/shar_clean/train_all/recording.000005.tar",]
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True
        )
    '''
    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train cuts of all languages"
        )
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/train_datasets'
        cuts = sorted(glob(os.path.join(data_path,'cuts*')))
        recordings =  sorted(glob(os.path.join(data_path,'recording*')))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True,
            ).repeat()

    @lru_cache()
    def train_all_unsup_cuts(self) -> CutSet:
        logging.info(
            "About to get the unsupervised train cuts of all languages"
        )
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/unsup_train_filtered'
        cuts = sorted(glob(os.path.join(data_path,'cuts*')))
        recordings =  sorted(glob(os.path.join(data_path,'recording*')))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True,
            ).repeat()
    
    @lru_cache()
    def train_all_shuf_cuts_25(self) -> CutSet:
        logging.info(
            "About to get the shuffled 25% train cuts of all languages"
        )
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/train_datasets_25'
        cuts = sorted(glob(os.path.join(data_path,'cuts*')))
        recordings =  sorted(glob(os.path.join(data_path,'recording*')))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True,
            ).repeat()

    @lru_cache()
    def train_all_shuf_cuts_50(self) -> CutSet:
        logging.info(
            "About to get the shuffled 50% train cuts of all languages"
        )
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/train_datasets_50'
        cuts = sorted(glob(os.path.join(data_path,'cuts*')))
        recordings =  sorted(glob(os.path.join(data_path,'recording*')))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True,
            ).repeat()

    @lru_cache()
    def train_all_shuf_cuts_75(self) -> CutSet:
        logging.info(
            "About to get the shuffled 75% train cuts of all languages"
        )
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/train_datasets_75'
        cuts = sorted(glob(os.path.join(data_path,'cuts*')))
        recordings =  sorted(glob(os.path.join(data_path,'recording*')))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            },
            split_for_dataloading=True,
            shuffle_shards=True,
            stateful_shuffle=True,
            ).repeat()
    
    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get val cuts")

        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/train-clean-100/cuts.000000.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/train-clean-100/recording.000000.tar"]
            }
        )
    '''
    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get val cuts")

        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/cuts.000000.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/cuts.000001.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/cuts.000002.jsonl.gz","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/cuts.000003.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/recording.000000.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/recording.000001.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/recording.000002.tar","/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/fleurs_merged/recording.000003.tar"]
            }
        )
    '''
    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get val cuts")

        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/dev-clean/cuts.000000.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/dev-clean/recording.000000.tar"]
            }
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get val cuts")

        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/dev-other/cuts.000000.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/dev-other/recording.000000.tar"]
            }
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-clean/cuts.000000.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-clean/recording.000000.tar"]
            }
        )

    @lru_cache()
    def test_ucla_cuts(self) -> CutSet:
        logging.info("About to get UCLA test cuts")
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/vox_angeles'
        cuts = sorted(glob(os.path.join(data_path,"**",'cuts*'),recursive=True))
        recordings =  sorted(glob(os.path.join(data_path,"**",'recording*'),recursive=True))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            }
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return CutSet.from_shar(
            {
                "cuts": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-other/cuts.000000.jsonl.gz"],
                "recording": ["/scratch/lingjzhu_root/lingjzhu1/lingjzhu/datasets/clean/librispeech_shar/test-other/recording.000000.tar"]
            }
        )
    
    @lru_cache()
    def test_doreco_cuts(self) -> CutSet:
        logging.info("About to get doreco test cuts")
        data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/doreco_shar'
        cuts = sorted(glob(os.path.join(data_path,"**",'cuts*'),recursive=True))
        recordings =  sorted(glob(os.path.join(data_path,"**",'recording*'),recursive=True))
        
        return CutSet.from_shar(
            fields={
                "cuts": cuts,
                "recording": recordings
            }
        )

    @lru_cache()
    def test_aishell_cuts(self) -> CutSet:
        logging.info("About to get aishell test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/aishell_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/aishell_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_german_cuts(self) -> CutSet:
        logging.info("About to get mls german test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_german_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_german_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_italian_cuts(self) -> CutSet:
        logging.info("About to get mls italian test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_italian_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_italian_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_french_cuts(self) -> CutSet:
        logging.info("About to get mls french test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_french_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_french_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_spanish_cuts(self) -> CutSet:
        logging.info("About to get mls spanish test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_spanish_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_spanish_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_portuguese_cuts(self) -> CutSet:
        logging.info("About to get mls portuguese test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_portuguese_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_portuguese_shar/test/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_mls_dutch_cuts(self) -> CutSet:
        logging.info("About to get mls dutch test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_dutch_shar/test/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/compressed_datasets/mls_dutch_shar/test/recording.000000.tar']
            }
        )
    
    
    @lru_cache()
    def test_buckeye_cuts(self) -> CutSet:
        logging.info("About to get buckeye test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/multilingual/buckeye_shar/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/multilingual/buckeye_shar/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_l2_arctic_cuts(self) -> CutSet:
        logging.info("About to get l2_arctic test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_shar/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_shar/recording.000000.tar']
            }
        )
    
    
    @lru_cache()
    def test_l2_arctic_preceived_cuts(self) -> CutSet:
        logging.info("About to get l2_arctic_perceived test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_perceived_shar/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/l2arctic_perceived_shar/recording.000000.tar']
            }
        )

    @lru_cache()
    def test_azerbaijani_cuts(self) -> CutSet:
        logging.info("About to get azerbaijani test cuts")
         
        return CutSet.from_shar(
            fields={
                "cuts": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/Q138-aze-shar/cuts.000000.jsonl.gz'],
                "recording": ['/scratch/lingjzhu_root/lingjzhu1/lingjzhu/Q138-aze-shar/recording.000000.tar']
            }
        )
    
if __name__ == "__main__":

    logging.info(
        "About to get the shuffled train cuts of all languages")

    data_path = '/scratch/lingjzhu_root/lingjzhu1/lingjzhu/clean_train_datasets'
    cuts = sorted(glob(os.path.join(data_path,'cuts*')))
    recordings =  sorted(glob(os.path.join(data_path,'recording*')))

    cutset = CutSet.from_shar(
        fields={
            "cuts": cuts,
            "recording": recordings
        },
        split_for_dataloading=True,
        shuffle_shards=True,
        stateful_shuffle=True,
    )
        
    cutset.describe()