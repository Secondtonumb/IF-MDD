"""
SpeechOcean762 Dataset Preparation Script

This script loads the SpeechOcean762 dataset from HuggingFace and converts it 
to the same format as L2-ARCTIC for mispronunciation detection.

Dataset: https://huggingface.co/datasets/mispeech/speechocean762

Author: Haopeng (Kevin) Geng
Institution: The University of Tokyo
Year: 2025
"""

import os
import json
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
from pathlib import Path
import re


def phoneme_to_arpabet(phoneme, for_train_target=False):
    """
    Convert phoneme representations to ARPAbet format (matching L2-ARCTIC).
    SpeechOcean762 uses CMU phoneme notation with stress markers (e.g., IY0, AO0).
    
    Args:
        phoneme: The phoneme to convert
        for_train_target: If True, skip <del> tokens (used for perceived_train_target)
    """
    # Handle special error tokens
    if phoneme == '<del>':
        if for_train_target:
            return None  # Skip deletion in train target
        else:
            return 'sil'  # Deletion errors -> silence in aligned
    
    if phoneme == '<unk>':
        return 'err'  # Unknown phonemes -> error token
    
    # Handle phonemes with asterisk (uncertain/error phonemes)
    # e.g., "ey*" -> "err"
    if phoneme.endswith('*'):
        return 'err'
    
    # Remove stress markers (0, 1, 2) if present
    phoneme_clean = re.sub(r'[012]', '', phoneme)
    
    # Convert to lowercase for consistency with L2-ARCTIC
    phoneme_lower = phoneme_clean.lower()
    
    # CMU to ARPAbet mapping (most are already compatible)
    phoneme_map = {
        # Vowels (CMU format already mostly ARPAbet)
        'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw',
        'ax': 'ah', 'ay': 'ay', 'eh': 'eh', 'er': 'er', 'ey': 'ey',
        'ih': 'ih', 'iy': 'iy', 'ow': 'ow', 'oy': 'oy', 'uh': 'uh',
        'uw': 'uw',
        # Consonants
        'b': 'b', 'ch': 'ch', 'd': 'd', 'dh': 'dh', 'f': 'f',
        'g': 'g', 'hh': 'hh', 'jh': 'jh', 'k': 'k', 'l': 'l',
        'm': 'm', 'n': 'n', 'ng': 'ng', 'p': 'p', 'r': 'r',
        's': 's', 'sh': 'sh', 't': 't', 'th': 'th', 'v': 'v',
        'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'zh',
        # Special tokens
        'sil': 'sil', 'sp': 'sil', 'spn': 'sil',
    }
    
    return phoneme_map.get(phoneme_lower, phoneme_lower)


def process_phoneme_sequence(phonemes, for_train_target=False):
    """
    Process phoneme sequence to match L2-ARCTIC format.
    
    Args:
        phonemes: List of phonemes or space-separated string
        for_train_target: If True, skip <del> tokens (for perceived_train_target)
        
    Returns:
        Processed phoneme string with 'sil' tokens
    """
    if isinstance(phonemes, str):
        phonemes = phonemes.strip().split()
    
    # Convert to ARPAbet and add sil tokens
    processed = ['sil']
    for phn in phonemes:
        converted = phoneme_to_arpabet(phn, for_train_target=for_train_target)
        if converted is not None:  # Skip None values (e.g., <del> in train_target)
            processed.append(converted)
    processed.append('sil')
    
    return ' '.join(processed)


def deduplicate_phonemes(phoneme_seq):
    """
    Remove consecutive duplicate phonemes (except 'sil').
    This creates the 'perceived_train_target' format.
    
    Args:
        phoneme_seq: Space-separated phoneme string
        
    Returns:
        Deduplicated phoneme string
    """
    phonemes = phoneme_seq.strip().split()
    if not phonemes:
        return ""
    
    result = [phonemes[0]]
    for phn in phonemes[1:]:
        if phn != result[-1] or phn == 'sil':
            result.append(phn)
    
    return ' '.join(result)


def prepare_speechocean762(output_dir="./data/speechocean762", 
                           audio_output_dir=None,
                           save_audio=True):  # Changed default to True
    """
    Load SpeechOcean762 dataset from HuggingFace and prepare JSON files.
    
    Args:
        output_dir: Directory to save JSON annotation files
        audio_output_dir: Directory to save audio files (if save_audio=True)
        save_audio: Whether to save audio files locally (default: True)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Always save audio by default
    if audio_output_dir is None:
        audio_output_dir = os.path.join(output_dir, "wav")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    print("Loading SpeechOcean762 dataset from HuggingFace...")
    
    # Load the dataset
    # The dataset has train and test splits
    try:
        dataset = load_dataset("mispeech/speechocean762")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative method...")
        dataset = load_dataset("mispeech/speechocean762", trust_remote_code=True)
    
    print(f"Dataset loaded. Splits: {list(dataset.keys())}")
    
    # Process each split
    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]
        
        json_data = {}
        
        for idx, example in enumerate(tqdm(split_data)):
            # Extract information from the dataset
            # Note: Field names may vary - adjust based on actual dataset structure
            
            # Common fields in SpeechOcean762:
            # - audio: audio data
            # - text: transcription
            # - phonemes: phoneme sequence
            # - accuracy_score: accuracy score per phoneme
            # - speaker_id: speaker identifier
            
            audio = example.get('audio', {})
            audio_array = audio.get('array', None)
            sample_rate = audio.get('sampling_rate', 16000)
            
            # Get file path or create one
            if 'path' in audio and audio['path']:
                audio_path = audio['path']
                file_id = Path(audio_path).stem
            else:
                # Create file ID from index
                file_id = f"{split_name}_{idx:05d}"
                audio_path = None
            
            # Always save audio to local directory
            if audio_array is not None:
                audio_filename = f"{file_id}.wav"
                audio_save_path = os.path.join(audio_output_dir, audio_filename)
                
                # Save using soundfile
                import soundfile as sf
                sf.write(audio_save_path, audio_array, sample_rate)
                
                # Use absolute path
                wav_path = os.path.abspath(audio_save_path)
            else:
                # Fallback if no audio data (shouldn't happen)
                wav_path = os.path.abspath(os.path.join(audio_output_dir, f"{file_id}.wav"))
            
            # Calculate duration
            if audio_array is not None:
                duration = len(audio_array) / sample_rate
            else:
                duration = 0.0
            
            # Get speaker ID
            speaker_id = example.get('speaker', example.get('speaker_id', 'unknown'))
            
            # Get text
            text = example.get('text', '')
            
            # Get phoneme sequences from words
            # SpeechOcean762 structure:
            # - 'words': list of word objects
            # - each word has 'phones' (canonical) and 'phones-accuracy' (scores)
            # - 'mispronunciations': list of mispronounced phonemes
            
            words = example.get('words', [])
            
            canonical_list = []
            perceived_list = []
            all_accuracy_scores = []
            
            for word in words:
                word_phones = word.get('phones', [])
                word_accuracy = word.get('phones-accuracy', [])
                mispronunciations = word.get('mispronunciations', [])
                
                # Create a mapping of phone index to mispronounced phone
                # mispronunciations format: [{"canonical-phone": "L", "index": 0, "pronounced-phone": "D"}, ...]
                mispron_map = {}
                for mispron in mispronunciations:
                    idx = mispron.get('index', -1)
                    pronounced = mispron.get('pronounced-phone', None)
                    if idx >= 0 and pronounced:
                        mispron_map[idx] = pronounced
                
                # Build canonical and perceived phoneme lists
                for i, phone in enumerate(word_phones):
                    canonical_list.append(phone)
                    
                    # Get accuracy score for this phone
                    if i < len(word_accuracy):
                        accuracy = word_accuracy[i]
                        all_accuracy_scores.append(accuracy)
                        
                        # Check if there's a mispronunciation for this phone
                        if i in mispron_map:
                            # Use the actual pronounced phone
                            perceived_list.append(mispron_map[i])
                        else:
                            # No mispronunciation, use canonical phone
                            perceived_list.append(phone)
                    else:
                        perceived_list.append(phone)
                        all_accuracy_scores.append(2.0)  # Default to correct
            
            # If no words found, try alternative field names
            if not canonical_list:
                canonical_phonemes = example.get('phonemes', example.get('reference_phonemes', ''))
                if isinstance(canonical_phonemes, str):
                    canonical_list = canonical_phonemes.strip().split()
                else:
                    canonical_list = canonical_phonemes if canonical_phonemes else []
                perceived_list = canonical_list.copy()
                all_accuracy_scores = None
            
            # Process phoneme sequences to match L2-ARCTIC format
            canonical_aligned = process_phoneme_sequence(canonical_list, for_train_target=False)
            perceived_aligned = process_phoneme_sequence(perceived_list, for_train_target=False)
            perceived_train_target = process_phoneme_sequence(perceived_list, for_train_target=True)
            # Then deduplicate
            perceived_train_target = deduplicate_phonemes(perceived_train_target)
            
            # Create entry in L2-ARCTIC format
            entry = {
                "wav": wav_path,
                "duration": duration,
                "spk_id": str(speaker_id),
                "canonical_aligned": canonical_aligned,
                "perceived_aligned": perceived_aligned,
                "perceived_train_target": perceived_train_target,
                "wrd": text,
            }
            
            # Add optional fields if available
            if all_accuracy_scores:
                entry["accuracy_scores"] = all_accuracy_scores
            
            # Add word-level accuracy and other metrics
            entry["total_score"] = example.get('total', None)
            entry["accuracy_score"] = example.get('accuracy', None)
            entry["fluency_score"] = example.get('fluency', None)
            entry["completeness_score"] = example.get('completeness', None)
            entry["prosodic_score"] = example.get('prosodic', None)
            
            # Use full path as key (matching L2-ARCTIC format)
            json_data[wav_path] = entry
        
        # Save JSON file
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(json_data)} examples to {output_file}")
    
    # Create train/dev split from train if needed
    # (similar to L2-ARCTIC's train-train.json and train-dev.json)
    if 'train' in dataset:
        print("\nCreating train/dev split...")
        train_file = os.path.join(output_dir, "train.json")
        
        with open(train_file, 'r') as f:
            full_train_data = json.load(f)
        
        # Split 90/10 for train/dev
        all_keys = list(full_train_data.keys())
        split_idx = int(len(all_keys) * 0.9)
        
        train_keys = all_keys[:split_idx]
        dev_keys = all_keys[split_idx:]
        
        train_train_data = {k: full_train_data[k] for k in train_keys}
        train_dev_data = {k: full_train_data[k] for k in dev_keys}
        
        # Save split files
        with open(os.path.join(output_dir, "train-train.json"), 'w') as f:
            json.dump(train_train_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, "train-dev.json"), 'w') as f:
            json.dump(train_dev_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created train-train.json: {len(train_train_data)} examples")
        print(f"Created train-dev.json: {len(train_dev_data)} examples")
    
    print(f"\n✅ Dataset preparation complete! Files saved to {output_dir}")
    print("\nTo use this dataset, update your YAML config:")
    print(f'  data_folder_save: "{output_dir}"')
    print(f'  train_annotation: "{output_dir}/train-train.json"')
    print(f'  valid_annotation: "{output_dir}/train-dev.json"')
    print(f'  test_annotation: "{output_dir}/test.json"')


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SpeechOcean762 dataset for mispronunciation detection"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/speechocean762",
        help="Directory to save JSON annotation files"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory to save audio files (if --save_audio is used)"
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        default=True,  # Changed default to True
        help="Save audio files locally (default: True)"
    )
    
    args = parser.parse_args()
    
    prepare_speechocean762(
        output_dir=args.output_dir,
        audio_output_dir=args.audio_dir,
        save_audio=args.save_audio
    )


if __name__ == "__main__":
    main()
