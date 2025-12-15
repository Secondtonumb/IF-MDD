"""
完整版本：处理所有 2500 个 utterance 的音素级特征提取
使用 WavLM-Large SSL 编码器 + mean pooling + (num_phns, 50, 1024) 输出格式
"""

import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
import torch
import torchaudio
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/kevingenghaopeng/MDD/IF-MDD/trainer')
from AutoSSLoader import AutoSSLLoader


class FullPhonemeSSLExtractor:
    def __init__(self):
        self.json_file = '/home/kevingenghaopeng/MDD/IF-MDD/data/speechocean762_with_word_scores/test.json'
        self.ctm_file = '/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/te_phones_nosil.ctm'
        self.output_dir = Path('/home/kevingenghaopeng/MDD/IF-MDD/data_so762/phoneme_ssl_features')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_phn_frames = 50
        self.feature_dim = 1024
        
        # 加载数据
        print("Loading JSON metadata...")
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        
        print("Loading CTM file...")
        self.ctm_data = self._load_ctm_file()
        
        # 加载模型
        print("Loading WavLM-Large model...")
        self.ssl_encoder = AutoSSLLoader(
            model_name='wavlm_large',
            freeze=True,
            freeze_feature_extractor=True,
            save_path='/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models',
            output_all_hiddens=False
        )
        self.ssl_encoder = self.ssl_encoder.to(self.device)
        self.ssl_encoder.eval()
        print(f"Model loaded on device: {self.device}")
    
    def _load_ctm_file(self):
        """加载 CTM 文件"""
        ctm_data = defaultdict(list)
        
        with open(self.ctm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                utt_id = parts[0]
                start_time = float(parts[2])
                duration = float(parts[3])
                phone_with_info = parts[4]
                
                # 提取纯音素（去掉音调和位置信息）
                match = re.match(r'([A-Z]+)', phone_with_info)
                phone = match.group(1) if match else phone_with_info
                
                ctm_data[utt_id].append({
                    'phone': phone,
                    'phone_with_info': phone_with_info,
                    'start_time': start_time,
                    'duration': duration,
                    'end_time': start_time + duration,
                })
        
        return ctm_data
    
    def _time_to_frame_idx(self, time_sec, sample_rate=16000, hop_length=320):
        """将时间转换为帧索引"""
        return int(round(time_sec * sample_rate / hop_length))
    
    def extract_sample(self, utt_id, wav_path):
        """提取单个样本的音素级特征"""
        try:
            # 加载音频
            waveform, sr = torchaudio.load(wav_path)
            
            # 重采样到 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # 单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            waveform = waveform.to(self.device)
            
            # 提取特征
            with torch.no_grad():
                wav_lens = torch.tensor([1.0], device=self.device)
                features = self.ssl_encoder(waveform, wav_lens)
            
            features = features.cpu().numpy()[0]  # (num_frames, 1024)
            
            # 按音素进行 mean pooling
            phones_info = self.ctm_data[utt_id]
            phn_features_list = []
            
            for phn_info in phones_info:
                start_frame = self._time_to_frame_idx(phn_info['start_time'])
                end_frame = self._time_to_frame_idx(phn_info['end_time'])
                
                start_frame = max(0, start_frame)
                end_frame = min(end_frame, features.shape[0])
                
                if start_frame < end_frame:
                    phn_frames = features[start_frame:end_frame]
                    pooled = phn_frames.mean(axis=0).astype(np.float32)
                else:
                    pooled = np.zeros(self.feature_dim, dtype=np.float32)
                
                phn_features_list.append(pooled)
            
            # 堆叠并 padding
            if phn_features_list:
                phn_features = np.stack(phn_features_list, axis=0)  # (num_phns, 1024)
                num_phns = phn_features.shape[0]
                
                # Padding 到 (num_phns, 50, 1024)
                padded = np.zeros((num_phns, self.target_phn_frames, self.feature_dim), 
                                   dtype=np.float32)
                # 在第一个位置放置 mean-pooled 特征，其余用 0 padding
                padded[:, 0, :] = phn_features
                
                return padded, num_phns
            
            return None, 0
            
        except Exception as e:
            print(f"  Error: {e}")
            return None, 0
    
    def run(self):
        """处理所有 2500 个 utterance"""
        # 构建 utt_id 到 wav 路径的映射
        utt_id_to_wav = {}
        for wav_path in self.json_data.keys():
            utt_id = Path(wav_path).stem
            utt_id_to_wav[utt_id] = wav_path
        
        # 获取需要处理的 utt ids（CTM 和 JSON 的交集）
        # dont sort to preserve original order
        # use ctm data's order 
        utt_ids = sorted(set(self.ctm_data.keys()) & set(utt_id_to_wav.keys()))
        
        print(f"\nFound {len(utt_ids)} utterances to process (CTM: {len(self.ctm_data)}, JSON: {len(self.json_data)})")
        
        # 提取特征
        output_data = {}
        stats = {
            'successful': 0,
            'failed': 0,
            'total_phonemes': 0,
            'max_phonemes': 0,
            'phoneme_dist': defaultdict(int)
        }
        
        for utt_id in tqdm(utt_ids, desc="Extracting features"):
            wav_path = utt_id_to_wav[utt_id]
            
            # 验证文件存在
            if not Path(wav_path).exists():
                stats['failed'] += 1
                continue
            
            padded, num_phns = self.extract_sample(utt_id, wav_path)
            
            if padded is not None:
                output_data[utt_id] = padded
                stats['successful'] += 1
                stats['total_phonemes'] += num_phns
                stats['max_phonemes'] = max(stats['max_phonemes'], num_phns)
                stats['phoneme_dist'][num_phns] += 1
            else:
                stats['failed'] += 1
        
        print(f"\n{'='*60}")
        print("Extraction Complete")
        print(f"{'='*60}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total phonemes: {stats['total_phonemes']}")
        print(f"Max phonemes per utterance: {stats['max_phonemes']}")
        print(f"Avg phonemes per utterance: {stats['total_phonemes']/max(1, stats['successful']):.1f}")
        
        # 合并为单个数组
        print(f"\nOrganizing output: {len(output_data)} utterances...")
        # import pdb; pdb.set_trace()
        sorted_utts = sorted(output_data.keys())
        max_phns = stats['max_phonemes']
        
        final_array = np.zeros((len(sorted_utts), max_phns, self.target_phn_frames, self.feature_dim),
                                dtype=np.float32)
        
        for i, utt_id in enumerate(sorted_utts):
            num_phns = output_data[utt_id].shape[0]
            final_array[i, :num_phns] = output_data[utt_id]
        
        # 保存
        output_file = self.output_dir / 'phoneme_ssl_features_wavlm_large_mean.npy'
        np.save(output_file, final_array)
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Shape: {final_array.shape}")
        print(f"  Format: (num_utterances={final_array.shape[0]}, max_phonemes={final_array.shape[1]}, "
              f"phn_frames={final_array.shape[2]}, feature_dim={final_array.shape[3]})")
        print(f"  Feature range: [{final_array.min():.3f}, {final_array.max():.3f}]")
        print(f"  File size: {output_file.stat().st_size / (1024**3):.2f} GB")
        
        # 保存元数据
        metadata = {
            'model': 'wavlm_large',
            'pooling_method': 'mean',
            'target_phn_frames': self.target_phn_frames,
            'feature_dim': self.feature_dim,
            'num_utterances': len(sorted_utts),
            'max_phonemes': max_phns,
            'total_phonemes': stats['total_phonemes'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'utterance_ids': sorted_utts,
        }
        
        metadata_file = self.output_dir / 'metadata.json'
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_file}")
        
        return final_array, metadata


if __name__ == '__main__':
    extractor = FullPhonemeSSLExtractor()
    final_array, metadata = extractor.run()
