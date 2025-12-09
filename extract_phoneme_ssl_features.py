"""
使用 SSL 编码器提取音素级别特征
从音频文件使用 WavLM-Large 编码器提取帧级特征，
然后根据 CTM 时间戳按音素进行 max/mean pooling
"""

import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
import torchaudio
import sys
import argparse

# 添加 trainer 目录到 Python 路径
sys.path.insert(0, '/home/kevingenghaopeng/MDD/IF-MDD/trainer')
from AutoSSLoader import AutoSSLLoader


class PhonemeSSLExtractorWithEncoder:
    def __init__(self,
                 json_file: str,
                 ctm_file: str,
                 output_dir: str,
                 ssl_model_name: str = "wavlm_large",
                 pooling_method: str = "mean",
                 target_phn_frames: int = 50,
                 pretrained_model_path: str = "/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化特征提取器
        
        Args:
            json_file: JSON 格式的音频文件和元数据
            ctm_file: CTM 格式的音素时间戳文件
            output_dir: 输出目录
            ssl_model_name: SSL 编码器名称（默认 wavlm_large）
            pooling_method: 池化方法 ("mean" 或 "max")
            target_phn_frames: 目标音素帧数（用于 padding）
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.json_file = Path(json_file)
        self.ctm_file = Path(ctm_file)
        self.output_dir = Path(output_dir)
        self.ssl_model_name = ssl_model_name
        self.pooling_method = pooling_method
        self.target_phn_frames = target_phn_frames
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 JSON 数据
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        
        # 加载 CTM 数据
        self.ctm_data = self._load_ctm_file()
        
        # 加载 SSL 编码器
        print(f"Loading SSL encoder: {ssl_model_name}...")
        self.ssl_encoder = AutoSSLLoader(
            model_name=ssl_model_name,
            freeze=True,
            freeze_feature_extractor=True,
            save_path=self.pretrained_model_path,
            output_all_hiddens=False,
            encoder_type=None
        )
        self.ssl_encoder = self.ssl_encoder.to(self.device)
        self.ssl_encoder.eval()
        
        print(f"SSL encoder loaded successfully")
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
        print(f"Feature dimension: {self.feature_dim}")
    
    def _get_feature_dim(self) -> int:
        """从模型配置获取特征维度"""
        if hasattr(self.ssl_encoder, 'model'):
            if hasattr(self.ssl_encoder.model, 'config'):
                return self.ssl_encoder.model.config.hidden_size
        # 默认 WavLM Large 的维度是 1024
        return 1024
    
    def _load_ctm_file(self) -> dict:
        """加载 CTM 文件并按 utterance ID 组织"""
        ctm_data = defaultdict(list)
        
        with open(self.ctm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                utt_id = parts[0]
                start_time = float(parts[2])
                duration = float(parts[3])
                phone_with_info = parts[4]  # 例如 "AA0_I"
                
                # 提取纯音素标签（去掉音调和位置信息）
                phone = self._extract_phone_label(phone_with_info)
                
                ctm_data[utt_id].append({
                    'phone': phone,
                    'phone_with_info': phone_with_info,
                    'start_time': start_time,
                    'duration': duration,
                    'end_time': start_time + duration,
                })
        
        return ctm_data
    
    @staticmethod
    def _extract_phone_label(phone_with_info: str) -> str:
        """
        从带有位置和音调信息的音素标签中提取纯音素
        例如 "AA0_I" -> "AA", "M_B" -> "M"
        
        Args:
            phone_with_info: 带信息的音素标签
        
        Returns:
            纯音素标签
        """
        # 匹配模式：字母 + 可选数字 + 可选 _X
        # 例如 AA0_I -> AA, M_B -> M
        match = re.match(r'([A-Z]+)', phone_with_info)
        if match:
            return match.group(1)
        return phone_with_info
    
    def _load_and_extract_features(self, wav_path: str) -> np.ndarray:
        """
        加载音频并使用 SSL 编码器提取帧级特征
        
        Args:
            wav_path: 音频文件路径
        
        Returns:
            (num_frames, feature_dim) 的特征数组
        """
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # 重采样到 16kHz（如果需要）
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # 确保是单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 移到 GPU
            waveform = waveform.to(self.device)
            
            # 提取特征
            with torch.no_grad():
                # 计算 wav_lens（相对长度，0-1之间）
                wav_lens = torch.tensor([1.0], device=self.device)
                
                # SSL encoder 的调用方式：model(wavs, wav_lens)
                # 返回值通常是 (batch_size, num_frames, feature_dim)
                features = self.ssl_encoder(waveform, wav_lens)
            
            # 如果返回的是多层的，取最后一层
            if isinstance(features, (list, tuple)):
                features = features[-1] if isinstance(features, (list, tuple)) else features
            
            # 转换为 numpy
            features = features.cpu().numpy()  # (batch_size, num_frames, feature_dim)
            
            # 移除批次维度
            if len(features.shape) == 3:
                features = features[0]  # 第一个样本
            elif len(features.shape) == 2:
                pass  # 已经是 (num_frames, feature_dim)
            
            return features.astype(np.float32)
        
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _time_to_frame_idx(self, time_sec: float, sample_rate: int = 16000, 
                           hop_length: int = 320) -> int:
        """
        将时间（秒）转换为帧索引
        WavLM 的默认 hop_length 是 320（20ms at 16kHz）
        """
        return int(round(time_sec * sample_rate / hop_length))
    
    def _pool_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        对帧进行池化操作
        
        Args:
            frames: (num_frames, feature_dim)
        
        Returns:
            (feature_dim,) 的池化特征
        """
        if frames.shape[0] == 0:
            # 如果没有帧，返回零向量
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        if self.pooling_method == "mean":
            return frames.mean(axis=0).astype(np.float32)
        elif self.pooling_method == "max":
            return frames.max(axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    def _pad_features(self, phn_features: np.ndarray) -> np.ndarray:
        """
        将音素特征 padding 到目标长度
        
        Args:
            phn_features: (num_phonemes, feature_dim) 池化后的特征
        
        Returns:
            (target_phn_frames, feature_dim) 的 padding 特征
            其中如果 num_phonemes < target_phn_frames，用 0 padding
        """
        num_phns = phn_features.shape[0]
        
        # 创建目标大小的数组
        padded = np.zeros((self.target_phn_frames, self.feature_dim), dtype=np.float32)
        
        # 将实际的音素特征放在前面
        copy_len = min(num_phns, self.target_phn_frames)
        padded[:copy_len] = phn_features[:copy_len]
        
        return padded
    
    def extract_utterance(self, utt_id: str, wav_path: str) -> np.ndarray:
        """
        提取单个 utterance 的音素级特征
        
        Args:
            utt_id: utterance ID
            wav_path: 音频文件路径
        
        Returns:
            (num_phonemes, target_phn_frames, feature_dim) 的特征数组
        """
        # 提取帧级特征
        frame_features = self._load_and_extract_features(wav_path)
        
        # 获取该 utterance 的音素信息
        if utt_id not in self.ctm_data:
            print(f"Warning: {utt_id} not found in CTM file")
            return np.array([], dtype=np.float32)
        
        phones_info = self.ctm_data[utt_id]
        phn_features_list = []
        
        # 对每个音素进行池化
        for phn_info in phones_info:
            start_frame = self._time_to_frame_idx(phn_info['start_time'])
            end_frame = self._time_to_frame_idx(phn_info['end_time'])
            
            # 提取该音素的帧
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frame_features.shape[0])
            
            if start_frame < end_frame:
                phn_frames = frame_features[start_frame:end_frame]
            else:
                phn_frames = np.array([], dtype=np.float32).reshape(0, self.feature_dim)
            
            # 池化
            pooled = self._pool_frames(phn_frames)
            phn_features_list.append(pooled)
        
        # 堆叠所有音素特征
        if phn_features_list:
            phn_features = np.stack(phn_features_list, axis=0)  # (num_phonemes, feature_dim)
        else:
            phn_features = np.array([], dtype=np.float32).reshape(0, self.feature_dim)
        
        # Padding
        padded_features = self._pad_features(phn_features)
        return padded_features
    
    def extract_all(self) -> dict:
        """
        提取所有 utterance 的音素级特征
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_processed': 0,
            'total_phonemes': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'phoneme_count_distribution': {}
        }
        
        output_data = {}
        
        # 构建 utt_id 到 wav 路径的映射
        utt_id_to_wav = {}
        for wav_path, info in self.json_data.items():
            # 从路径中提取 utterance ID（如 "000030012" 从 "...000030012.WAV"）
            utt_id = Path(wav_path).stem
            utt_id_to_wav[utt_id] = wav_path
        
        # 获取 CTM 中的所有 utterance IDs
        utt_ids = sorted(set(self.ctm_data.keys()) & set(utt_id_to_wav.keys()))
        
        print(f"Found {len(utt_ids)} utterances to process")
        print(f"(CTM: {len(self.ctm_data)}, JSON: {len(self.json_data)})")
        
        for utt_id in tqdm(utt_ids, desc="Extracting features"):
            try:
                wav_path = utt_id_to_wav[utt_id]
                
                # 验证文件存在
                if not Path(wav_path).exists():
                    stats['failed'] += 1
                    stats['errors'].append(f"Audio file not found: {wav_path}")
                    continue
                
                # 提取特征
                phn_features = self.extract_utterance(utt_id, wav_path)
                
                # 检查返回的特征
                if phn_features is None or phn_features.size == 0:
                    stats['failed'] += 1
                    stats['errors'].append(f"Failed to extract features for {utt_id}")
                    continue
                
                output_data[utt_id] = phn_features
                    
                # 统计
                num_phns = phn_features.shape[0]
                if num_phns not in stats['phoneme_count_distribution']:
                    stats['phoneme_count_distribution'][num_phns] = 0
                stats['phoneme_count_distribution'][num_phns] += 1
                
                stats['successful'] += 1
                stats['total_phonemes'] += num_phns
                    
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(f"Error processing {utt_id}: {str(e)}")
                continue
            
            stats['total_processed'] += 1
        
        # 整合所有数据为单个数组 [2500, 50, 1024]
        print(f"\nOrganizing output data...")
        
        # 检查是否有成功提取的特征
        if len(output_data) == 0:
            print("ERROR: No features extracted! Check error messages above.")
            print(f"Total processed: {stats['total_processed']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            if stats['errors']:
                print("\nFirst 10 errors:")
                for error in stats['errors'][:10]:
                    print(f"  - {error}")
            return None, stats, None, None
        
        # 按 utterance ID 排序确保一致性
        sorted_utt_ids = sorted(output_data.keys())
        
        # 直接堆叠，因为每个特征已经是 (50, 1024)
        final_features = np.stack([output_data[utt_id] for utt_id in sorted_utt_ids], axis=0)
        
        # 保存为 NPY 文件
        output_file = self.output_dir / f"phoneme_ssl_features_{self.ssl_model_name}_{self.pooling_method}.npy"
        np.save(output_file, final_features)
        print(f"\nSaved to: {output_file}")
        print(f"Output shape: {final_features.shape}")
        
        # 打印统计信息
        print("\n" + "="*60)
        print("Extraction Statistics:")
        print("="*60)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total phonemes: {stats['total_phonemes']}")
        print(f"Average phonemes per utterance: {stats['total_phonemes']/max(1, stats['successful']):.1f}")
        print(f"Model: {self.ssl_model_name}, Pooling: {self.pooling_method}")
        print(f"\nPhoneme count distribution (first 10):")
        for count in sorted(stats['phoneme_count_distribution'].keys())[:10]:
            freq = stats['phoneme_count_distribution'][count]
            print(f"  {count} phonemes: {freq} utterances")
        
        if stats['errors']:
            print(f"\nFirst 5 errors:")
            for error in stats['errors'][:5]:
                print(f"  - {error}")
        
        return output_data, stats, sorted_utt_ids, final_features, sorted_utt_ids, final_features


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='提取音素级别的 SSL 特征')
    
    parser.add_argument('--json-file', type=str, 
                        default='/home/kevingenghaopeng/MDD/IF-MDD/data/speechocean762_with_word_scores/test.json',
                        help='JSON 格式的音频文件和元数据')
    
    parser.add_argument('--ctm-file', type=str,
                        default='/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/te_phones_nosil.ctm',
                        help='CTM 格式的音素时间戳文件')
    
    parser.add_argument('--output-dir', type=str,
                        default='/home/kevingenghaopeng/MDD/IF-MDD/data_so762/phoneme_ssl_features',
                        help='输出目录')
    
    parser.add_argument('--ssl-model', type=str, default='wavlm_large',
                        choices=['wavlm_base', 'wavlm_large', 'hubert_base', 'wav2vec2_base'],
                        help='SSL 模型名称')
    
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='池化方法')
    
    parser.add_argument('--target-phn-frames', type=int, default=50,
                        help='目标音素帧数（padding 维度）')
    
    parser.add_argument('--pretrained-model-path', type=str,
                        default='/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models',
                        help='预训练模型路径')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='运行设备')
    
    args = parser.parse_args()
    
    # 创建配置字典
    config = {
        'json_file': args.json_file,
        'ctm_file': args.ctm_file,
        'output_dir': args.output_dir,
        'ssl_model_name': args.ssl_model,
        'pooling_method': args.pooling,
        'target_phn_frames': args.target_phn_frames,
        'pretrained_model_path': args.pretrained_model_path,
        'device': args.device,
    }
    
    # 打印配置
    print("="*60)
    print("Configuration:")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 创建提取器
    extractor = PhonemeSSLExtractorWithEncoder(
        json_file=config['json_file'],
        ctm_file=config['ctm_file'],
        output_dir=config['output_dir'],
        ssl_model_name=config['ssl_model_name'],
        pooling_method=config['pooling_method'],
        target_phn_frames=config['target_phn_frames'],
        pretrained_model_path=config['pretrained_model_path'],
        device=config['device']
    )
    
    # 执行提取
    output_data, stats, sorted_utt_ids, final_features, sorted_utt_ids, final_features = extractor.extract_all()
    
    # 检查是否提取成功
    if final_features is None:
        print("\nExtraction failed. Exiting...")
        return None, stats, None, None
    
    # 验证数据集
    print("\n" + "="*60)
    print("Dataset Verification:")
    print("="*60)
    print(f"Total utterances processed: {len(output_data)}")
    print(f"Expected: 2500")
    print(f"Final output shape: {final_features.shape}")
    print(f"Expected shape: (2500, 50, 1024)")
    print(f"Data type: {final_features.dtype}")
    print(f"Feature range: [{final_features.min():.3f}, {final_features.max():.3f}]")
    
    # 计算文件大小
    file_size_gb = final_features.nbytes / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    if len(output_data) > 0:
        first_utt = sorted_utt_ids[0]
        print(f"\nFirst utterance ({first_utt}):")
        print(f"  Shape: {output_data[first_utt].shape}")
        print(f"  Non-zero phonemes: {(output_data[first_utt].sum(axis=1) != 0).sum()}")
        print(f"  Feature range: [{output_data[first_utt].min():.3f}, {output_data[first_utt].max():.3f}]")
    
    return output_data, stats, sorted_utt_ids, final_features, sorted_utt_ids, final_features


if __name__ == '__main__':
    output_data, stats, sorted_utt_ids, final_features, sorted_utt_ids, final_features = main()