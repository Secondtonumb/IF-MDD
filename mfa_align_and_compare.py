"""
使用 MFA (Montreal Forced Aligner) 对 L2-ARCTIC 数据进行对齐，
并与人工标注的 ground truth 进行比较。

依赖:
- Montreal Forced Aligner (mfa)
- textgrid
- numpy

使用方法:
    python mfa_align_and_compare.py \
        --src_dir /common/db/L2-ARCTIC \
        --output_dir /home/kevingenghaopeng/MDD/IF-MDD/data/l2arctic_mfa \
        --test_spks TLV NJS TNI TXHC ZHAA YKWK
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import defaultdict
import numpy as np

try:
    from textgrid import TextGrid, IntervalTier
    TEXTGRID_AVAILABLE = True
except ImportError:
    TEXTGRID_AVAILABLE = False
    TextGrid = None
    IntervalTier = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None


def is_sil(s: str) -> bool:
    """判断是否为静音标记"""
    return s.lower() in {"sil", "sp", "spn", "pau", "", "sil"}


def remove_stress(phone: str) -> str:
    """
    去掉音素的 stress 标记
    例如: ao1 -> ao, iy0 -> iy
    """
    # 去除数字（stress 标记）
    return re.sub(r'\d+', '', phone)


def normalize_phone_for_mfa(s: str) -> str:
    """
    将 L2-ARCTIC 的音素标记转换为 ARPA 格式用于 MFA
    L2-ARCTIC 使用如 "AH,AX,A" 或 "B,B,S" 的格式
    """
    t = s.lower()
    # 去除非字母和逗号
    t = re.sub(r"[^a-z,]", "", t)
    
    if is_sil(t):
        return "sil"
    
    if len(t) == 0:
        return "sil"
    
    parts = t.split(",")
    # 使用 canonical phoneme (第一个部分)
    cano = parts[0]
    
    # ax -> ah 映射
    if cano == "ax":
        return "ah"
    
    return cano


def extract_ground_truth_from_textgrid(
    textgrid_path: Path
) -> Tuple[List[str], List[float], List[float]]:
    """
    从 L2-ARCTIC TextGrid 提取 ground truth 对齐信息
    返回: (phonemes, start_times, end_times)
    """
    tg = TextGrid.fromFile(str(textgrid_path))
    
    # L2-ARCTIC TextGrid 通常有 'phones' tier
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() == "phones":
            phone_tier = tier
            break
    
    if phone_tier is None:
        raise ValueError(f"No 'phones' tier found in {textgrid_path}")
    
    phonemes = []
    start_times = []
    end_times = []
    
    for interval in phone_tier:
        phone = normalize_phone_for_mfa(interval.mark)
        if phone:  # 跳过空标记
            # 去除 stress 标记
            phone = remove_stress(phone)
            phonemes.append(phone)
            start_times.append(interval.minTime)
            end_times.append(interval.maxTime)
    
    return phonemes, start_times, end_times


def extract_mfa_alignment(textgrid_path: Path) -> Tuple[List[str], List[float], List[float]]:
    """
    从 MFA 生成的 TextGrid 提取对齐信息
    返回: (phonemes, start_times, end_times)
    """
    tg = TextGrid.fromFile(str(textgrid_path))
    
    # MFA 通常使用 'phones' tier
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in {"phones", "phone"}:
            phone_tier = tier
            break
    
    if phone_tier is None:
        raise ValueError(f"No 'phones' tier found in {textgrid_path}")
    
    phonemes = []
    start_times = []
    end_times = []
    
    for interval in phone_tier:
        phone = interval.mark.lower()
        if phone and phone != "":  # 跳过空标记
            # 去除 stress 标记
            phone = remove_stress(phone)
            phonemes.append(phone)
            start_times.append(interval.minTime)
            end_times.append(interval.maxTime)
    
    return phonemes, start_times, end_times


def compute_alignment_metrics(
    gt_phonemes: List[str],
    gt_starts: List[float],
    gt_ends: List[float],
    mfa_phonemes: List[str],
    mfa_starts: List[float],
    mfa_ends: List[float]
) -> Dict:
    """
    计算对齐质量指标
    """
    metrics = {
        "gt_length": len(gt_phonemes),
        "mfa_length": len(mfa_phonemes),
        "phoneme_accuracy": 0.0,
        "boundary_mae_start": 0.0,
        "boundary_mae_end": 0.0,
        "boundary_mae_midpoint": 0.0,
        "matched_pairs": 0,
    }
    
    # 简单的对齐: 假设长度相同或接近
    min_len = min(len(gt_phonemes), len(mfa_phonemes))
    
    if min_len == 0:
        return metrics
    
    # 音素准确率
    correct = sum(1 for i in range(min_len) if gt_phonemes[i] == mfa_phonemes[i])
    metrics["phoneme_accuracy"] = correct / min_len
    metrics["matched_pairs"] = correct
    
    # 边界误差 (MAE)
    start_errors = [abs(gt_starts[i] - mfa_starts[i]) for i in range(min_len)]
    end_errors = [abs(gt_ends[i] - mfa_ends[i]) for i in range(min_len)]
    
    gt_mids = [(gt_starts[i] + gt_ends[i]) / 2 for i in range(min_len)]
    mfa_mids = [(mfa_starts[i] + mfa_ends[i]) / 2 for i in range(min_len)]
    mid_errors = [abs(gt_mids[i] - mfa_mids[i]) for i in range(min_len)]
    
    metrics["boundary_mae_start"] = np.mean(start_errors) if start_errors else 0.0
    metrics["boundary_mae_end"] = np.mean(end_errors) if end_errors else 0.0
    metrics["boundary_mae_midpoint"] = np.mean(mid_errors) if mid_errors else 0.0
    
    return metrics


def prepare_mfa_input(
    src_dir: Path,
    mfa_input_dir: Path,
    speaker_ids: List[str]
) -> None:
    """
    准备 MFA 输入目录结构:
    mfa_input_dir/
        <speaker>/
            <utt_id>.wav
            <utt_id>.lab  (transcript)
    """
    mfa_input_dir.mkdir(parents=True, exist_ok=True)
    
    for spk in speaker_ids:
        spk_path = src_dir / spk
        if not spk_path.exists():
            print(f"警告: Speaker {spk} 不存在于 {src_dir}")
            continue
        
        # 创建 speaker 目录
        spk_mfa_dir = mfa_input_dir / spk
        spk_mfa_dir.mkdir(exist_ok=True)
        
        # 复制 wav 文件
        wav_dir = spk_path / "wav"
        if wav_dir.exists():
            for wav_file in wav_dir.glob("*.wav"):
                dest_wav = spk_mfa_dir / wav_file.name
                shutil.copy2(wav_file, dest_wav)
        
        # 创建 .lab 文件 (从 transcript 提取)
        transcript_dir = spk_path / "transcript"
        if transcript_dir.exists():
            for txt_file in transcript_dir.glob("*.txt"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # 创建对应的 .lab 文件
                utt_id = txt_file.stem
                lab_file = spk_mfa_dir / f"{utt_id}.lab"
                with open(lab_file, "w", encoding="utf-8") as f:
                    f.write(text)
        
        print(f"已准备 speaker {spk} 的 MFA 输入数据")


def run_mfa_alignment(
    mfa_input_dir: Path,
    mfa_output_dir: Path,
    dictionary: str = "english_us_arpa",
    acoustic_model: str = "english_us_arpa"
) -> bool:
    """
    运行 MFA 对齐
    """
    mfa_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mfa", "align",
        str(mfa_input_dir),
        dictionary,
        acoustic_model,
        str(mfa_output_dir),
        "--clean"
    ]
    
    print(f"运行 MFA 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("MFA 对齐完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"MFA 对齐失败: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def generate_l2arctic_format_json(
    src_dir: Path,
    mfa_output_dir: Path,
    output_json_path: Path,
    speaker_ids: List[str]
) -> None:
    """
    生成与 l2arctic_ts 相同格式的 JSON 文件（使用 MFA 对齐结果）
    """
    all_data = {}
    
    for spk in speaker_ids:
        spk_src = src_dir / spk
        spk_mfa = mfa_output_dir / spk
        
        if not spk_src.exists() or not spk_mfa.exists():
            print(f"跳过 speaker {spk} (源目录或 MFA 输出不存在)")
            continue
        
        transcript_dir = spk_src / "transcript"
        wav_dir = spk_src / "wav"
        
        if not transcript_dir.exists():
            print(f"跳过 speaker {spk} (无 transcript 目录)")
            continue
        
        for mfa_textgrid in spk_mfa.glob("*.TextGrid"):
            utt_id = mfa_textgrid.stem
            wav_path = wav_dir / f"{utt_id}.wav"
            txt_path = transcript_dir / f"{utt_id}.txt"
            
            if not wav_path.exists() or not txt_path.exists():
                print(f"警告: 缺少文件 {spk}/{utt_id}")
                continue
            
            try:
                # 读取 transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # 获取音频时长
                duration = None
                if SOUNDFILE_AVAILABLE:
                    try:
                        audio_info = sf.info(str(wav_path))
                        duration = audio_info.duration
                    except:
                        pass
                
                if duration is None:
                    # 如果 soundfile 失败，尝试使用 librosa 或读取文件大小估算
                    try:
                        import librosa
                        y, sr = librosa.load(str(wav_path), sr=None)
                        duration = len(y) / sr
                    except:
                        # 最后的备选方案：使用 MFA TextGrid 的 maxTime
                        tg = TextGrid.fromFile(str(mfa_textgrid))
                        duration = tg.maxTime
                
                # 提取 MFA 对齐 (作为 canonical 和 perceived)
                mfa_phonemes, mfa_starts, mfa_ends = extract_mfa_alignment(mfa_textgrid)
                
                # 创建条目
                entry = {
                    "wav": str(wav_path),
                    "duration": round(duration, 2),
                    "spk_id": spk,
                    "canonical_aligned": " ".join(mfa_phonemes),
                    "perceived_aligned": " ".join(mfa_phonemes),  # MFA 只有一个输出
                    "perceived_train_target": " ".join(mfa_phonemes),
                    "wrd": transcript,
                    "canonical_starts": [round(t, 3) for t in mfa_starts],
                    "canonical_ends": [round(t, 3) for t in mfa_ends],
                    "target_starts": [round(t, 3) for t in mfa_starts],  # 与 canonical 相同
                    "target_ends": [round(t, 3) for t in mfa_ends],
                }
                
                all_data[str(wav_path)] = entry
                print(f"添加 {spk}/{utt_id}: {len(mfa_phonemes)} phonemes")
            
            except Exception as e:
                print(f"错误处理 {spk}/{utt_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存 JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 已生成 {len(all_data)} 个话语的 JSON 文件: {output_json_path}")


def compare_alignments(
    src_dir: Path,
    mfa_output_dir: Path,
    output_json_dir: Path,
    speaker_ids: List[str]
) -> None:
    """
    比较 ground truth 和 MFA 对齐结果
    """
    output_json_dir.mkdir(parents=True, exist_ok=True)
    
    all_comparisons = []
    overall_metrics = defaultdict(list)
    
    for spk in speaker_ids:
        spk_src = src_dir / spk
        spk_mfa = mfa_output_dir / spk
        
        if not spk_src.exists() or not spk_mfa.exists():
            print(f"跳过 speaker {spk} (源目录或 MFA 输出不存在)")
            continue
        
        annotation_dir = spk_src / "annotation"
        if not annotation_dir.exists():
            print(f"跳过 speaker {spk} (无 annotation 目录)")
            continue
        
        for gt_textgrid in annotation_dir.glob("*.TextGrid"):
            utt_id = gt_textgrid.stem
            mfa_textgrid = spk_mfa / f"{utt_id}.TextGrid"
            
            if not mfa_textgrid.exists():
                print(f"警告: MFA 结果不存在 {mfa_textgrid}")
                continue
            
            try:
                # 提取 ground truth
                gt_phonemes, gt_starts, gt_ends = extract_ground_truth_from_textgrid(gt_textgrid)
                
                # 提取 MFA 对齐
                mfa_phonemes, mfa_starts, mfa_ends = extract_mfa_alignment(mfa_textgrid)
                
                # 计算指标
                metrics = compute_alignment_metrics(
                    gt_phonemes, gt_starts, gt_ends,
                    mfa_phonemes, mfa_starts, mfa_ends
                )
                
                # 记录结果
                comparison = {
                    "utt_id": utt_id,
                    "speaker": spk,
                    "ground_truth": {
                        "phonemes": gt_phonemes,
                        "start_times": gt_starts,
                        "end_times": gt_ends
                    },
                    "mfa_alignment": {
                        "phonemes": mfa_phonemes,
                        "start_times": mfa_starts,
                        "end_times": mfa_ends
                    },
                    "metrics": metrics
                }
                
                all_comparisons.append(comparison)
                
                # 累积指标
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        overall_metrics[key].append(value)
                
                print(f"处理 {spk}/{utt_id}: "
                      f"Phoneme Acc={metrics['phoneme_accuracy']:.3f}, "
                      f"Boundary MAE={metrics['boundary_mae_midpoint']:.4f}s")
            
            except Exception as e:
                print(f"错误处理 {spk}/{utt_id}: {e}")
                continue
    
    # 计算总体统计
    summary = {
        "total_utterances": len(all_comparisons),
        "average_metrics": {}
    }
    
    for key, values in overall_metrics.items():
        if values:
            summary["average_metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    # 保存详细比较结果
    detailed_output = output_json_dir / "detailed_comparisons.json"
    with open(detailed_output, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细比较结果已保存到: {detailed_output}")
    
    # 保存摘要
    summary_output = output_json_dir / "comparison_summary.json"
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"摘要结果已保存到: {summary_output}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("对齐质量摘要")
    print("="*60)
    print(f"总话语数: {summary['total_utterances']}")
    for key, stats in summary["average_metrics"].items():
        print(f"\n{key}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 MFA 对 L2-ARCTIC 进行对齐并与 ground truth 比较"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/common/db/L2-ARCTIC",
        help="L2-ARCTIC 数据集路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kevingenghaopeng/MDD/IF-MDD/data/l2arctic_mfa",
        help="输出目录"
    )
    parser.add_argument(
        "--test_spks",
        nargs="+",
        default=["TLV", "NJS", "TNI", "TXHC", "ZHAA", "YKWK"],
        help="要处理的 speaker IDs"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="english_us_arpa",
        help="MFA 字典名称"
    )
    parser.add_argument(
        "--acoustic_model",
        type=str,
        default="english_us_arpa",
        help="MFA 声学模型名称"
    )
    parser.add_argument(
        "--skip_mfa",
        action="store_true",
        help="跳过 MFA 对齐，仅进行比较 (假设 MFA 输出已存在)"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not TEXTGRID_AVAILABLE:
        print("错误: 请安装 textgrid 库: pip install textgrid")
        return
    
    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)
    
    # 获取所有 speakers
    all_speakers = [d.name for d in src_dir.iterdir() if d.is_dir()]
    
    # 如果指定了 test_spks，使用它们；否则使用所有 speakers
    speakers_to_process = args.test_spks if args.test_spks else all_speakers
    
    print(f"将处理 {len(speakers_to_process)} 个 speakers: {speakers_to_process}")
    
    # MFA 输入和输出目录
    mfa_input_dir = output_dir / "mfa_input"
    mfa_output_dir = output_dir / "mfa_output"
    comparison_dir = output_dir / "comparisons"
    
    if not args.skip_mfa:
        # 步骤 1: 准备 MFA 输入
        print("\n步骤 1: 准备 MFA 输入数据...")
        prepare_mfa_input(src_dir, mfa_input_dir, speakers_to_process)
        
        # 步骤 2: 运行 MFA 对齐
        print("\n步骤 2: 运行 MFA 对齐...")
        success = run_mfa_alignment(
            mfa_input_dir,
            mfa_output_dir,
            args.dictionary,
            args.acoustic_model
        )
        
        if not success:
            print("MFA 对齐失败，退出")
            return
    else:
        print("跳过 MFA 对齐步骤")
    
    # 步骤 3: 比较对齐结果
    print("\n步骤 3: 比较 MFA 对齐与 ground truth...")
    compare_alignments(
        src_dir,
        mfa_output_dir,
        comparison_dir,
        speakers_to_process
    )
    
    # 步骤 4: 生成 L2-ARCTIC 格式的 JSON
    print("\n步骤 4: 生成 L2-ARCTIC 格式的 JSON (使用 MFA 对齐)...")
    mfa_json_path = output_dir / "l2arctic_mfa_format.json"
    generate_l2arctic_format_json(
        src_dir,
        mfa_output_dir,
        mfa_json_path,
        speakers_to_process
    )
    
    print("\n完成!")
    print(f"\n生成的文件:")
    print(f"  - MFA 格式 JSON: {mfa_json_path}")
    print(f"  - 详细比较: {comparison_dir}/detailed_comparisons.json")
    print(f"  - 统计摘要: {comparison_dir}/comparison_summary.json")


if __name__ == "__main__":
    main()
