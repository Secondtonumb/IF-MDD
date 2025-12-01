"""
生成包含 MFA 和 Ground Truth 对齐结果的 JSON 文件
- 去掉 stress 标记（如 ao1 -> ao）
- 生成 train.json 和 test.json
- 计算 onset/offset MSE 统计
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

try:
    from textgrid import TextGrid
except ImportError:
    print("请安装 textgrid: pip install textgrid")
    exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except:
    SOUNDFILE_AVAILABLE = False


def remove_stress(phone: str) -> str:
    """去掉音素的 stress 标记，如 ao1 -> ao"""
    return re.sub(r'\d+', '', phone)


def is_sil(s: str) -> bool:
    """判断是否为静音"""
    return s.lower() in {"sil", "sp", "spn", "pau", ""}


def normalize_phone(s: str) -> str:
    """
    标准化音素标记
    L2-ARCTIC 格式: "AH,AX,A"
    """
    t = s.lower()
    t = re.sub(r"[^a-z,]", "", t)
    
    if is_sil(t):
        return "sil"
    if len(t) == 0:
        return "sil"
    
    parts = t.split(",")
    cano = parts[0]
    
    if cano == "ax":
        return "ah"
    
    return cano


def extract_ground_truth(textgrid_path: Path) -> Tuple[List[str], List[float], List[float]]:
    """从 L2-ARCTIC TextGrid 提取 ground truth（去除所有 sil）"""
    tg = TextGrid.fromFile(str(textgrid_path))
    
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() == "phones":
            phone_tier = tier
            break
    
    if phone_tier is None:
        raise ValueError(f"No 'phones' tier in {textgrid_path}")
    
    phonemes, starts, ends = [], [], []
    
    for interval in phone_tier:
        phone = normalize_phone(interval.mark)
        if phone:
            phone = remove_stress(phone)  # 去除 stress
            # 过滤掉所有的 sil（静音标记）
            if phone != "sil":
                phonemes.append(phone)
                starts.append(interval.minTime)
                ends.append(interval.maxTime)
    
    return phonemes, starts, ends


def extract_mfa_alignment(textgrid_path: Path) -> Tuple[List[str], List[float], List[float]]:
    """从 MFA TextGrid 提取对齐（去除所有 sil）"""
    tg = TextGrid.fromFile(str(textgrid_path))
    
    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() in {"phones", "phone"}:
            phone_tier = tier
            break
    
    if phone_tier is None:
        raise ValueError(f"No 'phones' tier in {textgrid_path}")
    
    phonemes, starts, ends = [], [], []
    
    for interval in phone_tier:
        phone = interval.mark.lower()
        if phone and phone != "":
            phone = remove_stress(phone)  # 去除 stress
            # 过滤掉所有的 sil（静音标记）
            if phone != "sil" and phone != "sp" and phone != "spn":
                phonemes.append(phone)
                starts.append(interval.minTime)
                ends.append(interval.maxTime)
    
    return phonemes, starts, ends


def generate_combined_json(
    src_dir: Path,
    mfa_output_dir: Path,
    output_dir: Path,
    train_speakers: List[str],
    test_speakers: List[str]
) -> Tuple[Path, Path, Dict]:
    """
    生成包含 MFA 和 GT 的 JSON（train 和 test）
    
    返回: (train_json_path, test_json_path, statistics)
    """
    train_data = {}
    test_data = {}
    
    # 用于计算 MSE
    onset_errors = []  # (gt_start, mfa_start) 差值
    offset_errors = []  # (gt_end, mfa_end) 差值
    
    all_speakers = train_speakers + test_speakers
    
    for spk in all_speakers:
        is_test = spk in test_speakers
        target_dict = test_data if is_test else train_data
        
        spk_src = src_dir / spk
        spk_mfa = mfa_output_dir / spk
        
        if not spk_src.exists() or not spk_mfa.exists():
            print(f"跳过 {spk}")
            continue
        
        annotation_dir = spk_src / "annotation"
        transcript_dir = spk_src / "transcript"
        wav_dir = spk_src / "wav"
        
        if not annotation_dir.exists() or not transcript_dir.exists():
            print(f"跳过 {spk} (缺少目录)")
            continue
        
        for gt_textgrid in annotation_dir.glob("*.TextGrid"):
            utt_id = gt_textgrid.stem
            mfa_textgrid = spk_mfa / f"{utt_id}.TextGrid"
            wav_path = wav_dir / f"{utt_id}.wav"
            txt_path = transcript_dir / f"{utt_id}.txt"
            
            if not all([mfa_textgrid.exists(), wav_path.exists(), txt_path.exists()]):
                continue
            
            try:
                # 读取文本
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # 获取时长
                if SOUNDFILE_AVAILABLE:
                    try:
                        duration = sf.info(str(wav_path)).duration
                    except:
                        duration = TextGrid.fromFile(str(gt_textgrid)).maxTime
                else:
                    duration = TextGrid.fromFile(str(gt_textgrid)).maxTime
                
                # 提取对齐
                gt_phones, gt_starts, gt_ends = extract_ground_truth(gt_textgrid)
                mfa_phones, mfa_starts, mfa_ends = extract_mfa_alignment(mfa_textgrid)
                
                # 计算误差（用于统计）
                min_len = min(len(gt_phones), len(mfa_phones))
                for i in range(min_len):
                    if gt_phones[i] == mfa_phones[i]:  # 只计算匹配的音素
                        onset_errors.append(abs(gt_starts[i] - mfa_starts[i]))
                        offset_errors.append(abs(gt_ends[i] - mfa_ends[i]))
                
                # 创建条目
                entry = {
                    "wav": str(wav_path),
                    "duration": round(duration, 2),
                    "spk_id": spk,
                    "wrd": transcript,
                    # Ground Truth
                    "gt_canonical_aligned": " ".join(gt_phones),
                    "gt_perceived_aligned": " ".join(gt_phones),
                    "gt_perceived_train_target": " ".join(gt_phones),
                    "gt_canonical_starts": [round(t, 3) for t in gt_starts],
                    "gt_canonical_ends": [round(t, 3) for t in gt_ends],
                    "gt_target_starts": [round(t, 3) for t in gt_starts],
                    "gt_target_ends": [round(t, 3) for t in gt_ends],
                    # MFA
                    "mfa_canonical_aligned": " ".join(mfa_phones),
                    "mfa_perceived_aligned": " ".join(mfa_phones),
                    "mfa_perceived_train_target": " ".join(mfa_phones),
                    "mfa_canonical_starts": [round(t, 3) for t in mfa_starts],
                    "mfa_canonical_ends": [round(t, 3) for t in mfa_ends],
                    "mfa_target_starts": [round(t, 3) for t in mfa_starts],
                    "mfa_target_ends": [round(t, 3) for t in mfa_ends],
                }
                
                target_dict[str(wav_path)] = entry
                split = "test" if is_test else "train"
                print(f"[{split:5}] {spk}/{utt_id}: GT={len(gt_phones)}, MFA={len(mfa_phones)}")
            
            except Exception as e:
                print(f"错误 {spk}/{utt_id}: {e}")
                continue
    
    # 保存 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_json = output_dir / "train.json"
    test_json = output_dir / "test.json"
    
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # 计算统计（1 frame = 20ms = 0.02s）
    frame_size = 0.02  # 20ms per frame
    
    stats = {
        "train_count": len(train_data),
        "test_count": len(test_data),
        "total_count": len(train_data) + len(test_data),
        # 秒单位
        "onset_mse": float(np.mean(np.array(onset_errors) ** 2)) if onset_errors else 0.0,
        "offset_mse": float(np.mean(np.array(offset_errors) ** 2)) if offset_errors else 0.0,
        "onset_mae": float(np.mean(onset_errors)) if onset_errors else 0.0,
        "offset_mae": float(np.mean(offset_errors)) if offset_errors else 0.0,
        "onset_std": float(np.std(onset_errors)) if onset_errors else 0.0,
        "offset_std": float(np.std(offset_errors)) if offset_errors else 0.0,
        # 毫秒单位
        "onset_mae_ms": float(np.mean(onset_errors) * 1000) if onset_errors else 0.0,
        "offset_mae_ms": float(np.mean(offset_errors) * 1000) if offset_errors else 0.0,
        "onset_std_ms": float(np.std(onset_errors) * 1000) if onset_errors else 0.0,
        "offset_std_ms": float(np.std(offset_errors) * 1000) if offset_errors else 0.0,
        # 帧数单位 (1 frame = 20ms)
        "onset_mae_frames": float(np.mean(onset_errors) / frame_size) if onset_errors else 0.0,
        "offset_mae_frames": float(np.mean(offset_errors) / frame_size) if offset_errors else 0.0,
        "onset_std_frames": float(np.std(onset_errors) / frame_size) if onset_errors else 0.0,
        "offset_std_frames": float(np.std(offset_errors) / frame_size) if offset_errors else 0.0,
        "matched_phoneme_count": len(onset_errors),
        "frame_size_ms": 20,  # 帧大小：20ms
    }
    
    print(f"\n✓ train.json: {stats['train_count']} 话语")
    print(f"✓ test.json: {stats['test_count']} 话语")
    print(f"✓ 总计: {stats['total_count']} 话语")
    
    return train_json, test_json, stats


def main():
    parser = argparse.ArgumentParser(description="生成 MFA + GT 对齐 JSON")
    parser.add_argument("--src_dir", type=str, default="/common/db/L2-ARCTIC")
    parser.add_argument("--mfa_output_dir", type=str,
                       default="/home/kevingenghaopeng/MDD/IF-MDD/data/l2arctic_mfa/mfa_output")
    parser.add_argument("--output_dir", type=str,
                       default="/home/kevingenghaopeng/MDD/IF-MDD/data/l2arctic_mfa_gt")
    parser.add_argument("--test_spks", nargs="+",
                       default=["TLV", "NJS", "TNI", "TXHC", "ZHAA", "YKWK"])
    
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir)
    mfa_output_dir = Path(args.mfa_output_dir)
    output_dir = Path(args.output_dir)
    
    # 获取所有 speakers
    all_speakers = [d.name for d in src_dir.iterdir() if d.is_dir()]
    test_speakers = args.test_spks
    train_speakers = [s for s in all_speakers if s not in test_speakers]
    
    print(f"Train speakers ({len(train_speakers)}): {sorted(train_speakers)}")
    print(f"Test speakers ({len(test_speakers)}): {sorted(test_speakers)}")
    print()
    
    # 生成 JSON
    train_json, test_json, stats = generate_combined_json(
        src_dir,
        mfa_output_dir,
        output_dir,
        train_speakers,
        test_speakers
    )
    
    # 保存统计
    stats_path = output_dir / "alignment_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("对齐统计摘要")
    print(f"{'='*60}")
    print(f"匹配的音素对数: {stats['matched_phoneme_count']}")
    print(f"\nOnset (Start) 边界:")
    print(f"  MSE:  {stats['onset_mse']:.6f} 秒²")
    print(f"  MAE:  {stats['onset_mae']:.6f} 秒 = {stats['onset_mae_ms']:.2f} ms = {stats['onset_mae_frames']:.2f} frames")
    print(f"  STD:  {stats['onset_std']:.6f} 秒 = {stats['onset_std_ms']:.2f} ms = {stats['onset_std_frames']:.2f} frames")
    print(f"\nOffset (End) 边界:")
    print(f"  MSE:  {stats['offset_mse']:.6f} 秒²")
    print(f"  MAE:  {stats['offset_mae']:.6f} 秒 = {stats['offset_mae_ms']:.2f} ms = {stats['offset_mae_frames']:.2f} frames")
    print(f"  STD:  {stats['offset_std']:.6f} 秒 = {stats['offset_std_ms']:.2f} ms = {stats['offset_std_frames']:.2f} frames")
    print(f"\n注: 1 frame = {stats['frame_size_ms']} ms")
    print(f"\n保存位置: {stats_path}")


if __name__ == "__main__":
    main()
