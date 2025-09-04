"""示例: 将 HuggingFace 的 mispeech/speechocean762 数据集全部字段
通过 SpeechBrain 的 DynamicItemDataset 封装，并生成可直接迭代的样本。

运行:
    python speechocean762.py --split train --limit 5

说明:
1. 自动探测样本字段 (除 audio 的子字段会被拆成 wav, sr, audio_path)。
2. 所有原始字段均可通过 output_keys 取出。
3. 可设置 --limit 仅加载前 N 条做快速调试，避免一次性占用过多内存。
"""

from __future__ import annotations
import argparse
from datasets import load_dataset
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import takes, provides
import torch
from typing import Dict, Any, List


def build_dynamic_items(first_sample: Dict[str, Any]):
    """构建符合 SpeechBrain 要求的 dynamic_items 列表。

    返回: list[callable] 其中每个 callable 已用 @takes/@provides 装饰。
    """

    dynamic_items: List = []

    if "audio" in first_sample and isinstance(first_sample["audio"], dict):
        @takes("audio")
        @provides("wav")
        def extract_wav(audio):
            return torch.as_tensor(audio["array"]).float()

        dynamic_items.append(extract_wav)

        if "sampling_rate" in first_sample["audio"]:
            @takes("audio")
            @provides("sr")
            def extract_sr(audio):
                return audio.get("sampling_rate")
            dynamic_items.append(extract_sr)

        if "path" in first_sample["audio"]:
            @takes("audio")
            @provides("audio_path")
            def extract_path(audio):
                return audio.get("path")
            dynamic_items.append(extract_path)

    # 其它字段无需动态函数，直接作为 static key 输出；若需要可在此加入装饰函数。
    # 增加基于 words 的 canonical/perceived 序列动态项
    if "words" in first_sample:
        @takes("words")
        @provides("canonical_aligned", "perceived_aligned", "perceived_train_target")
        def build_phoneme_sequences(words_list):
            """从 words 列表生成 canonical / perceived 序列。

            规则:
            - canonical: 直接拼接每个 word['phones']
            - perceived: 拷贝 canonical, 若 mispronunciations 列表存在则尝试替换
              支持的 mispronunciation 结构:
                * 字符串: 目前忽略 (可扩展)
                * 字典: 尝试读取 replacement/produced/phone 等字段作为替换音素;
                  若存在 index / idx / position 字段作为该 word 内的相对索引
            - perceived_train_target 暂与 perceived_aligned 相同，可后续再加清理
            - 规范化: 全部转小写，并去掉数字重音标记 (例如 IY0 -> iy, EH1 -> eh)
            """
            def norm_phone(p):
                # 去掉所有数字并转小写
                if not isinstance(p, str):
                    return p
                return "".join(ch for ch in p if not ch.isdigit()).lower()
            canonical_phones = []
            perceived_phones = []
            # 记录每个 word 的起始 offset 以便调试（未返回）
            for w in words_list:
                w_phones = [norm_phone(ph) for ph in w.get("phones", [])]
                canonical_phones.extend(w_phones)
                # 初始同 canonical
                perceived_part = list(w_phones)
                mis_list = w.get("mispronunciations", []) or []
                if isinstance(mis_list, list) and mis_list:
                    for mis in mis_list:
                        try:
                            if isinstance(mis, dict):
                                # 猜测 index 字段名
                                idx = mis.get("index") or mis.get("idx") or mis.get("position")
                                # 替换音素候选字段
                                repl = (mis.get("produced") or mis.get("replacement") or mis.get("phone") or mis.get("perceived"))
                                if isinstance(repl, str):
                                    repl = norm_phone(repl)
                                if repl is not None and isinstance(idx, int) and 0 <= idx < len(perceived_part):
                                    perceived_part[idx] = repl
                            # 其它形式暂不处理
                        except Exception:
                            continue
                perceived_phones.extend(perceived_part)
            canonical = " ".join(canonical_phones)
            perceived = " ".join(perceived_phones)
            perceived_train_target = perceived  # 目前相同，留接口
            return canonical, perceived, perceived_train_target
        dynamic_items.append(build_phoneme_sequences)

    return dynamic_items


def hf_to_sb_dataset(hf_split, limit: int | None = None) -> DynamicItemDataset:
    """将 HuggingFace dataset split 转换为 SpeechBrain DynamicItemDataset。

    参数:
        hf_split: HuggingFace dataset (e.g. Dataset 对象)
        limit: 只取前 N 条样本 (调试用)
    返回:
        SpeechBrain DynamicItemDataset
    """
    if limit is not None:
        indices = list(range(min(limit, len(hf_split))))
    else:
        indices = list(range(len(hf_split)))

    # 构造底层 data dict (内部索引 -> sample dict); 添加统一 record_id 字段 (避免使用保留键 'id')
    data: Dict[str, Dict[str, Any]] = {}
    for i in indices:
        sample = dict(hf_split[i])  # copy
        # 生成 record_id: 原始 id > audio.path > 索引
        raw_id = sample.get("id")
        audio_info = sample.get("audio") or {}
        record_id = raw_id or audio_info.get("path") or f"sample_{i}"
        sample["record_id"] = record_id
        if "id" in sample:  # 移除原始 id 避免冲突
            sample.pop("id")
        data[str(i)] = sample

    if not data:
        raise RuntimeError("数据为空, 请检查 split 或 limit 设置。")

    first_sample = data[str(indices[0])]
    dynamic_items = build_dynamic_items(first_sample)

    # static keys (全部样本共有的顶层字段) + 动态 keys
    static_keys = set(first_sample.keys())
    # 动态 keys 明确列出（wav, sr, audio_path 可能存在）
    dynamic_keys = []
    for fn in dynamic_items:
        if hasattr(fn, "provides"):
            provided = fn.provides
            if isinstance(provided, (list, tuple)):
                dynamic_keys.extend(list(provided))
            else:
                dynamic_keys.append(provided)

    # 期望输出顺序
    preferred = ["record_id", "wav", "sr", "audio_path", "canonical_aligned", "perceived_aligned", "perceived_train_target"]
    output_keys: List[str] = []
    for k in preferred:
        if k in static_keys or k in dynamic_keys:
            output_keys.append(k)
    # 追加剩余 static keys
    for k in sorted(static_keys):
        if k not in output_keys and k != "audio":  # audio 原始字典不直接输出
            output_keys.append(k)

    # 某些 SpeechBrain 版本 DynamicItemDataset 不支持构造参数 output_keys
    try:
        sb_dataset = DynamicItemDataset(
            data=data,
            dynamic_items=dynamic_items,
            output_keys=output_keys,  # 若不支持会抛出 TypeError
        )
    except TypeError:
        sb_dataset = DynamicItemDataset(
            data=data,
            dynamic_items=dynamic_items,
        )
        # 尝试使用官方 API 设置输出键
        if hasattr(sb_dataset, "set_output_keys"):
            sb_dataset.set_output_keys(output_keys)
    # 缓存用户期望的输出键以便后续预览使用
    sb_dataset._user_output_keys = output_keys
    return sb_dataset


def preview_dataset(ds: DynamicItemDataset, n: int = 2):
    print(f"Dataset size: {len(ds)}")
    # 兼容不同版本: 优先使用缓存或属性
    keys = getattr(ds, "_user_output_keys", None)
    if keys is None:
        if hasattr(ds, "output_keys"):
            try:
                keys = list(ds.output_keys)  # 某些版本可能存在
            except Exception:
                keys = None
    if keys is None:
        # 回退: 取一个样本的键
        sample0 = ds[0]
        keys = list(sample0.keys())
    print("Output keys:", keys)
    for i in range(min(n, len(ds))):
        item = ds[i]
        print(f"--- Sample {i} ---")
        for k in keys:
            v = item.get(k)
            if isinstance(v, torch.Tensor):
                print(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                text = str(v)
                if len(text) > 120:
                    text = text[:117] + "..."
                print(f"{k}: {text}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "train+validation"], help="选择要加载的 split")
    parser.add_argument("--limit", type=int, default=10, help="调试: 仅加载前 N 条 (默认 10, 设置为 -1 代表全部)")
    parser.add_argument("--no_preview", action="store_true", help="不打印样本预览")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading HuggingFace dataset mispeech/speechocean762 split={args.split} ...")
    if args.split == "train+validation":
        ds_train = load_dataset("mispeech/speechocean762", split="train")
        ds_valid = load_dataset("mispeech/speechocean762", split="validation")
        from datasets import concatenate_datasets
        hf_split = concatenate_datasets([ds_train, ds_valid])
    else:
        hf_split = load_dataset("mispeech/speechocean762", split=args.split)

    limit = None if args.limit == -1 else args.limit
    sb_ds = hf_to_sb_dataset(hf_split, limit=limit)
    import pdb; pdb.set_trace()
    if not args.no_preview:
        preview_dataset(sb_ds, n=min(3, len(sb_ds)))

    # 访问示例
    # 打印发生“误读”的位置 (canonical != perceived)
    def print_mispronunciation_positions(sample_dict):
        cano = sample_dict.get("canonical_aligned", "").strip().split()
        perc = sample_dict.get("perceived_aligned", "").strip().split()
        if not cano or not perc:
            print("(No canonical/perceived data)")
            return
        L = min(len(cano), len(perc))
        diffs = []
        for i in range(L):
            if cano[i] != perc[i]:
                diffs.append((i, cano[i], perc[i]))
        # 若长度不同，也标记额外部分
        if len(cano) != len(perc):
            if len(cano) > L:
                for i in range(L, len(cano)):
                    diffs.append((i, cano[i], '<del>'))
            else:
                for i in range(L, len(perc)):
                    diffs.append((i, '<ins>', perc[i]))
        if not diffs:
            print("No mispronunciation positions (canonical == perceived)")
            return
        print(f"Mispronunciation count: {len(diffs)}")
        preview_n = 30
        for idx, cph, pph in diffs[:preview_n]:
            print(f"  idx={idx:03d}: {cph} -> {pph}")
        if len(diffs) > preview_n:
            print(f"  ... ({len(diffs)-preview_n} more)")
    for sample in sb_ds:
        print_mispronunciation_positions(sample)
    # print("\nFirst sample field types:")
    # for k, v in sample.items():
    #     t = type(v)
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: Tensor shape={tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {t.__name__}")


if __name__ == "__main__":
    main()