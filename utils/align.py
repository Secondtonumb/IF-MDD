import torch
import speechbrain as sb
from torchaudio.functional import forced_align, merge_tokens
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os


def compute_forced_alignment(
    p_ctc,
    targets,
    actual_target_len,
    actual_input_len,
    blank_index,
    device
):
    """
    计算强制对齐（Forced Alignment）- 使用 torchaudio
    
    Args:
        p_ctc: [1, T, C] CTC log probabilities
        targets: [1, T_target] 目标tokens（无padding）
        actual_target_len: 实际目标长度
        actual_input_len: 实际输入长度
        blank_index: blank token ID
        device: 计算设备
    
    Returns:
        tuple: (aligned_tokens, scores) 或 (None, None) 如果失败
    """
    try:
        forced_alignments, scores = forced_align(
            log_probs=p_ctc,
            targets=targets,
            target_lengths=torch.tensor([actual_target_len], dtype=torch.int32, device=device),
            input_lengths=torch.tensor([actual_input_len], dtype=torch.int32, device=device),
            blank=blank_index
        )
        forced_alignments = forced_alignments[0]
        scores = scores[0].exp()
        aligned_tokens = merge_tokens(forced_alignments, scores)
        return aligned_tokens, scores
    except Exception as e:
        return None, None


def compute_k2_alignment(
    audio_file,
    targets,
    blank_index,
    asr_model,
    tokenizer,
    device,
    frame_shift_ms=20
):
    """
    计算K2强制对齐 - 使用 SpeechBrain CTCAligner
    
    需要安装: pip install k2
    
    Args:
        audio_file: 音频文件路径
        targets: 目标token序列 (list of ints 或 list of str)
        blank_index: blank token ID
        asr_model: ASR模型（需要有 tokenizer 和 device 属性）
        tokenizer: 标记器（需要有 ind2lab 和 lab2ind 属性）
        device: 计算设备
        frame_shift_ms: 帧移(ms)
    
    Returns:
        dict: {
            'alignment': frame_level alignment (list of token ids),
            'text': phoneme sequence (list of strings),
            'text_frames': start frame of each phoneme (list of ints),
            'timestamps': (start_sec, end_sec, phoneme) tuples,
            'success': boolean
        }
    """
    try:
        from speechbrain.integrations.k2_fsa.align import CTCAligner
    except ImportError:
        print("⚠️  K2 not installed. Install with: pip install k2")
        return {
            'alignment': [],
            'text': [],
            'text_frames': [],
            'timestamps': [],
            'success': False
        }
    
    try:
        # 初始化 K2 aligner
        aligner = CTCAligner(
            model=asr_model,
            tokenizer=tokenizer,
            device=device
        )
        
        # 转换目标token为字符串（如果是id）
        if targets and isinstance(targets[0], int):
            transcript = [tokenizer.ind2lab[t] for t in targets if t != blank_index]
        else:
            transcript = targets
        
        # 执行对齐
        alignment = aligner.align_audio_to_tokens(
            audio_file=str(audio_file),
            transcript=transcript
        )
        
        # 将alignment转换为时间戳
        timestamps_result = ctc_alignment_to_timestamps(
            alignment=alignment,
            tokenizer=tokenizer,
            frame_shift_ms=frame_shift_ms,
            blank_index=blank_index
        )
        
        result = {
            'alignment': alignment,
            'text': timestamps_result['text'],
            'text_frames': timestamps_result['text_frames'],
            'timestamps': timestamps_result['timestamps'],
            'success': True
        }
        
        return result
        
    except Exception as e:
        print(f"⚠️  K2 alignment failed: {e}")
        return {
            'alignment': [],
            'text': [],
            'text_frames': [],
            'timestamps': [],
            'success': False
        }


def ctc_alignment_to_timestamps(
    alignment,
    tokenizer,
    frame_shift_ms=20,
    blank_index=0
):
    """
    将frame级别的CTC对齐转换为音素时间戳
    
    在CTC格式中，blank帧属于前一个音素的段。
    
    Args:
        alignment: List[int] - Frame级别对齐（音素ID），0是blank
        tokenizer: 标记器，需要 ind2lab 映射
        frame_shift_ms: 帧移(ms)，默认20ms
        blank_index: blank token ID，默认0
    
    Returns:
        dict with:
            - 'text': List[str] - 音素序列
            - 'text_frames': List[int] - 每个音素的起始帧
            - 'timestamps': List[Tuple[float, float, str]] - (start_sec, end_sec, 音素)
    """
    text = []
    text_frames = []
    timestamps = []
    
    if not alignment:
        return {'text': [], 'text_frames': [], 'timestamps': []}
    
    # 追踪当前音素及其起始帧
    current_phone_id = None
    start_frame = 0
    
    for frame_idx, phone_id in enumerate(alignment):
        # Blank帧 - 属于前一个音素的持续时间，继续
        if phone_id == blank_index:
            continue
        
        # 检测到新音素（与当前不同）
        if phone_id != current_phone_id:
            # 保存前一个音素
            if current_phone_id is not None and current_phone_id != blank_index:
                try:
                    phoneme = tokenizer.ind2lab[current_phone_id]
                except (KeyError, AttributeError):
                    phoneme = f"<{current_phone_id}>"
                
                text.append(phoneme)
                text_frames.append(start_frame)
                
                # 结束时间包含所有帧直到（但不包括）当前帧
                # 这包括最后一次出现该音素后的任何blank帧
                start_sec = start_frame * frame_shift_ms / 1000.0
                end_sec = (frame_idx - 1) * frame_shift_ms / 1000.0
                timestamps.append((start_sec, end_sec, phoneme))
            
            # 开始追踪新音素
            current_phone_id = phone_id
            start_frame = frame_idx
    
    # 处理最后一个音素（延伸到序列末尾，包括尾部blank）
    if current_phone_id is not None and current_phone_id != blank_index:
        try:
            phoneme = tokenizer.ind2lab[current_phone_id]
        except (KeyError, AttributeError):
            phoneme = f"<{current_phone_id}>"
        
        text.append(phoneme)
        text_frames.append(start_frame)
        
        start_sec = start_frame * frame_shift_ms / 1000.0
        end_sec = (len(alignment) - 1) * frame_shift_ms / 1000.0
        timestamps.append((start_sec, end_sec, phoneme))
    
    return {
        'text': text,
        'text_frames': text_frames,
        'timestamps': timestamps
    }


def plot_scores(word_spans, scores, label_encoder=None):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    span_xs, span_hs = [], []
    ax.axvspan(word_spans[0].start - 0.05, word_spans[-1].end + 0.05, facecolor="paleturquoise", edgecolor="none", zorder=-1)
    # for t_span in word_spans:
    for span in word_spans:
        for t in range(span.start, span.end):
            span_xs.append(t + 0.5)
            # scores shape: (time_steps, vocab_size), get score for specific token at time t
            span_hs.append(scores[t, span.token].item() if scores.ndim > 1 else scores[t].item())
        ax.annotate(label_encoder.decode_ndim(span.token), (span.start, -0.07))
        ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
    ax.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral")
    ax.set_title("Frame-level scores and word segments")
    ax.set_ylim(-0.1, None)
    ax.grid(True, axis="y")
    ax.axhline(0, color="black")
    fig.tight_layout()
    return fig

def plot_alignment_comparison(word_spans_gt, scores_gt, word_spans_pred, scores_pred, title="GT vs Predicted", label_encoder=None):
    """
    Plot GT and Predicted alignment side by side for comparison.
    
    Arguments
    ---------
    word_spans_gt : list of namedtuples
        Ground truth word/token spans
    scores_gt : tensor
        Ground truth alignment scores
    word_spans_pred : list of namedtuples or None
        Predicted word/token spans
    scores_pred : tensor or None
        Predicted alignment scores
    title : str
        Title for the figure
        
    Returns
    -------
    fig : matplotlib figure
        Figure with comparison plots
    """
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    # Determine layout
    if word_spans_pred is not None and len(word_spans_pred) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(14, 4))
        axes = [axes]
    
    # Plot GT alignment
    ax = axes[0]
    span_xs, span_hs = [], []
    if len(word_spans_gt) > 0:
        ax.axvspan(word_spans_gt[0].start - 0.05, word_spans_gt[-1].end + 0.05, 
                    facecolor="paleturquoise", edgecolor="none", zorder=-1)
        for span in word_spans_gt:
            for t in range(span.start, span.end):
                span_xs.append(t + 0.5)
                span_hs.append(scores_gt[t].item())
            token_name = label_encoder.decode_ndim(span.token)
            ax.annotate(token_name, (span.start, -0.07), fontsize=8)
            ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
    ax.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral", alpha=0.8)
    ax.set_title("🎯 Ground Truth Alignment", fontsize=11, fontweight='bold')
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(-0.1, None)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    
    # Plot Predicted alignment (if available)
    if len(axes) > 1:
        ax = axes[1]
        if word_spans_pred is not None and len(word_spans_pred) > 0 and scores_pred is not None:
            span_xs_pred, span_hs_pred = [], []
            ax.axvspan(word_spans_pred[0].start - 0.05, word_spans_pred[-1].end + 0.05, 
                        facecolor="lightgreen", edgecolor="none", zorder=-1, alpha=0.5)
            for span in word_spans_pred:
                for t in range(span.start, span.end):
                    span_xs_pred.append(t + 0.5)
                    span_hs_pred.append(scores_pred[t].item())
                token_name = self.label_encoder.decode_ndim(span.token)
                ax.annotate(token_name, (span.start, -0.07), fontsize=8)
                ax.axvspan(span.start - 0.05, span.end + 0.05, facecolor="lightcyan", edgecolor="none", zorder=-1)
            ax.bar(span_xs_pred, span_hs_pred, color="lightgreen", edgecolor="green", alpha=0.8)
            ax.set_title("📊 Predicted Alignment", fontsize=11, fontweight='bold')
            ax.set_ylabel("Score", fontsize=10)
            ax.set_xlabel("Frame Index", fontsize=10)
            ax.set_ylim(-0.1, None)
            ax.grid(True, axis="y", alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "❌ Predicted alignment unavailable\n(empty or failed)", 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def monitor_alignment(
    stage,
    batch,
    p_ctc,
    wav_lens,
    hparams,
    device,
    plot_scores_fn=plot_scores,
    plot_comparison_fn=plot_alignment_comparison,
    fixed_sample_id=None,
    test_predictions_list=None,
    asr_model=None,
    tokenizer=None,
    audio_dir=None,
    use_k2_alignment=False
):
    """
    对齐监控函数 - 处理VALID和TEST阶段的对齐可视化
    
    支持两种对齐方式:
    1. torchaudio forced alignment (默认，快速)
    2. K2 forced alignment (可选，更准确但需要k2库)
    
    Args:
        stage: sb.Stage.VALID 或 sb.Stage.TEST
        batch: 数据batch，需要包含:
            - batch.id: 样本ID列表
            - batch.phn_encoded_target: (targets, target_lens)
        p_ctc: [B, T, C] CTC预测logits
        wav_lens: [B] 音频长度比例
        hparams: 超参数对象，需要包含:
            - hparams.output_folder: 输出文件夹
            - hparams.blank_index: blank token ID
            - hparams.epoch_counter.current: 当前epoch
        device: 计算设备
        plot_scores_fn: 绘制对齐图的函数 (aligned_tokens, scores) -> fig
        plot_comparison_fn: 绘制对比图的函数 (gt, scores, pred, pred_scores, title) -> fig
        fixed_sample_id: VALID阶段固定监控的样本ID（可选）
        test_predictions_list: TEST阶段用于收集预测的列表（可选）
        asr_model: ASR模型（用于K2对齐，可选）
        tokenizer: 标记器（用于K2对齐，可选）
        audio_dir: 音频文件目录路径（用于K2对齐，可选）
        use_k2_alignment: 是否使用K2对齐（需要k2库）
    
    Returns:
        dict: 包含以下内容:
            - 'fixed_sample_id': 固定样本ID（VALID）
            - 'test_predictions': 测试预测列表（TEST）
            - 'alignment_results': 对齐结果列表
            - 'alignment_results': 对齐结果列表
    """
    from torchaudio.functional import forced_align, merge_tokens
    
    results = {
        'fixed_sample_id': fixed_sample_id,
        'test_predictions': test_predictions_list or [],
        'alignment_results': []
    }
    
    # ─── 确定处理模式 ───
    if stage == sb.Stage.VALID:
        # VALID: 监控固定样本
        if fixed_sample_id is None:
            # 首次运行时初始化
            fixed_sample_id = batch.id[0]
            results['fixed_sample_id'] = fixed_sample_id
            print(f"📌 VALID固定监控样本ID: {fixed_sample_id}")
        
        # 检查当前batch是否包含固定样本
        if fixed_sample_id not in batch.id:
            return results
        
        sample_indices = [batch.id.index(fixed_sample_id)]
        epoch = hparams.epoch_counter.current
        alignment_dir = os.path.join(hparams.output_folder, "alignment_monitoring")
        output_base = os.path.join(alignment_dir, f"epoch_{epoch:03d}")
        stage_label = f"VALID-Epoch{epoch}"
        
    elif stage == sb.Stage.TEST:
        # TEST: 处理所有样本
        sample_indices = list(range(len(batch.id)))
        alignment_dir = os.path.join(hparams.output_folder, "test_decoding")
        output_base = alignment_dir
        stage_label = "TEST"
        results['test_predictions'] = test_predictions_list or []
    else:
        return results
    
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    
    # ─── 处理每个样本 ───
    targets, target_lens = batch.phn_encoded_target
    
    for real_batch_idx in sample_indices:
        sample_id = batch.id[real_batch_idx]
        sample_stem = Path(sample_id).stem
        
        # 提取单个样本
        p_ctc_sample = p_ctc[real_batch_idx:real_batch_idx+1]  # [1, T, C]
        wav_lens_sample = wav_lens[real_batch_idx:real_batch_idx+1]  # [1]
        targets_sample = targets[real_batch_idx:real_batch_idx+1]
        target_lens_sample = target_lens[real_batch_idx:real_batch_idx+1]
        
        # 计算实际长度
        actual_target_len = int(target_lens_sample[0].item() * targets_sample.shape[-1])
        targets_sample_no_pad = targets_sample[:, :actual_target_len]
        actual_input_len = int(wav_lens_sample[0].item() * p_ctc_sample.shape[1])
        
        # ─── 计算GT对齐 ───
        aligned_tokens_gt, scores_gt = compute_forced_alignment(
            p_ctc=p_ctc_sample,
            targets=targets_sample_no_pad,
            actual_target_len=actual_target_len,
            actual_input_len=actual_input_len,
            blank_index=hparams.blank_index,
            device=device
        )
        
        if aligned_tokens_gt is None:
            if stage == sb.Stage.TEST:
                print(f"⚠️  [{sample_id}] GT alignment failed")
            continue
        
        # ─── Greedy解码预测 ───
        predict_target = sb.decoders.ctc_greedy_decode(
            p_ctc_sample, wav_lens_sample, blank_id=hparams.blank_index
        )
        predict_target_sample = predict_target[0] if predict_target else []
        
        # ─── 计算预测对齐 ───
        aligned_tokens_pred, scores_pred = None, None
        if len(predict_target_sample) > 0:
            predict_target_tensor = torch.tensor([predict_target_sample], dtype=torch.int32, device=device)
            aligned_tokens_pred, scores_pred = compute_forced_alignment(
                p_ctc=p_ctc_sample,
                targets=predict_target_tensor,
                actual_target_len=len(predict_target_sample),
                actual_input_len=actual_input_len,
                blank_index=hparams.blank_index,
                device=device
            )
        
        # ─── K2对齐（可选） ───
        k2_alignment_result = None
        if use_k2_alignment and asr_model is not None and tokenizer is not None and audio_dir is not None:
            # 获取音频文件路径
            audio_file = os.path.join(audio_dir, f"{sample_stem}.wav")
            if os.path.exists(audio_file):
                k2_alignment_result = compute_k2_alignment(
                    audio_file=audio_file,
                    targets=targets_sample_no_pad[0].tolist(),
                    blank_index=hparams.blank_index,
                    asr_model=asr_model,
                    tokenizer=tokenizer,
                    device=device,
                    frame_shift_ms=20  # 根据需要调整
                )
                if k2_alignment_result['success']:
                    print(f"✅ K2 alignment successful for {sample_stem}")
            else:
                if stage == sb.Stage.TEST:
                    print(f"⚠️  [{sample_id}] Audio file not found: {audio_file}")
        
        # ─── 保存结果 ───
        alignment_result = {
            'sample_id': sample_id,
            'sample_stem': sample_stem,
            'aligned_tokens_gt': aligned_tokens_gt,
            'scores_gt': scores_gt,
            'aligned_tokens_pred': aligned_tokens_pred,
            'scores_pred': scores_pred,
            'k2_alignment': k2_alignment_result  # 添加K2对齐结果
        }
        results['alignment_results'].append(alignment_result)
        
        # ─── 绘制并保存图表 ───
        if stage == sb.Stage.VALID:
            # VALID: 分开的GT和预测图
            if aligned_tokens_gt is not None and scores_gt is not None:
                try:
                    fig_gt = plot_scores_fn(aligned_tokens_gt, scores_gt)
                    fig_gt.suptitle(f"GT Alignment - {sample_stem} ({stage_label})")
                    gt_path = os.path.join(output_base, f"gt_alignment_{sample_stem}.png")
                    fig_gt.savefig(gt_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_gt)
                except Exception as e:
                    print(f"⚠️  [{sample_id}] Failed to save GT plot: {e}")
            
            if aligned_tokens_pred is not None and scores_pred is not None:
                try:
                    fig_pred = plot_scores_fn(aligned_tokens_pred, scores_pred)
                    fig_pred.suptitle(f"Pred Alignment - {sample_stem} ({stage_label})")
                    pred_path = os.path.join(output_base, f"pred_alignment_{sample_stem}.png")
                    fig_pred.savefig(pred_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_pred)
                except Exception as e:
                    print(f"⚠️  [{sample_id}] Failed to save pred plot: {e}")
        
        elif stage == sb.Stage.TEST:
            # TEST: 对比图
            try:
                # 提取speaker ID
                speaker_id = "unknown"
                if "/" in sample_id:
                    parts = sample_id.split("/")
                    for i, part in enumerate(parts):
                        if part in ["L2-ARCTIC", "L2ARCTIC"] and i+1 < len(parts):
                            speaker_id = parts[i+1]
                            break
                
                comparison_title = f"🎤 {speaker_id} - {sample_stem}\n GT vs Predicted Alignment"
                
                fig_compare = plot_comparison_fn(
                    aligned_tokens_gt, scores_gt,
                    aligned_tokens_pred if aligned_tokens_pred is not None else [],
                    scores_pred,
                    title=comparison_title
                )
                compare_path = os.path.join(output_base, f"alignment_compare_{sample_stem}.png")
                fig_compare.savefig(compare_path, dpi=150, bbox_inches='tight')
                plt.close(fig_compare)
                print(f"💾 Saved: {compare_path}")
            except Exception as e:
                print(f"⚠️  [{sample_id}] Failed to save comparison plot: {e}")
    
    return results
