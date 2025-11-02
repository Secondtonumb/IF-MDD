"""
Generate interactive HTML visualization for audio and phoneme alignment.
This script creates an HTML file that allows you to play audio and see
the corresponding phonemes highlighted in real-time.
"""

import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import io


def generate_spectrogram_with_annotations(audio_path, intervals_data, output_image_path=None):
    """
    Generate a spectrogram with phoneme annotations overlaid.
    
    Args:
        audio_path: Path to the audio file
        intervals_data: List of phoneme intervals
        output_image_path: Optional path to save the image
    
    Returns:
        Base64 encoded image string
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Create figure with high DPI for better quality
    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(18, 8), dpi=100,
                                            gridspec_kw={'height_ratios': [1, 3]})
    
    # Remove all margins and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)
    
    # === Waveform (top) ===
    time_wave = np.linspace(0, len(y) / sr, len(y))
    ax_wave.plot(time_wave, y, color='#2E86AB', linewidth=0.5)
    ax_wave.set_xlim([0, len(y) / sr])
    ax_wave.set_ylim([y.min(), y.max()])
    ax_wave.axis('off')  # Remove all axes
    ax_wave.margins(0, 0)
    
    # Add phoneme annotations on waveform
    y_max_wave = y.max()
    y_min_wave = y.min()
    y_range = y_max_wave - y_min_wave
    
    for i, interval in enumerate(intervals_data):
        start = interval['start']
        end = interval['end']
        phoneme = interval['content']
        center = (start + end) / 2
        
        # Draw vertical lines at boundaries
        ax_wave.axvline(x=start, color='red', linestyle='-', linewidth=1.5, alpha=0.6)
        
        # Add phoneme text
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         edgecolor='black', alpha=0.85, linewidth=1)
        ax_wave.text(center, y_max_wave + y_range * 0.1, phoneme, 
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=bbox_props)
    
    # Draw final boundary line
    if intervals_data:
        ax_wave.axvline(x=intervals_data[-1]['end'], color='red', linestyle='-', 
                       linewidth=1.5, alpha=0.6)
    
    # === Spectrogram (bottom) ===
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    
    # Plot spectrogram without axes
    librosa.display.specshow(D, sr=sr, hop_length=256, x_axis=None, y_axis=None,
                            ax=ax_spec, cmap='viridis', vmin=-80, vmax=0)
    
    # Set frequency range (0-8000 Hz is typical for speech)
    ax_spec.set_ylim([0, 8000])
    ax_spec.set_xlim([0, len(y) / sr])
    ax_spec.axis('off')  # Remove all axes
    ax_spec.margins(0, 0)
    
    # Add phoneme boundary lines on spectrogram
    for i, interval in enumerate(intervals_data):
        start = interval['start']
        ax_spec.axvline(x=start, color='white', linestyle='-', linewidth=1.2, alpha=0.5)
    
    # Draw final boundary line
    if intervals_data:
        ax_spec.axvline(x=intervals_data[-1]['end'], color='white', linestyle='-', 
                       linewidth=1.2, alpha=0.5)
    
    # Save to file if requested
    if output_image_path:
        plt.savefig(output_image_path, dpi=100, bbox_inches='tight', pad_inches=0)
        print(f"📊 Spectrogram saved to: {output_image_path}")
    
    # Save to bytes and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_clean_spectrogram(audio_path, intervals_data, output_image_path=None):
    """
    Generate a clean spectrogram and waveform without axes, labels, or legends.
    Perfect alignment with time for playhead overlay.
    
    Args:
        audio_path: Path to the audio file
        intervals_data: List of phoneme intervals
        output_image_path: Optional path to save the image
    
    Returns:
        Base64 encoded image string
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    # Create figure - exact dimensions with no margins
    fig = plt.figure(figsize=(16, 6), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    
    # Ratios must match overlay positioning in HTML
    wave_ratio = 0.3
    spec_ratio = 0.7
    
    # Create subplots with precise positioning (left, bottom, width, height)
    ax_wave = fig.add_axes([0, spec_ratio, 1, wave_ratio])  # Waveform at top
    ax_spec = fig.add_axes([0, 0, 1, spec_ratio])           # Spectrogram at bottom
    
    # === Waveform ===
    time_wave = np.linspace(0, duration, len(y))
    ax_wave.fill_between(time_wave, 0, y, color='#2E86AB', alpha=0.7, linewidth=0)
    ax_wave.plot(time_wave, y, color='#1A5276', linewidth=0.5)
    ax_wave.set_xlim([0, duration])
    ax_wave.set_ylim([y.min() * 1.2, y.max() * 1.2])
    ax_wave.axis('off')
    
    # Keep waveform clean (labels are rendered as HTML overlay spans)
    
    # === Spectrogram ===
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    
    # Plot spectrogram - no axes
    img = ax_spec.imshow(D, aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, duration, 0, sr/2], vmin=-80, vmax=0,
                         interpolation='bilinear')
    
    ax_spec.set_ylim([0, 8000])
    ax_spec.set_xlim([0, duration])
    ax_spec.axis('off')
    
    # Keep spectrogram clean (boundaries and labels are rendered as HTML overlays)
    
    # Save
    if output_image_path:
        plt.savefig(output_image_path, dpi=100, bbox_inches=None, pad_inches=0,
                    facecolor='white')
        print(f"📊 Clean spectrogram saved to: {output_image_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches=None, pad_inches=0,
                facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_interactive_html(audio_path, intervals_data, output_html_path):
    """
    Generate an interactive HTML file for audio-phoneme visualization.
    
    Args:
        audio_path: Path to the audio file (.wav)
        intervals_data: List of dicts with 'start', 'end', 'content' keys
        output_html_path: Path to save the HTML file
    """
    # Read audio file and encode to base64
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    # Generate spectrogram
    print("🎨 Generating clean spectrogram with waveform (no axes/labels)...")
    spectrogram_image_path = str(Path(output_html_path).with_suffix('.spectrogram.png'))
    spectrogram_base64 = generate_clean_spectrogram(
        audio_path, intervals_data, spectrogram_image_path
    )
    
    # Get audio duration from intervals
    audio_duration = intervals_data[-1]['end'] if intervals_data else 0
    
    # Convert intervals to JSON
    intervals_json = json.dumps(intervals_data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频-音素对齐查看器</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        
        .audio-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        audio {{
            width: 100%;
            margin-top: 10px;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
            font-weight: 500;
        }}
        
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #5a6268;
        }}
        
        .time-display {{
            display: inline-block;
            padding: 10px 20px;
            background: #e9ecef;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            color: #495057;
        }}
        
        .phonemes-container {{
            margin-top: 20px;
        }}
        
        .phonemes-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}
        
        .phoneme-box {{
            padding: 12px 18px;
            background: #e9ecef;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
            min-width: 60px;
            text-align: center;
            position: relative;
        }}
        
        .phoneme-box:hover {{
            background: #dee2e6;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .phoneme-box.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #5568d3;
            transform: scale(1.1);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            font-weight: bold;
        }}
        
        .phoneme-text {{
            font-size: 16px;
            font-weight: 500;
        }}
        
        .phoneme-time {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }}
        
        .phoneme-box.active .phoneme-time {{
            color: rgba(255, 255, 255, 0.9);
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin-top: 15px;
            overflow: hidden;
            cursor: pointer;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
        
        .waveform-placeholder {{
            background: #f8f9fa;
            height: 100px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            margin-top: 15px;
        }}
        
        .section-title {{
            font-size: 1.2em;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .spectrogram-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .spectrogram-container {{
            position: relative;
            margin-top: 15px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background: white;
            line-height: 0;  /* Remove any line spacing */
        }}
        
        .spectrogram-image {{
            width: 100%;
            height: auto;
            display: block;
            cursor: crosshair;
            margin: 0;
            padding: 0;
            border: none;
            -webkit-user-drag: none;
            user-select: none;
        }}
        
        .playhead {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: rgba(255, 0, 0, 0.85);
            box-shadow: 0 0 8px rgba(255, 0, 0, 0.6), 0 0 3px rgba(255, 0, 0, 1);
            pointer-events: none;
            transition: left 0.05s linear;
            z-index: 10;
            left: 0%;
        }}
        
        .spectrogram-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
            z-index: 2;
        }}

        /* Phoneme spans over spectrogram */
        .phoneme-span {{
            position: absolute;
            top: 0;
            height: 100%;
            background: rgba(255, 255, 255, 0.06);
            border-left: 1px solid rgba(255, 255, 255, 0.15);
            border-right: 1px solid rgba(0, 0, 0, 0.05);
            pointer-events: none; /* allow drag/select on spectrogram */
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }}

        .phoneme-span.alt {{
            background: rgba(0, 0, 0, 0.06);
        }}

        .phoneme-span .label {{
            margin-top: 6px;
            padding: 2px 6px;
            font-size: 12px;
            font-weight: 700;
            color: #000;
            background: rgba(255, 255, 0, 0.85);
            border: 1px solid rgba(0, 0, 0, 0.8);
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}

        .phoneme-span.active .label {{
            color: #fff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: rgba(85, 104, 211, 0.9);
            box-shadow: 0 4px 10px rgba(102,126,234,0.35);
        }}

        .selection-rect {{
            position: absolute;
            top: 0;
            height: 100%;
            background: rgba(102, 126, 234, 0.25);
            border: 2px solid #667eea;
            border-radius: 2px;
            display: none;
            pointer-events: none;
            z-index: 5;
        }}
        
        .view-toggle {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .toggle-btn {{
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        
        .toggle-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .toggle-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频-音素对齐查看器</h1>
        <p class="subtitle">播放音频以查看实时音素高亮 + 频谱图可视化</p>
        
        <!-- View Toggle -->
        <div class="view-toggle">
            <button class="toggle-btn active" onclick="showView('spectrogram')">📊 频谱图视图</button>
            <button class="toggle-btn" onclick="showView('phonemes')">🔤 音素列表视图</button>
            <button class="toggle-btn" onclick="showView('both')">📊 + 🔤 双视图</button>
        </div>
        
        <div class="audio-section">
            <h2 class="section-title">音频播放器</h2>
            <audio id="audioPlayer" controls>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                您的浏览器不支持音频播放。
            </audio>
            
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="playAudio()">▶ 播放</button>
                <button class="btn btn-secondary" onclick="pauseAudio()">⏸ 暂停</button>
                <button class="btn btn-secondary" onclick="resetAudio()">⏹ 重置</button>
                <span class="time-display" id="timeDisplay">0.00s / {audio_duration:.2f}s</span>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">总音素数</div>
                    <div class="stat-value" id="totalPhonemes">{len(intervals_data)}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">音频时长</div>
                    <div class="stat-value">{audio_duration:.2f}s</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">当前音素</div>
                    <div class="stat-value" id="currentPhoneme">-</div>
                </div>
            </div>
        </div>
        
        <!-- Spectrogram Section -->
        <div class="spectrogram-section" id="spectrogramSection">
            <h2 class="section-title">频谱图 + 音素标注 (点击跳转)</h2>
            <div class="spectrogram-container" id="spectrogramContainer">
                <img src="data:image/png;base64,{spectrogram_base64}" 
                     class="spectrogram-image" 
                     id="spectrogramImage"
                     alt="Spectrogram with phoneme annotations">
                <div class="playhead" id="playhead"></div>
                <!-- Overlay only on spectrogram portion (bottom 70%) -->
                <div class="selection-rect" id="selectionRect" style="top: 30%; height: 70%;"></div>
                <div class="spectrogram-overlay" id="spectrogramOverlay" style="top: 30%; height: 70%;"></div>
            </div>
        </div>
        
        <div class="phonemes-container" id="phonemesSection">
            <h2 class="section-title">音素序列 (点击跳转)</h2>
            <div class="phonemes-grid" id="phonemesGrid"></div>
        </div>
    </div>
    
    <script>
        const intervals = {intervals_json};
        const audio = document.getElementById('audioPlayer');
        const phonemesGrid = document.getElementById('phonemesGrid');
        const timeDisplay = document.getElementById('timeDisplay');
        const progressFill = document.getElementById('progressFill');
        const progressBar = document.getElementById('progressBar');
        const currentPhonemeDisplay = document.getElementById('currentPhoneme');
        const playhead = document.getElementById('playhead');
        const spectrogramContainer = document.getElementById('spectrogramContainer');
        const spectrogramImage = document.getElementById('spectrogramImage');
        const spectrogramSection = document.getElementById('spectrogramSection');
        const phonemesSection = document.getElementById('phonemesSection');
    const selectionRect = document.getElementById('selectionRect');
    const spectrogramOverlay = document.getElementById('spectrogramOverlay');
        
        let audioDuration = {audio_duration};
    let segmentStart = null;
    let segmentEnd = null;
    let selectionStart = null;
    let selectionEnd = null;
    let isSelecting = false;
    let selectStartX = 0;
    let selectCurrentX = 0;
        
        // View management
        function showView(view) {{
            const buttons = document.querySelectorAll('.toggle-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            if (view === 'spectrogram') {{
                spectrogramSection.style.display = 'block';
                phonemesSection.style.display = 'none';
            }} else if (view === 'phonemes') {{
                spectrogramSection.style.display = 'none';
                phonemesSection.style.display = 'block';
            }} else {{ // both
                spectrogramSection.style.display = 'block';
                phonemesSection.style.display = 'block';
            }}
        }}
        
        // Create phoneme boxes and spectrogram overlay spans
        const overlaySpans = [];
        intervals.forEach((interval, index) => {{
            const box = document.createElement('div');
            box.className = 'phoneme-box';
            box.dataset.index = index;
            box.innerHTML = `
                <div class="phoneme-text">${{interval.content}}</div>
                <div class="phoneme-time">${{interval.start.toFixed(2)}}s - ${{interval.end.toFixed(2)}}s</div>
            `;
            box.onclick = () => {{
                // Set segment to this phoneme and play only this range
                segmentStart = interval.start;
                segmentEnd = interval.end;
                
                // Update visual selection on spectrogram
                const rect = spectrogramImage.getBoundingClientRect();
                const left = (segmentStart / audioDuration) * rect.width;
                const right = (segmentEnd / audioDuration) * rect.width;
                selectionRect.style.display = 'block';
                selectionRect.style.left = left + 'px';
                selectionRect.style.width = (right - left) + 'px';

                audio.currentTime = segmentStart;
                audio.play();
            }};
            phonemesGrid.appendChild(box);

            // Create overlay span over spectrogram covering start..end
            const span = document.createElement('div');
            span.className = 'phoneme-span' + ((index % 2 === 1) ? ' alt' : '');
            const leftPct = (interval.start / audioDuration) * 100;
            const widthPct = ((interval.end - interval.start) / audioDuration) * 100;
            span.style.left = leftPct + '%';
            span.style.width = widthPct + '%';
            const label = document.createElement('div');
            label.className = 'label';
            label.textContent = interval.content;
            span.appendChild(label);
            spectrogramOverlay.appendChild(span);
            overlaySpans.push(span);
        }});
        
        // Update playhead position on spectrogram
        function updatePlayhead(currentTime) {{
            const percentage = (currentTime / audioDuration) * 100;
            playhead.style.left = percentage + '%';
        }}
        
        // Selection on spectrogram (click-drag to select, small click to seek)
        spectrogramContainer.addEventListener('mousedown', (e) => {{
            if (e.button !== 0) return; // left click only
            const rect = spectrogramImage.getBoundingClientRect();
            isSelecting = true;
            selectStartX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            selectCurrentX = selectStartX;
            selectionRect.style.display = 'block';
            selectionRect.style.left = selectStartX + 'px';
            selectionRect.style.width = '0px';
        }});

        window.addEventListener('mousemove', (e) => {{
            if (!isSelecting) return;
            const rect = spectrogramImage.getBoundingClientRect();
            selectCurrentX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            const left = Math.min(selectStartX, selectCurrentX);
            const width = Math.abs(selectCurrentX - selectStartX);
            selectionRect.style.left = left + 'px';
            selectionRect.style.width = width + 'px';
        }});

        window.addEventListener('mouseup', (e) => {{
            if (!isSelecting) return;
            const rect = spectrogramImage.getBoundingClientRect();
            isSelecting = false;
            const leftPx = Math.min(selectStartX, selectCurrentX);
            const rightPx = Math.max(selectStartX, selectCurrentX);
            const widthPx = rightPx - leftPx;

            // If selection is very small, treat as a seek
            if (widthPx < 5) {{
                selectionRect.style.display = 'none';
                const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                const percentage = x / rect.width;
                const seekTime = percentage * audioDuration;
                segmentStart = null;
                segmentEnd = null;
                audio.currentTime = seekTime;
                audio.play();
                return;
            }}

            // Convert to time selection and play only the selected segment
            selectionStart = (leftPx / rect.width) * audioDuration;
            selectionEnd = (rightPx / rect.width) * audioDuration;
            segmentStart = selectionStart;
            segmentEnd = selectionEnd;
            audio.currentTime = segmentStart;
            audio.play();
        }});
        
        // Update highlight on time update
        audio.addEventListener('timeupdate', () => {{
            const currentTime = audio.currentTime;
            const duration = audio.duration || audioDuration;

            // Stop at segment end if playing a segment
            if (segmentEnd !== null && currentTime >= (segmentEnd - 0.005)) {{
                audio.pause();
                audio.currentTime = segmentEnd;
            }}
            
            // Update time display
            timeDisplay.textContent = `${{currentTime.toFixed(2)}}s / ${{duration.toFixed(2)}}s`;
            
            // Update progress bar
            const progress = (currentTime / duration) * 100;
            progressFill.style.width = progress + '%';
            
            // Update playhead on spectrogram
            updatePlayhead(currentTime);
            
            // Find and highlight current phoneme
            let foundActive = false;
            intervals.forEach((interval, index) => {{
                const box = phonemesGrid.children[index];
                if (currentTime >= interval.start && currentTime < interval.end) {{
                    box.classList.add('active');
                    if (overlaySpans[index]) overlaySpans[index].classList.add('active');
                    currentPhonemeDisplay.textContent = interval.content;
                    foundActive = true;
                    
                    // Scroll into view
                    box.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    box.classList.remove('active');
                    if (overlaySpans[index]) overlaySpans[index].classList.remove('active');
                }}
            }});
            
            if (!foundActive) {{
                currentPhonemeDisplay.textContent = '-';
            }}
        }});
        
        // Progress bar click to seek
        progressBar.addEventListener('click', (e) => {{
            const rect = progressBar.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = x / rect.width;
            audio.currentTime = percentage * audio.duration;
        }});
        
        function playAudio() {{
            audio.play();
        }}
        
        function pauseAudio() {{
            audio.pause();
        }}
        
        function resetAudio() {{
            audio.pause();
            audio.currentTime = 0;
            progressFill.style.width = '0%';
            updatePlayhead(0);
            // Clear segment endpoints (keep visual selection if any)
            segmentStart = null;
            segmentEnd = null;
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{
                e.preventDefault();
                if (audio.paused) {{
                    playAudio();
                }} else {{
                    pauseAudio();
                }}
            }} else if (e.code === 'ArrowLeft') {{
                audio.currentTime = Math.max(0, audio.currentTime - 0.5);
            }} else if (e.code === 'ArrowRight') {{
                audio.currentTime = Math.min(audio.duration, audio.currentTime + 0.5);
            }}
        }});
        
        // Initialize
        updatePlayhead(0);
        console.log('Loaded', intervals.length, 'phoneme intervals');
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Interactive HTML generated: {output_html_path}")
    print(f"📊 Spectrogram image saved: {spectrogram_image_path}")
    print(f"📊 Total phonemes: {len(intervals_data)}")
    print(f"⏱️  Audio duration: {audio_duration:.2f}s")
    print(f"\n💡 功能特性：")
    print(f"   - 📊 波形图 + 频谱图（上下分布）")
    print(f"   - 🏷️  音素标注在波形图上方")
    print(f"   - 🔴 红色播放头精确跟随（时间完全对齐）")
    print(f"   - 🎯 点击频谱图任意位置跳转播放")
    print(f"   - ⌨️  空格键播放/暂停，左右箭头微调")
    print(f"   - 🔀 三种视图模式切换")


if __name__ == "__main__":
    import tgt
    
    # Example usage - read from TextGrid
    # audio_file = "examples/arctic_b0503.wav"
    # textgrid_file = "examples/arctic_b0503.TextGrid"
    # output_html = "examples/arctic_b0503_viewer.html"
    file_id = "arctic_b0503"
    audio_file = f"/home/kevingenghaopeng/MDD/IF-MDD/examples/{file_id}.wav"
    textgrid_file = f"/home/kevingenghaopeng/MDD/IF-MDD/examples/{file_id}.TextGrid"
    output_html = f"/home/kevingenghaopeng/MDD/IF-MDD/examples/{file_id}_viewer.html"
    
    # Read TextGrid
    print(f"📖 Reading TextGrid: {textgrid_file}")
    tg = tgt.io.read_textgrid(textgrid_file)
    
    # Extract intervals from the first tier (phonemes)
    intervals_data = []
    for tier in tg.tiers:
        if isinstance(tier, tgt.IntervalTier):
            for interval in tier.intervals:
                if interval.text:  # Skip empty intervals
                    intervals_data.append({
                        "start": round(interval.start_time, 4),
                        "end": round(interval.end_time, 4),
                        "content": interval.text,
                    })
            break  # Use only the first tier
    
    print(f"📊 Found {len(intervals_data)} phoneme intervals")
    
    # Generate HTML
    generate_interactive_html(audio_file, intervals_data, output_html)
    
    print(f"\n🌐 在浏览器中打开: {Path(output_html).absolute()}")
