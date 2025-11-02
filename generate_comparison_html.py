"""
Generate interactive HTML visualization comparing two alignment methods.
This allows side-by-side or vertical comparison of:
1. CTC Prefix Beam Search alignment
2. K2 Forced Alignment
"""

import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io


def generate_dual_spectrogram(audio_path, intervals_data1, intervals_data2, output_image_path=None):
    """
    Generate two spectrograms stacked vertically for comparison.
    
    Args:
        audio_path: Path to the audio file
        intervals_data1: List of phoneme intervals from method 1 (CTC Beam Search)
        intervals_data2: List of phoneme intervals from method 2 (K2 Alignment)
        output_image_path: Optional path to save the image
    
    Returns:
        Base64 encoded image string
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    # Create figure with two rows (vertical stack)
    fig, (ax_wave1, ax_spec1, ax_wave2, ax_spec2) = plt.subplots(
        4, 1, figsize=(18, 10), dpi=100,
        gridspec_kw={'height_ratios': [1, 3, 1, 3], 'hspace': 0.15}
    )
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Compute spectrogram once
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    time_wave = np.linspace(0, duration, len(y))
    
    # === Top: Method 1 (CTC Beam Search) ===
    # Waveform
    ax_wave1.fill_between(time_wave, 0, y, color='#2E86AB', alpha=0.7, linewidth=0)
    ax_wave1.plot(time_wave, y, color='#1A5276', linewidth=0.5)
    ax_wave1.set_xlim([0, duration])
    ax_wave1.set_ylim([y.min() * 1.2, y.max() * 1.2])
    ax_wave1.axis('off')
    ax_wave1.set_title('Method 1: CTC Prefix Beam Search', fontsize=14, fontweight='bold', 
                       pad=10, color='#2E86AB')
    
    # Spectrogram
    ax_spec1.imshow(D, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, duration, 0, sr/2], vmin=-80, vmax=0,
                    interpolation='bilinear')
    ax_spec1.set_ylim([0, 8000])
    ax_spec1.set_xlim([0, duration])
    ax_spec1.axis('off')
    
    # Add boundaries and labels for method 1
    for interval in intervals_data1:
        start = interval['start']
        end = interval['end']
        center = (start + end) / 2
        phoneme = interval['content']
        
        ax_wave1.axvline(x=start, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_spec1.axvline(x=start, color='red', linestyle='-', linewidth=1.2, alpha=0.6)
        
        # Add phoneme label on spectrogram
        ax_spec1.text(center, 7500, phoneme, ha='center', va='top',
                     fontsize=9, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7, edgecolor='none'))
    
    # === Bottom: Method 2 (K2 Forced Alignment) ===
    # Waveform
    ax_wave2.fill_between(time_wave, 0, y, color='#2EAB86', alpha=0.7, linewidth=0)
    ax_wave2.plot(time_wave, y, color='#1A7652', linewidth=0.5)
    ax_wave2.set_xlim([0, duration])
    ax_wave2.set_ylim([y.min() * 1.2, y.max() * 1.2])
    ax_wave2.axis('off')
    ax_wave2.set_title('Method 2: K2 Forced Alignment', fontsize=14, fontweight='bold', 
                       pad=10, color='#2EAB86')
    
    # Spectrogram
    ax_spec2.imshow(D, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, duration, 0, sr/2], vmin=-80, vmax=0,
                    interpolation='bilinear')
    ax_spec2.set_ylim([0, 8000])
    ax_spec2.set_xlim([0, duration])
    ax_spec2.axis('off')
    
    # Add boundaries and labels for method 2
    for interval in intervals_data2:
        start = interval['start']
        end = interval['end']
        center = (start + end) / 2
        phoneme = interval['content']
        
        ax_wave2.axvline(x=start, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        ax_spec2.axvline(x=start, color='lime', linestyle='-', linewidth=1.2, alpha=0.6)
        
        # Add phoneme label on spectrogram
        ax_spec2.text(center, 7500, phoneme, ha='center', va='top',
                     fontsize=9, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7, edgecolor='none'))
    
    # Save
    if output_image_path:
        plt.savefig(output_image_path, dpi=100, bbox_inches='tight', pad_inches=0.1,
                    facecolor='white')
        print(f"📊 Comparison spectrogram saved to: {output_image_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1,
                facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_comparison_html(audio_path, intervals_data1, intervals_data2, 
                             output_html_path, method1_name="CTC Beam Search",
                             method2_name="K2 Forced Alignment"):
    """
    Generate an interactive HTML file comparing two alignment methods.
    
    Args:
        audio_path: Path to the audio file (.wav)
        intervals_data1: List of dicts with 'start', 'end', 'content' keys (Method 1)
        intervals_data2: List of dicts with 'start', 'end', 'content' keys (Method 2)
        output_html_path: Path to save the HTML file
        method1_name: Display name for method 1
        method2_name: Display name for method 2
    """
    # Read audio file and encode to base64
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    # Generate comparison spectrogram
    print("🎨 Generating comparison spectrogram...")
    spectrogram_image_path = str(Path(output_html_path).with_suffix('.comparison.png'))
    spectrogram_base64 = generate_dual_spectrogram(
        audio_path, intervals_data1, intervals_data2, spectrogram_image_path
    )
    
    # Get audio duration
    audio_duration = max(
        intervals_data1[-1]['end'] if intervals_data1 else 0,
        intervals_data2[-1]['end'] if intervals_data2 else 0
    )
    
    # Convert intervals to JSON
    intervals1_json = json.dumps(intervals_data1)
    intervals2_json = json.dumps(intervals_data2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phoneme Alignment Comparison Viewer</title>
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
            max-width: 1400px;
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
            margin-bottom: 20px;
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
            align-items: center;
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
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            cursor: pointer;
            margin: 15px 0;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: #667eea;
            border-radius: 4px;
            width: 0%;
            transition: width 0.1s linear;
        }}
        
        .time-display {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        
        .comparison-section {{
            margin-top: 30px;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .method-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid transparent;
            transition: all 0.3s;
        }}
        
        .method-panel.method1 {{
            border-color: #2E86AB;
        }}
        
        .method-panel.method2 {{
            border-color: #2EAB86;
        }}
        
        .method-panel h3 {{
            margin-bottom: 15px;
            color: #333;
        }}
        
        .method1 h3 {{
            color: #2E86AB;
        }}
        
        .method2 h3 {{
            color: #2EAB86;
        }}
        
        .phoneme-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        
        .phoneme-box {{
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            font-weight: 500;
            border: 2px solid transparent;
        }}
        
        .method1 .phoneme-box {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        
        .method1 .phoneme-box:hover {{
            background: #bbdefb;
            transform: translateY(-2px);
        }}
        
        .method1 .phoneme-box.active {{
            background: #2E86AB;
            color: white;
            border-color: #1A5276;
            box-shadow: 0 2px 8px rgba(46, 134, 171, 0.4);
        }}
        
        .method2 .phoneme-box {{
            background: #e8f5e9;
            color: #388e3c;
        }}
        
        .method2 .phoneme-box:hover {{
            background: #c8e6c9;
            transform: translateY(-2px);
        }}
        
        .method2 .phoneme-box.active {{
            background: #2EAB86;
            color: white;
            border-color: #1A7652;
            box-shadow: 0 2px 8px rgba(46, 171, 134, 0.4);
        }}
        
        .stats {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        
        .stat-card h4 {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        
        .spectrogram-container {{
            margin: 30px 0;
            text-align: center;
            position: relative;
        }}
        
        .spectrogram-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            cursor: crosshair;
        }}
        
        .spectrogram-wrapper {{
            position: relative;
            display: inline-block;
        }}
        
        .selection-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        
        .selection-box {{
            position: absolute;
            background: rgba(255, 255, 0, 0.3);
            border: 2px solid yellow;
            pointer-events: none;
        }}
        
        .highlight-box {{
            position: absolute;
            background: rgba(255, 0, 0, 0.2);
            border: 2px solid red;
            pointer-events: none;
        }}
        
        .diff-highlight {{
            background: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        
        .current-phoneme {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 15px;
            margin: 20px 0;
        }}
        
        .current-phoneme-display {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }}
        
        .current-phoneme-display.method1 {{
            background: #e3f2fd;
            border: 2px solid #2E86AB;
        }}
        
        .current-phoneme-display.method2 {{
            background: #e8f5e9;
            border: 2px solid #2EAB86;
        }}
        
        .current-phoneme-display .label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .current-phoneme-display .phoneme {{
            font-size: 32px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Phoneme Alignment Comparison Viewer</h1>
        <p class="subtitle">Compare two different speech alignment methods</p>
        
        <!-- Audio Player -->
        <div class="audio-section">
            <h3>🎧 Audio Player</h3>
            <audio id="audio" controls>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support audio playback.
            </audio>
            
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="time-display">
                <span id="currentTime">0:00</span>
                <span id="duration">0:00</span>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="playAudio()">▶️ Play</button>
                <button class="btn btn-secondary" onclick="pauseAudio()">⏸️ Pause</button>
                <button class="btn btn-secondary" onclick="resetAudio()">🔄 Reset</button>
                <button class="btn btn-secondary" onclick="playSelection()" id="playSelectionBtn" style="display:none;">🎯 Play Selection</button>
            </div>
        </div>
        
        <!-- Current Phoneme Display -->
        <div class="current-phoneme">
            <div class="current-phoneme-display method1">
                <div class="label">{method1_name}</div>
                <div class="phoneme" id="currentPhoneme1">-</div>
            </div>
            <div class="current-phoneme-display method2">
                <div class="label">{method2_name}</div>
                <div class="phoneme" id="currentPhoneme2">-</div>
            </div>
        </div>
        
        <!-- Legend -->
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2E86AB;"></div>
                <span>{method1_name}</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2EAB86;"></div>
                <span>{method2_name}</span>
            </div>
        </div>
        
        <!-- Spectrogram Comparison -->
        <div class="spectrogram-container">
            <h3>📊 Spectrogram Comparison (Click and drag to select region)</h3>
            <div class="spectrogram-wrapper">
                <img id="spectrogramImg" src="data:image/png;base64,{spectrogram_base64}" alt="Spectrogram Comparison">
                <canvas id="selectionCanvas" class="selection-overlay"></canvas>
            </div>
        </div>
        
        <!-- Comparison Grid -->
        <div class="comparison-section">
            <h2>🔍 Phoneme Sequence Comparison</h2>
            <div class="comparison-grid">
                <!-- Method 1 -->
                <div class="method-panel method1">
                    <h3>📘 {method1_name}</h3>
                    <div class="phoneme-list" id="phonemeList1"></div>
                </div>
                
                <!-- Method 2 -->
                <div class="method-panel method2">
                    <h3>📗 {method2_name}</h3>
                    <div class="phoneme-list" id="phonemeList2"></div>
                </div>
            </div>
        </div>
        
        <!-- Statistics -->
        <div class="stats">
            <div class="stat-card">
                <h4>{method1_name} Phonemes</h4>
                <div class="value" id="count1">0</div>
            </div>
            <div class="stat-card">
                <h4>{method2_name} Phonemes</h4>
                <div class="value" id="count2">0</div>
            </div>
            <div class="stat-card">
                <h4>Audio Duration</h4>
                <div class="value">{audio_duration:.2f}s</div>
            </div>
        </div>
    </div>
    
    <script>
        const audio = document.getElementById('audio');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const currentTime = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');
        const phonemeList1 = document.getElementById('phonemeList1');
        const phonemeList2 = document.getElementById('phonemeList2');
        const currentPhoneme1 = document.getElementById('currentPhoneme1');
        const currentPhoneme2 = document.getElementById('currentPhoneme2');
        const count1 = document.getElementById('count1');
        const count2 = document.getElementById('count2');
        const spectrogramImg = document.getElementById('spectrogramImg');
        const selectionCanvas = document.getElementById('selectionCanvas');
        const ctx = selectionCanvas.getContext('2d');
        const playSelectionBtn = document.getElementById('playSelectionBtn');
        
        // Load intervals data
        const intervals1 = {intervals1_json};
        const intervals2 = {intervals2_json};
        
        // Selection state
        let isSelecting = false;
        let selectionStart = null;
        let selectionEnd = null;
        let selectedStartTime = null;
        let selectedEndTime = null;
        
        // Initialize canvas size
        function updateCanvasSize() {{
            selectionCanvas.width = spectrogramImg.width;
            selectionCanvas.height = spectrogramImg.height;
            selectionCanvas.style.width = spectrogramImg.offsetWidth + 'px';
            selectionCanvas.style.height = spectrogramImg.offsetHeight + 'px';
        }}
        
        spectrogramImg.onload = updateCanvasSize;
        window.addEventListener('resize', updateCanvasSize);
        setTimeout(updateCanvasSize, 100);
        
        // Update counts
        count1.textContent = intervals1.length;
        count2.textContent = intervals2.length;
        
        // Create phoneme boxes for method 1
        intervals1.forEach((interval, index) => {{
            const box = document.createElement('div');
            box.className = 'phoneme-box';
            box.textContent = interval.content;
            box.dataset.index = index;
            box.dataset.start = interval.start;
            box.dataset.end = interval.end;
            box.onclick = () => {{
                playPhonemeSegment(interval.start, interval.end);
                highlightSpectrogramRegion(interval.start, interval.end);
            }};
            phonemeList1.appendChild(box);
        }});
        
        // Create phoneme boxes for method 2
        intervals2.forEach((interval, index) => {{
            const box = document.createElement('div');
            box.className = 'phoneme-box';
            box.textContent = interval.content;
            box.dataset.index = index;
            box.dataset.start = interval.start;
            box.dataset.end = interval.end;
            box.onclick = () => {{
                playPhonemeSegment(interval.start, interval.end);
                highlightSpectrogramRegion(interval.start, interval.end);
            }};
            phonemeList2.appendChild(box);
        }});
        
        // Update time display
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        audio.addEventListener('loadedmetadata', () => {{
            durationDisplay.textContent = formatTime(audio.duration);
        }});
        
        audio.addEventListener('timeupdate', () => {{
            const progress = (audio.currentTime / audio.duration) * 100;
            progressFill.style.width = progress + '%';
            currentTime.textContent = formatTime(audio.currentTime);
            
            // Update active phoneme for method 1
            let foundActive1 = false;
            const boxes1 = phonemeList1.querySelectorAll('.phoneme-box');
            boxes1.forEach((box, index) => {{
                const interval = intervals1[index];
                if (audio.currentTime >= interval.start && audio.currentTime <= interval.end) {{
                    box.classList.add('active');
                    currentPhoneme1.textContent = interval.content;
                    foundActive1 = true;
                    box.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    box.classList.remove('active');
                }}
            }});
            if (!foundActive1) currentPhoneme1.textContent = '-';
            
            // Update active phoneme for method 2
            let foundActive2 = false;
            const boxes2 = phonemeList2.querySelectorAll('.phoneme-box');
            boxes2.forEach((box, index) => {{
                const interval = intervals2[index];
                if (audio.currentTime >= interval.start && audio.currentTime <= interval.end) {{
                    box.classList.add('active');
                    currentPhoneme2.textContent = interval.content;
                    foundActive2 = true;
                    box.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                }} else {{
                    box.classList.remove('active');
                }}
            }});
            if (!foundActive2) currentPhoneme2.textContent = '-';
        }});
        
        // Progress bar click to seek
        progressBar.addEventListener('click', (e) => {{
            const rect = progressBar.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = x / rect.width;
            audio.currentTime = percentage * audio.duration;
        }});
        
        function playAudio() {{
            selectedStartTime = null;
            selectedEndTime = null;
            audio.play();
        }}
        
        function pauseAudio() {{
            audio.pause();
        }}
        
        function resetAudio() {{
            audio.pause();
            audio.currentTime = 0;
            progressFill.style.width = '0%';
            selectedStartTime = null;
            selectedEndTime = null;
            clearSelection();
        }}
        
        // Play only a specific phoneme segment
        function playPhonemeSegment(start, end) {{
            selectedStartTime = start;
            selectedEndTime = end;
            audio.currentTime = start;
            audio.play();
        }}
        
        // Highlight region on spectrogram
        function highlightSpectrogramRegion(start, end) {{
            const duration = audio.duration || {audio_duration};
            const imgWidth = selectionCanvas.offsetWidth;
            const imgHeight = selectionCanvas.offsetHeight;
            
            const startX = (start / duration) * imgWidth;
            const endX = (end / duration) * imgWidth;
            const width = endX - startX;
            
            ctx.clearRect(0, 0, selectionCanvas.width, selectionCanvas.height);
            ctx.fillStyle = 'rgba(255, 0, 0, 0.25)';
            ctx.fillRect(startX, 0, width, imgHeight);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, 0, width, imgHeight);
        }}
        
        // Play selected region from canvas
        function playSelection() {{
            if (selectedStartTime !== null && selectedEndTime !== null) {{
                audio.currentTime = selectedStartTime;
                audio.play();
            }}
        }}
        
        // Clear selection overlay
        function clearSelection() {{
            ctx.clearRect(0, 0, selectionCanvas.width, selectionCanvas.height);
            selectionStart = null;
            selectionEnd = null;
            selectedStartTime = null;
            selectedEndTime = null;
            playSelectionBtn.style.display = 'none';
        }}
        
        // Mouse selection on spectrogram
        selectionCanvas.addEventListener('mousedown', (e) => {{
            isSelecting = true;
            const rect = selectionCanvas.getBoundingClientRect();
            selectionStart = e.clientX - rect.left;
            selectionEnd = selectionStart;
        }});
        
        selectionCanvas.addEventListener('mousemove', (e) => {{
            if (!isSelecting) return;
            const rect = selectionCanvas.getBoundingClientRect();
            selectionEnd = e.clientX - rect.left;
            
            // Draw selection
            const imgWidth = selectionCanvas.offsetWidth;
            const imgHeight = selectionCanvas.offsetHeight;
            const x = Math.min(selectionStart, selectionEnd);
            const width = Math.abs(selectionEnd - selectionStart);
            
            ctx.clearRect(0, 0, selectionCanvas.width, selectionCanvas.height);
            ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
            ctx.fillRect(x, 0, width, imgHeight);
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, 0, width, imgHeight);
        }});
        
        selectionCanvas.addEventListener('mouseup', (e) => {{
            if (!isSelecting) return;
            isSelecting = false;
            
            const rect = selectionCanvas.getBoundingClientRect();
            selectionEnd = e.clientX - rect.left;
            
            // Convert pixel positions to time
            const duration = audio.duration || {audio_duration};
            const imgWidth = selectionCanvas.offsetWidth;
            const startTime = Math.min(selectionStart, selectionEnd) / imgWidth * duration;
            const endTime = Math.max(selectionStart, selectionEnd) / imgWidth * duration;
            
            if (Math.abs(endTime - startTime) > 0.05) {{
                selectedStartTime = startTime;
                selectedEndTime = endTime;
                playSelectionBtn.style.display = 'inline-block';
                console.log(`Selected region: ${{startTime.toFixed(2)}}s - ${{endTime.toFixed(2)}}s`);
            }} else {{
                clearSelection();
            }}
        }});
        
        selectionCanvas.addEventListener('mouseleave', () => {{
            if (isSelecting) {{
                isSelecting = false;
            }}
        }});
        
        // Stop audio at segment end
        audio.addEventListener('timeupdate', () => {{
            if (selectedEndTime !== null && audio.currentTime >= selectedEndTime) {{
                audio.pause();
                selectedStartTime = null;
                selectedEndTime = null;
            }}
        }});
        
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
            }} else if (e.code === 'Escape') {{
                clearSelection();
            }}
        }});
        
        console.log('Method 1:', intervals1.length, 'phonemes');
        console.log('Method 2:', intervals2.length, 'phonemes');
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Comparison HTML generated: {output_html_path}")
    print(f"📊 Spectrogram image saved: {spectrogram_image_path}")
    print(f"📊 Method 1 ({method1_name}): {len(intervals_data1)} phonemes")
    print(f"📊 Method 2 ({method2_name}): {len(intervals_data2)} phonemes")
    print(f"⏱️  Audio duration: {audio_duration:.2f}s")
    print(f"\n💡 Features:")
    print(f"   - 📊 Vertical stack comparison of two alignment methods")
    print(f"   - 🏷️  Phoneme labels displayed on spectrogram")
    print(f"   - 🎯 Click phoneme to play that segment only")
    print(f"   - 🖱️  Click and drag on spectrogram to select and play region")
    print(f"   - 🔴 Real-time highlighting of current phoneme")
    print(f"   - ⌨️  Keyboard shortcuts: Space (play/pause), ESC (clear selection)")
    print(f"   - 📈 Statistics comparison")


if __name__ == "__main__":
    # Example usage
    from inference import asr_model, hyps, alignments, ctc_alignment_to_timestamps
    
    file_id = "arctic_b0503"
    audio_file = f"examples/{file_id}.wav"
    output_html = f"examples/{file_id}_comparison.html"
    
    # Method 1: CTC Beam Search results
    intervals1 = []
    text_frames = hyps[0][0].text_frames
    text = hyps[0][0].text
    for i in range(len(text)):
        start_frame = text_frames[i]
        end_frame = text_frames[i+1] if i+1 < len(text_frames) else len(alignments)
        start_sec = start_frame * 0.02
        end_sec = end_frame * 0.02
        intervals1.append({
            'start': start_sec,
            'end': end_sec,
            'content': text[i]
        })
    
    # Method 2: K2 Forced Alignment results
    result2 = ctc_alignment_to_timestamps(alignments, asr_model.tokenizer, frame_shift_ms=20)
    intervals2 = [
        {'start': start, 'end': end, 'content': phoneme}
        for start, end, phoneme in result2['timestamps']
    ]
    
    generate_comparison_html(
        audio_file, intervals1, intervals2, output_html,
        method1_name="CTC Prefix Beam Search",
        method2_name="K2 Forced Alignment"
    )
