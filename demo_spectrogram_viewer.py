#!/usr/bin/env python3
"""
快速演示脚本：生成带频谱图的交互式音频-音素可视化

用法:
    python demo_spectrogram_viewer.py <audio_file> <textgrid_file> [output_html]

示例:
    python demo_spectrogram_viewer.py examples/arctic_b0503.wav examples/arctic_b0503.TextGrid
"""

import sys
from pathlib import Path
import tgt
from generate_interactive_html import generate_interactive_html


def create_viewer(audio_file, textgrid_file, output_html=None):
    """
    创建交互式音频-音素查看器
    
    Args:
        audio_file: 音频文件路径 (.wav)
        textgrid_file: TextGrid 文件路径
        output_html: 输出 HTML 文件路径（可选）
    """
    audio_path = Path(audio_file)
    textgrid_path = Path(textgrid_file)
    
    # 验证文件存在
    if not audio_path.exists():
        print(f"❌ 错误: 音频文件不存在: {audio_file}")
        return False
    
    if not textgrid_path.exists():
        print(f"❌ 错误: TextGrid 文件不存在: {textgrid_file}")
        return False
    
    # 默认输出路径
    if output_html is None:
        output_html = audio_path.with_suffix('.viewer.html')
    
    print("=" * 60)
    print("🎨 生成交互式音频-音素可视化")
    print("=" * 60)
    print(f"📂 音频文件: {audio_path}")
    print(f"📂 TextGrid: {textgrid_path}")
    print(f"📂 输出HTML: {output_html}")
    print()
    
    # 读取 TextGrid
    print(f"📖 读取 TextGrid...")
    try:
        tg = tgt.io.read_textgrid(str(textgrid_path))
    except Exception as e:
        print(f"❌ 读取 TextGrid 失败: {e}")
        return False
    
    # 提取音素区间
    intervals_data = []
    tier_count = 0
    for tier in tg.tiers:
        if isinstance(tier, tgt.IntervalTier):
            tier_count += 1
            print(f"   发现层级: {tier.name} ({len(tier.intervals)} 个区间)")
            
            # 只使用第一个 IntervalTier
            if tier_count == 1:
                for interval in tier.intervals:
                    if interval.text:  # 跳过空白区间
                        intervals_data.append({
                            "start": round(interval.start_time, 4),
                            "end": round(interval.end_time, 4),
                            "content": interval.text,
                        })
    
    if not intervals_data:
        print(f"❌ 错误: TextGrid 中没有找到有效的音素区间")
        return False
    
    print(f"✅ 提取了 {len(intervals_data)} 个音素区间")
    print()
    
    # 生成 HTML
    try:
        generate_interactive_html(str(audio_path), intervals_data, str(output_html))
        print()
        print("=" * 60)
        print("✅ 生成完成！")
        print("=" * 60)
        print(f"\n📂 文件位置:")
        print(f"   HTML: {Path(output_html).absolute()}")
        print(f"   频谱图: {Path(output_html).with_suffix('.spectrogram.png').absolute()}")
        print(f"\n🌐 查看方式:")
        print(f"   1. 下载 HTML 到本地电脑双击打开")
        print(f"   2. 或使用 SSH 端口转发 + 启动 Web 服务器")
        print(f"      python start_viewer_server.py")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 生成 HTML 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n❌ 错误: 需要提供音频文件和 TextGrid 文件路径")
        print("\n示例:")
        print("  python demo_spectrogram_viewer.py examples/arctic_b0503.wav examples/arctic_b0503.TextGrid")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    textgrid_file = sys.argv[2]
    output_html = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = create_viewer(audio_file, textgrid_file, output_html)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
