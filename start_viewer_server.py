#!/usr/bin/env python3
"""
Simple HTTP server to view the interactive HTML files.
Run this on the server and access via browser using port forwarding.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8888
DIRECTORY = Path(__file__).parent / "examples"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        # Add CORS headers to allow local access
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print(f"🚀 服务器已启动！")
        print("=" * 60)
        print(f"\n📂 服务目录: {DIRECTORY}")
        print(f"🌐 端口: {PORT}")
        print(f"\n💡 访问方式：")
        print(f"\n1️⃣  如果在本地服务器（有图形界面）：")
        print(f"   浏览器打开: http://localhost:{PORT}/arctic_b0503_viewer.html")
        print(f"\n2️⃣  如果是远程服务器，需要端口转发：")
        print(f"   在你的本地电脑终端运行：")
        print(f"   ssh -L {PORT}:localhost:{PORT} 你的用户名@服务器地址")
        print(f"   然后在本地浏览器打开: http://localhost:{PORT}/arctic_b0503_viewer.html")
        print(f"\n3️⃣  或者直接下载 HTML 文件到本地电脑打开")
        print(f"   文件路径: {DIRECTORY}/arctic_b0503_viewer.html")
        print(f"\n按 Ctrl+C 停止服务器")
        print("=" * 60)
        print(f"\n⏳ 等待连接...\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")
