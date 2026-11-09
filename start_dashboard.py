import sys
import os
import subprocess
import time
import socket
import traceback

def is_port_in_use(port=8501):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
    except Exception:
        pass
    try:
        res = subprocess.run(["lsof", "-i", f":{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return "LISTEN" in res.stdout
    except Exception:
        return False

def start():
    port = 8501
    
    # Check if already running
    if is_port_in_use(port):
        print(f"✅ Streamlit is ALREADY RUNNING on port {port}!")
        print(f"👉 You can open Firefox directly to: http://localhost:{port}")
        return

    # Find app_dashboard.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "app_dashboard.py"),
        os.path.join(base_dir, "task_1w8k865d_rag_doubt_resolution", "app_dashboard.py"),
        "app_dashboard.py",
        "task_1w8k865d_rag_doubt_resolution/app_dashboard.py"
    ]
    script_path = None
    for c in candidates:
        if os.path.exists(c):
            script_path = os.path.abspath(c)
            break
            
    if not script_path:
        print("❌ Cannot find 'app_dashboard.py'. Make sure it exists in the directory.")
        return

    # Log file to capture everything
    log_file = os.path.join(os.path.dirname(script_path), "streamlit.log")
    
    print(f"📁 Target script: {script_path}")
    print(f"📝 Logging to: {log_file}")
    print(f"🐍 Python interpreter: {sys.executable}")
    print(f"📌 Python version: {sys.version.split()[0]}")

    # Check streamlit import with detailed traceback
    try:
        import streamlit
        print(f"📦 Found streamlit version: {streamlit.__version__}")
        print(f"📍 Streamlit installed at: {streamlit.__file__}")
    except Exception as e:
        print(f"❌ Error when importing streamlit: {type(e).__name__}: {e}")
        print("\n--- Full Import Traceback ---")
        traceback.print_exc()
        print("-----------------------------\n")
        print("👉 Try running: %pip install --upgrade streamlit typing_extensions")
        return

    # Start detached/background process with output to log file
    with open(log_file, "w") as out:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", script_path,
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(script_path)
        )

    print(f"🔄 Launching server (PID {proc.pid})...")
    time.sleep(3)

    # Check if process is still alive
    poll = proc.poll()
    if poll is not None:
        print(f"❌ Server process exited prematurely with code {poll}!")
        print("--- Log Output (streamlit.log) ---")
        if os.path.exists(log_file):
            with open(log_file) as f:
                print(f.read())
        print("---------------------------------")
    else:
        print("=" * 60)
        print("🚀 STREAMLIT SERVER IS NOW RUNNING!")
        print(f"👉 Open Firefox and go to: http://localhost:{port}")
        print("=" * 60)

if __name__ == "__main__":
    start()
