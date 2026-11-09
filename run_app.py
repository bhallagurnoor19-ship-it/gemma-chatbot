"""
Standalone Desktop Launcher for AI Coding Assistant & Chatbot
- Ensures Ollama is active
- Starts Flask server in background
- Opens standalone borderless App Window via Microsoft Edge App Mode
- Shuts down background server cleanly upon window exit
"""

import os
import sys
import time
import shutil
import logging
import ctypes
import subprocess
import urllib.request
import atexit

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 5000
SERVER_URL = f"http://localhost:{PORT}"
OLLAMA_URL = "http://localhost:11434/api/tags"

LAUNCHER_LOG = os.path.join(APP_DIR, "launcher.log")
SERVER_LOG = os.path.join(APP_DIR, "server.log")

logging.basicConfig(
    filename=LAUNCHER_LOG,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger("Launcher")

flask_proc = None
ollama_proc = None

def show_error(title, message):
    logger.error(f"{title}: {message}")
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10 | 0x10000)
        except Exception:
            pass

def is_ollama_running():
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def ensure_ollama():
    if is_ollama_running():
        logger.info("Ollama server is active and reachable.")
        return True
    
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        standard_path = os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(standard_path):
            ollama_bin = standard_path

    if ollama_bin:
        logger.info(f"Starting Ollama daemon from {ollama_bin}...")
        global ollama_proc
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            ollama_proc = subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
            for _ in range(15):
                time.sleep(1)
                if is_ollama_running():
                    logger.info("Ollama started successfully.")
                    return True
        except Exception as e:
            logger.warning(f"Could not auto-start Ollama: {e}")
    else:
        logger.warning("Ollama binary not found in PATH or standard locations.")
    return False

def is_server_ready():
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def get_python_executables():
    """Find python and pythonw executables."""
    base_exe = sys.executable
    exe_dir = os.path.dirname(base_exe)
    
    pythonw = os.path.join(exe_dir, "pythonw.exe")
    python = os.path.join(exe_dir, "python.exe")
    
    if not os.path.exists(pythonw):
        pythonw = shutil.which("pythonw") or base_exe
    if not os.path.exists(python):
        python = shutil.which("python") or base_exe
        
    return python, pythonw

def start_flask():
    global flask_proc
    if is_server_ready():
        logger.info("Flask server is already running on port 5000.")
        return True

    app_script = os.path.join(APP_DIR, "app.py")
    if not os.path.exists(app_script):
        show_error("Startup Error", f"Cannot find app.py at:\n{app_script}")
        return False

    python_bin, pythonw_bin = get_python_executables()
    launch_py = pythonw_bin if sys.platform == "win32" else python_bin

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    logger.info(f"Starting Flask backend with {launch_py}...")
    
    try:
        server_log_file = open(SERVER_LOG, "a", encoding="utf-8")
        server_log_file.write(f"\n--- Starting Server [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
        server_log_file.flush()

        flask_proc = subprocess.Popen(
            [launch_py, app_script],
            cwd=APP_DIR,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags
        )
    except Exception as e:
        show_error("Flask Startup Error", f"Failed to spawn Flask backend:\n{e}")
        return False

    # Poll until server is ready (up to 20 seconds)
    for _ in range(40):
        time.sleep(0.5)
        if is_server_ready():
            logger.info("Flask server is ready and responding 200 OK.")
            return True
        if flask_proc.poll() is not None:
            logger.error(f"Flask process exited prematurely with code {flask_proc.returncode}")
            break

    # Read error details from server.log
    err_snippet = ""
    try:
        if os.path.exists(SERVER_LOG):
            with open(SERVER_LOG, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                err_snippet = "".join(lines[-15:])
    except Exception:
        pass

    show_error(
        "AI Assistant - Server Error",
        f"Backend server failed to start on http://localhost:{PORT}.\n\n"
        f"Details from server.log:\n{err_snippet or 'No output in log.'}"
    )
    return False

def cleanup():
    global flask_proc
    if flask_proc and flask_proc.poll() is None:
        logger.info("Shutting down background Flask server...")
        try:
            flask_proc.terminate()
            flask_proc.wait(timeout=3)
        except Exception:
            try:
                flask_proc.kill()
            except Exception:
                pass
        logger.info("Flask server stopped.")

atexit.register(cleanup)

def find_browser():
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("msedge"),
    ]
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        shutil.which("chrome"),
    ]
    brave_paths = [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]

    for p in edge_paths + chrome_paths + brave_paths:
        if p and os.path.exists(p):
            return p
    return None

def launch_app_window():
    browser_bin = find_browser()
    profile_dir = os.path.join(APP_DIR, ".app_profile")
    os.makedirs(profile_dir, exist_ok=True)

    if browser_bin:
        logger.info(f"Launching app window with {browser_bin}...")
        cmd = [
            browser_bin,
            f"--app={SERVER_URL}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1220,900",
            "--use-fake-ui-for-media-stream",
            "--enable-features=AudioServiceOutOfProcess",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check"
        ]
        t0 = time.time()
        browser_proc = subprocess.Popen(cmd)
        browser_proc.wait()
        elapsed = time.time() - t0
        logger.info(f"App window closed after {elapsed:.1f}s.")
        
        # If Edge exited in < 1.5s (process delegation to existing window), do not kill Flask immediately
        if elapsed < 1.5:
            logger.info("Window exited rapidly (browser delegation detected). Keeping server active.")
            try:
                while is_server_ready():
                    time.sleep(3)
            except KeyboardInterrupt:
                pass
    else:
        logger.info("No Chromium browser detected. Falling back to default browser.")
        import webbrowser
        webbrowser.open(SERVER_URL)
        try:
            while is_server_ready():
                time.sleep(3)
        except KeyboardInterrupt:
            pass

def main():
    logger.info("=== Launching AI Assistant Desktop App ===")
    try:
        ensure_ollama()
        if not start_flask():
            logger.error("start_flask failed. Exiting.")
            sys.exit(1)
        launch_app_window()
    except Exception as e:
        logger.exception(f"Unhandled exception during launch: {e}")
        show_error("Launch Failure", f"An unexpected error occurred:\n{e}")
    finally:
        cleanup()
        logger.info("=== Application Finished ===")

if __name__ == "__main__":
    main()
