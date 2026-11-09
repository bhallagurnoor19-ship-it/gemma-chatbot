import os
import subprocess

def make_shortcut():
    desktop = os.path.expanduser(r"~\Desktop")
    app_dir = r"C:\Users\Gurnoor\Desktop\gemma-chatbot"
    shortcut_path = os.path.join(desktop, "AI Coding Assistant.lnk")
    icon_path = os.path.join(app_dir, "app_icon.ico")
    pythonw_path = r"C:\Users\Gurnoor\AppData\Local\Programs\Python\Python313\pythonw.exe"
    if not os.path.exists(pythonw_path):
        import shutil
        pythonw_path = shutil.which("pythonw.exe") or "pythonw.exe"

    run_app_py = os.path.join(app_dir, "run_app.py")

    ps_script = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
    $Shortcut.TargetPath = '{pythonw_path}'
    $Shortcut.Arguments = '"{run_app_py}"'
    $Shortcut.WorkingDirectory = '{app_dir}'
    $Shortcut.Description = 'AI Coding Assistant (DeepSeek-Coder-V2 + Qwen2.5-Coder + Gemma 4)'
    $Shortcut.IconLocation = '{icon_path}, 0'
    $Shortcut.Save()
    """

    res = subprocess.run(["powershell", "-Command", ps_script], capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(shortcut_path):
        print(f"Created shortcut successfully: {shortcut_path}")
        print(f"Target: {pythonw_path}")
        print(f"Arguments: {run_app_py}")
    else:
        print("Error creating shortcut:", res.stderr)

if __name__ == "__main__":
    make_shortcut()
