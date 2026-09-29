"""Create a local environment, install Chromium, and download the pinned model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def run(command, environment):
    print("\n> " + subprocess.list2cmdline([str(part) for part in command]), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if os.name == "nt" else "cpu")
    parser.add_argument("--venv", type=Path, default=Path(".venv"),
                        help="Environment directory relative to the project")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Run setup using Python 3.12: py -3.12 setup_project.py (Windows)")
    if args.device == "cuda" and os.name != "nt":
        parser.error("The bundled CUDA setup targets Windows. Use --device cpu on this platform.")
    destination = (ROOT / args.venv).resolve()
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(ROOT / ".cache/uv")
    environment["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / ".cache/ms-playwright")
    uv = shutil.which("uv")
    if python.is_file():
        installed = json.loads(subprocess.check_output([
            str(python), "-c", "import importlib.metadata as m,json; print(json.dumps([d.metadata['Name'].lower() for d in m.distributions()]))"
        ], text=True))
        incompatible = "onnxruntime" if args.device == "cuda" else "onnxruntime-gpu"
        if incompatible in installed:
            parser.error(f"{destination} already contains {incompatible}. Use --venv .venv-{args.device} to keep environments separate.")
    elif uv:
        run([uv, "venv", destination, "--python", sys.executable], environment)
    else:
        print(f"Creating {destination}", flush=True)
        venv.EnvBuilder(with_pip=True).create(destination)
    requirements = "requirements-web-lock.txt" if args.device == "cuda" else "requirements-web-cpu.txt"
    if uv:
        run([uv, "pip", "install", "--python", python, "-r", requirements], environment)
    else:
        # Handles an existing uv-created environment without pip as well.
        run([python, "-m", "ensurepip", "--upgrade"], environment)
        run([python, "-m", "pip", "install", "-r", requirements], environment)
    run([python, "-m", "playwright", "install", "chromium"], environment)
    run([python, ROOT / "download_model.py"], environment)
    run([python, ROOT / "depth_inference.py", "benchmark", "--device", args.device,
         "--size", "322", "--frames", "1", "--warmup", "1"], environment)
    print("\nSetup complete. Start the application with:")
    print(subprocess.list2cmdline([str(python), str(ROOT / "web_app.py"), "--device", args.device]))
    print("Then open http://127.0.0.1:8765/")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nSetup stopped because a command failed (exit {exc.returncode}). Fix the reported issue and rerun setup.", file=sys.stderr)
        sys.exit(exc.returncode)
