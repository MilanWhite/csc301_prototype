# CSC301 Prototype

A local browser with live depth visualization using Depth Anything V2.

Requires **Git and Python 3.12**. The default setup uses **Windows and an NVIDIA GPU with its driver installed**. Internet access is needed during setup.

```powershell
git clone https://github.com/MilanWhite/csc301_prototype.git
cd csc301_prototype
py -3.12 setup_project.py
.\.venv\Scripts\python.exe web_app.py
```

Open **http://127.0.0.1:8765/**, enter a website, and toggle **Depth view**.
Setup installs dependencies, Chromium, and model files; afterward, only the last command is needed.

**No NVIDIA GPU?** Replace the setup and launch commands with:

```powershell
py -3.12 setup_project.py --device cpu
.\.venv\Scripts\python.exe web_app.py --device cpu --size 322
```

On macOS/Linux (untested), use `python3.12` for setup and `.venv/bin/python` to launch, with the same CPU flags. Linux may need `.venv/bin/python -m playwright install-deps chromium`.

Tests: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
