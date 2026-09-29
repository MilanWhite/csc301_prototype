"""Download the pinned upstream ONNX weights and verify their SHA-256 hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi

REPO = "onnx-community/depth-anything-v2-small"
REVISION = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
DESTINATION = Path(__file__).resolve().parent / "cuda_storage/depth-anything-v2-small"
FILES = ["README.md", "config.json", "preprocessor_config.json",
         "onnx/model.onnx", "onnx/model_fp16.onnx"]


def main():
    hf = Path(sys.executable).parent / ("hf.exe" if sys.platform == "win32" else "hf")
    subprocess.run([str(hf), "download", REPO, *FILES, "--revision", REVISION,
                    "--local-dir", str(DESTINATION)], check=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    upstream = {entry.rfilename: entry for entry in info.siblings}
    checksums = {}
    for name in FILES:
        path = DESTINATION / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = upstream[name]
        if entry.lfs and digest != entry.lfs.sha256:
            raise RuntimeError(f"Checksum mismatch: {path}")
        if entry.size != path.stat().st_size:
            raise RuntimeError(f"Size mismatch: {path}")
        checksums[name] = {"sha256": digest, "bytes": path.stat().st_size}
    (DESTINATION / "download_manifest.json").write_text(
        json.dumps({"repository": REPO, "revision": REVISION, "files": checksums}, indent=2)
        + "\n", encoding="utf-8")
    print(f"Downloaded and verified model files: {DESTINATION}")


if __name__ == "__main__":
    main()
