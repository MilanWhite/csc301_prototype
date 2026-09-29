"""Reusable ONNX depth estimator, image inference, and a repeatable benchmark."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import site
import time

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parent / "cuda_storage/depth-anything-v2-small"
_DLL_HANDLES = []


def load_cuda_libraries():
    """Make pip-installed NVIDIA DLLs visible without changing the system PATH."""
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    elif os.name == "nt":
        directories = [path for root in site.getsitepackages()
                       for path in (Path(root) / "nvidia").glob("*/bin")]
        # cuDNN loads its graph/ops DLLs with the legacy Windows loader, which
        # also needs these locations in this process's PATH (not the system PATH).
        os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
        for path in directories:
            _DLL_HANDLES.append(os.add_dll_directory(str(path)))
        # Loading by absolute path also resolves ONNX Runtime's delayed imports.
        for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll",
                     "cufft64_11.dll", "cudnn64_9.dll"):
            for directory in directories:
                path = directory / name
                if path.is_file():
                    _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
                    break


class DepthEstimator:
    """One reusable session. predict() accepts uint8 BGR and returns float32 HxW.

    Output is relative inverse depth (larger values are closer), not meters.
    Instances should be used by one inference thread at a time.
    """

    def __init__(self, model_dir=MODEL_DIR, device="cuda", precision="auto", size=518):
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if precision not in ("auto", "fp16", "fp32"):
            raise ValueError("precision must be auto, fp16, or fp32")
        if size < 14 or size % 14:
            raise ValueError("size must be a positive multiple of 14 (e.g. 322 or 518)")
        self.size = size
        self.device = device
        self.precision = ("fp16" if device == "cuda" else "fp32") if precision == "auto" else precision
        if device == "cpu" and self.precision == "fp16":
            raise ValueError("Use fp32 for CPU inference")
        directory = Path(model_dir)
        filename = "model_fp16.onnx" if self.precision == "fp16" else "model.onnx"
        self.model_path = directory / "onnx" / filename
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing {self.model_path}. Run download_model.py first.")
        config = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.std = np.asarray(config["image_std"], dtype=np.float32)
        self.scale = np.float32(config["rescale_factor"])
        if device == "cuda":
            load_cuda_libraries()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider unavailable. Install requirements.txt, or use --device cpu.")
            providers = [("CUDAExecutionProvider", {"device_id": 0,
                          "cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA initialization failed. Check NVIDIA DLLs/driver or use --device cpu.")
        self.session.disable_fallback()
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0].name
        self.dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32

    def preprocess(self, frame):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Expected a nonempty uint8 BGR image with shape HxWx3")
        height, width = frame.shape[:2]
        # DPT's minimal resize policy: choose the scale closest to 1, then round
        # both dimensions to the ViT patch multiple, preserving aspect ratio.
        sh, sw = self.size / height, self.size / width
        scale = sw if abs(1 - sw) < abs(1 - sh) else sh
        h = max(14, round(height * scale / 14) * 14)
        w = max(14, round(width * scale / 14) * 14)
        rgb = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        pixels = (rgb.astype(np.float32) * self.scale - self.mean) / self.std
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=self.dtype)

    def predict(self, frame):
        pixels = self.preprocess(frame)
        result = self.session.run([self.output], {self.input.name: pixels})[0]
        depth = np.asarray(result[0], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise RuntimeError("Model returned an invalid depth map; try --precision fp32")
        return cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    def describe(self):
        return {"model": str(self.model_path), "precision": self.precision,
                "providers": self.session.get_providers(), "size": self.size}


def colorize(depth):
    low, high = float(depth.min()), float(depth.max())
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def add_model_arguments(parser):
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--size", type=int, default=518, help="DPT resize target, multiple of 14; try 322 for speed")


def estimator_from_args(args):
    return DepthEstimator(args.model_dir, args.device, args.precision, args.size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    image = commands.add_parser("image", help="Save raw float32 depth and a color preview")
    add_model_arguments(image)
    image.add_argument("input", type=Path)
    image.add_argument("--output-dir", type=Path, default=Path("outputs"))
    bench = commands.add_parser("benchmark", help="Measure warm inference including preprocessing and resizing")
    add_model_arguments(bench)
    bench.add_argument("--frames", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--width", type=int, default=640)
    bench.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.command == "benchmark" and (min(args.frames, args.width, args.height) <= 0 or args.warmup < 0):
        parser.error("frames/width/height must be positive; warmup must be nonnegative")
    model = estimator_from_args(args)
    print(json.dumps(model.describe(), indent=2))
    if args.command == "image":
        frame = cv2.imread(str(args.input))
        if frame is None:
            parser.error(f"Cannot read image: {args.input}")
        depth = model.predict(frame)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / f"{args.input.stem}_depth.npy", depth)
        output = args.output_dir / f"{args.input.stem}_depth.png"
        if not cv2.imwrite(str(output), colorize(depth)):
            raise RuntimeError(f"Failed to write {output}")
        print(f"Saved raw depth and preview to {args.output_dir.resolve()}")
    else:
        frame = np.random.default_rng(42).integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
        for _ in range(args.warmup):
            model.predict(frame)
        durations = []
        for _ in range(args.frames):
            start = time.perf_counter()
            model.predict(frame)
            durations.append((time.perf_counter() - start) * 1000)
        print(json.dumps({"frames": args.frames, "input_shape": list(model.preprocess(frame).shape),
                          "mean_ms": float(np.mean(durations)), "p95_ms": float(np.percentile(durations, 95)),
                          "fps": 1000 / float(np.mean(durations))}, indent=2))


if __name__ == "__main__":
    main()
