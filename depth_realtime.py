"""Run depth estimation on a webcam, video file, or live stream. Q/Esc exits."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from depth_inference import add_model_arguments, colorize, estimator_from_args


class LatestFrameCapture:
    """Continuously drain a live source into a one-frame buffer.

    The capture thread owns and releases the OpenCV capture. Consumers wait for
    a new sequence number so they never infer on the same frame twice.
    """

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.done = False
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                with self.condition:
                    self.frame = frame
                    self.timestamp = time.perf_counter()
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            self.capture.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self, previous_sequence, timeout=10):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > previous_sequence or self.done, timeout=timeout)
            if not ready:
                raise TimeoutError("No new frame from live source for 10 seconds")
            if self.error:
                raise RuntimeError("Live capture failed") from self.error
            if self.sequence <= previous_sequence:
                raise RuntimeError("Live source stopped delivering frames")
            return self.sequence, self.frame, self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL")
    parser.add_argument("--live", action="store_true", help="Use newest-frame capture for a stream URL")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true", help="Disable preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N predictions; 0 means unlimited")
    parser.add_argument("--output", type=Path, help="Write a side-by-side MP4 at source FPS")
    args = parser.parse_args()
    if min(args.width, args.height, args.camera_fps) <= 0 or args.max_frames < 0:
        parser.error("Camera dimensions/FPS must be positive; max-frames must be nonnegative")

    model = estimator_from_args(args)
    print(json.dumps(model.describe(), indent=2), flush=True)
    # Initialize GPU kernels before opening the camera.
    model.predict(np.zeros((args.height, args.width, 3), dtype=np.uint8))
    source = int(args.source) if args.source.isdecimal() else args.source
    is_live = isinstance(source, int) or args.live
    if args.output and isinstance(source, str) and "://" not in source:
        if args.output.resolve() == Path(source).resolve():
            parser.error("Output must differ from the input video")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open source: {args.source}")
    reader = None
    writer = None
    count = 0
    sequence = 0
    dropped = 0
    frame_age_ms = 0.0
    durations = []
    output_size = None
    start = time.perf_counter()
    try:
        if isinstance(source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = args.camera_fps
        if is_live:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            reader = LatestFrameCapture(capture)
        while not args.max_frames or count < args.max_frames:
            if reader:
                next_sequence, frame, captured_at = reader.read(sequence)
                dropped += next_sequence - sequence - 1
                sequence = next_sequence
            else:
                ok, frame = capture.read()
                if not ok:
                    if not count:
                        raise RuntimeError("Source contains no readable frames")
                    break
                captured_at = time.perf_counter()
            tick = time.perf_counter()
            depth = model.predict(frame)
            elapsed = time.perf_counter() - tick
            durations.append(elapsed)
            # Keep memory bounded even during a long-running camera session.
            if len(durations) > 120:
                durations.pop(0)
            count += 1
            frame_age_ms = (time.perf_counter() - captured_at) * 1000
            if not args.headless or args.output:
                preview = np.hstack((frame, colorize(depth)))
                label = f"{model.device.upper()} {1 / np.mean(durations):.1f} inference FPS | {frame_age_ms:.0f} ms frame age"
                cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if args.output:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        output_size = (preview.shape[1], preview.shape[0])
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {args.output}")
                    if (preview.shape[1], preview.shape[0]) != output_size:
                        preview = cv2.resize(preview, output_size)
                    writer.write(preview)
                if not args.headless:
                    cv2.imshow("Depth Anything V2 - Q/Esc to exit", preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty("Depth Anything V2 - Q/Esc to exit", cv2.WND_PROP_VISIBLE) < 1:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.close()
        else:
            capture.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({"frames": count, "processed_fps": count / max(time.perf_counter() - start, 1e-8),
                      "dropped_capture_frames": dropped, "last_frame_age_ms": frame_age_ms}, indent=2))


if __name__ == "__main__":
    main()
