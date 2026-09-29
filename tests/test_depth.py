import unittest

import numpy as np

from depth_inference import DepthEstimator, colorize
from depth_realtime import LatestFrameCapture


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DepthEstimator(device="cpu", size=322)

    def test_bgr_to_rgb_normalization_and_patch_dimensions(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        pixels = self.model.preprocess(frame)
        self.assertEqual(pixels.shape, (1, 3, 322, 434))
        self.assertTrue(pixels.flags.c_contiguous)
        expected = (np.array([1, 0, 0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np.testing.assert_allclose(pixels[0, :, 100, 100], expected, rtol=1e-6)

    def test_real_onnx_inference_restores_original_resolution(self):
        for shape in ((180, 320, 3), (320, 180, 3)):
            frame = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
            depth = self.model.predict(frame)
            self.assertEqual(depth.shape, shape[:2])
            self.assertEqual(depth.dtype, np.float32)
            self.assertTrue(np.isfinite(depth).all())
            self.assertGreater(float(depth.std()), 0)

    def test_invalid_input_and_size(self):
        with self.assertRaises(ValueError):
            self.model.preprocess(np.zeros((20, 30, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            DepthEstimator(size=320)
        with self.assertRaises(ValueError):
            DepthEstimator(device="cpu", precision="fp16")

    def test_constant_depth_has_valid_visualization(self):
        colored = colorize(np.ones((20, 30), dtype=np.float32))
        self.assertEqual(colored.shape, (20, 30, 3))
        self.assertEqual(colored.dtype, np.uint8)


class CaptureTests(unittest.TestCase):
    def test_only_newest_frame_is_retained_and_capture_is_released(self):
        class FakeCapture:
            count = 0
            released = False

            def read(self):
                self.count += 1
                return self.count <= 10, np.full((2, 2, 3), self.count, dtype=np.uint8)

            def release(self):
                self.released = True

        capture = FakeCapture()
        reader = LatestFrameCapture(capture)
        reader.thread.join(timeout=2)
        self.assertFalse(reader.thread.is_alive())
        sequence, frame, _ = reader.read(0)
        self.assertEqual(sequence, 10)
        self.assertEqual(int(frame[0, 0, 0]), 10)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            reader.read(sequence)
        reader.close()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
