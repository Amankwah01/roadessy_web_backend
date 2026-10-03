"""
DepthAnythingV2 Estimator - Relative Dense Depth Estimation.
Uses Hugging Face Transformers (ONNX-community models) for CPU-friendly inference.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


class DepthAnythingV2Estimator:

    # The official `-hf` repos, not `onnx-community/*`. The onnx-community
    # mirrors publish ONNX graphs only, so from_pretrained finds no
    # pytorch_model.bin or model.safetensors and raises. These carry
    # model.safetensors and are tagged model_type "depth_anything", which is
    # what transformers 5.x expects.
    MODEL_CONFIGS = {
        "vits": {
            "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        },
        "vitb": {
            "model_id": "depth-anything/Depth-Anything-V2-Base-hf",
        },
        "vitl": {
            "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        },
    }

    def __init__(
        self,
        encoder: str = "vits",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        try:
            # transformers 5.x renamed the class: the V2-suffixed name is gone and
            # the V2 checkpoints were re-tagged with model_type "depth_anything".
            # The fallback keeps older pinned versions working, so this is not a
            # reason to downgrade transformers.
            from transformers import (  # type: ignore[import-not-found]
                DepthAnythingForDepthEstimation,  # type: ignore[attr-defined]
                AutoImageProcessor,  # type: ignore[attr-defined]
            )
        except ImportError as new_name_error:
            try:
                from transformers import (  # type: ignore[import-not-found]
                    DepthAnythingV2ForDepthEstimation,  # type: ignore[attr-defined]
                    AutoImageProcessor,  # type: ignore[attr-defined]
                )
            except ImportError:
                # Surface the real reason. Reporting "install transformers" here
                # is what hid this: transformers was installed the whole time, and
                # the pipeline quietly ran with depth estimation switched off.
                raise ImportError(
                    "Could not import the DepthAnythingV2 model class from "
                    f"transformers ({new_name_error}). Either transformers is not "
                    "installed, or the installed version exposes neither "
                    "DepthAnythingForDepthEstimation nor the older "
                    "DepthAnythingV2ForDepthEstimation."
                ) from new_name_error

        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder: {encoder}")

        config = self.MODEL_CONFIGS[encoder]

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.encoder = encoder

        if model_path is None:
            model_path = config["model_id"]

        self.model_id = model_path
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_path,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()

        print(
            f"[DepthAnythingV2] encoder={encoder}, "
            f"device={self.device}"
        )

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided.")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        depth = outputs.predicted_depth
        depth = depth.squeeze().cpu().numpy().astype(np.float32)

        return depth

    @staticmethod
    def bbox_relative_depth(
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0)]

        if valid.size == 0:
            return 0.0

        return float(np.median(valid))

    @staticmethod
    def bbox_relative_dims(
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
        fx: float,
        fy: float,
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0)]

        if valid.size == 0:
            return 0.0, 0.0

        depth_m = float(np.median(valid))
        w_px = x2 - x1
        h_px = y2 - y1
        width_m = w_px * depth_m / fx
        length_m = h_px * depth_m / fy
        return width_m, length_m


def estimate_depth(image: np.ndarray, encoder: str = "vits", device: Optional[str] = None) -> np.ndarray:
    estimator = DepthAnythingV2Estimator(encoder=encoder, device=device)
    return estimator.estimate(image)
