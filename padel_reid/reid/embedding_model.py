"""OSNet-AIN appearance embedder with generic MSMT17 person Re-ID weights."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from padel_reid.reid.osnet_ain_architecture import osnet_ain_x1_0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_generic_weights(model: torch.nn.Module, weights_path: str | Path) -> None:
    """Load MSMT17 person Re-ID weights, skipping the 4101-way classifier."""
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model.load_state_dict(state, strict=False)


def preprocess_crops(crops: list[np.ndarray], input_size: tuple[int, int]) -> torch.Tensor:
    """BGR crops -> normalised (N, 3, H, W) float tensor."""
    height, width = input_size
    batch = np.empty((len(crops), height, width, 3), dtype=np.float32)
    for i, crop in enumerate(crops):
        rgb = cv2.cvtColor(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        batch[i] = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(batch).permute(0, 3, 1, 2)


class ReidEmbedder:
    """Embeds person crops into L2-normalised 512-d vectors."""

    def __init__(self, reid_cfg, device: str = "cuda"):
        self.input_size = tuple(reid_cfg.input_size)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = osnet_ain_x1_0(num_classes=1000, loss="softmax")
        load_generic_weights(self.model, reid_cfg.weights)
        self.model.eval().to(self.device)
        if self.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = preprocess_crops(crops, self.input_size).to(self.device)
        if self.device == "cuda":
            x = x.half()
        features = self.model(x)                 # eval mode returns the 512-d feature
        return F.normalize(features.float(), dim=1).cpu().numpy()
