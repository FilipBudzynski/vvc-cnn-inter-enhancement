import os
import re
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from features_parser.parser import VTMParser
from features_generator.generator import FeatureMapGenerator


class VTMDataset(Dataset):
    """VTM Dataset with SOTA metadata handling."""

    def __init__(
        self,
        decoded_yuv_filepath: str,
        original_yuv_filepath: str,
        vtm_trace_path: str,
        patch_size: int = 132,
    ) -> None:
        super().__init__()
        self.yuv_dec = decoded_yuv_filepath
        self.yuv_orig = original_yuv_filepath
        self.vtm_trace_path = vtm_trace_path
        self.patch_size = patch_size

        # 1. Parse dimensions from trace file
        self.width, self.height = 0, 0
        with open(vtm_trace_path, "r") as f:
            for i, line in enumerate(f):
                if "# Sequence size:" in line:
                    dims = re.findall(r"\d+", line)
                    if len(dims) >= 2:
                        self.width, self.height = int(dims[0]), int(dims[1])
                        break
                if i > 50:
                    break

        if self.width == 0 or self.height == 0:
            raise ValueError(f"Failed to parse dimensions for {vtm_trace_path}")

        # 2. FAST POC SCAN
        with open(vtm_trace_path, "r") as f:
            content = f.read()
            found_pocs = re.findall(r"POC (\d+)", content)
            if found_pocs:
                self.pocs = sorted(list(set(int(p) for p in found_pocs)))
            else:
                self.pocs = []

        # Initialize Lazy Loading Placeholders
        self.parser = VTMParser()
        self.grouped_tokens = None
        self.generator = None

        self.feature_order = [
            "QP",
            "PredMode",
            "Depth",
            "Boundary",
            "MVL0_X",
            "MVL0_Y",
            "MVL1_X",
            "MVL1_Y",
        ]

    def _lazy_load(self):
        if self.grouped_tokens is None:
            cache_path = Path(self.vtm_trace_path).with_suffix(".pkl")

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    self.grouped_tokens = pickle.load(f)
            else:
                self.grouped_tokens = self.parser.parse_file(self.vtm_trace_path)
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.grouped_tokens, f)
                except Exception as e:
                    print(f"⚠️ Could not save cache: {e}")

            if self.generator is None:
                self.generator = FeatureMapGenerator(self.width, self.height)

    def _normalize_metadata(self, name: str, data: np.ndarray) -> torch.Tensor:
        """Normalize metadata to [0, 1] range."""
        t = torch.from_numpy(data).float()

        if name == "QP":
            return (t / 63.0).clamp(0, 1)
        elif name == "Depth":
            return (t / 7.0).clamp(0, 1)
        elif name == "PredMode":
            return (t / 3.0).clamp(0, 1)
        elif name == "Boundary":
            return (t / 1.0).clamp(0, 1)
        elif "MV" in name:
            return (t / 128.0).clamp(-1, 1)

        return t

    def __len__(self):
        return len(self.pocs)

    def _read_yuv_frame(self, path: str, poc: int) -> torch.Tensor:
        """Reads YUV 4:2:0 frame."""
        y_size = self.width * self.height
        uv_width, uv_height = self.width // 2, self.height // 2
        uv_size = uv_width * uv_height

        with open(path, "rb") as f:
            f.seek(poc * (y_size + 2 * uv_size))
            y_data = np.frombuffer(f.read(y_size), dtype=np.uint8).copy()
            u_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).copy()
            v_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).copy()

        y_tensor = (
            torch.from_numpy(y_data.reshape(self.height, self.width))
            .float()
            .div(255.0)
        )
        u_tensor = (
            torch.from_numpy(u_data.reshape(uv_height, uv_width)).float().div(255.0)
        )
        v_tensor = (
            torch.from_numpy(v_data.reshape(uv_height, uv_width)).float().div(255.0)
        )

        u_up = F.interpolate(
            u_tensor.unsqueeze(0).unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        v_up = F.interpolate(
            v_tensor.unsqueeze(0).unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )

        return torch.cat(
            [y_tensor.unsqueeze(0), u_up.squeeze(0), v_up.squeeze(0)], dim=0
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self._lazy_load()
        poc = self.pocs[idx]

        decoded_yuv = self._read_yuv_frame(self.yuv_dec, poc)
        original_yuv = self._read_yuv_frame(self.yuv_orig, poc)

        tokens = self.grouped_tokens.get(poc, [])
        maps_dict = self.generator.generate_maps_for_frame(tokens)

        _, curr_h, curr_w = decoded_yuv.shape

        feature_list = []
        for feat in self.feature_order:
            m_raw = maps_dict.get(feat, np.zeros((curr_h, curr_w), dtype=np.float32))
            m_norm = self._normalize_metadata(feat, m_raw)
            feature_list.append(
                m_norm if torch.is_tensor(m_norm) else torch.from_numpy(m_norm)
            )

        feature_tensor = torch.stack(feature_list, dim=0)

        th, tw = self.patch_size, self.patch_size
        h, w = decoded_yuv.shape[1], decoded_yuv.shape[2]

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_h > 0 or pad_w > 0:
            decoded_yuv = F.pad(decoded_yuv, (0, pad_w, 0, pad_h))
            original_yuv = F.pad(original_yuv, (0, pad_w, 0, pad_h))
            feature_tensor = F.pad(feature_tensor, (0, pad_w, 0, pad_h))
            h, w = decoded_yuv.shape[1], decoded_yuv.shape[2]

        max_y = h - th
        max_x = w - tw

        top_crop = random.choice(range(0, max_y + 1, 8)) if max_y > 0 else 0
        left_crop = random.choice(range(0, max_x + 1, 8)) if max_x > 0 else 0

        decoded_patch = decoded_yuv[
            :, top_crop : top_crop + th, left_crop : left_crop + tw
        ]
        original_patch = original_yuv[
            :, top_crop : top_crop + th, left_crop : left_crop + tw
        ]
        features_patch = feature_tensor[
            :, top_crop : top_crop + th, left_crop : left_crop + tw
        ]

        # Return: (yuv, original, metadata, info)
        return decoded_patch, original_patch, features_patch, {"poc": poc}
