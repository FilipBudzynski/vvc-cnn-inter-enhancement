import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Any, Dict
from torch.utils.data import Dataset

from features_parser.parser import VTMParser
from features_generator.generator import FeatureMapGenerator
import random


class VTMDataset(Dataset):
    def __init__(
        self,
        decoded_yuv_filepath: str,
        original_yuv_filepath: str,
        vtm_trace_path: str,
        patch_size: int = 128,  # Default patch size
    ) -> None:
        super().__init__()
        self.yuv_dec = decoded_yuv_filepath
        self.yuv_orig = original_yuv_filepath
        self.patch_size = patch_size

        self.parser = VTMParser()
        if os.path.exists(vtm_trace_path):
            self.parser.parse_file(vtm_trace_path)
        
        self.height, self.width = self.parser.get_heigth_and_width()
        self.grouped_tokens = self.parser.group_on_poc()
        self.generator = FeatureMapGenerator(self.width, self.height)
        self.pocs = sorted(list(self.grouped_tokens.keys()))

        self.feature_order = [
            "QP", "PredMode", "Depth", 
            "MVL0_X", "MVL0_Y", "MVL1_X", "MVL1_Y"
        ]

    def _normalize_metadata(self, name: str, data: np.ndarray) -> torch.Tensor:
        """Scales metadata to [0, 1] or [-1, 1] for stable training."""
        t = torch.from_numpy(data).float()
        if name == "QP":
            return t / 63.0             # Max VVC QP is 63
        elif name == "Depth":
            return t / 6.0              # Max depth in VVC is 6
        elif name == "PredMode":
            return t                    # Already 0 or 1
        elif "MV" in name:
            return torch.tanh(t / 32.0) # Squish MVs into [-1, 1] range
        return t

    def __len__(self):  
        return len(self.pocs)

    def _read_yuv_frame(self, path: str, poc: int) -> torch.Tensor:
        """
        Reads YUV 4:2:0 frame and upsamples Chroma to match Luma resolution.
        """
        y_size = self.width * self.height
        uv_width, uv_height = self.width // 2, self.height // 2
        uv_size = uv_width * uv_height

        frame_size = int(y_size * 1.5)
        offset = poc * frame_size

        if not os.path.exists(path):
            return torch.zeros((3, self.height, self.width))

        with open(path, "rb") as f:
            f.seek(offset)
            #raw_bytes = f.read(y_size)
            #y_data_flat = np.frombuffer(raw_bytes, dtype=np.uint8)
            # try:
            #     y_data = y_data_flat.reshape((self.height, self.width))
            # except ValueError:
            #     print(f"ERROR: Cannot reshape size {len(y_data_flat)} into ({self.height}, {self.width})")
            
            y_data = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape( (self.height, self.width))
            u_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape( (uv_height, uv_width))
            v_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape( (uv_height, uv_width))

        # Normalization [0.0, 1.0]
        y_tensor = torch.from_numpy(y_data).float() / 255.0
        u_tensor = torch.from_numpy(u_data).float() / 255.0
        v_tensor = torch.from_numpy(v_data).float() / 255.0

        # Upsample U and V from 4:2:0 to 4:4:4 resolution using Bilinear Interpolation
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

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        poc = self.pocs[idx]

        # 1. Load 3-channel Tensors (Y, U_up, V_up)
        decoded_yuv = self._read_yuv_frame(self.yuv_dec, poc)
        original_yuv = self._read_yuv_frame(self.yuv_orig, poc)

        # 2. Generate 7-channel Metadata Maps
        tokens = self.grouped_tokens.get(poc, [])
        maps_dict = self.generator.generate_maps_for_frame(tokens)

        feature_list = []
        for feat in self.feature_order:
            m = maps_dict.get( feat, np.zeros((self.height, self.width), dtype=np.float32))
            feature_list.append(torch.from_numpy(m).float())

        feature_tensor = torch.stack(feature_list, dim=0)  # [7, H, W]

        # 3. Random Crop to 128x128
        height, width = decoded_yuv.shape[1], decoded_yuv.shape[2]
        temp_height, temp_width = min(height, self.patch_size), min(width, self.patch_size)

        top = random.randint(0, height - temp_height)
        left = random.randint(0, width - temp_width)

        decoded_patch = decoded_yuv[:, top:top+temp_height, left:left+temp_width]
        original_patch = original_yuv[:, top:top+temp_height, left:left+temp_width]
        features_patch = feature_tensor[:, top:top+temp_height, left:left+temp_width]
        
        # 4. input assembly
        x = torch.cat([decoded_patch, features_patch], dim=0)
        y = original_patch

        return x, y, {"poc": poc, "top": top, "left": left}
