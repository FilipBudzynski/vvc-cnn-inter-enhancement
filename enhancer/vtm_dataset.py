import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Any, Dict
from torch.utils.data import Dataset

from features_parser.parser import VTMParser
from features_generator.generator import FeatureMapGenerator


class VTMDataset(Dataset):
    """
    VTM Dataset for Full YUV (10 channels: 3 Pixels + 7 Metadata).
    NOTE: This implementation assumes YUV 4:2:0 input.
    If using 4:4:4 or 4:2:2, the chroma upsampling logic must be modified.
    """


    def __init__(
        self,
        decoded_yuv_filepath: str,
        original_yuv_filepath: str,
        vtm_trace_path: str,
        width: int,
        height: int,
        chunk_transform: Any = None,
    ) -> None:
        if not decoded_yuv_filepath:
            raise ValueError("decoded_yuv_filepath cannot be empty. Check config.yaml")
        super().__init__()
        self.yuv_dec = decoded_yuv_filepath
        self.yuv_orig = original_yuv_filepath
        self.width = width
        self.height = height
        self.chunk_transform = chunk_transform

        self.parser = VTMParser()
        if os.path.exists(vtm_trace_path):
            self.parser.parse_file(vtm_trace_path)
        self.grouped_tokens = self.parser.group_on_poc()
        self.generator = FeatureMapGenerator(width, height)

        self.pocs = sorted(list(self.grouped_tokens.keys()))
        if not self.pocs:
            file_size = os.path.getsize(decoded_yuv_filepath)
            # Assuming 4:2:0 (1.5 bytes per pixel)
            frame_count = int(file_size / (width * height * 1.5))
            self.pocs = list(range(frame_count))

        self.feature_order = [
            "QP",
            "PredMode",
            "Depth",
            "MVL0_X",
            "MVL0_Y",
            "MVL1_X",
            "MVL1_Y",
        ]

    def __len__(self):  
        return len(self.pocs)

    def _read_yuv_frame(self, path: str, poc: int) -> torch.Tensor:
        """
        Reads YUV 4:2:0 frame and upsamples Chroma to match Luma resolution.
        """
        y_size = self.width * self.height
        # 4:2:0 specific: Chroma is half width and half height
        uv_width, uv_height = self.width // 2, self.height // 2
        uv_size = uv_width * uv_height

        frame_size = int(y_size * 1.5)
        offset = poc * frame_size

        if not os.path.exists(path):
            return torch.zeros((3, self.height, self.width))

        with open(path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(y_size)
            y_data_flat = np.frombuffer(raw_bytes, dtype=np.uint8)
            print(f"First 10 pixels: {y_data_flat[:10]}")

            try:
                y_data = y_data_flat.reshape((self.height, self.width))
            except ValueError:
                print(f"ERROR: Cannot reshape size {len(y_data_flat)} into ({self.height}, {self.width})")
            # y_data = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape( (self.height, self.width))
            u_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(
                (uv_height, uv_width)
            )
            v_data = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(
                (uv_height, uv_width)
            )

        # Normalization [0.0, 1.0]
        y_tensor = torch.from_numpy(y_data).float() / 255.0
        u_tensor = torch.from_numpy(u_data).float() / 255.0
        v_tensor = torch.from_numpy(v_data).float() / 255.0

        # Upsample U and V from 4:2:0 to 4:4:4 resolution using Bilinear Interpolation
        # .unsqueeze(0).unsqueeze(0) converts [H, W] to [Batch, Channel, H, W] for F.interpolate
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
        dec_yuv = self._read_yuv_frame(self.yuv_dec, poc)
        orig_yuv = self._read_yuv_frame(self.yuv_orig, poc)

        # 2. Generate 7-channel Metadata Maps
        tokens = self.grouped_tokens.get(poc, [])
        maps_dict = self.generator.generate_maps_for_frame(tokens)

        feature_list = []
        for feat in self.feature_order:
            m = maps_dict.get(
                feat, np.zeros((self.height, self.width), dtype=np.float32)
            )
            feature_list.append(torch.from_numpy(m).float())

        feature_tensor = torch.stack(feature_list, dim=0)  # [7, H, W]

        # 3. Handle Crops/Augmentations
        if self.chunk_transform:
            # We stack everything to ensure the same RandomCrop is applied to all
            # Total 13 channels: 3 (dec) + 3 (orig) + 7 (meta)
            combined = torch.cat([dec_yuv, orig_yuv, feature_tensor], dim=0)
            combined = self.chunk_transform(combined)

            # Unpack after transformation
            dec_yuv = combined[0:3, :, :]
            orig_yuv = combined[3:6, :, :]
            feature_tensor = combined[6:13, :, :]

        return (dec_yuv, orig_yuv, feature_tensor, {"poc": poc})
