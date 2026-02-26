import torch
import os
import numpy as np
from typing import Tuple, Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pydantic import validate_arguments
from glob import glob
from pathlib import Path
from .config import SubDatasetConfig
from .features_parser.parser import VTMParser
from .features_generator.generator import FeatureMapGenerator


@validate_arguments
@dataclass
class Metadata:
    file: str
    profile: str
    qp: int
    alf: bool
    sao: bool
    db: bool
    frame: int
    is_intra: bool
    height: int
    width: int
    decoder_csv_path: Optional[str] = None  # Path to VTM decoder statistics


@validate_arguments
@dataclass
class Chunk:
    position: Tuple[int, int]
    corner: str
    metadata: Any


def chunk_to_tuple(chunk: Chunk) -> Tuple:
    return (
        chunk.position[0],
        chunk.position[1],
        chunk.corner,
        chunk.metadata.file,
        chunk.metadata.profile,
        chunk.metadata.qp,
        chunk.metadata.alf,
        chunk.metadata.sao,
        chunk.metadata.db,
        chunk.metadata.frame,
        chunk.metadata.is_intra,
        chunk.metadata.height,
        chunk.metadata.width,
    )


class VTMDataset(torch.utils.data.Dataset):
    """
    Enhanced VVC Dataset with VTM decoder statistics integration
    
    Loads compressed chunks along with VTM decoder feature maps
    """

    CHUNK_GLOB = "{folder}/*/*/*.png"
    ORIG_CHUNK_GLOB = "{folder}/*/*/*.png" 
    DECODER_CSV_PATTERN = "{folder}../decoded/{profile}_QP{qp}.csv"

    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform: Any,
        metadata_transform: Any,
        use_vtm_features: bool = True,
    ) -> None:
        super().__init__()

        self.chunk_folder = settings.chunk_folder
        self.orig_chunk_folder = settings.orig_chunk_folder
        self.use_vtm_features = use_vtm_features

        self.chunk_height = settings.chunk_height
        self.chunk_width = settings.chunk_width

        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform

        # Load chunk files
        self.chunk_files = glob(self.CHUNK_GLOB.format(folder=self.chunk_folder))
        self.orig_chunk_files = glob(self.ORIG_CHUNK_GLOB.format(folder=self.orig_chunk_folder))
        
        # Cache for parsed VTM decoder statistics
        self.vtm_cache: Dict[str, Dict] = {}
        
        print(f"Loaded {len(self.chunk_files)} compressed chunks")
        print(f"Loaded {len(self.orig_chunk_files)} original chunks")

    def get_chunk(self, fname: str) -> Chunk:
        """Parse chunk filename to extract metadata"""
        parts = Path(fname).parts
        profile_qp_part = parts[-2]  # e.g., "fast_QP23"
        frame_part = parts[-3]  # e.g., "0_0"
        position_part = Path(fname).stem  # e.g., "0_0_tl"
        
        # Parse profile and QP
        profile_parts = profile_qp_part.split("_")
        profile = profile_parts[0]
        qp = int(profile_parts[1][2:])  # Extract number from "QP23"
        
        # Parse frame info
        frame, is_intra = frame_part.split("_")
        frame_num = int(frame)
        is_intra_flag = is_intra == "1"
        
        # Parse position
        pos_parts = position_part.split("_")
        pos_x, pos_y = int(pos_parts[0]), int(pos_parts[1])
        corner = pos_parts[2] if len(pos_parts) > 2 else ""
        
        # Find decoder CSV path
        decoder_csv_path = self.DECODER_CSV_PATTERN.format(
            folder=self.chunk_folder, profile=profile, qp=qp
        )
        
        metadata = Metadata(
            file=fname,
            profile=profile,
            qp=qp,
            alf=False,  # Will be extracted from filename if needed
            sao=False,
            db=False,
            frame=frame_num,
            is_intra=is_intra_flag,
            height=1080,  # Will be updated from actual image
            width=1920,   # Will be updated from actual image
            decoder_csv_path=decoder_csv_path if os.path.exists(decoder_csv_path) else None
        )
        
        return Chunk(position=(pos_x, pos_y), corner=corner, metadata=metadata)

    def load_vtm_features(self, metadata: Metadata, chunk_x: int, chunk_y: int) -> Optional[Dict[str, np.ndarray]]:
        """Load VTM decoder features for a specific chunk region"""
        if not self.use_vtm_features or not metadata.decoder_csv_path:
            return None
            
        cache_key = metadata.decoder_csv_path
        if cache_key not in self.vtm_cache:
            # Parse the entire CSV file
            parser = VTMParser()
            with open(metadata.decoder_csv_path, 'r') as f:
                tokens_by_frame = parser.parse_file(f)
            
            # Generate feature maps for each frame
            self.vtm_cache[cache_key] = {}
            generator = FeatureMapGenerator(metadata.width, metadata.height)
            
            for frame_num, tokens in tokens_by_frame.items():
                maps = generator.generate_maps_for_frame(tokens)
                self.vtm_cache[cache_key][frame_num] = maps
        
        # Extract features for the chunk region
        frame_features = self.vtm_cache[cache_key].get(metadata.frame)
        if not frame_features:
            return None
            
        chunk_features = {}
        for feature_name, feature_map in frame_features.items():
            # Extract chunk region from full feature map
            chunk_features[feature_name] = feature_map[
                chunk_y:chunk_y + self.chunk_height,
                chunk_x:chunk_x + self.chunk_width
            ]
            
        return chunk_features

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Chunk]:
        chunk_file = self.chunk_files[idx]
        
        # Find corresponding original chunk
        chunk = self.get_chunk(chunk_file)
        orig_chunk_file = chunk_file.replace(self.chunk_folder, self.orig_chunk_folder)
        
        # Load images
        chunk_img = self.chunk_transform(chunk_file)
        orig_chunk_img = self.chunk_transform(orig_chunk_file)
        
        # Load VTM features
        chunk_x, chunk_y = chunk.position
        vtm_features = self.load_vtm_features(chunk.metadata, chunk_x, chunk_y)
        
        # Convert VTM features to tensor
        if vtm_features:
            # Stack all feature maps as additional channels
            feature_channels = []
            for feature_name in sorted(vtm_features.keys()):  # Consistent ordering
                feature_map = vtm_features[feature_name]
                feature_channels.append(torch.from_numpy(feature_map).float())
            
            if feature_channels:
                vtm_tensor = torch.stack(feature_channels, dim=0)
            else:
                vtm_tensor = torch.zeros((1, self.chunk_height, self.chunk_width))
        else:
            vtm_tensor = torch.zeros((1, self.chunk_height, self.chunk_width))
        
        # Combine metadata with VTM features
        metadata_tensor = self.metadata_transform([chunk.metadata.qp, chunk.metadata.is_intra])
        
        return chunk_img, orig_chunk_img, metadata_tensor, chunk, vtm_tensor


class VTMFrameDataset(torch.utils.data.Dataset):
    """
    Dataset for full-frame processing with VTM features
    """
    
    def __init__(
        self,
        settings: SubDatasetConfig,
        chunk_transform: Any,
        metadata_transform: Any,
        use_vtm_features: bool = True,
    ):
        super().__init__()
        
        self.settings = settings
        self.chunk_transform = chunk_transform
        self.metadata_transform = metadata_transform
        self.use_vtm_features = use_vtm_features
        
        # Find all frames
        self.frame_files = glob(f"{settings.chunk_folder}/*/*.png")
        self.frame_files = [f for f in self.frame_files if not any(corner in f for corner in ['_tl', '_tr', '_bl', '_br'])]
        
        print(f"Loaded {len(self.frame_files)} frames")
        
        # Cache for VTM features
        self.vtm_cache: Dict[str, Dict] = {}

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_file = self.frame_files[idx]
        
        # Load frame
        frame = self.chunk_transform(frame_file)
        original_frame = self.chunk_transform(frame_file.replace('_decoded', '_original'))
        
        # Extract metadata from filename
        parts = Path(frame_file).parts
        profile_qp = parts[-2]
        profile, qp_str = profile_qp.split('_')
        qp = int(qp_str[2:])
        
        metadata = [qp, False]  # qp, is_intra
        metadata_tensor = self.metadata_transform(metadata)
        
        return frame, original_frame, metadata_tensor


# Keep backward compatibility
VVCDataset = VTMDataset
FrameDataset = VTMFrameDataset