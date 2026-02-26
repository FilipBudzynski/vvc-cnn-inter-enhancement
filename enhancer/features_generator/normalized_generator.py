"""
Normalized Feature Map Generator
===============================

Fixed version that normalizes VTM features to prevent training instability
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict
from features_parser.parser import BlockStatToken


class NormalizedFeatureMapGenerator:
    """Enhanced feature map generator with proper normalization"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Feature normalization parameters based on typical VTM ranges
        self.normalization_params = {
            'QP': {'min': 0, 'max': 51, 'scale': 'linear'},      # QP typically 0-51
            'PredMode': {'min': 0, 'max': 10, 'scale': 'linear'}, # Prediction modes
            'Depth': {'min': 0, 'max': 6, 'scale': 'linear'},      # CU depth levels
            'MVL0_X': {'min': -64, 'max': 64, 'scale': 'tanh'},   # Motion vectors (pixels)
            'MVL0_Y': {'min': -64, 'max': 64, 'scale': 'tanh'},
            'MVL1_X': {'min': -64, 'max': 64, 'scale': 'tanh'},
            'MVL1_Y': {'min': -64, 'max': 64, 'scale': 'tanh'},
        }
    
    def _normalize_feature(self, feature_map: np.ndarray, param_name: str) -> np.ndarray:
        """Normalize feature map to [0, 1] or [-1, 1] range"""
        if param_name not in self.normalization_params:
            print(f"Warning: Unknown feature {param_name}, using min-max normalization")
            # Simple min-max normalization to [0, 1]
            min_val, max_val = feature_map.min(), feature_map.max()
            if max_val > min_val:
                return (feature_map - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(feature_map)
        
        params = self.normalization_params[param_name]
        min_val, max_val = params['min'], params['max']
        scale_type = params['scale']
        
        if scale_type == 'linear':
            # Normalize to [0, 1]
            normalized = np.clip((feature_map - min_val) / (max_val - min_val), 0, 1)
        elif scale_type == 'tanh':
            # Normalize to [-1, 1] using tanh-like scaling
            normalized = np.tanh(feature_map / 32.0)  # 32 is a scaling factor for motion vectors
            normalized = (normalized + 1.0) / 2.0    # Convert to [0, 1]
        else:
            normalized = np.clip((feature_map - min_val) / (max_val - min_val), 0, 1)
        
        return normalized.astype(np.float32)
    
    def generate_maps_for_frame(
        self, tokens: List["BlockStatToken"]
    ) -> Dict[str, np.ndarray]:
        """Creates normalized feature maps for each feature of a frame"""
        # First, generate raw maps using existing token paint logic
        raw_maps = {}
        for token in tokens:
            token.paint(raw_maps, self.width, self.height)
        
        # Then normalize each feature map
        normalized_maps = {}
        for param_name, map_data in raw_maps.items():
            if isinstance(map_data, np.ndarray):
                normalized_maps[param_name] = self._normalize_feature(map_data, param_name)
        
        return normalized_maps
    
    def generate_tensor_features(self, tokens: List["BlockStatToken"]) -> torch.Tensor:
        """Generate normalized feature tensor directly"""
        maps = self.generate_maps_for_frame(tokens)
        
        # Ensure consistent order and convert to tensor
        feature_names = ['QP', 'PredMode', 'Depth', 'MVL0_X', 'MVL0_Y', 'MVL1_X', 'MVL1_Y']
        features = []
        
        for name in feature_names:
            if name in maps:
                features.append(torch.tensor(maps[name], dtype=torch.float32))
            else:
                # If feature is missing, create zeros
                features.append(torch.zeros(self.height, self.width, dtype=torch.float32))
        
        return torch.stack(features)  # [7, H, W]


def create_normalized_generator(width: int, height: int) -> NormalizedFeatureMapGenerator:
    """Factory function to create normalized feature generator"""
    return NormalizedFeatureMapGenerator(width, height)
