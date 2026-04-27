import numpy as np
from typing import List, Dict
from features_parser.parser import BlockStatToken


class FeatureMapGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def generate_maps_for_frame(
        self, tokens: List["BlockStatToken"]
    ) -> Dict[str, np.ndarray]:
        """
        Creates maps for each feature of a frame
        """
        maps = {}

        for token in tokens:
            token.paint(maps, self.width, self.height)

        boundary_map = np.zeros((self.height, self.width), dtype=np.float32)

        for token in tokens:
            if "Depth" in token.param:
                boundary_map[token.y, token.x : token.x + token.w] = 1.0
                boundary_map[token.y : token.y + token.h, token.x] = 1.0

        maps["Boundary"] = boundary_map
        
        # Infer frame type from PredMode: I=0.0, P=0.5, B=1.0
        pred_mode_vals = [t.value for t in tokens if t.param == "PredMode"]
        if pred_mode_vals:
            intra_count = sum(1 for v in pred_mode_vals if v == 1.0)
            inter_count = sum(1 for v in pred_mode_vals if v in [2.0, 3.0])
            total = len(pred_mode_vals)
            intra_ratio = intra_count / total if total > 0 else 0
            
            if intra_ratio > 0.9:
                frame_type = 0.0  # I-frame
            elif inter_count > intra_count:
                frame_type = 1.0  # B-frame
            else:
                frame_type = 0.5  # P-frame
            
            maps["FrameType"] = np.full((self.height, self.width), frame_type, dtype=np.float32)
        
        return maps
