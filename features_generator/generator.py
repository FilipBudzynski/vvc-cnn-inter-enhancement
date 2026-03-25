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

        return maps
