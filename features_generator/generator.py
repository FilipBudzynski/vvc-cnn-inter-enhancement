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
        return maps
