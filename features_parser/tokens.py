from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np


@dataclass()
class BlockStatToken:
    poc: int
    x: int
    y: int
    w: int
    h: int
    param: str
    value: Any

    @abstractmethod
    def paint(self, maps: dict, width: int, height: int):
        pass


@dataclass
class ScalarToken(BlockStatToken):
    """Token for Scalar values (QP, Depth)"""

    value: float

    def paint(self, maps: dict, width: int, height: int):
        """
        Paints scalar value to the map, e.g.
        paint value on (2x 2) block starting at (1,1) with  value 9
        [[0, 0, 0, 0],
         [0, 9, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
        """
        if self.param not in maps:
            maps[self.param] = np.zeros((height, width), dtype=np.float32)

        maps[self.param][
            self.y : self.y + self.h, self.x : self.x + self.w
        ] = self.value


class MotionVector(NamedTuple):
    x: float = 0.0
    y: float = 0.0


@dataclass
class VectorToken(BlockStatToken):
    """Token for motion vectors (MV)"""

    value: MotionVector

    def paint(self, maps: dict, width: int, height: int):
        """
        Paints motion vector to the map, e.g.
        paint vector on (2x 2) block starting at (1,1) with  value 9
        [[0, 0, 0, 0],
         [0, 9, 9, 0],
         [0, 9, 9, 0],
         [0, 0, 0, 0]]
        This happens two times for x and y vectors
        """
        name_x, name_y = f"{self.param}_X", f"{self.param}_Y"

        if name_x not in maps:
            maps[name_x] = np.zeros((height, width), dtype=np.float32)
            maps[name_y] = np.zeros((height, width), dtype=np.float32)

        maps[name_x][self.y : self.y + self.h, self.x : self.x + self.w] = self.value.x
        maps[name_y][self.y : self.y + self.h, self.x : self.x + self.w] = self.value.y
