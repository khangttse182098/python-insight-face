from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, List
import numpy as np


class Pose(Enum):
    FRONT = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


@dataclass
class FaceRecord:
    id: int
    code: str
    pose: Pose
    vector: np.ndarray
