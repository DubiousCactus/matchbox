from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Task(Enum):
    IDLE = -1
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


@dataclass
class Plot_BestModel:
    """Dataclass for representing a best model mark in the plotter widget."""

    epoch: int
    loss: float
    metrics: Dict[str, float]
