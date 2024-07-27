from enum import Enum


class Task(Enum):
    IDLE = -1
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2
