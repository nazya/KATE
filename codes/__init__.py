from .utils import DictEnum#, haha_seed

from enum import auto
# from codes import DictEnum


class Loss(DictEnum):
    CrossEntropyLoss = auto()


class Scheduler(DictEnum):
    MultiStepLR = auto()
    CosineAnnealingLR = auto()
    ConstantLR = auto()
    LinearLR = auto()
    # = auto()
