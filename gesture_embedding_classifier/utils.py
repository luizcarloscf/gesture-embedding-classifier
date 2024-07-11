from typing import Callable, Iterable, Union, Type

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import Dataset


DatasetType = Type[Dataset]
DatasetCallable = Callable[[Iterable], Dataset]

ModuleType = Type[nn.Module]
ModuleCallable = Callable[[Iterable], nn.Module]

OptimizerType = Type[Optimizer]
OptimizerCallable = Callable[[Iterable], Optimizer]

LRSchedulerType = Type[Union[LRScheduler, ReduceLROnPlateau]]
LRSchedulerCallable = Callable[[Optimizer], Union[LRScheduler, ReduceLROnPlateau]]
