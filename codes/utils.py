import os
import random as random
import numpy as np
import torch
# from torch import manual_seed
from enum import Enum


# def seed(seed):
def fix_seed(seed, env=True):
    # print(f"{os.environ['PYTHONHASHSEED']=}")
    if env:
        os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# class FrozenDict(dict):
#     def __init__(self, *args, **kwargs):
#         self._hash = None
#         super(FrozenDict, self).__init__(*args, **kwargs)

#     def __hash__(self):
#         if self._hash is None:
#             self._hash = hash(tuple(sorted(self.items())))  # iteritems() on py2
#         return self._hash

#     def _immutable(self, *args, **kws):
#         raise TypeError('cannot change object - object is immutable')

#     # makes (deep)copy alot more efficient
#     def __copy__(self):
#         return self

#     def __deepcopy__(self, memo=None):
#         if memo is not None:
#             memo[id(self)] = self
#         return self

#     __setitem__ = _immutable
#     __delitem__ = _immutable
#     pop = _immutable
#     popitem = _immutable
#     clear = _immutable
#     update = _immutable
#     setdefault = _immutable


class DictEnum(dict, Enum):
    def _generate_next_value_(name, start, count, last_values):
        # return name
        return {'name': name, 'id': count}
        # return FrozenDict({'name': name, 'id': count})
        # return "{'%s': %i}" % (name, count)

    # def __class__(self):
    #     return str

    def __repr__(self):
        # print(type(self._value_))
        return self._name_

    def __str__(self):
        # print(type(self._name_))
        return self._name_

    # def __getstate__(self):
    #     return self._name_