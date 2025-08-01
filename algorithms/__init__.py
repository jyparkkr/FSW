from .base import Heuristic
from .imbalance import Heuristic2 as Weighting
from .sensitive import Heuristic3


__all__ = ['Heuristic',
           'Weighting',
           'Heuristic3',
          ]