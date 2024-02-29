from .base import ContinualTrainer1 as ContinualTrainer
from .fair_trainer import FairContinualTrainer

__all__ = ['ContinualTrainer',
           'FairContinualTrainer',
]