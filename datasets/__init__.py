from .biased_mnist import BiasedMNIST
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .fair_mnist import FairMNIST
from .drug import Drug
from .base import SplitDataset1, SplitDataset3

__all__ = ['BiasedMNIST',
           'MNIST',
           'FashionMNIST',
           'FairMNIST',
           'Drug',
]

fairness_datasets = ['BiasedMNIST',
                     'FairMNIST',
                     'Drug',
]