from .biased_mnist import BiasedMNIST
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .fair_mnist import FairMNIST
from .cifar import CIFAR10, CIFAR100
from .base import SplitDataset2, SplitDataset3

__all__ = ['BiasedMNIST',
           'MNIST',
           'FashionMNIST',
           'FairMNIST',
           'CIFAR10',
           'CIFAR100',
]

fairness_dataset = ['BiasedMNIST',
                    'FairMNIST',
]