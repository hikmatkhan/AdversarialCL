
# Ready to use continual learning benchmarks
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, CelebA, TinyImagenet, CUB200, OpenLORIS

# Models
import avalanche
from avalanche.models import SimpleCNN
from avalanche.models import SimpleMLP
from avalanche.models import MTSimpleMLP
from avalanche.models import SimpleMLP_TinyImageNet
from avalanche.models import MobilenetV1

# Build-in strategies
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb, LwF, GEM, AGEM, EWC

# Base strategy classes
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from typing import Optional, Sequence, Union
from avalanche.benchmarks import Experience, nc_benchmark, ni_benchmark, tensors_benchmark

# Logger
from avalanche.logging import InteractiveLogger

# Evaluation
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import StreamAccuracy
from avalanche.evaluation.plot_utils import learning_curves_plot

from avalanche.training.strategies import BaseStrategy
from typing import Optional, Sequence, Union
from avalanche.benchmarks import Experience

from avalanche.training.plugins import ReplayPlugin, EWCPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.training.plugins import EvaluationPlugin

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
from avalanche.logging import InteractiveLogger

from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.models.dynamic_modules import MultiHeadClassifier
from avalanche.models.dynamic_modules import MultiTaskModule
# Config
