from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import default_logger
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from typing import Optional, Sequence, Union, List

from agents.ARCLPlugin import ARCLPlugin


class ARCLStrategy(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):

        arcl = ARCLPlugin()

        if plugins is None:
            plugins = [arcl]
        else:
            plugins += [arcl]

        super().__init__(
            model, optimizer, criterion=criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    # def train(self, experience):
    #     # super().train(self, experience)
    #     print("Data:", len(experience.dataset), " Task-Labels:", experience.task_label)
    #
    # def eval(self, experience):
    #     print("Data:", len(experience))
    #
    # def before_training(self, **kwargs):
    #     print("Happy birthday")
