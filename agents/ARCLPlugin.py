from avalanche.training.plugins import StrategyPlugin


class ARCLPlugin(StrategyPlugin):

    def __init__(self):
        pass

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        print("Before Training.")
        return super().before_training(strategy, **kwargs)

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        print("Before training data adaptation.")
        return super().before_train_dataset_adaptation(strategy, **kwargs)






