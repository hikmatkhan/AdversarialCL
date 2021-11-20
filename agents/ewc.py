import argparse
import sys

import avalanche
import torch.nn
import torchvision.datasets
from avalanche.benchmarks import SplitMNIST, nc_benchmark, ni_benchmark, tensors_benchmark
from avalanche.benchmarks.datasets import MNIST, CIFAR100, CIFAR10
from avalanche.evaluation.metrics import StreamAccuracy, accuracy_metrics, forgetting_metrics
from avalanche.evaluation.plot_utils import learning_curves_plot
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.models import SimpleCNN, SimpleMLP, MultiHeadClassifier, pytorchcv_wrapper
from avalanche.training import EWC, Naive, JointTraining
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.storage_policy import StoragePolicy
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset

import utils
from agents.ARCLStrategy import ARCLStrategy


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    # parser.add_argument('--model_type', type=str, default='lenet',
    #                     help="The type (mlp|lenet|vgg|resnet) of backbone network")
    # parser.add_argument('--model_name', type=str, default='lenet', help="The name of actual model for the backbone")
    # parser.add_argument('--force_out_dim', type=int, default=2,
    #                     help="Set 0 to let the task decide the required output dimension")
    # parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--outdir', type=str, default='default', help="Output results directory")
    # parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='./data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--device', type=str, default=utils.get_compute_device(), help="Computational unit")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    # parser.add_argument('--first_split_size', type=int, default=2)
    # parser.add_argument('--other_split_size', type=int, default=2)
    # parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
    #                     help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    # parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
    #                     help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--seed', type=int, default=101, help="No randomization")
    parser.add_argument('--batch_size', type=int, default=1024 * 10)
    parser.add_argument('--shuffle', dest='shuffle', default=True, action='store_false',
                        help="Dataset shuffle.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[10],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--epochs', nargs="+", type=int, default=1,
                        help="Number of epochs")

    parser.add_argument('--print_freq', type=float, default=25, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--ewc_lambda', nargs="+", type=float, default=0,
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    # parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
    #                     help="Force the evaluation on train set")
    # parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
    #                     help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    # parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
    #                     help="The number of output node in the single-headed model increases along with new categories.")

    # Misc
    misc = parser.add_argument_group('Misc')
    misc.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers to use for data-loading (default: 1).')
    # Logging
    misc.add_argument('--wand-project', type=str, default="PY_CIFAR",
                      help='Wandb project name should go here')
    misc.add_argument('--wand-note', type=str, default="Test Run Note",
                      help='To identify run')

    misc.add_argument('--username', type=str, default="hikmatkhan",
                      help='Wandb username should go here')
    misc.add_argument('--wandb-logging', type=int, default=1,
                      help='If True, Logs will be reported on wandb.')
    #Avalanche
    avalanche = parser.add_argument_group('Avalanche')
    avalanche.add_argument('--run', type=str, default="Avalanche Run",
                      help='Provide run name for wandb.')
    avalanche.add_argument('--project', type=str, default="Avalanche Project",
                      help='Define the name of the wandb project.')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    args = get_args(sys.argv[1:])
    utils.fix_seeds(seed=args.seed)
    # utils.init_wandb(args=args)
    print(args)


    # benchmark = SplitMNIST(n_experiences=5, shuffle=True, dataset_root=args.dataroot)
    trainset = CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))
    testset = CIFAR10(root="./data", train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor()
                                          ]))
    #NC BENCHMARK (CLASS INCREMENTAL LEANRING)
    benchmark = nc_benchmark(train_dataset=trainset, test_dataset=testset, task_labels=True,
                             shuffle=True, n_experiences=10, seed=args.seed)

    #NI BENCHMARK (DOMAIN INCREMENTAL LEARNING)
    # benchmark = ni_benchmark(train_dataset=trainset, test_datasett=testset, shuffle=True, n_experiences=10,
    #                          task_labels=True, seed=args.seed)

    # classes = [[0, 1], 2, 3, 4, 5, 6, 7, 8, 9]

    # benchmark = utils.nc_benchmark_by_ids(trainset=trainset, testset=testset, classes=classes)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    # wandb_logger = WandBLogger(project_name=args.project, run_name=args.run, config=vars(args))

    # main_metric = StreamAccuracy()
    eval_plugin = EvaluationPlugin(
        StreamAccuracy(),
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #     timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        #     cpu_usage_metrics(experience=True),
        #     confusion_matrix_metrics(num_classes=split_mnist.n_classes, save_image=False, stream=True),
        #     disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=interactive_logger
    )

    # model = SimpleMLP(num_classes=10, hidden_layers=2, hidden_size=512)
    model = pytorchcv_wrapper.resnet(dataset="cifar10", depth=20, pretrained=False)

    # model.classifier = MultiHeadClassifier(in_features=model.classifier.in_features, initial_out_features=2)
    optimizer = utils.get_optimizer(model=model, args=args)
    criterion = CrossEntropyLoss()
    print(optimizer)
    replay = ReplayPlugin(mem_size=10000)
    # strategy = Naive(model=model, optimizer=optimizer,
    #                criterion=criterion, evaluator=eval_plugin, train_mb_size=args.batch_size,
    #                eval_mb_size=args.batch_size,
    #                device=args.device, train_epochs=args.epochs, plugins=[replay])
    # strategy = EWC(model=model, ewc_lambda=args.ewc_lambda, optimizer=optimizer,
    #                criterion=criterion, evaluator=eval_plugin, train_mb_size=args.batch_size,
    #                eval_mb_size=args.batch_size,
    #                device=args.device, train_epochs=args.epochs)

    # print("Strategy:", strategy)
    # strategy = ARCLStrategy(model=model, optimizer=optimizer, criterion=criterion)
    # strategy = JointTraining(model=model, optimizer=optimizer,
    #                criterion=criterion, evaluator=eval_plugin, train_mb_size=args.batch_size,
    #                eval_mb_size=args.batch_size,
    #                device=args.device, train_epochs=args.epochs)
    strategy = ARCLStrategy(model=model, optimizer=optimizer,
                   criterion=criterion, evaluator=eval_plugin, train_mb_size=args.batch_size,
                   eval_mb_size=args.batch_size,
                   device=args.device, train_epochs=2)#args.epochs)

    for experience in benchmark.train_stream:
        print(len(experience.dataset))
        strategy.train(experience)


        # print(model)
        # strategy.train(experience)
    #     cl_strategy.eval(benchmark.test_stream)
    strategy.eval(benchmark.test_stream)

    # learning_curves_plot(eval_plugin.get_all_metrics())
