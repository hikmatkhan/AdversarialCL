import argparse
import random
import numpy as np
import torch
import wandb
from avalanche.benchmarks import tensors_benchmark


def fix_seeds(seed=101):
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
    return seed


def get_compute_device():
    device = torch.device('cpu')
    if torch.cuda.device_count():
        device = torch.device('cuda')
    return device


def init_wandb(args):
    if args.wandb_logging:
        wandb.init(project=args.wand_project, entity="hikmatkhan-", reinit=True)
        wandb.config.update(args)


def nc_benchmark_by_ids(trainset, testset, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    trnset = []
    tstset = []

    for ids in classes:
        if isinstance(ids, list):
            train_x_temp = None
            test_x_temp  = None
            train_y_temp = None
            test_y_temp  = None
            for id in ids:
                if train_x_temp is None:
                    train_x_temp = trainset.data[trainset.targets == id]
                    train_y_temp = trainset.targets[trainset.targets == id]
                    test_x_temp = trainset.data[trainset.targets == id]
                    test_y_temp = trainset.targets[trainset.targets == id]
                else:
                    train_x_temp = torch.cat([train_x_temp, trainset.data[trainset.targets == id]])
                    train_y_temp = torch.cat([train_y_temp, trainset.targets[trainset.targets == id]])
                    test_x_temp = torch.cat([test_x_temp, trainset.data[trainset.targets == id]])
                    test_y_temp = torch.cat([test_y_temp, trainset.targets[trainset.targets == id]])

            trnset.append((train_x_temp, train_y_temp))
            tstset.append((test_x_temp, test_y_temp))
        else:
            trnset.append((trainset.data[trainset.targets == ids],
                           trainset.targets[trainset.targets == ids]))
            tstset.append((testset.data[testset.targets == ids],
                           testset.targets[testset.targets == ids]))

    # print("Tasks in training set:", len(trnset))
    # print("Tasks in testing set:", len(tstset))
    print("Task_labels=", list(range(0, len(classes))))
    return tensors_benchmark(train_tensors=trnset,
                             test_tensors=tstset, task_labels=list(range(0, len(classes))),
                             # complete_test_set_only=True
                             )
                             # list(range(0,   len(classes))))


def get_optimizer(model, args):
    optimizer_arg = {'params': model.parameters(),
                     'lr': args.lr,
                     'weight_decay': args.weight_decay}
    if args.optimizer in ['SGD', 'RMSprop']:
        optimizer_arg['momentum'] = args.momentum
    elif args.optimizer in ['Rprop']:
        optimizer_arg.pop('weight_decay')
    elif args.optimizer == 'amsgrad':
        optimizer_arg['amsgrad'] = True
        args.optimizer['optimizer'] = 'Adam'

    optimizer = torch.optim.__dict__[args.optimizer](**optimizer_arg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule,
                                                     gamma=0.1)
    return optimizer


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='lenet',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='lenet', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2,
                        help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--outdir', type=str, default='default', help="Output results directory")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='./data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--device', type=str, default=get_compute_device(), help="Computational unit")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=1, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=1024 * 4)
    parser.add_argument('--shuffle', dest='shuffle', default=True, action='store_false',
                        help="Dataset shuffle.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[10000],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")

    parser.add_argument('--print_freq', type=float, default=25, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.],
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")

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

    args = parser.parse_args(argv)
    return args
