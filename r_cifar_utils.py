import os

import torch
import torchvision
from torchvision.transforms import transforms
from avalanche.benchmarks.generators import tensors_benchmark

R_CIFAR_PATH = "/home/hikmat/Desktop/JWorkspace/CL/RCL/PY_CIFAR/datasets/release_datasets"


def load_cifar(ds_name="d_robust_CIFAR"):
    data_path = "{0}/{1}".format(R_CIFAR_PATH, ds_name)
    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transform_test)

    return (train_data, train_labels), test_dataset


def get_examples(train_data, train_labels, classes=[0, 1]):
    tasks_x = None
    tasks_y = None
    for idx in classes:
        #         idxs.append(train_labels==idx)
        print(train_labels[train_labels == idx])
        if (tasks_x == None):
            tasks_x = train_data[train_labels == idx]
            tasks_y = train_labels[train_labels == idx]
        else:
            tasks_x = torch.cat([tasks_x, train_data[train_labels == idx]])
            tasks_y = torch.cat([tasks_y, train_labels[train_labels == idx]])
        print("Concatenated", len(tasks_y))
    return (tasks_x, tasks_y)


def get_cifar_experience(ds_name="d_robust_CIFAR",
                         classes=[[0, 1], [2], [3], [4], [5], [6], [7], [8], [9]]):
    (train_data, train_labels), test_dataset = load_cifar(ds_name=ds_name)

    train_experiences = []
    for i in classes:
        #     (train_data, train_labels) = get_examples(train_data, train_labels, classes=i)
        train_experiences.append(get_examples(train_data, train_labels, classes=i))
        print(len(train_labels))

    generic_scenario = tensors_benchmark(
        train_tensors=train_experiences,
        test_tensors=[(test_dataset.data, test_dataset.targets)],
        task_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8],  # Task label of each train exp
        complete_test_set_only=True
    )
    return generic_scenario
