# GITPASSWORD : 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
# Load cmd arguments
import sys

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# from PYTorchPlayground.Optims import torch_optimizers
from Optims import torch_optimizers
from models.resnet import ResNet20_cifar
from utils import evaluate, cifar_data_tranforms
from utils import fix_seeds, init_wandb, get_args


if __name__ == '__main__':
    # Initial setup
    args = get_args(sys.argv[1:])
    init_wandb(args)
    fix_seeds()

    #Data transform
    data_transforms = cifar_data_tranforms()

    # Load Dataset
    trainset = CIFAR10(root=args.dataroot, train=True, download=True, transform=data_transforms['train'])
    testset = CIFAR10(root=args.dataroot, train=False, download=True, transform=data_transforms['test'])
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=args.shuffle,
                             num_workers=args.workers)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=args.shuffle,
                            num_workers=args.workers)

    model = ResNet20_cifar(out_dim=10).to(args.device)

    optimizer = torch_optimizers(model=model, args=args)
    print(model)
    # lr_schedulr = ReduceLROnPlateau(optimizer=optimizer, min_lr=0.00001, patience=50, factor=0.95, mode="min")
    loss = torch.nn.CrossEntropyLoss()
    # # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
    # #                                                       gamma=0.1)

    for epoch in range(args.schedule[0]):
        print("E:", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)
            b_loss = loss(outputs, labels)
            b_loss.backward()
            optimizer.step()

            running_loss += b_loss.item()
            if i % args.print_freq == args.print_freq - 1:
                running_loss = running_loss / args.print_freq
                # print("[%d, %d]  b-loss:%.3f  r-loss:%.3f" % (epoch, i, b_loss.item(), running_loss))
                if args.wandb_logging:
                    wandb.log({"running_loss": running_loss})
                print("[%d, %d]  b-loss:%.3f  r-loss:%.3f l-rate:%0.3f Lr: %0.3f" % (
                epoch, i, b_loss.item(), running_loss, optimizer.param_groups[0]["lr"]))
                running_loss = 0.0
            else:
                print("[%d, %d]  b-loss:%.3f  Lr: %0.3f" % (epoch, i, b_loss.item(), optimizer.param_groups[0]["lr"]))

            # Update bn statistics for the swa_model at the end

            testaccuracy, testloss = evaluate(model=model, dataloader=testloader, loss=loss)
            # lr_schedulr.step(testloss)
            if args.wandb_logging:
                wandb.log({"batch_loss": b_loss.item(),
                           "test_loss": testloss,
                           "lr": optimizer.param_groups[0]["lr"],
                           "test_accuracy": testaccuracy})
