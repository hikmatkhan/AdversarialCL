import torch


def torch_optimizers(model, args):

    optimizer_arg = {'params': model.parameters(),
                     'lr': args.lr,
                     'weight_decay': args.weight_decay}

    if args.optimizer in ['SGD', 'RMSprop']:
        optimizer_arg['momentum'] = args.momentum
    elif args.optimizer in ['Rprop']:
        optimizer_arg.pop('weight_decay')
    elif args.optimizer == 'amsgrad':
        optimizer_arg['amsgrad'] = True
        args.optimizer = 'Adam'

    return torch.optim.__dict__[args.optimizer](**optimizer_arg)