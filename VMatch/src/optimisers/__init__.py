# Code adapted from EfficientLoFTR [https://github.com/zju3dv/EfficientLoFTR/]

import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    '''
    Initialise the optimizer
    Input:
        model (torch.nn.Module): model to be trained
        config (dict): configuration dictionary
    Output:
        optimizer (torch.optim.Optimizer): optimizer
    '''
    name = config.optimizer
    lr = config.true_lr

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=config.betas, weight_decay=config.adam_decay, eps=1e-8 if not config.fp16_optimizer else 1e-4)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=config.betas, weight_decay=config.adamw_decay, eps=1e-8 if not config.fp16_optimizer else 1e-4)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer) -> dict:
    '''
    Optimizer scheduler
    Input:
        config (dict): configuration dictionary
        optimizer (torch.optim.Optimizer): optimizer
    Output:
        scheduler (dict): scheduler dictionary
    '''
    scheduler = {'interval': config.scheduler_interval}
    name = config.scheduler

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.mslr_milestones, gamma=config.mslr_gamma)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.cosa_tmax)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.elr_gamma)})
    else:
        raise NotImplementedError(f"Scheduler {name} not implemented.")

    return scheduler