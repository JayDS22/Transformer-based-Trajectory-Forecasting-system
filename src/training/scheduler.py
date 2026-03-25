"""Learning rate scheduling utilities."""

import math
import torch.optim as optim


def get_scheduler(optimizer, config: dict):
    """
    Create LR scheduler from config.

    Supports:
      - cosine_annealing: Cosine decay with optional warmup
      - step: Step decay
      - plateau: Reduce on plateau
    """
    scheduler_type = config.get("scheduler", "cosine_annealing")
    epochs = config.get("epochs", 100)
    warmup = config.get("warmup_epochs", 5)

    if scheduler_type == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-7
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    if warmup > 0:
        scheduler = WarmupScheduler(optimizer, scheduler, warmup)

    return scheduler


class WarmupScheduler:
    """Linear warmup wrapper for any LR scheduler."""

    def __init__(self, optimizer, base_scheduler, warmup_epochs: int):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, *args, **kwargs):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * factor
        else:
            self.base_scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / self.warmup_epochs
            return [lr * factor for lr in self.base_lrs]
        return [pg["lr"] for pg in self.optimizer.param_groups]
