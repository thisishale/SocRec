import math

class custom_schd(object):
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, epoch,args):
        """
        Update parameters and rate
        """
        if epoch < args.warmup:
        # Warm-up phase: Linearly increase the learning rate
            new_lr = (args.max_lr - args.initial_lr_) / args.warmup * epoch + args.initial_lr_
        else:
        # Exponential decay phase: Decrease the learning rate exponentially
            new_lr = args.max_lr * math.exp(-0.1 * (epoch - args.warmup))
            new_lr = max(new_lr, args.min_lr)
        self.optimizer.step()
        return new_lr
        

