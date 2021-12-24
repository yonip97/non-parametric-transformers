"""Learning rate scheduler."""

import numpy as np
import torch
#from dotmap import DotMap
#from fairseq.optim.fairseq_optimizer import FairseqOptimizer
#from fairseq.optim.lr_scheduler import cosine_lr_scheduler
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from functools import wraps
import warnings
import math
from torch.optim.optimizer import Optimizer
import weakref

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)
# from transformers import (
#     get_constant_schedule,
#     get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]



def clip_gradient(model, clip: float):
    nn.utils.clip_grad_norm_(model.parameters(), clip)


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    From Over9000
    https://github.com/mgrankin/over9000/blob/master/train.py
    """
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps,
                 pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        self.curr_epoch = 0
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.curr_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        self.curr_epoch += 1
        super().step()

    def get_lr(self):
        if self.curr_epoch <= self.step_start:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()


class TradeoffAnnealer:
    def __init__(self, c, num_steps=None):
        """
        Anneal the tradeoff between label and augmentation loss according
            to some schedule.

        :param c: config
        :param num_steps: int, provide when loading from checkpoint to fast-
            forward to that tradeoff value.
        """
        self.c = c
        self.name = self.c.exp_tradeoff_annealing

        self.num_steps = 0
        self.init_tradeoff = self.c.exp_tradeoff
        self.curr_tradeoff = self.c.exp_tradeoff
        self.max_steps = self.get_max_steps()
        self.step_map = {
            'constant': self.constant_step,
            'cosine': self.cosine_step,
            'linear_decline': self.linear_decline_step}

        if self.name not in self.step_map.keys():
            raise NotImplementedError

        self.step = self.step_map[self.name]

        if num_steps > 0:
            # If we are loading a model from checkpoint,
            # should update the annealer to that number of steps.
            for _ in range(num_steps):
                self.step()

            print(f'Fast-forwarded tradeoff annealer to step {num_steps}.')

        print(
            f'Initialized "{self.name}" augmentation/label tradeoff annealer. '
            f'Annealing to minimum value in {self.max_steps} steps.')

    def get_max_steps(self):
        # If annealing proportion is set to -1,
        if self.c.exp_tradeoff_annealing_proportion == -1:
            # and the optimizer proportion is set, we use the optimizer
            # proportion to determine how long it takes for the tradeoff to
            # anneal to 0.
            if self.c.exp_optimizer_warmup_proportion != -1:
                return int(np.ceil(self.c.exp_optimizer_warmup_proportion
                                   * self.c.exp_num_total_steps))
            # and the optimizer proportion is not set,
            # we take all steps to anneal.
            else:
                return self.c.exp_num_total_steps

        if (self.c.exp_tradeoff_annealing_proportion < 0
                or self.c.exp_tradeoff_annealing_proportion > 1):
            raise Exception('Invalid tradeoff annealing proportion.')

        # Otherwise, we use the tradeoff annealing proportion to determine
        # for how long we anneal.
        return int(np.ceil(self.c.exp_tradeoff_annealing_proportion
                           * self.c.exp_num_total_steps))

    def constant_step(self):
        self.num_steps += 1
        return self.curr_tradeoff

    def linear_decline_step(self):
        curr = self.num_steps
        max_val = self.init_tradeoff

        if self.num_steps <= self.max_steps:
            self.curr_tradeoff = max_val - (curr / self.max_steps) * max_val
        else:
            self.curr_tradeoff = 0

        self.num_steps += 1

        return self.curr_tradeoff

    def cosine_step(self):
        if self.num_steps <= self.max_steps:
            self.curr_tradeoff = self.init_tradeoff * (1 / 2) * (
                np.cos(np.pi * (self.num_steps / self.max_steps)) + 1)
        else:
            self.curr_tradeoff = 0

        self.num_steps += 1

        return self.curr_tradeoff


class LRScheduler:
    def __init__(self, c, name, optimizer):
        self.c = c
        self.name = name
        self.optimizer = optimizer
        self.num_steps = 0

        self.construct_auto_scheduler()

        print(f'Initialized "{name}" learning rate scheduler.')

    def construct_auto_scheduler(self):
        total_steps = self.c.exp_num_total_steps

        if self.c.exp_optimizer_warmup_proportion >= 0:
            num_warmup_steps = (
                    total_steps * self.c.exp_optimizer_warmup_proportion)
        else:
            num_warmup_steps = self.c.exp_optimizer_warmup_fixed_n_steps

        print(f'Warming up for {num_warmup_steps}/{total_steps} steps.')

        if self.name == 'constant':
            self.scheduler = get_constant_schedule(optimizer=self.optimizer)
        elif self.name == 'linear_warmup':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps)
        # elif self.name == 'cosine_cyclic':
        #     args = dict(
        #         warmup_updates=num_warmup_steps,
        #         warmup_init_lr=1e-7,
        #         max_lr=self.c.exp_lr,
        #         lr=[1e-7],
        #         t_mult=2.,
        #         lr_period_updates=num_warmup_steps * 2,
        #         lr_shrink=0.5)
        #     optim = FairseqOptimizer(None)
        #     optim._optimizer = optim.optimizer = self.optimizer
        #     self.scheduler = cosine_lr_scheduler.CosineSchedule(
        #         optimizer=optim, args=DotMap(args))
        elif self.name == 'polynomial_decay_warmup':
            # Based on the fairseq implementation, which is based on BERT
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=1e-7,
                power=1.0)
        elif self.name == 'flat_and_anneal':
            def d(x):
                return 1

            assert self.c.exp_optimizer_warmup_proportion >= 0

            # We use exp_optimizer_warmup_proportion to denote the
            # flat LR regime, prior to annealing
            dummy = LambdaLR(self.optimizer, d)
            cosine = CosineAnnealingLR(
                self.optimizer, int(total_steps * (
                    1 - self.c.exp_optimizer_warmup_proportion)))
            self.scheduler = ConcatLR(
                self.optimizer, dummy, cosine, total_steps,
                self.c.exp_optimizer_warmup_proportion)
        else:
            raise NotImplementedError

    def step(self):
        self.num_steps += 1
        c_lr = self.c.exp_lr
        num = self.num_steps
        tot = self.c.exp_num_total_steps

        if self.name == 'cosine_cyclic':
            self.scheduler.step_update(num_updates=num)
        else:
            self.scheduler.step()
