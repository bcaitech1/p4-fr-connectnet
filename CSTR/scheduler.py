import numpy as np
import warnings
import weakref
from functools import wraps
from torch.optim import Optimizer


class _Iter_LRScheduler(object):

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 last_iter=-1,
                 iter_based=True):
        self._iter_based = iter_based
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.niter_per_epoch = niter_per_epoch
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_iter = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i))  # noqa 501
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = int(last_iter / niter_per_epoch)

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
        self.iter_nums(last_iter)
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def iter_nums(self, iter_=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden "
                    "after learning rate scheduler initialization. Please, "
                    "make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",  # noqa 501
                    UserWarning)

            # Just check if there were two first lr_scheduler.step()
            # calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "  # noqa 501
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "  # noqa 501
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "  # noqa 501
                    "will result in PyTorch skipping the first value of the learning rate schedule."  # noqa 501
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",  # noqa 501
                    UserWarning)
        self._step_count += 1

        if iter_ is None:
            iter_ = self.last_iter + 1
        self.last_iter = iter_
        self.last_epoch = np.int(iter_ / self.niter_per_epoch)

    def step(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StepLR(_Iter_LRScheduler):

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 max_epochs,
                 milestones,
                 gamma=0.1,
                 last_iter=-1,
                 warmup_epochs=0,
                 iter_based=True):
        self.max_iters = niter_per_epoch * max_epochs
        self.milestones = milestones
        self.count = 0
        self.gamma = gamma
        self.warmup_iters = int(niter_per_epoch * warmup_epochs)
        super(StepLR, self).__init__(optimizer, niter_per_epoch, last_iter,
                                     iter_based)

    def get_lr(self):
        if self._iter_based and self.last_iter in self.milestones:
            self.count += 1
        elif not self._iter_based and self.last_epoch in self.milestones:
            self.count += 1

        if self.last_iter < self.warmup_iters:
            multiplier = self.last_iter / float(self.warmup_iters)
        else:
            multiplier = self.gamma**self.count
        return [base_lr * multiplier for base_lr in self.base_lrs]
