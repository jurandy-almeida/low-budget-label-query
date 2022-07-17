import torch
import torch.nn as nn

import functional as F


class EntropyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input):
        return F.entropy(input, reduction=self.reduction)


class ConsistencyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.consistency(input, target, reduction=self.reduction)


class ConsensusLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(ConsensusLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.consensus(input, target, reduction=self.reduction)


class IICLoss(nn.Module):

    def forward(self, input, target):
        return F.iic(input, target)


class CrossEntropyLossWithProbs(nn.Module):
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N, C)`  where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.
    """
    __constants__ = ['weight']

    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLossWithProbs, self).__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, input, target):
        return F.cross_entropy_with_probs(input, target, weight=self.weight,
                                          reduction=self.reduction)

    def extra_repr(self):
        s = ('weight={weight}')
        return s.format(**self.__dict__)


class MultiTaskLoss(nn.Module):
    '''https://arxiv.org/abs/1705.07115'''

    __constants__ = ['num_tasks', 'weight']

    def __init__(self, num_tasks, weight=None, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.reduction = reduction
        self.num_tasks = num_tasks
        self.weight = weight
        self.log_vars = nn.Parameter(torch.Tensor(num_tasks))
        self.reset_parameters()

    def forward(self, losses):
        if self.weight is not None:
            losses = losses * self.weight.to(losses.device).to(losses.dtype)
        multi_task_losses = torch.exp(-self.log_vars) * losses + self.log_vars

        if self.reduction == "none":
            return multi_task_losses
        elif self.reduction == "sum":
            return multi_task_losses.sum()
        elif self.reduction == "mean":
            return multi_task_losses.mean()
        else:
            raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")

    def reset_parameters(self):
        nn.init.zeros_(self.log_vars)

    def extra_repr(self):
        s = ('num_tasks={num_tasks}, weight={weight}')
        return s.format(**self.__dict__)

