import sys
import torch
import torch.nn.functional as F


def map(value, mapping=None):
    """Re-maps a number from one range to another.
    Args:
        value (int): Value to be mapped.
        mapping (dict): Dictionary with the new range of values.
    Returns:
        int: Mapped value.
    """
    if mapping is None:
        return value

    if not value in mapping.keys():
        raise KeyError('{} was not mapped in the new range of values.'.format(value))

    return mapping[value]


def entropy(input, reduction='mean'):
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    p = F.softmax(input, 1)
    q = F.log_softmax(input, 1)
    b = p * q
    b = -1.0 * b.sum(-1)

    if reduction == "none":
        return b
    elif reduction == "mean":
        return b.mean()
    elif reduction == "sum":
        return b.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def consistency(input, target, reduction='mean'):
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    return F.kl_div(F.log_softmax(input, 1), F.softmax(target, 1), reduction=reduction)


def consensus(input, target, reduction='mean'):
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    p = F.log_softmax(input, 1)
    q = F.log_softmax(target, 1)
    b = p + q
    b = -0.5 * b.max(-1)[0]

    if reduction == "none":
        return b
    elif reduction == "mean":
        return b.mean()
    elif reduction == "sum":
        return b.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def compute_joint(input, target):
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    # produces variable that requires grad (since args require grad)

    bn, k = input.size()
    assert (target.size(0) == bn and target.size(1) == k)

    p_i_j = input.unsqueeze(2) * target.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def iic(input, target):
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    # has had softmax applied
    _, k = input.size()
    p_i_j = compute_joint(input, target)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                             k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < sys.float_info.epsilon).data] = sys.float_info.epsilon
    p_j[(p_j < sys.float_info.epsilon).data] = sys.float_info.epsilon
    p_i[(p_i < sys.float_info.epsilon).data] = sys.float_info.epsilon

    loss = - p_i_j * (torch.log(p_i_j) \
                      - torch.log(p_j) \
                      - torch.log(p_i))

    return loss.sum()


def cross_entropy_with_probs(input, target, weight=None, reduction='mean'):
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    dim = input.dim()
    if dim < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
