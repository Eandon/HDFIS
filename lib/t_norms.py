def softmin(x, q=-12, dim=0):
    """
    softer version of minimum T-norm
    :param x: inputs, tensor type, get the minimum from them
    :param q: the index parameter of the softmin
    :param dim: {int}, get the minimum on which dimension
    :return: minimum value results
    """
    return (x ** q).mean(dim) ** (1 / q)


def adasoftmin(x, dim=0):
    """
    adaptive softmin, the index parameter is adaptively determined according to x
    :param x: inputs, tensor type, get the minimum from them
    :param dim: {int}, get the minimum on which dimension
    :return: minimum values results
    """
    x = x.double()
    q = 690 / x.data.min(dim=dim).values.log()
    q = q.ceil() + 1
    q[q < -1000] = -1000
    return (x ** q.unsqueeze(dim)).mean(dim) ** (1 / q)


def softmin2(x, q, dim=0):
    """
    Ln-Exp softmin
    :param x: inputs, tensor type, get the minimum from them
    :param q: the index parameter of the softmin
    :param dim: {int}, get the minimum on which dimension
    :return:
    """
    return (x * (x * q).exp()).sum(dim=dim) / (x * q).exp().sum(dim=dim)


def adasoftmin2(x, dim=0):
    """
    adaptive Ln-Exp softmin
    the index parameter is adaptively determined according to x
    :param x: inputs, tensor type, get the minimum from them
    :param dim: {int}, get the minimum on which dimension
    :return: minimum values results
    """
    x = x.double()
    q = -700 / x.data.max(dim=dim).values

    return (x * (x * q.unsqueeze(dim=dim)).exp()).sum(dim=dim) / (x * q.unsqueeze(dim=dim)).exp().sum(dim=dim)
