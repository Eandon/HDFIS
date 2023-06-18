"""
membership functions used in the fuzzy system
"""

from torch import tensor


def gauss(x, m, sigma):
    """
    gaussian membership function
    exp(-(x-m)^2/(2*sigma^2))
    :param x: independent variable
    :param m: center
    :param sigma: spread
    :return: membership values
    """
    return (-(x - m) ** 2 / (2 * sigma ** 2)).exp()


def gauss_dmf_sig(x, m, sigma, dimension, rho=None):
    """
    gaussian dimension-dependent membership function (DMF)
    The spread is D^ρ+σ^2, where ρ is the lower bound adaptively computed from the input dimension.
    σ is the system parameter to be optimized
    :param x: independent variable
    :param m: center
    :param sigma: system parameter to be optimized
    :param dimension: input dimension
    :param rho: scale parameter
    :return: membership values
    """
    rho = 1 - tensor(745).log() / tensor(dimension).log() if rho is None else rho
    return (-(x - m) ** 2 / (dimension ** rho + sigma ** 2)).exp()


def tri(x, a, t, b):
    """
    triangular membership function
    :param x: independent variable
    :param a: left end point
    :param t: peak
    :param b: right end point
    :return:
    """
    return ((x - a) / (t - a)).minimum((b - x) / (b - t)).maximum(tensor(0.))


def tri_dmf(x, a, t, b, dimension):
    """
    triangular DMF
    In the interval [a,b], the minimum is not 0, and the maximum is 1.
    :param x: independent variable
    :param a: left end point
    :param t: peak
    :param b: right end point
    :param dimension: input dimension, i.e., the number of features
    :return:
    """
    rho_tilde = 1 - tensor(745).log() / tensor(dimension).log()
    y1 = (x - a) / (t - a) * (1 - (-1 / dimension ** rho_tilde).exp()) + (-1 / dimension ** rho_tilde).exp()
    y2 = (b - x) / (b - t) * (1 - (-1 / dimension ** rho_tilde).exp()) + (-1 / dimension ** rho_tilde).exp()

    return y1.minimum(y2).maximum(tensor(0.))


def trap(x, a, t1, t2, b):
    """
    trapezoidal membership function
    max{ min{ (x-a)/(t1-a), 1, (b-x)/(b-t2) }, 0 }
    :param x: independent variable
    :param a: left end point
    :param t1: left peak
    :param t2: right peak
    :param b: right end point
    :return:
    """
    return ((x - a) / (t1 - a)).minimum(tensor(1.)).minimum((b - x) / (b - t2)).maximum(tensor(0.))


def trap_dmf(x, a, t1, t2, b, dimension):
    """
    trapezoidal DMF
    :param x: independent variable
    :param a: left end point
    :param t1: left peak
    :param t2: right peak
    :param b: right end point
    :param dimension: input dimension, i.e., the number of features
    :return:
    """
    rho_tilde = 1 - tensor(745).log() / tensor(dimension).log()
    y1 = (x - a) / (t1 - a) * (1 - (-1 / dimension ** rho_tilde).exp()) + (-1 / dimension ** rho_tilde).exp()
    y2 = (b - x) / (b - t2) * (1 - (-1 / dimension ** rho_tilde).exp()) + (-1 / dimension ** rho_tilde).exp()

    return y1.minimum(tensor(1.)).minimum(y2).maximum(tensor(0.))


def gauss_htsk(x, m, sigma, dimension):
    """
    Equivalent membership function used in HTSK, where the spread is 2 * D * σ^2.
    :param x: independent variable
    :param m: center
    :param sigma: system parameter to be optimized
    :param dimension: input dimension
    :return:
    """
    return (-(x - m) ** 2 / (2 * dimension * sigma ** 2)).exp()
