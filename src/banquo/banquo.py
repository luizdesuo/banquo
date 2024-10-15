#!/usr/bin/env python3
"""The module contains building blocks for Nonparanormal models."""

###############################################################################
# Imports #####################################################################
###############################################################################


from typing import Any

from array_api_compat import array_namespace


###############################################################################
# Custom types for annotation #################################################
###############################################################################


array = Any
"""Type annotation for array objects.

    For more information, please refer to `array-api
    <https://data-apis.org/array-api/latest/API_specification/array_object.html>`__.
"""


###############################################################################
# Auxiliary functions #########################################################
###############################################################################


def chol2inv(spd_chol: array) -> array:
    r"""Invert a SPD square matrix from its Choleski decomposition.

    Given a Choleski decomposition :math:`\Sigma` of a matrix :math:`\Sigma`,
    i.e. :math:`\Sigma = LL^T`, this function returns the inverse
    :math:`\Sigma^{-1}`.

    Parameters
    ----------
    spd_chol : array
        Cholesky factor of the correlation/covariance matrix.

    Returns
    -------
    array
        Inverse matrix.
    """
    xp = array_namespace(spd_chol)
    spd_chol_inv = xp.linalg.inv(spd_chol)
    return spd_chol_inv.T @ spd_chol_inv


###############################################################################
# Copula functions ############################################################
###############################################################################


def multi_normal_cholesky_copula_lpdf(marginal: array, omega_chol: array) -> float:
    r"""Compute multivariate normal copula lpdf (Cholesky parameterisation).

    Considering the copula function :math:`C:[0,1]^d\rightarrow [0,1]`
    and any :math:`(u_1,\dots,u_d)\in[0,1]^d`, such that
    :math:`u_i = F_i(X_i) = P(X_i \leq x)` are cumulative distribution
    functions. The multivariate normal copula is given by
    :math:`C_\Omega(u) = \Phi_\Omega\left(\Phi^{-1}(u_1),\dots, \Phi^{-1}(u_d) \right)`.
    It is parameterized by the correlation matrix :math:`\Omega = LL^T`, from which
    :math:`L` is the Cholesky decomposition. Then, the copula density function is
    given by

    .. math::
        c_\Omega(u) = \frac{\partial^d C_\Omega(u)}{\partial \Phi(u_1)\cdots \partial \Phi(u_d)} \,,

    and this function computes its log density :math:`\log\left(c_\Omega(u)\right)`.


    Parameters
    ----------
    marginal : array
        Matrix of outcomes from marginal calculations.
        In this function, :math:`\text{marginal} = \Phi^{-1}(u)`.
    omega_chol : array
        Cholesky factor of the correlation matrix.

    Returns
    -------
    float
        log density of distribution.
    """  # noqa: B950
    xp = array_namespace(marginal, omega_chol)
    n, d = marginal.shape
    precision = chol2inv(omega_chol)
    log_density: float = -n * xp.sum(xp.log(xp.diagonal(omega_chol))) - 0.5 * xp.sum(
        xp.multiply(precision - xp.eye(d), marginal.T @ marginal)
    )
    return log_density
