#!/usr/bin/env python3
"""The module contains building blocks for Nonparanormal models."""

###############################################################################
# Imports #####################################################################
###############################################################################

import jax.numpy as jnp
from jax.typing import ArrayLike


###############################################################################
# Auxiliary functions #########################################################
###############################################################################


def chol2inv(spd_chol: ArrayLike) -> ArrayLike:
    r"""Invert a SPD square matrix from its Choleski decomposition.

    Given a Choleski decomposition :math:`\Sigma` of a matrix :math:`\Sigma`,
    i.e. :math:`\Sigma = LL^T`, this function returns the inverse
    :math:`\Sigma^{-1}`.

    Parameters
    ----------
    spd_chol : ArrayLike
        Cholesky factor of the correlation/covariance matrix.

    Returns
    -------
    ArrayLike
        Inverse matrix.
    """
    spd_chol_inv = jnp.linalg.inv(spd_chol)
    return spd_chol_inv.T @ spd_chol_inv


###############################################################################
# Copula functions ############################################################
###############################################################################


def multi_normal_cholesky_copula_lpdf(
    marginal: ArrayLike, omega_chol: ArrayLike
) -> float:
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
    marginal : ArrayLike
        Matrix of outcomes from marginal calculations.
        In this function, :math:`\text{marginal} = \Phi^{-1}(u)`.
    omega_chol : ArrayLike
        Cholesky factor of the correlation matrix.

    Returns
    -------
    float
        log density of distribution.
    """  # noqa: B950
    n, d = marginal.shape
    gammainv = chol2inv(omega_chol)
    log_density: float = -n * jnp.sum(
        jnp.log(jnp.diagonal(omega_chol))
    ) - 0.5 * jnp.sum(jnp.multiply(gammainv - jnp.eye(d), marginal.T @ marginal))
    return log_density
