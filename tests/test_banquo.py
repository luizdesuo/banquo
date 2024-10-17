#!/usr/bin/env python3
"""The module is a test suite for banquo package."""

###############################################################################
# Imports #####################################################################
###############################################################################


from typing import no_type_check

import numpy as nxp  # ! from numpy import array_api as nxp not working
from hypothesis import given, settings
from scipy.stats import multivariate_normal, norm

from banquo import (
    array,
    chol2inv,
    multi_normal_cholesky_copula_lpdf,
    normalize_covariance,
)

from .hypothesis_arrays_strategy import spd_square_matrix_builder_float64


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


spd_matrix_builder_float64 = spd_square_matrix_builder_float64(size=5)


###############################################################################
# Tests for chol2inv  #########################################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(X=spd_matrix_builder_float64)
def test_chol2inv_equals_inverting_spd_matrix_float64(X: array):
    """Test if :func:`chol2inv` equals directly inverting a SPD matrix.

    Parameters
    ----------
    X : array
        SPD matrix.
    """
    X_inv = nxp.linalg.inv(X)
    spd_chol = nxp.linalg.cholesky(X)
    X_chol2inv = chol2inv(spd_chol)
    assert nxp.allclose(X_inv, X_chol2inv)


###############################################################################
# Tests for multi_normal_cholesky_copula_lpdf  ################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(X=spd_matrix_builder_float64)
def test_multi_normal_cholesky_copula_lpdf_equals_multivariate_gaussian_float64(
    X: array,
):
    """Test for :func:`multi_normal_cholesky_copula_lpdf`.
    This test compares the Gaussian copula combined with Gaussian
    marginal distributions and a multivariate normal distribution lpdf.

    Parameters
    ----------
    X : array
        SPD matrix.
    """

    rng = nxp.random.default_rng()

    # Transform covariance into a correlation matrix
    corr = normalize_covariance(X)

    # Get the Cholesky factor of corr matrix
    corr_chol = nxp.linalg.cholesky(corr)

    d = corr.shape[0]  # number of dimensions
    mean = nxp.zeros(d)  # mean vector

    # Sample data from multivariate normal
    samples = rng.multivariate_normal(mean=mean, cov=corr)

    # Distributions for multivariate normal and normal
    joint_dist = multivariate_normal(mean=mean, cov=corr)
    marginals_dist = norm(loc=0, scale=1)

    # Sum of marginals lpdf
    marginals_lpdf = nxp.sum(nxp.log(marginals_dist.pdf(samples)))

    # Joint distribution lpdf
    joint_lpdf = nxp.log(joint_dist.pdf(samples))

    # Here \Phi^{-1}(\Phi(x)) = x
    copula_lpdf = multi_normal_cholesky_copula_lpdf(samples[nxp.newaxis, :], corr_chol)

    # copula lpdf + marginals_lpdf = joint_lpdf
    assert nxp.isclose(copula_lpdf, joint_lpdf - marginals_lpdf)
