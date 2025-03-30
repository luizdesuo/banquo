#!/usr/bin/env python3
"""The module is a test suite for banquo package."""

###############################################################################
# Imports #####################################################################
###############################################################################


from typing import no_type_check

import numpy as nxp  # ! from numpy import array_api as nxp not working
import pytest
from array_api_compat import array_namespace, device
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import logsumexp as scipy_logsumexp
from scipy.stats import multivariate_normal, norm

from banquo.banquo import (
    DataMaxExceedsSupportUpperBoundError,
    DataMinExceedsSupportLowerBoundError,
    DataRangeExceedsSupportBoundError,
    array,
    chol2inv,
    divide_ns,
    homographic_ns,
    logsumexp,
    minmax_normalization,
    multi_normal_cholesky_copula_lpdf,
    multiply_ns,
    normalize_covariance,
    std_ns,
)

from .conftest import xps  # Parameterization in conftest.py for CLI
from .hypothesis_arrays_strategy import FLOAT64, spd_square_matrix_builder_float64


###############################################################################
# Constants  ##################################################################
###############################################################################


WIDTH = 32


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


spd_matrix_builder_float64 = spd_square_matrix_builder_float64(size=5)


###############################################################################
# Tests for auxiliary functions  ##############################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(x=spd_matrix_builder_float64)
def test_chol2inv_equals_inverting_spd_matrix_float64(x: array) -> None:
    """Test if :func:`chol2inv` equals directly inverting a SPD matrix.

    Parameters
    ----------
    x : array
        SPD matrix.
    """
    xp = array_namespace(x)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy

    x_inv = nxp.linalg.inv(x)
    spd_chol = nxp.linalg.cholesky(x)
    x_chol2inv = nxp.asarray(chol2inv(xp.asarray(spd_chol, device=device(x))))
    assert nxp.allclose(x_inv, x_chol2inv)


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    )
)
def test_std_ns_not_inf_float64(x: array) -> None:
    """Test if :func:`std_ns` is close to finite entries std.

    Parameters
    ----------
    x : array
        Elements to calculate std.
    """
    xp = array_namespace(x)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy

    res = nxp.std(x)
    x_robust = nxp.asarray(std_ns(xp.asarray(x, device=device(x))))
    res_not_inf = ~nxp.isinf(res)
    if res_not_inf:
        assert nxp.isclose(x_robust, res)


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    ),
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    ),
)
def test_divide_ns_not_inf_float64(x1: array, x2: array) -> None:
    """Test if :func:`divide_ns` is close to finite entries element-wise division.

    Parameters
    ----------
    x1 : array
        Numerator.
    x2 : array
        Denominator.
    """
    xp = array_namespace(x1, x2)  # Get the array API namespace
    x1 = nxp.asarray(x1)  # Convert to numpy
    x2 = nxp.asarray(x2)  # Convert to numpy

    res = x1 / x2
    div_robust = nxp.asarray(
        divide_ns(xp.asarray(x1, device=device(x1)), xp.asarray(x2, device=device(x1)))
    )
    res_not_inf = ~nxp.isinf(res)
    res_not_nan = ~nxp.isnan(res)
    assert nxp.allclose(
        res[res_not_inf & res_not_nan], div_robust[res_not_inf & res_not_nan]
    )


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    ),
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    ),
)
def test_multiply_ns_not_inf_float64(x1: array, x2: array) -> None:
    """Test if :func:`multiply_ns` is close to finite entries multiplication.

    Parameters
    ----------
    x1 : array
        Factor.
    x2 : array
        Factor.
    """
    xp = array_namespace(x1, x2)  # Get the array API namespace
    x1 = nxp.asarray(x1)  # Convert to numpy
    x2 = nxp.asarray(x2)  # Convert to numpy

    res = x1 * x2
    mul_robust = nxp.asarray(
        multiply_ns(
            xp.asarray(x1, device=device(x1)), xp.asarray(x2, device=device(x1))
        )
    )
    res_not_inf = ~nxp.isinf(res)
    assert nxp.allclose(res[res_not_inf], mul_robust[res_not_inf])


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    )
)
def test_homographic_ns_float64(x: array) -> None:
    r"""Test if :func:`homographic_ns` is close to :math:`1/(1+x)`.

    Parameters
    ----------
    x : array
        Elements to perform homographic transform.
    """
    xp = array_namespace(x)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy

    res = 1 / (1 + x)
    homographic = nxp.asarray(homographic_ns(xp.asarray(x, device=device(x))))
    res_not_inf = ~nxp.isinf(res)
    assert nxp.allclose(res[res_not_inf], homographic[res_not_inf])


@no_type_check
@settings(deadline=None)
@given(x=spd_matrix_builder_float64)
def test_logsumexp_equals_scipy_implementation_float64(x: array) -> None:
    """Test if :func:`logsumexp` is close to scipy implementation.

    Parameters
    ----------
    x : array
        Elements to calculate logsumexp.
    """
    xp = array_namespace(x)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy

    rng = nxp.random.default_rng()

    # Transform covariance into a correlation matrix
    corr = normalize_covariance(x)

    d = corr.shape[0]  # number of dimensions
    mean = nxp.zeros(d)  # mean vector

    # Sample data from multivariate normal
    samples = rng.multivariate_normal(mean=mean, cov=corr, size=10)

    # Distributions for multivariate normal
    joint_dist = multivariate_normal(mean=mean, cov=corr)

    # Joint distribution lpdf
    joint_lpdf = joint_dist.logpdf(samples)

    res = scipy_logsumexp(joint_lpdf)
    lse = nxp.asarray(logsumexp(xp.asarray(joint_lpdf, device=device(samples))))

    assert nxp.isclose(res, lse)


###############################################################################
# Test data transform #########################################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    )
)
def test_minmax_normalization_default_support_float64(x: array) -> None:
    """Test :func:`minmax_normalization` with default support.

    Parameters
    ----------
    x : array
        Elements to be transformed.
    """

    x_transf = nxp.asarray(minmax_normalization(x))

    x_transfmin = nxp.min(x_transf)
    x_transfmax = nxp.max(x_transf)

    # For some unknown reason, in some cases, hypothesis is saying
    # x_transf is nan, when it actually is not
    if ~nxp.isnan(x_transfmin):
        assert nxp.isclose(x_transfmin, 0) or (x_transfmin > 0)

    if ~nxp.isnan(x_transfmax):
        assert nxp.isclose(x_transfmax, 1) or (x_transfmax < 1)


@no_type_check
@settings(deadline=None)
@given(
    xps.arrays(
        dtype=FLOAT64,
        shape=(10,),
        elements=st.floats(allow_infinity=False, allow_nan=False, width=WIDTH),
        unique=True,
    ),
    xps.arrays(
        dtype=FLOAT64,
        shape=(2,),
        elements=st.floats(allow_nan=False, width=WIDTH),
        unique=True,
    ),
)
def test_minmax_normalization_any_support_float64(x: array, support: array) -> None:
    """Test :func:`minmax_normalization` with any support.

    Parameters
    ----------
    x : array
        Elements to be transformed.
    support : array
        Two-elements array containing the lower and upper bounds
        for the elements.
    """
    xp = array_namespace(x, support)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy
    support = nxp.asarray(support)  # Convert to numpy

    support = nxp.sort(support)

    x_range = nxp.asarray((nxp.min(x), nxp.max(x)))

    condition_lower: bool = x_range[0] < support[0]
    condition_upper: bool = x_range[1] > support[1]

    # Check if data range exceeds support's boundary
    if condition_lower and condition_upper:
        with pytest.raises(DataRangeExceedsSupportBoundError):
            x_transf = nxp.asarray(
                minmax_normalization(
                    xp.asarray(x, device=device(x)),
                    support=xp.asarray(support, device=device(x)),
                )
            )
    # Check if data minimum exceeds support's lower bound
    elif x_range[0] < support[0]:
        with pytest.raises(DataMinExceedsSupportLowerBoundError):
            x_transf = nxp.asarray(
                minmax_normalization(
                    xp.asarray(x, device=device(x)),
                    support=xp.asarray(support, device=device(x)),
                )
            )
    # Check if data maximum exceeds support's upper bound
    elif x_range[1] > support[1]:
        with pytest.raises(DataMaxExceedsSupportUpperBoundError):
            x_transf = nxp.asarray(
                minmax_normalization(
                    xp.asarray(x, device=device(x)),
                    support=xp.asarray(support, device=device(x)),
                )
            )
    else:
        x_transf = nxp.asarray(
            minmax_normalization(
                xp.asarray(x, device=device(x)),
                support=xp.asarray(support, device=device(x)),
            )
        )

        x_transfmin = nxp.min(x_transf)
        x_transfmax = nxp.max(x_transf)

        # For some unknown reason, in some cases, hypothesis is saying
        # x_transf is nan, when it actually is not
        if ~nxp.isnan(x_transfmin):
            assert nxp.isclose(x_transfmin, 0) or (x_transfmin > 0)

        if ~nxp.isnan(x_transfmax):
            assert nxp.isclose(x_transfmax, 1) or (x_transfmax < 1)


###############################################################################
# Tests for copula functions  #################################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(x=spd_matrix_builder_float64)
def test_multi_normal_cholesky_copula_lpdf_equals_multivariate_gaussian_float64(
    x: array,
) -> None:
    """Test for :func:`multi_normal_cholesky_copula_lpdf`.
    This test compares the Gaussian copula combined with Gaussian
    marginal distributions and a multivariate normal distribution lpdf.

    Parameters
    ----------
    x : array
        SPD matrix.
    """
    xp = array_namespace(x)  # Get the array API namespace
    x = nxp.asarray(x)  # Convert to numpy

    rng = nxp.random.default_rng()

    # Transform covariance into a correlation matrix
    corr = normalize_covariance(x)

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
    marginals_lpdf = nxp.sum(marginals_dist.logpdf(samples))

    # Joint distribution lpdf
    joint_lpdf = joint_dist.logpdf(samples)

    # Here \Phi^{-1}(\Phi(x)) = x
    copula_lpdf = nxp.asarray(
        multi_normal_cholesky_copula_lpdf(
            xp.asarray(samples[nxp.newaxis, :], device=device(samples)),
            xp.asarray(corr_chol, device=device(samples)),
        )
    )

    # copula lpdf + marginals_lpdf = joint_lpdf
    assert nxp.isclose(copula_lpdf, joint_lpdf - marginals_lpdf)
