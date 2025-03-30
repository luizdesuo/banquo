#!/usr/bin/env python3
"""The module is a test suite for numpyro submodule."""


###############################################################################
# Imports #####################################################################
###############################################################################


from dataclasses import dataclass, field
from typing import Any, no_type_check

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from arviz import hdi
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import random
from numpyro.distributions import Dirichlet, LKJCholesky
from numpyro.infer import MCMC, NUTS
from scipy import stats
from scipy.spatial.distance import jensenshannon

from banquo.banquo import (
    array,
    bernstein_cdf,
    bernstein_pdf,
    extract_minmax_parameters,
    minmax_normalization,
    normalize_covariance,
    shape_handle_wT_posterior,
    shape_handle_x,
)
from banquo.numpyro import Bernstein, NonparanormalBernstein

from .hypothesis_arrays_strategy import spd_square_matrix_builder_float64


###############################################################################
# Numpyro configuration  ######################################################
###############################################################################


numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


###############################################################################
# Marginal distributions  #####################################################
###############################################################################


MAX_EXAMPLES = 10

DISTRIBUTIONS: dict[str, Any] = {
    "norm": stats.norm(0, 1),
    "genextreme": stats.genextreme(c=0, loc=0, scale=1),
    "expon": stats.expon(1),
    "beta": stats.beta(10, 10),
}

SUPPORTS: dict[str, Any] = {
    "norm": np.asarray((-np.inf, np.inf)),
    "genextreme": np.asarray((-np.inf, np.inf)),
    "expon": np.asarray((0, np.inf)),
    "beta": np.asarray((0, 1)),
}

SEED: int = 37  # Seed for pseudo-random generation

# Adaptive threshold for Jensen-Shannon divergence (JSD) for pdf
# and Kolmogorov-Smirnov metrics (KS) for cdf
THRESHOLD_MIN: float = 0.05
THRESHOLD_MAX: float = 0.40

# Adaptative probability threshold for JSD for pdf and KS for cdf
PROB_THRESHOLD_MIN = 0.70
PROB_THRESHOLD_MAX = 0.95


K_MIN = 10  # Minimum number of basis functions
K_MAX = 30  # Maximum number of basis functions
N_MIN = 100  # Minimum number of samples
N_MAX = 300  # Maximum number of samples
D_MIN = 1  # Minimum number of dimensions
D_MAX = 5  # Maximum number of dimensions


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


spd_matrix_builder_float64 = spd_square_matrix_builder_float64(size=D_MAX)

###############################################################################
# Auxiliary functions  ########################################################
###############################################################################


def adapt_thresholds(
    k: int, n: int, k_weight: float = 0.5, n_weight: float = 0.5
) -> tuple[float, float]:
    """Calculate adaptive threshold and probability threshold.

    This function uses linear interpolation between minimum and maximum values
    for threshold and probability thresholds, based on the positions of `k`
    and `n` within their defined ranges. The influence
    of `k` and `n` can be adjusted with weights.

    Parameters
    ----------
    k : int
        Number of Bernstein basis functions, expected to be in the range
        [K_MIN, K_MAX].
    n : int
        Number of samples, expected to be in the range [N_MIN, N_MAX].
    k_weight : float, optional
        The weight assigned to k's influence on the adaptive
        thresholds, by default 0.5.
    n_weight : float, optional
        The weight assigned to n's influence on the adaptive
        thresholds, by default 0.5.

    Returns
    -------
    tuple[float, float]
        tuple containing:
        - threshold (float): The adaptive threshold based on k and n.
        - prob_threshold (float): The adaptive probability threshold
          based on k and n.
    """
    # Ensure weights sum to 1
    total_weight = k_weight + n_weight
    k_weight /= total_weight
    n_weight /= total_weight

    # Calculate scale factors for k and n within their respective ranges
    k_scale = (k - K_MIN) / (K_MAX - K_MIN)
    n_scale = (n - N_MIN) / (N_MAX - N_MIN)

    # Combine scales with weights
    combined_scale = k_scale * k_weight + n_scale * n_weight

    # Calculate the adaptive threshold and probability threshold based on the combined scale
    threshold = THRESHOLD_MAX - (THRESHOLD_MAX - THRESHOLD_MIN) * combined_scale
    prob_threshold = (
        PROB_THRESHOLD_MIN + (PROB_THRESHOLD_MAX - PROB_THRESHOLD_MIN) * combined_scale
    )

    return threshold, prob_threshold


###############################################################################
# Auxiliary objects for marginal models  ######################################
###############################################################################


@dataclass
class ScipyBeta:
    """A Scipy interface for a Beta distribution.

    This protocol outlines the required attributes and methods for working
    with a Beta distribution, including the log probability density function
    (lpdf), probability density function (pdf),cumulative distribution
    function (cdf) and inverse cumulative distribution function (icdf)
    or quantile function.

    Parameters
    ----------
    a : array
        The first shape parameter (alpha) of the Beta distribution.
        It is an array to allow for vectorized operations over multiple
        distributions.
    b: array
        The second shape parameter (beta) of the Beta distribution.
        Similar to `a`, it is an array to allow for vectorized operations
        over multiple distributions.
    """

    a: array = field()
    b: array = field()

    def lpdf(self, x: array) -> array:
        """Calculate the log probability density function of the beta distribution."""
        return stats.beta(self.a, self.b).logpdf(x)

    def pdf(self, x: array) -> array:
        """Calculate the probability density function of the beta distribution."""
        return stats.beta(self.a, self.b).pdf(x)

    def cdf(self, x: array) -> array:
        """Calculate the cumulative distribution function of the beta distribution."""
        return stats.beta(self.a, self.b).cdf(x)

    def icdf(self, x: array) -> array:
        """Calculate the quantile function of the beta distribution."""
        return stats.beta.ppf(self.a, self.b)(x)


def bernstein_density_model(x: array, k: int) -> None:
    """Numpyro fixture to use in bernstein density tests."""

    n, d = x.shape

    zeta = 0.3 * jnp.ones((d, k))

    # Define the Bernstein-Dirichlet distribution
    w = numpyro.sample("w", Dirichlet(zeta))

    # Sample x from the Bernstein-Dirichlet density
    numpyro.sample("marginals", Bernstein(w), obs=x)


###############################################################################
# Tests for marginal models  ##################################################
###############################################################################


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    dist_name=st.sampled_from(list(DISTRIBUTIONS.keys())),
    k=st.integers(min_value=K_MIN, max_value=K_MAX),
    d=st.integers(min_value=D_MIN, max_value=D_MAX),
    n=st.integers(min_value=N_MIN, max_value=N_MAX),
)
def test_bernstein_density_approximation(
    dist_name: str, k: int, d: int, n: int
) -> None:
    """Test if bernstein density approximate true pdf for different supports.

    Parameters
    ----------
    dist_name : str
        Distribution name.
    k : int
        Number of Bernstein basis.
    d : int
        Number of dimensions.
    n : int
        Number of samples.
    """
    seed: int = SEED

    threshold = THRESHOLD_MIN
    prob_threshold = PROB_THRESHOLD_MAX

    dist = DISTRIBUTIONS[dist_name]
    support = SUPPORTS[dist_name]

    np.random.seed(seed=seed)
    sample = dist.rvs((n, d))

    coeffs = np.apply_along_axis(
        extract_minmax_parameters, 0, sample, support=support
    )  # Shape: (2, d)

    norm_coeff = coeffs[1, :]

    norm_coeff = norm_coeff[None, None, :]  # Shape: (1, 1, d)

    x = np.apply_along_axis(minmax_normalization, 0, sample)  # Shape: (n, d)

    rng_key = random.PRNGKey(seed=seed)

    num_samples = 2000
    num_chains = 4

    mcmc = MCMC(
        NUTS(bernstein_density_model),
        num_warmup=num_samples // 2,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(rng_key, x=jnp.asarray(x), k=k)

    posterior_samples = mcmc.get_samples()

    w = np.asarray(posterior_samples["w"])  # Shape: (s, d, k)

    pdf = np.squeeze(
        bernstein_pdf(
            ScipyBeta, shape_handle_x(x), shape_handle_wT_posterior(w), keepdims=True
        ),
        axis=(0, 3),
    )  # Shape: (s, n, d)

    pdf_norm = pdf * norm_coeff  # Shape: (s, n, d)

    res = dist.pdf(sample)[None, :, :]  # Shape: (1, n, d)

    jsd = jensenshannon(pdf_norm, res, axis=1) ** 2  # Shape: (s, d)

    prob_less_than_threshold = np.mean(jsd < threshold, axis=0)  # Shape: (d)

    assert np.all(prob_less_than_threshold > prob_threshold)


@pytest.mark.skip("Test taking too long to finish")
@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    dist_name=st.sampled_from(list(DISTRIBUTIONS.keys())),
    k=st.integers(min_value=K_MIN, max_value=K_MAX),
    d=st.integers(min_value=D_MIN, max_value=D_MAX),
    n=st.integers(min_value=N_MIN, max_value=N_MAX),
)
def test_bernstein_cdf_approximation(dist_name: str, k: int, d: int, n: int) -> None:
    """Test if bernstein cdf approximate true cdf for different supports.

    Parameters
    ----------
    dist_name : str
        Distribution name.
    k : int
        Number of Bernstein basis.
    d : int
        Number of dimensions.
    n : int
        Number of samples.
    """
    seed: int = SEED

    threshold, prob_threshold = adapt_thresholds(k, n, k_weight=0.4, n_weight=0.6)

    dist = DISTRIBUTIONS[dist_name]
    support = SUPPORTS[dist_name]

    np.random.seed(seed=seed)
    sample = dist.rvs((n, d))

    coeffs = np.apply_along_axis(
        extract_minmax_parameters, 0, sample, support=support
    )  # Shape: (2, d)

    norm_coeff = coeffs[1, :]

    norm_coeff = norm_coeff[None, None, :]  # Shape: (1, 1, d)

    x = np.apply_along_axis(minmax_normalization, 0, sample)  # Shape: (n, d)

    rng_key = random.PRNGKey(seed=seed)

    num_samples = 2000
    num_chains = 4

    mcmc = MCMC(
        NUTS(bernstein_density_model),
        num_warmup=num_samples // 2,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(rng_key, x=jnp.asarray(x), k=k)

    posterior_samples = mcmc.get_samples()

    w = np.asarray(posterior_samples["w"])  # Shape: (s, d, k)

    cdf = np.squeeze(
        bernstein_cdf(
            ScipyBeta, shape_handle_x(x), shape_handle_wT_posterior(w), keepdims=True
        ),
        axis=(0, 3),
    )  # Shape: (s, n, d)

    res = dist.cdf(sample)[None, :, :]  # Shape: (1, n, d)

    ks, _ = stats.ks_2samp(cdf, res, axis=1)  # Shape: (s, d)

    prob_less_than_threshold = np.mean(ks < threshold, axis=0)  # Shape: (d)

    assert np.all(prob_less_than_threshold > prob_threshold)


###############################################################################
# Auxiliary objects for Nonparanormal models  #################################
###############################################################################


def nonparanormal_bernstein_model(x: array, k: int) -> None:
    """Numpyro fixture to use in Nonparanormal-bernstein density tests."""

    n, d = x.shape

    zeta = 0.3 * jnp.ones((d, k))

    # Define the Bernstein-Dirichlet distribution
    w = numpyro.sample("w", Dirichlet(zeta))

    corr_chol = numpyro.sample("corr_chol", LKJCholesky(d))

    # Sample x from the NonparanormalBernstein density
    numpyro.sample(
        "NonparanormalBernstein",
        NonparanormalBernstein(weights=w, correlation_cholesky=corr_chol),
        obs=x,
    )


###############################################################################
# Tests for Nonparanormal models  #############################################
###############################################################################


@pytest.mark.skip("Test taking too long to finish")
@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    cov=spd_matrix_builder_float64,
    dist_name=st.sampled_from(list(DISTRIBUTIONS.keys())),
    k=st.integers(min_value=K_MIN, max_value=K_MAX),
    n=st.integers(min_value=N_MIN, max_value=N_MAX),
)
def test_nonparanormal_bernstein_corr_recovery(
    cov: array, dist_name: str, k: int, n: int
) -> None:
    """Test parameters recovery from NonparanormalBernstein model."""
    seed: int = SEED

    threshold = THRESHOLD_MIN
    prob_threshold = PROB_THRESHOLD_MAX

    dist = DISTRIBUTIONS[dist_name]
    support = SUPPORTS[dist_name]

    # Transform covariance into a correlation matrix
    corr = normalize_covariance(cov)

    # Get the Cholesky factor of corr matrix
    res = np.linalg.cholesky(corr)

    d = corr.shape[0]  # number of dimensions

    # Sample data from multivariate normal
    Z = np.random.RandomState(seed=seed).normal(size=(n, d))
    U = stats.norm.cdf(Z @ res)

    np.random.seed(seed=seed)
    sample = dist.ppf(U)

    coeffs = np.apply_along_axis(
        extract_minmax_parameters, 0, sample, support=support
    )  # Shape: (2, d)

    norm_coeff = coeffs[1, :]

    norm_coeff = norm_coeff[None, None, :]  # Shape: (1, 1, d)

    x = np.apply_along_axis(minmax_normalization, 0, sample)  # Shape: (n, d)

    rng_key = random.PRNGKey(seed=seed)

    num_samples = 2000
    num_chains = 4

    mcmc = MCMC(
        NUTS(nonparanormal_bernstein_model),
        num_warmup=num_samples // 2,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(rng_key, x=jnp.asarray(x), k=k)

    posterior_samples = mcmc.get_samples()

    w = np.asarray(posterior_samples["w"])  # Shape: (s, d, k)

    pdf = np.squeeze(
        bernstein_pdf(
            ScipyBeta, shape_handle_x(x), shape_handle_wT_posterior(w), keepdims=True
        ),
        axis=(0, 3),
    )  # Shape: (s, n, d)

    pdf_norm = pdf * norm_coeff  # Shape: (s, n, d)

    pdf_res = dist.pdf(sample)[None, :, :]  # Shape: (1, n, d)

    jsd = jensenshannon(pdf_norm, pdf_res, axis=1) ** 2  # Shape: (s, d)

    prob_less_than_threshold = np.mean(jsd < threshold, axis=0)  # Shape: (d)

    assert np.all(prob_less_than_threshold > prob_threshold)

    corr_chol = np.asarray(posterior_samples["corr_chol"])  # Shape: (s, d, d)

    hdi_corr_chol = hdi(corr_chol[None, :, :, :])  # Shape: (d, d, 2)

    leq = (res > hdi_corr_chol[:, :, 0]) | np.isclose(res, hdi_corr_chol[:, :, 0])

    geq = (res < hdi_corr_chol[:, :, 1]) | np.isclose(res, hdi_corr_chol[:, :, 1])

    assert np.all(leq & geq)
