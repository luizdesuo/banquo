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
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import random
from numpyro.distributions import Dirichlet
from numpyro.infer import MCMC, NUTS
from scipy import stats
from scipy.spatial.distance import jensenshannon

from banquo import (
    array,
    bernstein_pdf,
    extract_minmax_parameters,
    minmax_normalization,
    shape_handle_wT_posterior,
    shape_handle_x,
)
from banquo.numpyro import Bernstein


###############################################################################
# Numpyro configuration  ######################################################
###############################################################################


numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


###############################################################################
# Marginal distributions  #####################################################
###############################################################################


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
JSD_THRESHOLD: float = 0.05  # Threshold for Jensen-Shannon divergence (JSD)
PROB_JSD_LESS_THAN_THRESHOLD = 0.95  # Probability that lpdf of observation
# is similar to true distribution under JSD


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
@settings(deadline=None, max_examples=10)
@given(
    st.sampled_from(list(DISTRIBUTIONS.keys())),
    st.integers(min_value=5, max_value=30),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=50, max_value=200),
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
    threshold: float = JSD_THRESHOLD
    prob_jsd_less_than_threshold: float = PROB_JSD_LESS_THAN_THRESHOLD

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

    assert np.all(prob_less_than_threshold > prob_jsd_less_than_threshold)
