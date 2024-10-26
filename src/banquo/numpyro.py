#!/usr/bin/env python3
"""The module contains Numpyro Nonparanormal models implementation."""

###############################################################################
# Imports #####################################################################
###############################################################################


from dataclasses import dataclass, field

import numpyro
from array_api_compat import array_namespace
from numpyro.distributions import Beta, Dirichlet

from banquo import array, bernstein_lpdf, shape_handle_wT, shape_handle_x


###############################################################################
# Numpyro Interface ###########################################################
###############################################################################


@dataclass
class NumpyroBeta:
    """A Numpyro interface for a Beta distribution.

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
        return Beta(self.a, self.b).log_prob(x)

    def pdf(self, x: array) -> array:
        """Calculate the probability density function of the beta distribution."""
        xp = array_namespace(x)  # Get the array API namespace
        return xp.exp(self.lpdf(x))

    def cdf(self, x: array) -> array:
        """Calculate the cumulative distribution function of the beta distribution."""
        return Beta(self.a, self.b).cdf(x)

    def icdf(self, x: array) -> array:
        """Calculate the quantile function of the beta distribution."""
        return Beta(self.a, self.b).icdf(x)


###############################################################################
# Models ######################################################################
###############################################################################


def bernstein_log_prob(x: array, zeta: array) -> None:
    """Compute the Bernstein polynomial-based lpdf in a Numpyro model.

    This function samples weights `w` from a Dirichlet distribution with
    concentration parameter `zeta` and computes the log-probability of
    observing the data `x` using a Bernstein polynomial likelihood
    function. It adds the resulting lpdf as a factor to the joint
    log-probability in a NumPyro model. This function does not return any value.
    Instead, it affects the state of the NumPyro probabilistic model by
    adding `w` to the model trace and modifying the joint log-probability.

    Parameters
    ----------
    x : array
        The observed data or values to evaluate the lpdf. The array should
        have shape `(n, d)`, where `n` is the number of samples,
        and `d` is the number of dimensions. If `x` is one-dimensional
        with shape `(n,)`, it will be reshaped to `(n, 1)`. Each element
        represents a sample to be evaluated under the Bernstein polynomial
        model.
    zeta : array
        The concentration parameter of the Dirichlet distribution.
        It controls the distribution of the simplex weights `w`.
        The array should have shape `(d, k)`, where `k` is the number of
        basis functions (order of the Bernstein polynomial) and `d` the
        number of dimensions. If the shape is `k`, the system will be
        considered as a one-dimensional array.

    Side Effects
    ------------
    - Samples weights `w` from a Dirichlet distribution using
      :func:`numpyro.sample`.
    - Adds a log-probability factor to the joint model log-probability
      using :func:`numpyro.factor`. The factor is computed using the
      :func:`bernstein_lpdf` function.

    Notes
    -----
    - The Dirichlet distribution samples weights `w` sum to 1 across
      across each dimension. They are used as coefficients for the
      Bernstein polynomial model.
    - The :func:`bernstein_lpdf` function is called to compute the
      log-likelihood of the observed data `x` given the sampled weights `w`.
      This likelihood function relies on a Beta distribution
      parameterization.
    """
    # Sample 'w' from the Dirichlet distribution
    w = numpyro.sample("w", Dirichlet(zeta))

    # Add a factor that affects the joint log-probability
    numpyro.factor(
        "bernstein_lpdf",
        bernstein_lpdf(NumpyroBeta, shape_handle_x(x), shape_handle_wT(w)),
    )
