#!/usr/bin/env python3
"""The module contains Numpyro Nonparanormal models implementation."""

###############################################################################
# Imports #####################################################################
###############################################################################


from dataclasses import dataclass, field

import jax.random as random
from array_api_compat import array_namespace
from numpyro.distributions import Beta, Distribution, GaussianCopula, constraints
from numpyro.util import is_prng_key

from banquo import (
    array,
    bernstein_cdf,
    bernstein_icdf,
    bernstein_lpdf,
    shape_handle_wT,
    shape_handle_x,
)


###############################################################################
# Namespace ###################################################################
###############################################################################


__all__ = ["Bernstein", "NonparanormalBernstein"]


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


# TODO: Combine this with Dirichlet distribution
class Bernstein(Distribution):  # type: ignore
    """Bernstein polynomial-based model.

    The `Bernstein` class implements a probability distribution
    for nonparametric modeling densities with Bernstein polynomials.
    It uses a set of `weights` as coefficients for the basis functions
    and supports computing the log-probability density and cumulative
    distribution function (CDF).

    Parameters
    ----------
    weights : array
        Array of weights (simplex) with shape `(d, k)`, where `d` is the number
        of dimensions, and `k` is the number of basis functions (Bernstein
        polynomial order). If the shape is `k`, the system will be considered
        as a one-dimensional array. The weights are, for each dimension `d`,
        a `k`-dimensional unit simplex. The weights are applied as coefficients
        for the Bernstein polynomial basis in each dimension.

    validate_args : bool | None, optional
        If True, validates the input parameters. By default, None (no
        validation is applied).

    Raises
    ------
    ValueError
        If `weights` is not at least one-dimensional.

    Notes
    -----
    - :func:`Bernstein.log_prob` and :func:`Bernstein.cdf` use
      :func:`banquo.bernstein_lpdf` and :func:`banquo.bernstein_cdf` respectively, with
      Beta distribution based Bernstein basis functions.
    - The weights must sum to 1 across each dimension, fulfilling the
      Dirichlet distribution constraint.
    """

    arg_constraints = {
        "weights": constraints.simplex,
    }
    reparametrized_params = ["weights"]
    support = constraints.unit_interval

    def __init__(self, weights: array, *, validate_args: bool | None = None) -> None:
        """Init Bernstein model."""
        if weights.ndim < 1:
            raise ValueError("`weights` parameter must be at least one-dimensional.")

        self.weights = weights

        # batch_shape = d, event_shape = k
        batch_shape, _ = weights.shape[:-1], weights.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args,
        )

    def sample(self, key: random.PRNGKey, sample_shape: tuple[int, ...] = ()) -> array:
        """Sample from Bernstein distribution.

        Returns a sample from the distribution with the shape specified
        by `sample_shape + batch_shape + event_shape`. Exchangeable draws from
        the distribution instance will fill the leading dimensions
        (of size `sample_shape`) of the returned sample when `sample_shape`
        is not empty.

        Parameters
        ----------
        key : random.PRNGKey
            The rng_key to be used for generating random numbers.
        sample_shape : tuple[int, ...], optional
            The desired shape of independent samples to draw, by default ().
            If it is set to an empty tuple, it corresponds to a single sample.

        Returns
        -------
        array
            Samples drawn from the distribution with shape
            `sample_shape + self.batch_shape + self.event_shape`.

        Raises
        ------
        ValueError
            If the provided key is not a valid PRNG key.
        """
        if not is_prng_key(key):
            raise ValueError("The provided key must be a valid PRNG key.")
        random_shape = sample_shape + self.batch_shape + self.event_shape
        unifs = random.uniform(key, shape=random_shape)
        return self.icdf(unifs)

    def log_prob(self, value: array) -> array:
        """Compute the lpdf of `value` using the Bernstein polynomial model.

        Parameters
        ----------
        value : array
            The observed data or values to evaluate the lpdf. The array should
            have shape `(n, d)`, where `n` is the number of samples,
            and `d` is the number of dimensions. If `value` is one-dimensional
            with shape `(n,)`, it will be reshaped to `(n, 1)`. Each element
            represents a sample to be evaluated under the Bernstein polynomial
            model.

        Returns
        -------
        array
            The log-probability density function of `value`
            under the Bernstein model.

        Notes
        -----
        The :func:`banquo.bernstein_lpdf` function is used to calculate the lpdf,
        based on Beta distribution for Bernstein basis functions.
        """
        return bernstein_lpdf(
            NumpyroBeta, shape_handle_x(value), shape_handle_wT(self.weights)
        )

    def cdf(self, value: array) -> array:
        """Compute the cdf of `value` using the Bernstein polynomial model.

        Parameters
        ----------
        value : array
            The observed data or values to evaluate the cdf. The array should
            have shape `(n, d)`, where `n` is the number of samples,
            and `d` is the number of dimensions. If `value` is one-dimensional
            with shape `(n,)`, it will be reshaped to `(n, 1)`. Each element
            represents a sample to be evaluated under the Bernstein polynomial
            model.

        Returns
        -------
        array
            The cumulative distribution function of `value`
            under the Bernstein model.

        Notes
        -----
        The :func:`banquo.bernstein_cdf` function is used to calculate the cdf,
        based on Beta distribution for Bernstein basis functions.
        """
        return bernstein_cdf(
            NumpyroBeta, shape_handle_x(value), shape_handle_wT(self.weights)
        )

    def icdf(self, value: array) -> array:
        """Compute the icdf of `value` using the Bernstein polynomial model.

        Parameters
        ----------
        value : array
            The observed data or values to evaluate the icdf. The array should
            have shape `(n, d)`, where `n` is the number of samples,
            and `d` is the number of dimensions. If `value` is one-dimensional
            with shape `(n,)`, it will be reshaped to `(n, 1)`. Each element
            represents a value in the interval [0, 1] to return
            quantile under the Bernstein polynomial model.

        Returns
        -------
        array
            The quantile function of `value` under the Bernstein model.

        Notes
        -----
        The :func:`banquo.bernstein_icdf` function is used to calculate the icdf,
        based on Beta distribution for Bernstein basis functions.
        """
        return bernstein_icdf(
            NumpyroBeta, shape_handle_x(value), shape_handle_wT(self.weights)
        )


class NonparanormalBernstein(GaussianCopula):  # type: ignore
    r"""Nonparanormal model with Bernstein polynomial-based marginals.

    The `NonparanormalBernstein` class implements a probability distribution
    for nonparametric marginal densities modeling with Bernstein polynomials.
    It uses a set of `weights` as coefficients for the basis functions
    and supports computing the log-probability density and cumulative
    distribution function (CDF). A Gaussian copula is used to create the
    joint distribution. The nonparanormal model is formed up of both
    the nonparametric marginals and the Gaussian copula, and its probability
    density function is as follows:

    .. math::
        p(\mathbf{x}) = \lvert\mathbf{\Sigma}\rvert^{-1/2} \, \exp \left( -\frac{1}{2} \mathbf{\Psi}(\mathbf{x})^\mathrm{T} \, \left(\mathbf{\Sigma}^{-1} - \mathbf{I}\right) \, \mathbf{\Psi}(\mathbf{x})\right) \prod_{i=1}^d \lvert p(x_i)\rvert,

    with :math:`\mathbf{\Psi}(\mathbf{x}) = (\Psi_1(x_1), \ldots, \Psi_d(x_d))^\mathrm{T}`,
    where :math:`\Psi_i = \Phi^{-1} \circ F_i`. In this case, :math:`F_i` are
    Bernstein cdf estimators given by:

    .. math::
        F_i(x) \approx \mathbf{w_i}^\mathrm{T} \mathbf{B}(x),

    and the marginal density estimators are given by:

    .. math::
        p_i(x) = \approx \mathbf{w_i}^\mathrm{T} \mathbf{\beta}(x).

    Both functions are implemented by :class:`Bernstein` model.

    Parameters
    ----------
    weights : array
        Array of weights (simplex) with shape `(d, k)`, where `d` is the number
        of dimensions, and `k` is the number of basis functions (Bernstein
        polynomial order). If the shape is `k`, the system will be considered
        as a one-dimensional array. The weights are, for each dimension `d`,
        a `k`-dimensional unit simplex. The weights are applied as coefficients
        for the Bernstein polynomial basis in each dimension.
    correlation_matrix : array | None, optional
            Correlation matrix of coupling multivariate normal
            distribution, by default None.
    correlation_cholesky : array | None, optional
        Correlation Cholesky factor of coupling multivariate normal
        distribution, by default None.
    validate_args : bool | None, optional
        If True, validates the input parameters. By default, None (no
        validation is applied).

    Raises
    ------
    ValueError
        If `weights` is not at least one-dimensional.
    """  # noqa: B950

    arg_constraints = {
        "weights": constraints.simplex,
        "correlation_matrix": constraints.corr_matrix,
        "correlation_cholesky": constraints.corr_cholesky,
    }
    support = constraints.independent(constraints.unit_interval, 1)
    pytree_data_fields = "weights"

    def __init__(
        self,
        weights: array,
        correlation_matrix: array | None = None,
        correlation_cholesky: array | None = None,
        *,
        validate_args: bool | None = None,
    ):
        """Init NonparanormalBernstein model."""
        # set initially to allow argument validation
        self.weights = weights

        super().__init__(
            Bernstein(weights),
            correlation_matrix,
            correlation_cholesky,
            validate_args=validate_args,
        )
