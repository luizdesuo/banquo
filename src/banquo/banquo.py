#!/usr/bin/env python3
"""The module contains building blocks for Nonparanormal models."""

###############################################################################
# Imports #####################################################################
###############################################################################


from dataclasses import dataclass, field
from typing import Any, Protocol

from array_api_compat import array_namespace
from array_api_compat import device as _device
from array_api_compat import is_jax_array


###############################################################################
# Namespace ###################################################################
###############################################################################


__all__ = [
    "BanquoError",
    "BetaProtocol",
    "DataMaxExceedsSupportUpperBoundError",
    "DataMinExceedsSupportLowerBoundError",
    "DataRangeExceedsSupportBoundError",
    "Device",
    "MinMaxNormalizationError",
    "add_intercept_column",
    "array",
    "bernstein_cdf",
    "bernstein_icdf",
    "bernstein_lpdf",
    "bernstein_pdf",
    "chol2inv",
    "device",
    "diag",
    "divide_ns",
    "extract_minmax_parameters",
    "homographic_ns",
    "kahan_sum",
    "logsumexp",
    "minmax_normalization",
    "multi_normal_cholesky_copula_lpdf",
    "multiply_ns",
    "normalize_covariance",
    "shape_handle_wT",
    "shape_handle_wT_posterior",
    "shape_handle_x",
    "std_ns",
]


###############################################################################
# Custom types for annotation #################################################
###############################################################################


array = Any
"""Type annotation for array objects.

    For more information, please refer to `array-api
    <https://data-apis.org/array-api/latest/API_specification/array_object.html>`__.
"""


class BetaProtocol(Protocol):
    """A protocol that defines the interface for a Beta distribution.

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

    Notes
    -----
    - The Beta distribution is a continuous probability distribution defined on
      the interval [0, 1].
    - The `a` and `b` parameters define the shape of the distribution. For instance:

      - `a = b = 1` gives a uniform distribution.
      - `a > b` gives a distribution skewed toward 1.
      - `a < b` gives a distribution skewed toward 0.

    - All methods operate on arrays, allowing for efficient vectorized
      computation of the log-pdf, pdf, cdf, and icdf across multiple
      Beta distributions and samples.

    Example
    -------
    >>> import numpy as np
    >>> from scipy.stats import beta
    >>> class ScipyBeta:
    >>>     def __init__(self, a, b):
    >>>         self.a = a
    >>>         self.b = b
    >>>
    >>>     def lpdf(self, x):
    >>>         return beta(self.a, self.b).logpdf(x)
    >>>
    >>>     def pdf(self, x):
    >>>         return beta(self.a, self.b).pdf(x)
    >>>
    >>>     def cdf(self, x):
    >>>         return beta(self.a, self.b).cdf(x)
    >>>
    >>>     def icdf(self, x):
    >>>         return beta.ppf(self.a, self.b)(x)
    >>>
    >>> beta_dist = ScipyBeta(a=np.array([2]), b=np.array([5]))
    >>> x = np.array([0.2, 0.5])
    >>> beta_dist.lpdf(x)
    array([ 0.89918526, -0.06453852])
    """

    a: array
    b: array

    def lpdf(self, x: array) -> array:
        """Calculate the log probability density function of the beta distribution."""

    def pdf(self, x: array) -> array:
        """Calculate the probability density function of the beta distribution."""

    def cdf(self, x: array) -> array:
        """Calculate the cumulative distribution function of the beta distribution."""

    def icdf(self, x: array) -> array:
        """Calculate the quantile function of the beta distribution."""


Device = Any
"""Type annotation for device objects."""

###############################################################################
# Exceptions ##################################################################
###############################################################################


class BanquoError(Exception):
    """Base class for exceptions in Banquo."""


class MinMaxNormalizationError(BanquoError):
    """Base class for min-max normalization exceptions in Banquo."""


@dataclass
class DataRangeExceedsSupportBoundError(MinMaxNormalizationError):
    """Exception for data range exceeding support's boundary."""

    support: array = field()
    data_range: array = field()
    msg_data: str = field(init=False, repr=False)
    msg_support: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Set error message's parameters."""
        self.msg_data: str = f"range {(self.data_range[0], self.data_range[1])}"
        self.msg_support: str = f"boundary {(self.support[0], self.support[1])}"

    def __str__(self) -> str:
        """Set error message."""
        return f"Data {self.msg_data} exceeds the support's {self.msg_support}."


@dataclass
class DataMinExceedsSupportLowerBoundError(DataRangeExceedsSupportBoundError):
    """Exception for data minimum exceeding support's lower bound."""

    def __post_init__(self) -> None:
        """Set error message's parameters."""
        self.msg_data: str = f"minimum {self.data_range[0]}"
        self.msg_support: str = f"lower boundary {self.support[0]}"


@dataclass
class DataMaxExceedsSupportUpperBoundError(DataRangeExceedsSupportBoundError):
    """Exception for data maximum exceeding support's upper bound."""

    def __post_init__(self) -> None:
        """Set error message's parameters."""
        self.msg_data: str = f"maximum {self.data_range[1]}"
        self.msg_support: str = f"upper boundary {self.support[1]}"


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
    xp = array_namespace(spd_chol)  # Get the array API namespace
    spd_chol_inv = xp.linalg.inv(spd_chol)
    return spd_chol_inv.T @ spd_chol_inv


def diag(x: array) -> array:
    """Generate a diagonal matrix from array `x`.

    Parameters
    ----------
    x : array
        One-dimensional array.

    Returns
    -------
    array
        Diagonal matrix from input array.
    """
    xp = array_namespace(x)  # Get the array API namespace
    n = x.shape[0]
    res = xp.zeros((n, n), dtype=x.dtype, device=device(x))
    ii = xp.arange(n, device=device(x))  # Generate indices for the diagonal
    if is_jax_array(res):
        res = res.at[ii, ii].set(x)
    else:
        res[ii, ii] = x  # Set the diagonal elements
    return res


def normalize_covariance(cov: array) -> array:
    r"""Normalize a covariance matrix.

    Assuming a covariance matrix :math:`\Sigma`, the correlation matrix
    :math:`\Omega` entries are given by:

    .. math::
        \Omega_{ij} = \Sigma_{ij}/\sqrt{\Sigma_{ii} \Sigma_{jj}}\,.

    Parameters
    ----------
    cov : array
        SPD covariance matrix

    Returns
    -------
    array
        SPD correlation matrix.
    """
    xp = array_namespace(cov)  # Get the array API namespace

    # Get the standard deviations (sqrt of diagonal elements)
    std_devs = xp.sqrt(xp.diagonal(cov))

    # Create a diagonal matrix with the reciprocal of the standard deviations
    inv_stddev_matrix = diag(1 / std_devs)

    # Transform covariance matrix to correlation matrix
    return inv_stddev_matrix @ cov @ inv_stddev_matrix


def std_ns(x: array, axis: int | None = None, keepdims: bool = False) -> array | float:
    """Numerically stable calculation of standard deviation.

    If the standard deviation tends to infinity,
    it is substituted by the interquartile range (IQR).

    Parameters
    ----------
    x : array
        Elements to extract the standard deviation.
    axis : int | None, optional
        Axis or axes along which the standard deviation calculation
        is performed. If axis is negative it counts from the last
        to the first axis, by default None will sum all of
        the elements of the input array.
    keepdims : bool, optional
        If True, retains reduced axes as dimensions with size 1,
        by default False. If False, the dimensions are removed.

    Returns
    -------
    array | float
        An array with the same shape as a, with the specified axis removed.
        If a is a 0-d array, or if axis is None, a scalar is returned.
        If an output array is specified, a reference to out is returned.
        If keepdims is True, retains reduced axes as dimensions with size 1.
    """
    xp = array_namespace(x)  # Get the array API namespace

    x_std = xp.std(x, axis=axis, keepdims=keepdims)

    if xp.any(xp.isinf(x_std)):
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        q1 = xp.percentile(x, 25, axis=axis, keepdims=keepdims)
        q3 = xp.percentile(x, 75, axis=axis, keepdims=keepdims)

        # Compute IQR (Interquartile Range)
        iqr = q3 - q1

        # Use IQR as a measure of spread instead of std
        x_std = 20 * iqr / 27  # IQR can be scaled to approximate std

    return x_std


def kahan_sum(x1: array, x2: array) -> array:
    r"""Element-wise Kahan summation algorithm.

    Given two arrays of the same shape, `x1` and `x2`, this function
    performs a numerically stable element-wise summation `x1 + x2`.
    For the subtraction :math:`x_1 - x_2`, it is sufficient to
    use :func:`kahan_sum` with parameters `x1` and `-x2`.

    Parameters
    ----------
    x1 : array
        First term.
    x2 : array
        Second term.

    Returns
    -------
    array
        Sum of x1 and x2.
    """
    temp = x1 + x2
    compensation = (
        x1 - temp
    ) + x2  # Here we perform compensation for floating-point errors
    result = temp + compensation
    return result


def divide_ns(x1: array, x2: array) -> array:
    r"""Numerically stable division.

    Given two arrays of the same shape, `x1` and `x2`, this function
    performs a numerically stable element-wise division :math:`x_1 / x_2`.
    The function relies on the formula:

    .. math::
        \frac{x_1}{x_2} = \text{sign}(x_1)\text{sign}(x_2) \exp\left(\log(\lvert x_1\rvert) - \log(\lvert x_2\rvert)\right)\,.

    Parameters
    ----------
    x1 : array
        Numerator.
    x2 : array
        Denominator.

    Returns
    -------
    array
        Quotient.
    """  # noqa: B950
    xp = array_namespace(x1, x2)  # Get the array API namespace

    # Handle signs of x1 and x2, using absolute values for log
    sign = xp.sign(x1) * xp.sign(x2)
    log_x1 = xp.log(xp.abs(x1))
    log_x2 = xp.log(xp.abs(x2))

    return sign * xp.exp(log_x1 - log_x2)


def multiply_ns(x1: array, x2: array | None = None) -> array | float:
    r"""Numerically stable multiplication.

    Given two arrays of the same shape, `x1` and `x2`, this function
    performs a numerically stable element-wise division :math:`x_1 \times x_2`.
    The function relies on the formula:

    .. math::
        x_1 \times x_2 = \text{sign}(x_1)\text{sign}(x_2) \exp\left(\log(\lvert x_1\rvert) + \log(\lvert x_2\rvert)\right)\,.

    Parameters
    ----------
    x1 : array
        Factor.
    x2 : array | None, optional
        Factor, by default None. If None, the product will be
        performed through array `x1`.

    Returns
    -------
    array | float
        Product. If `x2` is None, the product will be performed in
        array `x1`, resulting in a float. If otherwise, the function returns
        the element-wise multiplication between `x1` and `x2` resulting in an
        array of the same shape.
    """  # noqa: B950
    if x2 is None:
        xp = array_namespace(x1)  # Get the array API namespace
        sign = xp.prod(xp.sign(x1))
        log_x1 = xp.log(xp.abs(x1))
        return sign * xp.exp(xp.sum(log_x1))
    else:
        xp = array_namespace(x1, x2)  # Get the array API namespace
        sign = xp.sign(x1) * xp.sign(x2)
        log_x1 = xp.log(xp.abs(x1))
        log_x2 = xp.log(xp.abs(x2))
        return sign * xp.exp(log_x1 + log_x2)


def homographic_ns(x: array) -> array:
    r"""Numerically stable homographic function.

    Given an array `x`, this function
    performs a numerically stable calculation of :math:`1/(1+x)`.
    The function applies :func:`divide_ns` with
    :math:`x_1 = 1` and :math:`x_2 = 1 + x`.

    Parameters
    ----------
    x : array
        Elements to extract the homographic function.

    Returns
    -------
    array
        homographic function, :math:`1/(1+x)`.
    """
    xp = array_namespace(x)  # Get the array API namespace
    one = xp.ones_like(x, device=device(x))
    y = one + x
    return divide_ns(one, y)


def add_intercept_column(x: array, const: float | int = 1) -> array:
    """Include intercept column to array `x`.

    The intercept can be any constant number.

    Parameters
    ----------
    x : array
        Array to add a constant (intercept) column.
    const : float | int, optional
        constant to be included into array `x`, by default 1.

    Returns
    -------
    array
        For an input `x` with dimensions :math:`(n, d)`, it includes a
        constant column to `x`, resulting in an array with dimensions
        :math:`(n, d+1)`.
    """
    xp = array_namespace(x)  # Get the array API namespace

    n = x.shape[0]  # Get the number of rows in x

    return xp.concat(
        (const * xp.ones((n, 1), dtype=x.dtype, device=device(x)), x[:, None]),
        axis=1,
    )


def logsumexp(x: array, axis: int | None = None, keepdims: bool = False) -> array:
    """Compute the log of the sum of exponentials of input elements over a given axis.

    Parameters
    ----------
    x : array
        Input array to compute the logsumexp over.
    axis : int | None, optional
        Axis or axes along which to perform the reduction, by default None.
        If None, the operation is performed over all elements.
    keepdims : bool, optional
        If True, retains reduced axes as dimensions with size 1,
        by default False. If False, the dimensions are removed.

    Returns
    -------
    array
        An array with the same type as `x` containing the log of the
        sum of exponentials. If `keepdims` is True, the result will
        have the same number of dimensions as the input. Otherwise, the
        reduced dimensions are removed.

    Notes
    -----
    This implementation follows a numerically stable approach to compute
    the log of the sum of exponentials by:

     - Shifting input values by the maximum along the specified axis.
     - Computing the exponentials of the shifted values to avoid
       overflow or underflow.
    """
    xp = array_namespace(x)  # Get the array API namespace

    # Step 1: Compute the max value along the specified axis to stabilize the computation
    max_val = xp.max(x, axis=axis, keepdims=keepdims)

    # Step 2: Subtract the max value from the input array to stabilize the computation
    shifted_x = x - max_val

    # Step 3: Compute the sum of the exponentials of the shifted array
    exp_sum = xp.sum(xp.exp(shifted_x), axis=axis, keepdims=keepdims)

    # Step 4: Take the log of the sum and add back the max_val
    log_sum_exp = xp.log(exp_sum)

    # Return the final logsumexp result
    return log_sum_exp + max_val


def device(x: array) -> Device:
    """Wrap function device from array_api_compat.

    JIT-compiled function include JAX transforms, e.g. `DynamicJaxprTracer`,
    that has no `device` attribute. This function fill this gap
    by considering this case.

    Parameters
    ----------
    x : array
        a array object.

    Returns
    -------
    Device
        A device object.

    Notes
    -----
    - `array_api_compat.device
      <https://data-apis.org/array-api-compat/helper-functions.html#array_api_compat.device>`__.
    - `JIT mechanics
      <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables>`__.

    """
    if is_jax_array(x):
        try:
            return _device(x)
        except AttributeError:
            return None
    else:
        return _device(x)


###############################################################################
# Data transform ##############################################################
###############################################################################


def extract_minmax_parameters(x: array, support: array | None = None) -> array:
    r"""Extract the intercept and slop from `x` for the support :math:`[0, 1]`.

    These parameters can be applied into the linear transformation, given by,

    .. math::
        y = \frac{-a}{b-a} + \frac{1}{b-a} x,

    to make the data bounded by :math:`[0, 1]`. Where :math:`a` and
    :math:`b`. are given by:

    .. math::

        a & = \max\{X_{(1)} - \sqrt{S^2/n}, a'\},\\
        b & = \min\{X_{(n)} + \sqrt{S^2/n}, b'\},

    with :math:`S^2` representing the sample variance, and
    :math:`X_{(1)}` and :math:`X_{(n)}` denoting the first and last
    order statistics, respectively. In this formula :math:`x \in [a', b']`


    Parameters
    ----------
    x : array
        Elements to be transformed.
    support : array | None, optional
        Two-elements array containing the lower and upper bounds
        for the elements, by default None. If None, `support`
        is the unbounded interval :math:`(-\infty, \infty)`.

    Returns
    -------
    array
        Two-elements array containing the intercept and slope for
        a linear transformation.

    Raises
    ------
    DataRangeExceedsSupportBoundError
        If the data range exceeds support's boundary.
    DataMinExceedsSupportLowerBoundError
        If data minimum exceeds support's lower bound.
    DataMaxExceedsSupportUpperBoundError
        If data maximum exceeds support's upper bound.
    Note
    ----
    See `Unimodal density estimation using Bernstein polynomials
    <https://www.sciencedirect.com/science/article/pii/S0167947313003757>`__.
    """
    if support is None:
        xp = array_namespace(x)  # Get the array API namespace
        support = xp.asarray(
            (-xp.inf, xp.inf), device=device(x)
        )  # Default: unbounded support
    else:
        xp = array_namespace(x, support)  # Get the array API namespace

    x_range = xp.asarray((xp.min(x), xp.max(x)), device=device(x))

    condition_lower: bool = x_range[0] < support[0]
    condition_upper: bool = x_range[1] > support[1]

    # Check if data range exceeds support's boundary
    if condition_lower and condition_upper:
        raise DataRangeExceedsSupportBoundError(support, x_range)
    # Check if data minimum exceeds support's lower bound
    elif condition_lower:
        raise DataMinExceedsSupportLowerBoundError(support, x_range)
    # Check if data maximum exceeds support's upper bound
    elif condition_upper:
        raise DataMaxExceedsSupportUpperBoundError(support, x_range)

    n = x.shape[0]

    x_std = std_ns(x)

    adjustment = xp.asarray(x_std / xp.sqrt(n), device=device(x))

    # Defines the minimum value
    a_min = kahan_sum(xp.asarray(x_range[0], device=device(x)), -adjustment)
    a = xp.max(xp.asarray((a_min, support[0]), device=device(x)))

    # Defines the maximum value
    b_max = kahan_sum(xp.asarray(x_range[1], device=device(x)), adjustment)
    b = xp.min(xp.asarray((b_max, support[1]), device=device(x)))

    # b-a
    denominator = kahan_sum(b, -a)

    # -a/(b-a)
    coeff1 = divide_ns(-a, denominator)

    # 1/(b-a)
    coeff2 = divide_ns(
        xp.ones_like(denominator, device=device(denominator)), denominator
    )

    return xp.asarray((coeff1, coeff2), device=device(x))


def minmax_normalization(
    x: array, *, support: array | None = None, coeffs: array | None = None
) -> array:
    r"""Transform `x` to the range :math:`[0, 1]`.

    Linear transform is applied to `x`, given by,

    .. math::
        y = c_1 + c_2 x,

    See :func:`extract_minmax_parameters` for more information
    on how :math:`c_1` and :math:`c_2` can be calculated.

    Parameters
    ----------
    x : array
        Elements to be transformed.
    support : array | None, optional
        Two-elements array containing the lower and upper bounds
        for the elements, by default None. If None, `support`
        is the unbounded interval :math:`(-\infty, \infty)`.
    coeffs : array | None, optional
        Two-elements array containing the intercept and slope for
        a linear transformation, by default None. If None, the
        both parameters will be calculated by
        :func:`extract_minmax_parameters`.

    Returns
    -------
    array
        Elements in the the range :math:`[0, 1]`.
    """
    if coeffs is None:
        coeffs = extract_minmax_parameters(x, support)

    # Transform the data
    return add_intercept_column(x) @ coeffs


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
    xp = array_namespace(marginal, omega_chol)  # Get the array API namespace
    n, d = marginal.shape
    precision = chol2inv(omega_chol)
    log_density: float = -n * xp.sum(xp.log(xp.diagonal(omega_chol))) - 0.5 * xp.sum(
        xp.multiply(
            precision - xp.eye(d, device=device(precision)), marginal.T @ marginal
        )
    )
    return log_density


###############################################################################
# Marginal modeling ###########################################################
###############################################################################


def shape_handle_x(x: array) -> array:
    """
    Reshape observations `x` for use with Bernstein functions.

    This function reshapes the input array `x` to ensure compatibility
    with functions such as :func:`bernstein_lpdf`, :func:`bernstein_pdf`,
    and :func:`bernstein_cdf`, leveraging broadcasting and vectorized
    operations.

    Parameters
    ----------
    x : array
        An array of observations, with shape `(n, d)` where `n` is the
        number of samples, and `d` is the number of dimensions. If
        `x` is one-dimensional, it represents `n` samples with shape `(n,)`.

    Returns
    -------
    array
        Reshaped array of `x` suitable for Bernstein functions.

         - If `x` has shape `(n,)`, it will be reshaped to `(1, 1, n, 1, 1)`.
         - If `x` has shape `(n, d)`, it will be reshaped to `(1, 1, n, 1, d)`.

    Raises
    ------
    ValueError
        If `x` has more than 2 dimensions.
    """
    if x.ndim == 1:
        return x[
            None, None, :, None, None
        ]  # Shape (1, 1, n, 1, 1) for single dimension
    elif x.ndim == 2:
        return x[None, None, :, None, :]  # Shape (1, 1, n, 1, d)
    else:
        raise ValueError(
            f"Input `x` has too many dimensions (ndim={x.ndim})."
            f" Expected shape (n,) or (n, d), but received {x.shape}."
        )


def shape_handle_wT(w: array) -> array:  # noqa: N802
    """
    Reshape and transpose weights `w` for use with Bernstein functions.

    This function reshapes the weights array `w` and transposes it for
    compatibility with functions like :func:`bernstein_lpdf`, :func:`bernstein_pdf`,
    and :func:`bernstein_cdf`, enabling broadcasting.

    Parameters
    ----------
    w : array
        An array of weights with shape `(d, k)`, where `d` is the number of
        dimensions, and `k` is the number of basis functions. If `w` is one-dimensional,
        it represents `k` basis functions with shape `(k,)`.

    Returns
    -------
    array
        Reshaped and transposed array of `w` for Bernstein functions.

         - If `w` has shape `(k,)`, it will be reshaped to `(1, 1, 1, k, 1)`.
         - If `w` has shape `(d, k)`, it will be reshaped to `(1, 1, 1, k, d)`.

    Raises
    ------
    ValueError
        If `w` has more than 2 dimensions.
    """
    if w.ndim == 1:
        return w[
            None, None, None, :, None
        ]  # Shape (1, 1, 1, k, 1) for single dimension
    elif w.ndim == 2:
        return w.T[None, None, None, :, :]  # Shape (1, 1, 1, k, d)
    else:
        raise ValueError(
            f"Input `w` has too many dimensions (ndim={w.ndim})."
            f" Expected shape (k,) or (d, k), but received {w.shape}."
        )


def shape_handle_wT_posterior(w: array, chains: bool = False) -> array:  # noqa: N802
    """
    Reshape posterior weights `w` with optional chain handling for MCMC output.

    This function reshapes and transposes the input weights array `w`
    based on whether MCMC chains are included. It ensures compatibility
    with functions like :func:`bernstein_lpdf`, :func:`bernstein_pdf`,
    and :func:`bernstein_cdf`.

    Parameters
    ----------
    w : array
        An array of weights with shape `(s, k)` or `(s, d, k)` when `chains=False`.
        If `chains=True`, `w` should have shape `(c, s, k)` or `(c, s, d, k)`.

    chains : bool, optional
        Specifies if `w` includes MCMC chain data, by default False.

    Returns
    -------
    array
        Reshaped array `w` suitable for Bernstein functions.

         - If `chains=True`, reshapes `w` to `(c, s, 1, k, 1)` or `(c, s, 1, k, d)`.
         - If `chains=False`, reshapes `w` to `(1, s, 1, k, 1)` or `(1, s, 1, k, d)`.

    Raises
    ------
    ValueError
        If `w` has an incompatible number of dimensions.
    """
    if chains:
        if w.ndim < 3:
            raise ValueError(
                "Input `w` must have at least 3 dimensions when `chains=True`;"
                f" received shape {w.shape} (ndim={w.ndim})."
                " Expected shape (c, s, k) or (c, s, d, k)."
            )
        elif w.ndim == 3:
            return w[:, :, None, :, None]  # Shape (c, s, 1, k, 1)
        elif w.ndim == 4:
            xp = array_namespace(w)
            return xp.transpose(w, axes=(0, 1, 3, 2))[
                :, :, None, :, :
            ]  # Shape (c, s, 1, k, d)
        else:
            raise ValueError(
                f"Input `w` has too many dimensions (ndim={w.ndim}) for `chains=True`."
                f" Expected shape (c, s, k) or (c, s, d, k), but received {w.shape}."
            )
    else:
        if w.ndim < 2:
            raise ValueError(
                "Input `w` must have at least 2 dimensions when `chains=False`;"
                f" received shape {w.shape} (ndim={w.ndim})."
                " Expected shape (s, k) or (s, d, k)."
            )
        elif w.ndim == 2:
            return w[None, :, None, :, None]  # Shape (1, s, 1, k, 1)
        elif w.ndim == 3:
            xp = array_namespace(w)
            return xp.transpose(w, axes=(0, 2, 1))[
                None, :, None, :, :
            ]  # Shape (1, s, 1, k, d)
        else:
            raise ValueError(
                f"Input `w` has too many dimensions (ndim={w.ndim}) for `chains=False`."
                f"Expected shape (s, k) or (s, d, k), but received {w.shape}."
            )


def bernstein_lpdf(
    beta: type[BetaProtocol], x: array, w: array, keepdims: bool = False
) -> array:
    r"""Compute the lpdf for a Bernstein-Dirichlet polynomial model.

    This function evaluates the lpdf of a weighted sum of Beta distributions,
    where each Beta distribution forms a basis function in the Bernstein
    polynomial. The weights (simplex) for each basis function are
    specified by `w`, and the inputs to the Beta distributions are given by `x`.

    Considering the beta density function,

    .. math::
        b(x \mid \alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \, x^{\alpha-1}(1-x)^{\beta-1},

    with :math:`\alpha = j` and :math:`\beta=k-j+1` for :math:`k` basis functions
    and :math:`j \in \{1, \ldots k\}`. The Bernstein density approximation
    is given by:

    .. math::
        p(x) \approx \mathbf{w}^\mathrm{T} \mathbf{b}(x),

    with :math:`\mathbf{b}(x) = (b_{1,k}(x), \ldots, b_{k,k}(x))^\mathrm{T}`
    and weights :math:`\mathbf{w}` are elements in a k-dimensional simplex.
    This function returns :math:`\log(\mathbf{w}^\mathrm{T} \mathbf{b}(x))`.

    Parameters
    ----------
    beta : type[BetaProtocol]
        A class implementing the protocol that defines the interface
        for a Beta distribution for each basis function, see
        :class:`BetaProtocol`. It should accept two arguments
        `a = j` and `b = k_j`, where:

         - `j`: index of the basis function.
         - `k_j`: the complement index for the Beta distribution's
           second shape parameter.

    x : array
        An array of shape `(1, 1, n, 1, d)`, where:

         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each element represents an observation for fitting the
        Bernstein polynomial model. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_x`.
    w : array
        An array of shape `(c, s, 1, k, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `k` is the number of basis functions.
         - `d` is the number of dimensions.

        The elements of `w` are the weights  assigned to each of
        the `k` basis functions. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_wT`. In case of the weights
        are the product of a MCMC sampling algorithm, i.e., it has
        dimensions `(c, s, 1, k, d)`, it can be
        used directly into Bernstein functions.
    keepdims : bool, optional
        If True, retains reduced axes as dimensions with size 1,
        by default False. If False, the dimensions are removed.

    Returns
    -------
    array
        The log-probability density function evaluated at each observation in `x`,
        returned as an array of shape `(c, s, n, 1, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each entry corresponds to the lpdf of a sample for a specific
        dimension in the Bernstein polynomial model with parameters `w`. In case
        of `keepdims` is `False`, the axes of length one will be removed.

    Raises
    ------
    ValueError
        If `x` or `w` don't have exactly 5 dimensions.

    Notes
    -----
    - This function leverages broadcasting and reshaping to ensure that the log-pdf
      of the Beta distributions and the weight vectors can be combined element-wise
      across dimensions and samples.
    - The Beta distributions are parameterized by `j` and `k_j`, which vary across
      the number of basis functions `k`. The :func:`BetaProtocol.lpdf` method
      computes the log-pdf for the inputs in `x`.
    - The :func:`logsumexp` function is used to aggregate the weighted
      log-probabilities across the basis functions, ensuring numerical stability.
    """  # noqa: B950
    xp = array_namespace(x, w)  # Get the array API namespace

    # Shape: (c, s, n, k, d)
    if x.ndim != 5:
        raise ValueError(f"Input x must have exactly 5 dimensions, but has {x.ndim}")

    # Shape: (c, s, n, k, d)
    if w.ndim != 5:
        raise ValueError(f"Input w must have exactly 5 dimensions, but has {x.ndim}")

    # Number of Bernstein basis function for each dimension
    k = w.shape[-2]

    j = xp.arange(1, k + 1, device=device(x))  # j = 1, 2, ..., k
    k_j = k - j + 1  # k-j+1 for each j

    # Expand j and k_j for broadcasting over dimensions and samples
    j = j[None, None, None, :, None]  # Shape: (1, 1, 1, k, 1)
    k_j = k_j[None, None, None, :, None]  # Shape: (1, 1, 1, k, 1)

    # The beta parameters are broadcasted over dimensions
    beta_dist = beta(j, k_j)  # type: ignore [call-arg]

    # Compute log-pdf of the beta distribution
    beta_lpdf = beta_dist.lpdf(x)

    # Log of the weights
    w_log = xp.log(w)

    # Add log-weights to the log-pdf, equivalent to multiplication in normal scale
    weighted_lpdf = w_log + beta_lpdf

    # Compute log-sum-exp over the k basis functions
    sum_weighted_lpdf = logsumexp(weighted_lpdf, axis=-2, keepdims=True)

    if keepdims:
        return sum_weighted_lpdf
    else:
        return xp.squeeze(sum_weighted_lpdf)


def bernstein_pdf(
    beta: type[BetaProtocol], x: array, w: array, keepdims: bool = False
) -> array:
    r"""Compute the pdf for a Bernstein-Dirichlet polynomial model.

    This function evaluates the pdf of a weighted sum of Beta distributions,
    where each Beta distribution forms a basis function in the Bernstein
    polynomial. The weights (simplex) for each basis function are
    specified by `w`, and the inputs to the Beta distributions are given by `x`.
    This function exponentiate the :func:`bernstein_lpdf`.

    Considering the beta density function,

    .. math::
        b(x \mid \alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \, x^{\alpha-1}(1-x)^{\beta-1},

    with :math:`\alpha = j` and :math:`\beta=k-j+1` for :math:`k` basis functions
    and :math:`j \in \{1, \ldots k\}`. The Bernstein density approximation
    is given by:

    .. math::
        p(x) \approx \mathbf{w}^\mathrm{T} \mathbf{b}(x),

    with :math:`\mathbf{b}(x) = (b_{1,k}(x), \ldots, b_{k,k}(x))^\mathrm{T}`
    and weights :math:`\mathbf{w}` are elements in a k-dimensional simplex.

    Parameters
    ----------
    beta : type[BetaProtocol]
        A class implementing the protocol that defines the interface
        for a Beta distribution for each basis function, see
        :class:`BetaProtocol`. It should accept two arguments
        `a = j` and `b = k_j`, where:

         - `j`: index of the basis function.
         - `k_j`: the complement index for the Beta distribution's
           second shape parameter.

    x : array
        An array of shape `(1, 1, n, 1, d)`, where:

         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each element represents an observation for fitting the
        Bernstein polynomial model. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_x`.
    w : array
        An array of shape `(c, s, 1, k, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `k` is the number of basis functions.
         - `d` is the number of dimensions.

        The elements of `w` are the weights  assigned to each of
        the `k` basis functions. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_wT`. In case of the weights
        are the product of a MCMC sampling algorithm, i.e., it has
        dimensions `(c, s, 1, k, d)`, it can be
        used directly into Bernstein functions.
    keepdims : bool, optional
        If True, retains reduced axes as dimensions with size 1,
        by default False. If False, the dimensions are removed.

    Returns
    -------
    array
        The probability density function evaluated at each observation in `x`,
        returned as an array of shape `(c, s, n, 1, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each entry corresponds to the pdf of a sample for a specific
        dimension in the Bernstein polynomial model with parameters `w`. In case
        of `keepdims` is `False`, the axes of length one will be removed.

    Raises
    ------
    ValueError
        If `x` or `w` don't have exactly 5 dimensions.
    """  # noqa: B950
    xp = array_namespace(x, w)  # Get the array API namespace

    return xp.exp(bernstein_lpdf(beta, x, w, keepdims))


def bernstein_cdf(
    beta: type[BetaProtocol], x: array, w: array, keepdims: bool = False
) -> array:
    r"""Compute the cdf for a Bernstein-Dirichlet polynomial model.

    This function evaluates the cdf of a weighted sum of Beta distributions,
    where each Beta distribution forms a basis function in the Bernstein
    polynomial. The weights (simplex) for each basis function are
    specified by `w`, and the inputs to the Beta distributions are given by `x`.

    Considering the beta cumulative distribution function,

    .. math::
        B(x \mid \alpha, \beta) = \int_0^x \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \, x^{\alpha-1}(1-x)^{\beta-1} \, \mathrm{d}z,

    with :math:`\alpha = j` and :math:`\beta=k-j+1` for :math:`k` basis functions
    and :math:`j \in \{1, \ldots k\}`. The Bernstein cdf approximation
    is given by:

    .. math::
        F(x) \approx \mathbf{w}^\mathrm{T} \mathbf{B}(x),

    with :math:`\mathbf{B}(x) = (B_{1,k}(x), \ldots, B_{k,k}(x))^\mathrm{T}`
    and weights :math:`\mathbf{w}` are elements in a k-dimensional simplex.

    Parameters
    ----------
    beta : type[BetaProtocol]
        A class implementing the protocol that defines the interface
        for a Beta distribution for each basis function, see
        :class:`BetaProtocol`. It should accept two arguments
        `a = j` and `b = k_j`, where:

         - `j`: index of the basis function.
         - `k_j`: the complement index for the Beta distribution's
           second shape parameter.

    x : array
        An array of shape `(1, 1, n, 1, d)`, where:

         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each element represents an observation for fitting the
        Bernstein polynomial model. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_x`.
    w : array
        An array of shape `(c, s, 1, k, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `k` is the number of basis functions.
         - `d` is the number of dimensions.

        The elements of `w` are the weights  assigned to each of
        the `k` basis functions. The other dimensions
        assigned with 1 are for array broadcasting,
        see :func:`shape_handle_wT`. In case of the weights
        are the product of a MCMC sampling algorithm, i.e., it has
        dimensions `(c, s, 1, k, d)`, it can be
        used directly into Bernstein functions.
    keepdims : bool, optional
        If True, retains reduced axes as dimensions with size 1,
        by default False. If False, the dimensions are removed.

    Returns
    -------
    array
        The cumulative distribution function evaluated at each observation in `x`,
        returned as an array of shape `(c, s, n, 1, d)`, where:

         - `c` is the number of MCMC chains.
         - `s` is the number of MCMC samples per chain.
         - `n` is the number of samples.
         - `d` is the number of dimensions.

        Each entry corresponds to the cdf of a sample for a specific
        dimension in the Bernstein polynomial model with parameters `w`. In case
        of `keepdims` is `False`, the axes of length one will be removed.

    Raises
    ------
    ValueError
        If `x` or `w` don't have exactly 5 dimensions.
    """  # noqa: B950
    xp = array_namespace(x, w)  # Get the array API namespace

    # Shape: (c, s, n, k, d)
    if x.ndim != 5:
        raise ValueError(f"Input x must have exactly 5 dimensions, but has {x.ndim}")

    # Shape: (c, s, n, k, d)
    if w.ndim != 5:
        raise ValueError(f"Input w must have exactly 5 dimensions, but has {x.ndim}")

    # Number of Bernstein basis function for each dimension
    k = w.shape[-2]

    j = xp.arange(1, k + 1, device=device(x))  # j = 1, 2, ..., k
    k_j = k - j + 1  # k-j+1 for each j

    # Expand j and k_j for broadcasting over dimensions and samples
    j = j[None, None, None, :, None]  # Shape: (1, 1, 1, k, 1)
    k_j = k_j[None, None, None, :, None]  # Shape: (1, 1, 1, k, 1)

    # The beta parameters are broadcasted over dimensions
    beta_dist = beta(j, k_j)  # type: ignore [call-arg]

    # Compute cdf of the beta distribution
    beta_cdf = beta_dist.cdf(x)

    # Multiply weights to the cdf
    weighted_cdf = w * beta_cdf

    # Compute sum over the k basis functions
    sum_weighted_cdf = xp.sum(weighted_cdf, axis=-2, keepdims=True)

    if keepdims:
        return sum_weighted_cdf
    else:
        return xp.squeeze(sum_weighted_cdf)


def bernstein_icdf(
    beta: type[BetaProtocol], x: array, w: array, keepdims: bool = False
) -> array:
    r"""Compute the icdf for a Bernstein-Dirichlet polynomial model."""
    raise NotImplementedError
