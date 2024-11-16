#!/usr/bin/env python3
"""The module contains power spectral densities and kernels implementation."""

###############################################################################
# Imports #####################################################################
###############################################################################


from array_api_compat import array_namespace

from banquo import array


###############################################################################
# Auxiliary functions #########################################################
###############################################################################


def squared_fractional_graph_laplacian(
    eigenvalues: array, kappa: float, alpha: float
) -> array:
    r"""Compute the squared fractional graph Laplacian.

    This function applies the squared fractional transformation to the graph
    Laplacian `eigenvalues` :math:`\lambda` with a shifting factor `kappa`
    and exponent `alpha`, which must both be positive. The fractional operator
    eigenvalues are given by,


    .. math::
        \widetilde{\lambda} = \left(\kappa^2 \mathbf{I} + \lambda\right)^{\alpha}.

    Parameters
    ----------
    eigenvalues : array
        Array of eigenvalues of the graph Laplacian matrix.
    kappa : float
        Shifting factor applied to the eigenvalues, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.

    Returns
    -------
    array
        Transformed eigenvalues using the squared fractional Laplacian
        formula.

    Raises
    ------
    ValueError
        If `kappa` or `alpha` is less than or equal to zero.

    Notes
    -----
    - `Non-separable Spatio-temporal Graph Kernels via SPDEs
      <https://proceedings.mlr.press/v151/nikitin22a>`__.
    """
    if kappa <= 0:
        raise ValueError("Parameter `kappa` must be positive.")
    if alpha <= 0:
        raise ValueError("Parameter `alpha` must be positive.")

    return (kappa**2 + eigenvalues) ** (alpha)


def flat_index(
    node_i: int, node_j: int, time_i: int, time_j: int, dim_t: int
) -> tuple[int, int]:
    """Calculate flattened indices for combining node and time indices.

    Given node indices and time indices, this function returns the equivalent
    flat indices for a 2D matrix where each node spans `dim_t` time steps.

    Parameters
    ----------
    node_i : int
        Index of the first node.
    node_j : int
        Index of the second node.
    time_i : int
        Index of the time step for the first node.
    time_j : int
        Index of the time step for the second node.
    dim_t : int
        Number of time steps for each node.

    Returns
    -------
    tuple[int, int]
        Flattened indices corresponding to the combined node-time positions.
    """
    return node_i * dim_t + time_i, node_j * dim_t + time_j


###############################################################################
# Discrete stochastic heat equation kernel  ###################################
###############################################################################


def psd_diag_discrete_stochastic_heat_equation(
    eta: tuple[array, array], tau: float, gamma: float, kappa: float, alpha: float
) -> array:
    r"""Compute the PSD for the discrete stochastic heat equation.

    This function calculates the power spectral density (PSD) based on the
    eigenvalues associated with spatial and temporal frequencies of the system.
    The following is the expression for the stochastic heat equation PSD:

    .. math::
        S_\mathcal{X}(\omega_t, \lambda_s) = \left[(2\pi)^{d+1} \tau^2 \left(\lvert\omega_t\rvert^2 + \lambda_s^2\right)\right]^{-1},

    where :math:`\lambda_s` are eigenvalues of the graph Laplacian.

    This function uses the :func:`squared_fractional_graph_laplacian`
    for computing the fractional graph Laplacian through the
    graph Laplacian eigenvalues.

    Parameters
    ----------
    eta : tuple[array, array]
        Tuple containing the eigenvalues for time (`eta[1]`)
        and graph Laplacian (`eta[0]`).
    tau : float
        Precision parameter, must be positive.
    gamma : float
        Medium's (thermal) diffusivity, must be positive.
    kappa : float
        Shifting factor applied to the spatial eigenvalues of the graph
        Laplacian, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.

    Returns
    -------
    array
        Array of PSD values for each combination of spatial and temporal
        frequency, representing the power contribution of each mode in
        the heat equation. It must be combined with the respective
        eigenvectors so it becomes a kernel.

    Raises
    ------
    ValueError
        If any of `tau`, `gamma`, `kappa` or `alpha` is less
        than or equal to zero.
    """  # noqa: B950
    if tau <= 0:
        raise ValueError("Parameter `tau` must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter `gamma` must be positive.")
    if kappa <= 0:
        raise ValueError("Parameter `kappa` must be positive.")
    if alpha <= 0:
        raise ValueError("Parameter `alpha` must be positive.")

    # omega = (omega_time, omega_space)
    xp = array_namespace(*eta)  # Get the array API namespace

    omega_space_abs = xp.abs(eta[0])
    omega_time_abs = xp.abs(eta[1])

    # d + 1
    dim = omega_space_abs.shape[0] + 1

    Gamma_squared = gamma**2 * squared_fractional_graph_laplacian(  # noqa: N806
        eigenvalues=omega_space_abs, kappa=kappa, alpha=alpha
    )

    return 1 / ((2 * xp.pi) ** dim * tau**2 * (omega_time_abs**2 + Gamma_squared))


# TODO: check ill-conditioning of the kernel
# TODO: when tau, gamma, kappa and alpha tends to 0
def hs_discrete_stochastic_heat_equation_kernel(
    hs_eigenpair: tuple[array, array],
    graph_eigenpair: tuple[array, array],
    tau: float,
    gamma: float,
    kappa: float,
    alpha: float,
    epsilon: float = 1e-8,
) -> array:
    r"""Approximate the discrete stochastic heat equation kernel.

    This function builds an approximation for the Gram matrix of the kernel
    resulted from the discrete stochastic heat equation on a graph.
    The following is the expression for the stochastic heat equation:

    .. math::
        \left[\frac{\partial}{\partial t} + \gamma \left(\kappa^2 + \Delta\right)^{\alpha/2}\right] \tau \mathbf{X}(\xi) = \mathbf{W}(\xi).

    For the spatiotemporal Brownian motion, :math:`\mathbf{W}(\xi)` is the
    derivative, and :math:`\xi = (\mathbf{s}, t) \in \mathcal{D}`.
    Given the spatial domain :math:`\mathcal{S} \subseteq \mathbb{R}^d` and the
    time domain :math:`\mathcal{T} \subseteq \mathbb{R}`, the spatiotemporal
    domain is :math:`\mathcal{D} = \mathcal{S} \times \mathcal{T}`.
    With :math:`\tau > 0`, the dispersion parameter is represented as
    :math:`1/\tau`. The thermal diffusivity of the medium in the process
    :math:`\mathbf{X}(\xi)` is :math:`\gamma > 0`. Here, the graph
    Laplacian operator :math:`L` is used in place of the continuous one
    :math:`\Delta`.

    This function leverages the Hilbert space approximation:

    .. math::
       k(\xi, \xi ') \approx \sum_{j \in \mathcal{I}} S_\mathcal{X}(\sqrt{\lambda}_j) \phi_j(\xi) \phi_j(\xi ').

    It computes the Gram matrix by combining Hilbert space approximation
    eigenvalues and eigenfunctions for time with graph Laplacian eigenvalues and
    eigenvectors for space. The Gram matrix shape reflects node-time structure.
    For regularization and numerical stability, a Marquardt-Levenberg coefficient
    is added to the diagonal.

    :func:`flat_index` can be used to provide human-friendly access
    to the matrix elements.

    Parameters
    ----------
    hs_eigenpair : tuple[array, array]
        Eigenvalues (sqrt) and eigenfunctions of the Hilbert space
        approximation.
    graph_eigenpair : tuple[array, array]
        Eigenvalues and eigenvectors of the graph Laplacian.
    tau : float
        Precision parameter, must be positive.
    gamma : float
        Medium's (thermal) diffusivity, must be positive.
    kappa : float
        Shifting factor applied to the spatial eigenvalues of the graph
        Laplacian, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.
    epsilon : float, optional
        Marquardt-Levenberg coefficient, by default 1e-8

    Returns
    -------
    array
        Non-separable spatiotemporal covariance matrix.
        The spatial domain is discretized as a graph.

    Raises
    ------
    ValueError
        If any of `tau`, `gamma`, `kappa` or `alpha` is less
        than or equal to zero.

    Notes
    -----
    - `Non-separable Spatio-temporal Graph Kernels via SPDEs
      <https://proceedings.mlr.press/v151/nikitin22a>`__.
    - `Practical Hilbert space approximate Bayesian
      Gaussian processes for probabilistic programming
      <https://link.springer.com/article/10.1007/s11222-022-10167-2>`__.
    - `Hilbert space methods for reduced-rank Gaussian process regression
      <https://link.springer.com/article/10.1007/s11222-019-09886-w>`__.
    """  # noqa: B950
    if tau <= 0:
        raise ValueError("Parameter `tau` must be positive.")
    if gamma <= 0:
        raise ValueError("Parameter `gamma` must be positive.")
    if kappa <= 0:
        raise ValueError("Parameter `kappa` must be positive.")
    if alpha <= 0:
        raise ValueError("Parameter `alpha` must be positive.")

    sqrt_lambdas, phi = hs_eigenpair
    S, Q = graph_eigenpair  # noqa: N806

    xp = array_namespace(phi, sqrt_lambdas, Q, S)  # Get the array API namespace

    # Sum of all combinations of HS eigenvalues with graph Laplacian matrix eigenvalues
    lambdas = psd_diag_discrete_stochastic_heat_equation(
        eta=(S[:, None], sqrt_lambdas), tau=tau, gamma=gamma, kappa=kappa, alpha=alpha
    )

    d, m = lambdas.shape
    t, _ = phi.shape

    eye_d = xp.eye(d)
    eye_m = xp.eye(m)

    # Tensor diag
    lambdas_diag = lambdas.T[:, :, None] * eye_d

    # Recomposing graph domain
    psd = Q[None, :, :] @ lambdas_diag @ Q.T[None, :, :]

    # Tensor diag
    psd_diag = psd.T[:, :, :, None] * eye_m

    # Recomposing time domain
    kernel = phi[None, None, :, :] @ psd_diag @ phi.T[None, None, :, :]

    reg_coeff = epsilon * xp.eye(d * t)

    # Reshape to covariance matrix and add Marquardt-Levenberg
    # coefficient for ill-conditioned matrices
    return xp.transpose(kernel, (0, 2, 1, 3)).reshape(d * t, d * t) + reg_coeff
