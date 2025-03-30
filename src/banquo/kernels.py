#!/usr/bin/env python3
"""The module contains power spectral densities and kernels implementation."""

###############################################################################
# Imports #####################################################################
###############################################################################


from array_api_compat import array_namespace

from banquo.banquo import array


###############################################################################
# Namespace ###################################################################
###############################################################################


__all__ = [
    "squared_fractional_graph_laplacian",
    "flat_index",
    "discrete_stochastic_heat_equation_corr",
    "discrete_whittle_matern_fields_corr",
]


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

    Notes
    -----
    - `Non-separable Spatio-temporal Graph Kernels via SPDEs
      <https://proceedings.mlr.press/v151/nikitin22a>`__.
    """
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


def discrete_stochastic_heat_equation_corr(
    t: array,
    graph_eigenpair: tuple[array, array],
    gamma: float,
    kappa: float,
    alpha: float,
) -> array:
    r"""Approximate the discrete stochastic heat equation normalized kernel.

    This function builds an approximation for the Gram matrix of the normalized
    kernel (correlation) resulted from the discrete stochastic heat equation
    on a graph. The following is the expression for the stochastic
    heat equation:

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

    This correlation function is given by:

    .. math::
       k\left(\lvert t-t'\rvert\right) = \exp\left(-\mathbf{\Gamma} \lvert t-t'\rvert\right).

    :func:`flat_index` can be used to provide human-friendly access
    to the matrix elements.

    Parameters
    ----------
    t : array
        Time indices.
    graph_eigenpair : tuple[array, array]
        Eigenvalues and eigenvectors of the graph Laplacian.
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
        Non-separable spatiotemporal correlation matrix.
        The spatial domain is discretized as a graph.

    Notes
    -----
    - `Non-separable Spatio-temporal Graph Kernels via SPDEs
      <https://proceedings.mlr.press/v151/nikitin22a>`__.
    """  # noqa: B950
    S, Q = graph_eigenpair  # noqa: N806

    D = S.shape[0]  # noqa: N806
    T = t.shape[0]  # noqa: N806

    xp = array_namespace(Q, S, t)  # Get the array API namespace

    r = xp.abs(t[:, None] - t[None, :])  # Time delta matrix

    # Drift operator eigenvalues
    Gamma = gamma * xp.sqrt(  # noqa: N806
        squared_fractional_graph_laplacian(S, kappa, alpha)
    )

    eye_d = xp.eye(D)

    # Tensor diag
    lambda_diag = xp.exp(-r * Gamma[:, None, None]).T[..., None] * eye_d

    kernel = Q[None, None, ...] @ lambda_diag @ Q.T[None, None, ...]

    return xp.transpose(kernel, (2, 0, 3, 1)).reshape(D * T, D * T)


def discrete_whittle_matern_fields_corr(
    graph_eigenpair: tuple[array, array],
    kappa: float,
    alpha: float,
) -> array:
    r"""Compute the correlation matrix approximating a Whittle–Matérn field.

    This function constructs an approximation to the kernel corresponding to
    a Whittle–Matérn field defined as the solution to the SPDE

    .. math::
        \tau \, \left(\kappa^2 - \Delta_{\mathbf{s}}\right)^{\alpha/2} \mathbf{X}(\mathbf{s}) = \mathbf{W}(\mathbf{s}),

    where :math:`\mathbf{W}(\mathbf{s})` is Gaussian white noise and
    :math:`\Delta` is the Laplacian operator on a spatial domain
    :math:`\Omega`. In the discrete setting, the continuous domain
    :math:`\Omega` is replaced by a tessellated mesh :math:`\Omega_h`
    with grid spacing :math:`h`, and the continuous Laplacian is approximated
    by the graph Laplacian :math:`\mathbf{L}`. Assuming the spectral
    decomposition

    .. math::
        \mathbf{L} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{\mathrm{T}},

    with eigenvalues :math:`\mathbf{\Lambda}` and eigenvectors
    :math:`\mathbf{Q}`, the covariance matrix of the associated
    Gaussian Markov random field (GMRF) is given by

    .. math::
        \mathbf{\Sigma} = \mathbf{Q} \tau^{-2}\left(\kappa^2 \mathbf{I} + \mathbf{\Lambda}\right)^{-\alpha}  \mathbf{Q}^{\mathrm{T}},

    Removing the precision parameter :math:`\tau` via a diagonal scaling
    produces the correlation matrix

    .. math::
        \mathbf{\Omega} = \mathbf{S}^{-1} \mathbf{Q} \left(\kappa^2 \mathbf{I} + \mathbf{\Lambda}\right)^{-\alpha}  \mathbf{Q}^{\mathrm{T}} \mathbf{S}^{-1} ,

    where the diagonal entries of :math:`\mathbf{S}` are defined by

    .. math::
        \mathbf{S}_{ii} = \sqrt{\sum_{r=1}^d \mathbf{Q}_{ir}^2 \left( \kappa^2 + \lambda_r \right)^{-\alpha}}.

    Parameters
    ----------
    graph_eigenpair : tuple[array, array]
        A tuple (:math:`\mathbf{\Lambda}`, :math:`\mathbf{Q}`) where
        :math:`\mathbf{\Lambda}` is an array of eigenvalues and
        :math:`\mathbf{Q}` is the corresponding array of eigenvectors.
    kappa : float
        A positive shifting factor applied to the eigenvalues.
        It adjusts the spatial scale of the field.
    alpha : float
        A positive exponent controlling the fractional power
        of the Laplacian transformation. This parameter is directly
        related to the smoothness of the field. It must be greater
        than :math:`d/2`, where :math:`d` is the number of nodes.

    Returns
    -------
    array
        The normalized correlation matrix approximating
        the Whittle–Matérn field on the discretized domain. This matrix
        converges to the true continuous correlation structure as the
        discretization is refined.

    Notes
    -----
    - `Non-separable Spatio-temporal Graph Kernels via SPDEs
      <https://proceedings.mlr.press/v151/nikitin22a>`__.
    """  # noqa: B950
    Lambda, Q = graph_eigenpair  # noqa: N806

    xp = array_namespace(Lambda, Q)  # Get the array API namespace

    squared_L_tilde = squared_fractional_graph_laplacian(  # noqa: N806
        Lambda, kappa, alpha
    )

    # Compute normalizing factor
    S = xp.sqrt(xp.sum(Q**2 * squared_L_tilde, axis=1))  # noqa: N806

    Q_scaled = Q * xp.sqrt(squared_L_tilde)  # noqa: N806

    Q_normalized = Q_scaled / S[:, None]  # noqa: N806

    return Q_normalized @ Q_normalized.T
