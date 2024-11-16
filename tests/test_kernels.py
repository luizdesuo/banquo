#!/usr/bin/env python3
"""The module is a test suite for numpyro submodule."""


###############################################################################
# Imports #####################################################################
###############################################################################

from typing import no_type_check

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st
from numpyro.contrib.hsgp.laplacian import eigenfunctions, sqrt_eigenvalues

from banquo import array, normalize_covariance
from banquo.kernels import hs_discrete_stochastic_heat_equation_kernel

# TODO: implement to other APIs (for now jax).
from .hypothesis_arrays_strategy import spd_square_matrix_builder_float64


###############################################################################
# Constants  ##################################################################
###############################################################################


WIDTH = 64
MAX_EXAMPLES = 10
ELL = 1.2
SPD_SIZE = 5
M_MIN = 5  # Minimum number of basis functions of Hilbert space approximation
M_MAX = 50  # Maximum number of basis functions of Hilbert space approximation
T_MIN = 10  # Minimum number of samples
T_MAX = 100  # Maximum number of samples


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


spd_matrix_builder_float64 = spd_square_matrix_builder_float64(size=SPD_SIZE)


###############################################################################
# Tests for discrete stochastic heat equation kernel  #########################
###############################################################################


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    graph_laplacian=spd_matrix_builder_float64,
    m=st.integers(min_value=M_MIN, max_value=M_MAX),
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    tau=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
    epsilon=st.floats(min_value=1.0e-8, max_value=1.0e-6, allow_nan=False, width=WIDTH),
)
def test_hs_discrete_stochastic_heat_equation_kernel_shape(
    graph_laplacian: array,
    m: int,
    t: int,
    tau: float,
    gamma: float,
    kappa: float,
    alpha: float,
    epsilon: float,
) -> None:
    """Test of hs_discrete_stochastic_heat_equation_kernel resulting shape.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    m : int
        Number of Hilbert space basis functions
    t : int
        Number of time stamps or number of samples.
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
        Marquardt-Levenberg coefficient.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    phi = eigenfunctions(x=x, ell=ELL, m=m)
    sqrt_lambdas = sqrt_eigenvalues(ell=ELL, m=m, dim=1)  # One time dimension

    k = hs_discrete_stochastic_heat_equation_kernel(
        (sqrt_lambdas, phi),
        (S, Q),
        tau=tau,
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
        epsilon=epsilon,
    )

    assert k.shape == (d * t, d * t)


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    graph_laplacian=spd_matrix_builder_float64,
    m=st.integers(min_value=M_MIN, max_value=M_MAX),
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    tau=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
    epsilon=st.floats(min_value=1.0e-8, max_value=1.0e-6, allow_nan=False, width=WIDTH),
)
def test_hs_discrete_stochastic_heat_equation_kernel_is_spd(
    graph_laplacian: array,
    m: int,
    t: int,
    tau: float,
    gamma: float,
    kappa: float,
    alpha: float,
    epsilon: float,
) -> None:
    """Test if hs_discrete_stochastic_heat_equation_kernel results in SPD.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    m : int
        Number of Hilbert space basis functions
    t : int
        Number of time stamps or number of samples.
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
        Marquardt-Levenberg coefficient.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    phi = eigenfunctions(x=x, ell=ELL, m=m)
    sqrt_lambdas = sqrt_eigenvalues(ell=ELL, m=m, dim=1)  # One time dimension

    k = hs_discrete_stochastic_heat_equation_kernel(
        (sqrt_lambdas, phi),
        (S, Q),
        tau=tau,
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
        epsilon=epsilon,
    )

    eigenvals = jnp.linalg.eigvalsh(k)

    assert jnp.all(eigenvals > 0)


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    graph_laplacian=spd_matrix_builder_float64,
    m=st.integers(min_value=M_MIN, max_value=M_MAX),
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    tau=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
    epsilon=st.floats(min_value=1.0e-8, max_value=1.0e-6, allow_nan=False, width=WIDTH),
)
def test_hs_discrete_stochastic_heat_equation_kernel_normalization(
    graph_laplacian: array,
    m: int,
    t: int,
    tau: float,
    gamma: float,
    kappa: float,
    alpha: float,
    epsilon: float,
) -> None:
    """Test normalization of hs_discrete_stochastic_heat_equation_kernel.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    m : int
        Number of Hilbert space basis functions
    t : int
        Number of time stamps or number of samples.
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
        Marquardt-Levenberg coefficient.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    phi = eigenfunctions(x=x, ell=ELL, m=m)
    sqrt_lambdas = sqrt_eigenvalues(ell=ELL, m=m, dim=1)  # One time dimension

    k = hs_discrete_stochastic_heat_equation_kernel(
        (sqrt_lambdas, phi),
        (S, Q),
        tau=tau,
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
        epsilon=epsilon,
    )

    k_norm = normalize_covariance(k)

    eigenvals = jnp.linalg.eigvalsh(k_norm)

    k_norm_abs = jnp.abs(k_norm)

    assert jnp.all(eigenvals > 0)

    assert jnp.all((k_norm_abs < 1.0) | jnp.isclose(k_norm_abs, 1.0))
