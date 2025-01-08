#!/usr/bin/env python3
"""The module is a test suite for numpyro submodule."""


###############################################################################
# Imports #####################################################################
###############################################################################

from typing import no_type_check

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from banquo.kernels import discrete_stochastic_heat_equation_corr

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
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
)
def test_discrete_stochastic_heat_equation_corr_shape(
    graph_laplacian: array,
    t: int,
    gamma: float,
    kappa: float,
    alpha: float,
) -> None:
    """Test of discrete_stochastic_heat_equation_corr resulting shape.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    t : int
        Number of time stamps or number of samples.
    gamma : float
        Medium's (thermal) diffusivity, must be positive.
    kappa : float
        Shifting factor applied to the spatial eigenvalues of the graph
        Laplacian, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    k = discrete_stochastic_heat_equation_corr(
        x,
        (S, Q),
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
    )

    assert k.shape == (d * t, d * t)


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    graph_laplacian=spd_matrix_builder_float64,
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
)
def test_discrete_stochastic_heat_equation_corr_is_spd(
    graph_laplacian: array,
    t: int,
    gamma: float,
    kappa: float,
    alpha: float,
) -> None:
    """Test if discrete_stochastic_heat_equation_corr results in SPD.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    t : int
        Number of time stamps or number of samples.
    gamma : float
        Medium's (thermal) diffusivity, must be positive.
    kappa : float
        Shifting factor applied to the spatial eigenvalues of the graph
        Laplacian, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    k = discrete_stochastic_heat_equation_corr(
        x,
        (S, Q),
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
    )

    eigenvals = jnp.linalg.eigvalsh(k)

    assert jnp.all(eigenvals > 0)


@no_type_check
@settings(deadline=None, max_examples=MAX_EXAMPLES)
@given(
    graph_laplacian=spd_matrix_builder_float64,
    t=st.integers(min_value=T_MIN, max_value=T_MAX),
    gamma=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    kappa=st.floats(min_value=0.1, max_value=5, allow_nan=False, width=WIDTH),
    alpha=st.floats(
        min_value=SPD_SIZE / 2,
        max_value=10,
        exclude_min=True,
        allow_nan=False,
        width=WIDTH,
    ),
)
def test_discrete_stochastic_heat_equation_corr_normalization(
    graph_laplacian: array,
    t: int,
    gamma: float,
    kappa: float,
    alpha: float,
) -> None:
    """Test normalization of discrete_stochastic_heat_equation_corr.

    Parameters
    ----------
    graph_laplacian : array
        SPD matrix.
    t : int
        Number of time stamps or number of samples.
    gamma : float
        Medium's (thermal) diffusivity, must be positive.
    kappa : float
        Shifting factor applied to the spatial eigenvalues of the graph
        Laplacian, must be positive.
    alpha : float
        Exponent controlling the fractional power of the transformed Laplacian,
        must be positive. It has a linear relation with the smoothness.
    """
    graph_laplacian = jnp.asarray(graph_laplacian)

    d = graph_laplacian.shape[0]

    S, Q = jnp.linalg.eigh(jnp.asarray(graph_laplacian))

    x = jnp.linspace(-1, 1, t)

    k = discrete_stochastic_heat_equation_corr(
        x,
        (S, Q),
        gamma=gamma,
        kappa=kappa,
        alpha=alpha,
    )

    eigenvals = jnp.linalg.eigvalsh(k)

    k_norm_abs = jnp.abs(k)

    assert jnp.all(eigenvals > 0)

    assert jnp.all((k_norm_abs < 1.0) | jnp.isclose(k_norm_abs, 1.0))
