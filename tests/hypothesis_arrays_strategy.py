"""
Provides the :mod:`hypothesis` strategy for arrays-api.
These strategies can be used for building specific types of arrays.
This module will soon have its own package, known as a hypothesis-extension.
"""

###############################################################################
# Imports #####################################################################
###############################################################################

from typing import Any, no_type_check

import numpy as nxp  # ! from numpy import array_api as nxp not working
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

from banquo import diag


xps = make_strategies_namespace(nxp)


###############################################################################
# Data types  #################################################################
###############################################################################


INT_SIZES = [8, 16, 32, 64]
FLOAT_SIZES = [32, 64]

INT_DTYPES = [f"int{size}" for size in INT_SIZES]
UINT_DTYPES = [f"uint{size}" for size in INT_SIZES]
FLOAT_DTYPES = [f"float{size}" for size in FLOAT_SIZES]
REAL_DTYPES = INT_DTYPES + UINT_DTYPES + FLOAT_DTYPES

FLOAT64 = "float64"


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


# TODO: Generalize to REAL_DTYPES
# Define the composite SPD matrix builder using array_api for float64
@no_type_check
@st.composite
def spd_square_matrix_builder_float64(draw: Any, size: int) -> st.SearchStrategy:
    """spd_square_matrix_builder_float64

    Generate a random symmetric, positive-definite matrix for hypothesis
    strategy. The function is conditioned to float64.

    Parameters
    ----------
    draw : Any
        For internal hypothesis use.
    size : int
        It is the size of the square matrix, which enforces a shape (size, size).

    Returns
    -------
    st.SearchStrategy
        Composite strategy for sampling SPD matrix for float64 dtype.
    """

    # Generate a matrix of the given size with values of the given dtype between 0 and 1
    A = draw(
        xps.arrays(
            dtype=FLOAT64,
            shape=(size, 2 * size),
            elements=st.floats(min_value=0, max_value=1),
            unique=True,
        )
    )

    # Use the same SVD-based approach to generate an SPD matrix
    U, _, Vt = nxp.linalg.svd(A @ A.T)

    # Generate diagonal values with uniform distribution to ensure positive definiteness
    diag_values = draw(
        xps.arrays(
            dtype=FLOAT64,
            shape=(size),
            elements=st.floats(min_value=1, max_value=2),
            unique=True,
        )
    )

    return U @ diag(diag_values) @ Vt
