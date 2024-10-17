#!/usr/bin/env python3
"""The module is a test suite for banquo package."""

###############################################################################
# Imports #####################################################################
###############################################################################


from typing import no_type_check

import numpy as nxp  # ! from numpy import array_api as nxp not working
from hypothesis import given, settings

from banquo import array, chol2inv

from .hypothesis_arrays_strategy import spd_square_matrix_builder_float64


###############################################################################
# SPD square matrix builders  #################################################
###############################################################################


spd_matrix_builder_float64 = spd_square_matrix_builder_float64(size=5)


###############################################################################
# Tests for chol2inv  #########################################################
###############################################################################


@no_type_check
@settings(deadline=None)
@given(X=spd_matrix_builder_float64)
def test_chol2inv_equals_inverting_spd_matrix_float64(X: array):
    """Test if :func:`chol2inv` equals directly inverting a SPD matrix.

    Parameters
    ----------
    X : array
        SPD matrix.
    """
    X_inv = nxp.linalg.inv(X)
    spd_chol = nxp.linalg.cholesky(X)
    X_chol2inv = chol2inv(spd_chol)
    assert nxp.allclose(X_inv, X_chol2inv)
