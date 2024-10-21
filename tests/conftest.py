#!/usr/bin/env python3
"""This module provides CLI parameterization for pytest.

To select and use different array APIs, such as NumPy or JAX.
It also displays the selected array API in the pytest report header.

Available Array APIs:
- numpy
- jax-numpy

To use:
    pytest --array-api <api>
"""

###############################################################################
# Imports #####################################################################
###############################################################################

import logging
from typing import Optional, no_type_check

import jax.numpy as jnp
import numpy as nxp  # ! from numpy import array_api as nxp not working
import pytest
from hypothesis.extra.array_api import make_strategies_namespace


###############################################################################
# Setup Logging ###############################################################
###############################################################################


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###############################################################################
# Parameterization  ###########################################################
###############################################################################

ARRAY_API: dict[str, object] = {"numpy": nxp, "jax-numpy": jnp}  # array apis
selected_array_api: str = "numpy"  # Default array-api for tests
xps: object = make_strategies_namespace(
    ARRAY_API[selected_array_api]
)  # hypothesis array_api extra module


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds a command-line option `--array-api` to pytest.

    This option allows the user to select the array API to be used
    during the test session. The available choices are defined in the
    `ARRAY_API` dictionary.

    Parameters
    ----------
    parser : pytest.Parser
        The pytest parser object to which the option is added.
    """
    parser.addoption(
        "--array-api",
        action="store",
        default="numpy",
        help=f"select the array-api {tuple(ARRAY_API.keys())}",
        choices=tuple(ARRAY_API.keys()),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Hook to capture the selected array API early.

    This function stores the selected array API in the
    `selected_array_api` global variable, making it accessible to other
    parts of the code, such as fixtures or hooks.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object with command-line options.
    """
    global selected_array_api
    selected_array_api = config.getoption("array_api")  # Capture the selected array API


@no_type_check
@pytest.fixture(scope="session", autouse=True)
def setup_array_api(request: pytest.FixtureRequest) -> Optional[object]:
    """Initialize the hypothesis strategies namespace for the selected API.

    This fixture is automatically used in the test session to configure the
    `xps` variable, which contains the hypothesis strategies namespace
    based on the selected array API.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture request object.

    Yields
    ------
    Optional[object]
        The hypothesis strategies namespace for the selected array API.
    """
    global xps
    option_value = selected_array_api  # Use the captured value
    if option_value is not None:
        array_api = ARRAY_API[option_value]
        xps = make_strategies_namespace(array_api)

        # Log the selected array API
        logger.info(f"Using array API: {option_value}")

    yield xps


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Hook to add custom information to the pytest report header.

    This function appends the selected array API to the
    report header, allowing users to see which array API was
    chosen for the test session.

    Parameters
    ----------
    config : pytest.Config
        The pytest config object.

    Returns
    -------
    list[str]
        A list of strings to be added to the pytest report header.
    """
    header = []
    if selected_array_api:
        header.append(f"selected array-api: {selected_array_api}")
    return header
