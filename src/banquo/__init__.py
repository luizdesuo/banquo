"""Banquo."""

# read version from installed package
from importlib.metadata import version


__version__ = version("banquo")

# TODO: include other submodules
# populate package namespace
from banquo.banquo import *  # noqa: F401, F403
from banquo.kernels import *  # noqa: F401, F403
from banquo.numpyro import *  # noqa: F401, F403
