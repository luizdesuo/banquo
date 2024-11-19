# Banquo

a Python package for building Bayesian Nonparanormal models

[![Logo](https://raw.githubusercontent.com/luizdesuo/banquo/main/docs/_static/banquo-logo.png)](https://github.com/luizdesuo/banquo)

[![PyPI](https://img.shields.io/pypi/v/banquo.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/banquo.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/banquo)][pypi status]
[![License](https://img.shields.io/pypi/l/banquo)][license]

[![Read the documentation at https://banquo.readthedocs.io/](https://img.shields.io/readthedocs/banquo/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Release](https://github.com/luizdesuo/banquo/workflows/release/badge.svg)][release]
[![Codecov](https://codecov.io/gh/luizdesuo/banquo/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/banquo/
[read the docs]: https://banquo.readthedocs.io/
[release]: https://github.com/luizdesuo/banquo/actions?workflow=release
[codecov]: https://app.codecov.io/gh/luizdesuo/banquo
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- [Array-API standard]
- Bayesian inference
- Semiparametric Bernstein estimator for cumulative distribution function and
  probability density function
- Nonparanormal model with Bernstein marginals
- Hilbert space approximation for discrete stochastic heat equation kernel

See the Banquo architecture below:

![Architecture](https://raw.githubusercontent.com/luizdesuo/banquo/main/docs/_static/architecture.png)

## Requirements

- [array-api-compat]
- [NumPyro]
- [ArviZ]

## Installation

You can install _Banquo_ via [pip] from [PyPI]:

```bash
pip install banquo
```

## Usage

Please see the documentation at [Read the Docs] for details.

## Contributing

Contributions are very welcome. To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license], _Banquo_ is free and open
source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Credits

This project was generated from [Dinkin Flicka Cookiecutter] template.

[pypi]: https://pypi.org/project/banquo/
[dinkin flicka cookiecutter]: https://github.com/luizdesuo/cookiecutter-dinkin-flicka
[file an issue]: https://github.com/luizdesuo/banquo/issues
[pip]: https://pip.pypa.io/
[array-api standard]: https://data-apis.org/array-api/latest/
[array-api-compat]: https://github.com/data-apis/array-api-compat
[numpyro]: https://github.com/pyro-ppl/numpyro
[arviz]: https://github.com/arviz-devs/arviz

<!-- github-only -->

[license]: https://github.com/luizdesuo/banquo/blob/main/LICENSE
[contributor guide]: https://github.com/luizdesuo/banquo/blob/main/CONTRIBUTING.md
[command-line reference]: https://banquo.readthedocs.io/en/latest/usage.html
