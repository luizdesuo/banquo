# CHANGELOG


## v0.7.0 (2025-03-30)

### Bug Fixes

- Modules namespace
  ([`145f786`](https://github.com/luizdesuo/banquo/commit/145f786abcabae9f3deaf49daf794a3cbfa591c6))

### Features

- Add correlation matrix derived from matern fields approximation for graphs
  ([`032e7c4`](https://github.com/luizdesuo/banquo/commit/032e7c41fdd7b13ce1a12ff221de5d717060c78f))

### Testing

- Add test for matern correlation
  ([`6438b40`](https://github.com/luizdesuo/banquo/commit/6438b408b762434673145e2ffc64f9473baee594))

- Fix test namespace from submodules
  ([`44d5e21`](https://github.com/luizdesuo/banquo/commit/44d5e21e1f017d9686f65bc20da367dbc9c24884))


## v0.6.0 (2025-01-08)

### Build System

- Move arviz from dependency to test group
  ([`28fa4c0`](https://github.com/luizdesuo/banquo/commit/28fa4c0eda794116aadd06ee2276761ad0f72323))

- **deps**: Bump codecov/codecov-action from 4 to 5
  ([`cd0558a`](https://github.com/luizdesuo/banquo/commit/cd0558aa45ee2414aa94f80a4b48743f379f434c))

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 4 to 5. - [Release
  notes](https://github.com/codecov/codecov-action/releases) -
  [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/codecov/codecov-action/compare/v4...v5)

--- updated-dependencies: - dependency-name: codecov/codecov-action dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump python-semantic-release/python-semantic-release
  ([`d5ade96`](https://github.com/luizdesuo/banquo/commit/d5ade9678b7e73622c661cbde48676b2535cb9b4))

Bumps
  [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
  from 9.14.0 to 9.15.2. - [Release
  notes](https://github.com/python-semantic-release/python-semantic-release/releases) -
  [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
  -
  [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v9.14.0...v9.15.2)

--- updated-dependencies: - dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

### Documentation

- Add source code view extension
  ([`770fd7b`](https://github.com/luizdesuo/banquo/commit/770fd7bde1955133e7c675d7774a5cd66415fd65))

- Remove hilbert space approximation
  ([`fc703eb`](https://github.com/luizdesuo/banquo/commit/fc703eb11cee19d68e573fb915eebf74a8175ff7))

### Features

- Add sample method for bernstein model
  ([`96d5266`](https://github.com/luizdesuo/banquo/commit/96d5266514bf998d210c7c9eb709230c34dfe27f))

- Remove hilbert space approximations and add heat equation correlation
  ([`28270ff`](https://github.com/luizdesuo/banquo/commit/28270ff0f785063f48b7be5eb1bf2b49efc81f26))

### Testing

- Fix typing issue
  ([`dedd5bd`](https://github.com/luizdesuo/banquo/commit/dedd5bd2efa35aa382fb7658bb7ccc3709358557))

- Modify test to work with new correlation function
  ([`98e7422`](https://github.com/luizdesuo/banquo/commit/98e74223b71fb67ad263ca75c1024b957f550692))


## v0.5.0 (2024-11-19)

### Bug Fixes

- Populate namespace
  ([`e64fa93`](https://github.com/luizdesuo/banquo/commit/e64fa93edfc320fb565041df3e87395ac2b2f296))

- Remove circular import error
  ([`db7ac65`](https://github.com/luizdesuo/banquo/commit/db7ac6528dbee7a623da9013a6103ab45199a289))

### Build System

- Add development status
  ([`8e71d89`](https://github.com/luizdesuo/banquo/commit/8e71d891803b889510edefcc6457c9ee97c4dae4))

- **deps**: Bump python-semantic-release/python-semantic-release
  ([`d9b6083`](https://github.com/luizdesuo/banquo/commit/d9b60836149fef8b3d7e454d6ce56712cd102bc3))

Bumps
  [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
  from 9.12.0 to 9.14.0. - [Release
  notes](https://github.com/python-semantic-release/python-semantic-release/releases) -
  [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
  -
  [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v9.12.0...v9.14.0)

--- updated-dependencies: - dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

### Documentation

- Add initial content to README file
  ([`fb11eff`](https://github.com/luizdesuo/banquo/commit/fb11eff8a462329e99a2f6d963eea56cbc9374e5))

### Features

- Add nonparanormal model with bernstein marginals
  ([`d5911cf`](https://github.com/luizdesuo/banquo/commit/d5911cffc3707345a02fd3e4b184d4d6e1cfbe7f))

- Add prototype for bernstein_icdf
  ([`a8c57ff`](https://github.com/luizdesuo/banquo/commit/a8c57ff42193707354bacd068de085c517c1f1d3))

### Testing

- Add new tests (skipped) for cdf estimation and nonparanormal model
  ([`840532d`](https://github.com/luizdesuo/banquo/commit/840532dae999f3737e105f0ee459380847980dff))


## v0.4.0 (2024-11-17)

### Bug Fixes

- Jax arrays setting API with .at[idx].set(value)
  ([`66a1edf`](https://github.com/luizdesuo/banquo/commit/66a1edff61d6b69fe325faf32e516d846c4c45f9))

### Build System

- Safety check was deprecated, updated to scan
  ([`638f8da`](https://github.com/luizdesuo/banquo/commit/638f8da4477bbdf838d47913f5ab9cc49c7702f3))

- **deps**: Bump python-semantic-release/python-semantic-release
  ([`ba68db4`](https://github.com/luizdesuo/banquo/commit/ba68db4e26ae3833a2aea8c296b2e513dee417b2))

Bumps
  [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
  from 9.11.0 to 9.12.0. - [Release
  notes](https://github.com/python-semantic-release/python-semantic-release/releases) -
  [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
  -
  [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v9.11.0...v9.12.0)

--- updated-dependencies: - dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

### Continuous Integration

- Removed safety from ci workflow due to their API breaking change
  ([`7296d08`](https://github.com/luizdesuo/banquo/commit/7296d08b90125b4970d1b2bd921ae7c7ec1262ab))

### Documentation

- Fix references for functions in docstrings
  ([`65034ed`](https://github.com/luizdesuo/banquo/commit/65034ed83cad50f098e10fbcb39aed7dfc49e659))

### Features

- Add kernels module with stochastic heat equation kernel
  ([`2fae7eb`](https://github.com/luizdesuo/banquo/commit/2fae7ebc26184d4420bd087a4d9befcf46a50839))

- Remove bernstein_density and add Bernstein numpyro model
  ([`a647b2d`](https://github.com/luizdesuo/banquo/commit/a647b2d78ef9b031743e7eb81f6b81ff6eb6e279))

### Testing

- Add auxiliary functions
  ([`f2b5431`](https://github.com/luizdesuo/banquo/commit/f2b54319b7979d260ec12f93ad13e267ddb5dc3d))

- Add test suite for kernels module
  ([`2b4998a`](https://github.com/luizdesuo/banquo/commit/2b4998a9dcf40fdf247f58e7ab6562e3060f9c91))

- Replace bernstein_density with Bernstein model
  ([`13d2453`](https://github.com/luizdesuo/banquo/commit/13d24538a366a46d783600b75702de29aa9a3852))


## v0.3.0 (2024-10-27)

### Bug Fixes

- Error when expanding weights dimension
  ([`11275a2`](https://github.com/luizdesuo/banquo/commit/11275a259959495bce8aa2c63557cd00cd6b2088))

### Build System

- **deps**: Bump python-semantic-release/python-semantic-release
  ([`3c6d3fb`](https://github.com/luizdesuo/banquo/commit/3c6d3fb0fac0f3eff3db3bc2a336441e45cc8081))

Bumps
  [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
  from 9.10.1 to 9.11.0. - [Release
  notes](https://github.com/python-semantic-release/python-semantic-release/releases) -
  [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
  -
  [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v9.10.1...v9.11.0)

--- updated-dependencies: - dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

### Continuous Integration

- Remove step for jax-numpy array-api
  ([`7105a71`](https://github.com/luizdesuo/banquo/commit/7105a7187425c945b14a9c9af9ea5dfc4c90271b))

- Run tests for jax-numpy array api
  ([`e4534f5`](https://github.com/luizdesuo/banquo/commit/e4534f59ad7f4f277d7d4169de573f4e99b7afd7))

### Documentation

- Add bernstein approximation equations
  ([`b448193`](https://github.com/luizdesuo/banquo/commit/b448193c872f1e3f0b3ad4447cc6bfa124bbf3de))

- Fix BetaProtocol example should not subclass a protocol
  ([`6265af2`](https://github.com/luizdesuo/banquo/commit/6265af2f426571748592b4bb3043e0d8828638ab))

- Fix math equations
  ([`7d818d4`](https://github.com/luizdesuo/banquo/commit/7d818d4fe756c7b469617b97cce6689fb25e2d60))

- Fix output dimension of returns on bernstein functions
  ([`ce41600`](https://github.com/luizdesuo/banquo/commit/ce4160035fac0fe49200c571d687dfb839a954e8))

### Features

- Add bernstein-based marginal modeling functions
  ([`fad6648`](https://github.com/luizdesuo/banquo/commit/fad6648c0e64faed957df19938d3620f0e9490c5))

- Add logsumexp using array-api standard
  ([`9412c14`](https://github.com/luizdesuo/banquo/commit/9412c142cc71ed2d00fc80f1f7915adc1b19427d))

- Add numpyro module, build marginal models and beta protocol
  ([`722e073`](https://github.com/luizdesuo/banquo/commit/722e07309722c26a4d84bfd9cdad560c90d871cf))

- Add shape handling functions for compatibility with bernstein functions
  ([`e1b2a40`](https://github.com/luizdesuo/banquo/commit/e1b2a40c77c507694242b32e62be08f86ba295ab))

- Expand bernstein functions to multiple purpose (MCMC and posterior reconstruction)
  ([`0d74487`](https://github.com/luizdesuo/banquo/commit/0d74487781bf34ef8dd381210fec025a7f0718b7))

### Testing

- Add configuration for tests array-api from CLI and adjust the other modules
  ([`0f785d1`](https://github.com/luizdesuo/banquo/commit/0f785d1698a907ca8cd2a79956f8f71a97f0c731))

- Add integration test to bernstein density and functions
  ([`cca046d`](https://github.com/luizdesuo/banquo/commit/cca046d1d99efe472f44ec1e2a939c8f4c2b0fb3))

- Add test to logsumexp
  ([`c3053bd`](https://github.com/luizdesuo/banquo/commit/c3053bdb9a201773ab495910ff064ea7d20bc619))


## v0.2.0 (2024-10-19)

### Build System

- Add array-api tools and hypothesis for testing
  ([`0544d83`](https://github.com/luizdesuo/banquo/commit/0544d835ebdd728eeb7f21b44ed6f30ca48fac18))

- Manually add package version
  ([`7c77eb0`](https://github.com/luizdesuo/banquo/commit/7c77eb000d1362e62f66694d97d4cb78931a4376))

- Remove check-docstring-first hook
  ([`ed18f00`](https://github.com/luizdesuo/banquo/commit/ed18f0089b2cdce1bb1958e0cfcbde381417dd3d))

### Continuous Integration

- Add PSR configuration
  ([`239dea2`](https://github.com/luizdesuo/banquo/commit/239dea2a8f150cb5f1ed33cdd58bcf8b734135d6))

- Remove condition to publish into PyPI
  ([`5c7a9ef`](https://github.com/luizdesuo/banquo/commit/5c7a9effa8919baf70b7af223986993135dda5b2))

- Restore workflow
  ([`791e319`](https://github.com/luizdesuo/banquo/commit/791e319fa42007f4edf4a7820f331af46880a0d2))

- Temporary remove condition to publish into PyPI
  ([`2620d59`](https://github.com/luizdesuo/banquo/commit/2620d597da0369107daa3043603a4a7ecf42194f))

- Temporary remove Test install from TestPyPI
  ([`886408b`](https://github.com/luizdesuo/banquo/commit/886408bc8b888aff15a6f220b43c550b07a746aa))

### Features

- Add auxiliary functions
  ([`18eabb3`](https://github.com/luizdesuo/banquo/commit/18eabb37157d8248349036cc4e46b070a9dc3ce2))

- Add auxiliary functions, data transform functions their custom errors
  ([`3f82e1a`](https://github.com/luizdesuo/banquo/commit/3f82e1a79f3c5a7602d021d89245473201a0aaab))

- Extend the existing function to support array-api
  ([`738c2c5`](https://github.com/luizdesuo/banquo/commit/738c2c5e3d22d9ca7e79eec06935ef40a630098f))

### Refactoring

- Add array info to documentation
  ([`c5150fc`](https://github.com/luizdesuo/banquo/commit/c5150fc1cee78896c05783733c765100142ecd8d))

### Testing

- Add functional test for chol2inv function
  ([`3dc099c`](https://github.com/luizdesuo/banquo/commit/3dc099c88e0a98334f41a5f4ff4be82d2afb6bbd))

- Add functional test for multi_normal_cholesky_copula_lpdf function
  ([`c2ce33c`](https://github.com/luizdesuo/banquo/commit/c2ce33cd692a1cfd643488b2039114ff63486f5b))

- Add module to include array builders strategies
  ([`7c6ca1b`](https://github.com/luizdesuo/banquo/commit/7c6ca1bfb5177799205db17d77f9ec583bba6ae9))

- Add tests for usual cases of auxiliary and data transform functions
  ([`1455083`](https://github.com/luizdesuo/banquo/commit/1455083165a2fc5a11af152287cfb13ea06a0884))

- Change diag function to array-api standard
  ([`3ba49ff`](https://github.com/luizdesuo/banquo/commit/3ba49fff5e61f7c460f1a19aa5937766bf093f99))


## v0.1.0 (2024-10-11)

### Build System

- Add numpyro and arviz dependencies
  ([`53ad24e`](https://github.com/luizdesuo/banquo/commit/53ad24ed887c5a0193017a9a890fa20c832e1522))

### Features

- Add gaussian copula and auxiliary functions
  ([`a607ef0`](https://github.com/luizdesuo/banquo/commit/a607ef0cd15c514ffd4a0e6abb259a6790203ea9))


## v0.0.0 (2024-10-11)

### Build System

- Add safety policy file
  ([`1de3ef8`](https://github.com/luizdesuo/banquo/commit/1de3ef85ea312890952bcf3dcfd0dac216042e35))

- Remove sphinxcontrib-mermaid package
  ([`ac38f45`](https://github.com/luizdesuo/banquo/commit/ac38f45828b58529a88c6ad7fa36dd9c71e71e1e))

- Update pre-commit hooks versions
  ([`030306c`](https://github.com/luizdesuo/banquo/commit/030306c89d623875aea9f764b8e77536478b84a4))

- **deps**: Bump codecov/codecov-action from 3 to 4
  ([`7db76a5`](https://github.com/luizdesuo/banquo/commit/7db76a5fb8e0553a1a3f1e065d800fa9583cc09e))

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 3 to 4. - [Release
  notes](https://github.com/codecov/codecov-action/releases) -
  [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/codecov/codecov-action/compare/v3...v4)

--- updated-dependencies: - dependency-name: codecov/codecov-action dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump dependabot/fetch-metadata from 1 to 2
  ([`dc736a7`](https://github.com/luizdesuo/banquo/commit/dc736a7084a4b21994ce71b66676b40dc9a70206))

Bumps [dependabot/fetch-metadata](https://github.com/dependabot/fetch-metadata) from 1 to 2. -
  [Release notes](https://github.com/dependabot/fetch-metadata/releases) -
  [Commits](https://github.com/dependabot/fetch-metadata/compare/v1...v2)

--- updated-dependencies: - dependency-name: dependabot/fetch-metadata dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump python-semantic-release/python-semantic-release
  ([`c0e5bc9`](https://github.com/luizdesuo/banquo/commit/c0e5bc9d39e927b987e3173226dca5a0554f0c0a))

Bumps
  [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release)
  from 8.3.0 to 9.10.1. - [Release
  notes](https://github.com/python-semantic-release/python-semantic-release/releases) -
  [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
  -
  [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v8.3.0...v9.10.1)

--- updated-dependencies: - dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

### Documentation

- Add logo
  ([`6bc6314`](https://github.com/luizdesuo/banquo/commit/6bc6314c531c2893f63942d466be14d97cf3a1d0))
