# CHANGELOG


## v0.2.0 (2024-10-19)

### Build System

* build: remove check-docstring-first hook ([`ed18f00`](https://github.com/luizdesuo/banquo/commit/ed18f0089b2cdce1bb1958e0cfcbde381417dd3d))

* build: add array-api tools and hypothesis for testing ([`0544d83`](https://github.com/luizdesuo/banquo/commit/0544d835ebdd728eeb7f21b44ed6f30ca48fac18))

* build: manually add package version ([`7c77eb0`](https://github.com/luizdesuo/banquo/commit/7c77eb000d1362e62f66694d97d4cb78931a4376))

### Continuous Integration

* ci: restore workflow ([`791e319`](https://github.com/luizdesuo/banquo/commit/791e319fa42007f4edf4a7820f331af46880a0d2))

* ci: temporary remove condition to publish into PyPI ([`2620d59`](https://github.com/luizdesuo/banquo/commit/2620d597da0369107daa3043603a4a7ecf42194f))

* ci: remove condition to publish into PyPI ([`5c7a9ef`](https://github.com/luizdesuo/banquo/commit/5c7a9effa8919baf70b7af223986993135dda5b2))

* ci: temporary remove Test install from TestPyPI ([`886408b`](https://github.com/luizdesuo/banquo/commit/886408bc8b888aff15a6f220b43c550b07a746aa))

* ci: add PSR configuration ([`239dea2`](https://github.com/luizdesuo/banquo/commit/239dea2a8f150cb5f1ed33cdd58bcf8b734135d6))

### Features

* feat: add auxiliary functions, data transform functions their custom errors ([`3f82e1a`](https://github.com/luizdesuo/banquo/commit/3f82e1a79f3c5a7602d021d89245473201a0aaab))

* feat: add auxiliary functions ([`18eabb3`](https://github.com/luizdesuo/banquo/commit/18eabb37157d8248349036cc4e46b070a9dc3ce2))

* feat: extend the existing function to support array-api ([`738c2c5`](https://github.com/luizdesuo/banquo/commit/738c2c5e3d22d9ca7e79eec06935ef40a630098f))

### Refactoring

* refactor: add array info to documentation ([`c5150fc`](https://github.com/luizdesuo/banquo/commit/c5150fc1cee78896c05783733c765100142ecd8d))

### Testing

* test: add tests for usual cases of auxiliary and data transform functions ([`1455083`](https://github.com/luizdesuo/banquo/commit/1455083165a2fc5a11af152287cfb13ea06a0884))

* test: add functional test for multi_normal_cholesky_copula_lpdf function ([`c2ce33c`](https://github.com/luizdesuo/banquo/commit/c2ce33cd692a1cfd643488b2039114ff63486f5b))

* test: change diag function to array-api standard ([`3ba49ff`](https://github.com/luizdesuo/banquo/commit/3ba49fff5e61f7c460f1a19aa5937766bf093f99))

* test: add functional test for chol2inv function ([`3dc099c`](https://github.com/luizdesuo/banquo/commit/3dc099c88e0a98334f41a5f4ff4be82d2afb6bbd))

* test: add module to include array builders strategies ([`7c6ca1b`](https://github.com/luizdesuo/banquo/commit/7c6ca1bfb5177799205db17d77f9ec583bba6ae9))


## v0.1.0 (2024-10-11)

### Build System

* build: add numpyro and arviz dependencies ([`53ad24e`](https://github.com/luizdesuo/banquo/commit/53ad24ed887c5a0193017a9a890fa20c832e1522))

### Features

* feat: add gaussian copula and auxiliary functions ([`a607ef0`](https://github.com/luizdesuo/banquo/commit/a607ef0cd15c514ffd4a0e6abb259a6790203ea9))


## v0.0.0 (2024-10-11)

### Build System

* build: remove sphinxcontrib-mermaid package ([`ac38f45`](https://github.com/luizdesuo/banquo/commit/ac38f45828b58529a88c6ad7fa36dd9c71e71e1e))

* build(deps): bump python-semantic-release/python-semantic-release

Bumps [python-semantic-release/python-semantic-release](https://github.com/python-semantic-release/python-semantic-release) from 8.3.0 to 9.10.1.
- [Release notes](https://github.com/python-semantic-release/python-semantic-release/releases)
- [Changelog](https://github.com/python-semantic-release/python-semantic-release/blob/master/CHANGELOG.md)
- [Commits](https://github.com/python-semantic-release/python-semantic-release/compare/v8.3.0...v9.10.1)

---
updated-dependencies:
- dependency-name: python-semantic-release/python-semantic-release
  dependency-type: direct:production
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] <support@github.com> ([`c0e5bc9`](https://github.com/luizdesuo/banquo/commit/c0e5bc9d39e927b987e3173226dca5a0554f0c0a))

* build(deps): bump codecov/codecov-action from 3 to 4

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 3 to 4.
- [Release notes](https://github.com/codecov/codecov-action/releases)
- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)
- [Commits](https://github.com/codecov/codecov-action/compare/v3...v4)

---
updated-dependencies:
- dependency-name: codecov/codecov-action
  dependency-type: direct:production
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] <support@github.com> ([`7db76a5`](https://github.com/luizdesuo/banquo/commit/7db76a5fb8e0553a1a3f1e065d800fa9583cc09e))

* build(deps): bump dependabot/fetch-metadata from 1 to 2

Bumps [dependabot/fetch-metadata](https://github.com/dependabot/fetch-metadata) from 1 to 2.
- [Release notes](https://github.com/dependabot/fetch-metadata/releases)
- [Commits](https://github.com/dependabot/fetch-metadata/compare/v1...v2)

---
updated-dependencies:
- dependency-name: dependabot/fetch-metadata
  dependency-type: direct:production
  update-type: version-update:semver-major
...

Signed-off-by: dependabot[bot] <support@github.com> ([`dc736a7`](https://github.com/luizdesuo/banquo/commit/dc736a7084a4b21994ce71b66676b40dc9a70206))

* build: add safety policy file ([`1de3ef8`](https://github.com/luizdesuo/banquo/commit/1de3ef85ea312890952bcf3dcfd0dac216042e35))

* build: update pre-commit hooks versions ([`030306c`](https://github.com/luizdesuo/banquo/commit/030306c89d623875aea9f764b8e77536478b84a4))

### Documentation

* docs: add logo ([`6bc6314`](https://github.com/luizdesuo/banquo/commit/6bc6314c531c2893f63942d466be14d97cf3a1d0))

### Unknown

* Merge pull request #1 from luizdesuo/dependabot/github_actions/python-semantic-release/python-semantic-release-9.10.1

build(deps): bump python-semantic-release/python-semantic-release from 8.3.0 to 9.10.1 ([`a25ba68`](https://github.com/luizdesuo/banquo/commit/a25ba68e06c618eab03826d8283e61147ec6abe3))

* Merge pull request #2 from luizdesuo/dependabot/github_actions/codecov/codecov-action-4

build(deps): bump codecov/codecov-action from 3 to 4 ([`25806f9`](https://github.com/luizdesuo/banquo/commit/25806f9191477e995e6f188ff70b703c348924ce))

* Merge pull request #3 from luizdesuo/dependabot/github_actions/dependabot/fetch-metadata-2

build(deps): bump dependabot/fetch-metadata from 1 to 2 ([`4ff9494`](https://github.com/luizdesuo/banquo/commit/4ff9494d05d4a9d74374ba02bd6869a7c79980f0))

* initial package setup ([`f759bfb`](https://github.com/luizdesuo/banquo/commit/f759bfba6f9544cef862877786aa99b47c6bbd34))
