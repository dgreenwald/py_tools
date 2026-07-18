# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-18

### Added

- Add weighted-sum output to `collapse` and `Collapser.get_data` through the
  `as_sum` option.
- Expand regression-table generation with inferred variable names, custom
  variable and statistic labels, significance stars, observation counts, and
  direct LaTeX file output.
- Add `multi_regression_table` for aligning and displaying results from several
  regression specifications.
- Add absorbed fixed effects and configurable robust covariance estimators to
  regression helpers, including weighted formula regressions and local
  projections.
- Add hatching options to `double_hist` and a `figsize` option to `binscatter`.
- Add annual ZIP5 and annual ZIP3 support to the FHFA dataset loader.
- Add residential mortgage flows to the Financial Accounts dataset mapping.

### Changed

- Store `Collapser` metadata sidecars as JSON. Loading retains a deprecated
  fallback for existing pickle sidecars.
- Store FHFA dataset caches as Parquet instead of pickle and validate supported
  geography/frequency combinations explicitly.
- Add `pyarrow` and `openpyxl` to the `datasets` and `all` dependency extras.
- Configure GitHub Actions to use the Node.js 24 runtime.

### Fixed

- Avoid mutable default arguments in multiquantile collapsing and top-share
  calculations.
- Correct `binscatter` raw-data layering, weight handling, and limit checks.
- Support the current FHFA state-data column names and annual file formats.
- Support CRSP monthly files that provide `MthCalDt` instead of `caldt`.
- Correct the BVAR data-augmentation regular expression.
- Install dataset dependencies in CI so FHFA tests can import `openpyxl`.
- Report the installed package version using the `dgreenwald-py-tools`
  distribution metadata.

## [0.1.0] - 2026-02-22

- Initial packaged release.
