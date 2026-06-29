# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning: `MAJOR.MINOR.PATCH`.

## [0.1.0] - 2026-06-25

### Added

* Initial public release of `rp-segmentation`.
* Added sentence segmentation using NLTK.
* Added paragraph segmentation based on structural line breaks.
* Added stopword-based segmentation every `N` stopwords.
* Added Unicode-aware token extraction.
* Added optional stopword removal.
* Added explicit management of required NLTK resources.
* Added custom package exceptions for segmentation and NLTK resource errors.
* Added initial test suite using `pytest`.
* Added test coverage configuration with `pytest-cov`.
* Added linting configuration with Ruff.
* Added static type checking configuration with mypy.
* Added typed package support through `py.typed`.
* Added build configuration using `pyproject.toml`.
* Added continuous integration workflow with GitHub Actions.
* Added PyPI publishing workflow.
* Added project documentation files, including `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `LICENSE`.