# Contributing

Thank you for your interest in contributing to `rp-segmentation`.

## Local Development

Clone the repository:

```bash
git clone https://github.com/pablonicolasr/rp-segmentation.git
cd rp-segmentation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Install the required NLTK resources:

```bash
python -m nltk.downloader punkt_tab
python -m nltk.downloader stopwords
```

## Checks Before Opening a Pull Request

Before opening a Pull Request, run:

```bash
ruff check .
mypy src
pytest --cov=rp_segmentation
python -m build
twine check dist/*
```

## General Guidelines

* Add tests for new features.
* Maintain compatibility with Python 3.10+.
* Update the README if the public API changes.
* Update the CHANGELOG when appropriate.
* Keep names clear and behavior predictable.
* Prefer explicit, typed, and well-tested code.
* Avoid adding unnecessary runtime dependencies.
