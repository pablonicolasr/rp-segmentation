# rp-segmentation

`rp-segmentation` is a lightweight Python library for text segmentation, token normalization, and NLP-oriented preprocessing.

The package provides a simple and consistent API for splitting text into meaningful units, including sentences, paragraphs, and stopword-based segments. It is designed for text processing pipelines, NLP experimentation, semantic search, retrieval-augmented generation, and document preprocessing workflows.

## Features

* Sentence segmentation using NLTK.
* Paragraph segmentation based on structural line breaks.
* Stopword-based segmentation every `N` stopwords.
* Unicode-aware token extraction.
* Optional stopword removal.
* Typed package support through `py.typed`.
* Lightweight and easy to integrate into NLP pipelines.

## Installation

```bash
pip install rp-segmentation
```

## Requirements

* Python 3.10 or higher.
* NLTK.
* regex.

## NLTK Resources

`rp-segmentation` relies on external NLTK resources for sentence tokenization and stopword handling.

You can install the required resources manually:

```bash
python -m nltk.downloader punkt_tab
python -m nltk.downloader stopwords
```

Or install them directly from Python:

```python
from rp_segmentation import ensure_required_nltk_resources

ensure_required_nltk_resources()
```

## Basic Usage

```python
from rp_segmentation import (
    sentence_segmentation,
    paragraph_segmentation,
    n_stop_words_segmentation,
)

text = """
Hello, Pablo. This is a simple test.

This is another paragraph with additional content.
It can be used for text processing workflows.
"""

print(sentence_segmentation(text))
print(paragraph_segmentation(text))
print(n_stop_words_segmentation(text, n=3))
```

## Available Methods

### `sentence_segmentation`

```python
sentence_segmentation(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> list[str]
```

Segments a text into sentences using NLTK and applies the package's internal normalization strategy to each resulting segment.

#### Example

```python
from rp_segmentation import sentence_segmentation

text = "Hello, John. How are you?"

segments = sentence_segmentation(text)

print(segments)
```

Output:

```python
["hello john", "how are you"]
```

With stopword removal:

```python
segments = sentence_segmentation(
    text,
    language="english",
    remove_stopwords=True,
)

print(segments)
```

Output:

```python
["hello john"]
```

---

### `paragraph_segmentation`

```python
paragraph_segmentation(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> list[str]
```

Segments a text into paragraphs using double or multiple line breaks. Each paragraph is normalized before being returned.

#### Example

```python
from rp_segmentation import paragraph_segmentation

text = "First paragraph.\n\nSecond paragraph."

segments = paragraph_segmentation(text)

print(segments)
```

Output:

```python
["first paragraph", "second paragraph"]
```

---

### `n_stop_words_segmentation`

```python
n_stop_words_segmentation(
    text: str,
    language: str = "english",
    n: int = 5,
    remove_stopwords: bool = False,
) -> list[str]
```

Segments a text every `N` stopwords. This strategy is useful when working with natural language texts where stopword distribution can help define semantic or syntactic boundaries.

#### Example

```python
from rp_segmentation import n_stop_words_segmentation

text = "Alpha the beta and gamma is delta of omega."

segments = n_stop_words_segmentation(
    text,
    language="english",
    n=2,
)

print(segments)
```

Output:

```python
[
    "alpha the beta and",
    "gamma is delta of",
    "omega",
]
```

With stopword removal:

```python
segments = n_stop_words_segmentation(
    text,
    language="english",
    n=2,
    remove_stopwords=True,
)

print(segments)
```

Output:

```python
[
    "alpha beta",
    "gamma delta",
    "omega",
]
```

## Use Cases

`rp-segmentation` can be used in a wide range of text processing tasks, including:

* Natural Language Processing.
* Text normalization.
* Document preprocessing.
* Semantic search.
* Embedding preparation.
* Retrieval-Augmented Generation pipelines.
* Educational and research-oriented NLP projects.

## Local Development

Clone the repository:

```bash
git clone https://github.com/pablonicolasr777/rp-segmentation.git
cd rp-segmentation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

Install the required NLTK resources:

```bash
python -m nltk.downloader punkt_tab
python -m nltk.downloader stopwords
```

Run code quality checks:

```bash
ruff check .
mypy src
pytest --cov=rp_segmentation --cov-report=term-missing
```

## Project Structure

```text
rp-segmentation/
├── src/
│   └── rp_segmentation/
│       ├── __init__.py
│       ├── segmenters.py
│       ├── nltk_resources.py
│       ├── exceptions.py
│       └── py.typed
├── tests/
│   └── test_segmenters.py
├── docs/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
├── pyproject.toml
├── requirements-dev.txt
└── .gitignore
```

## Authors

* Pablo Nicolás Ramos
* Ricardo Daniel Perez

## License

This project is licensed under the MIT License.