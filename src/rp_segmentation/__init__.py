from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from rp_segmentation.nltk_resources import ensure_required_nltk_resources
from rp_segmentation.segmenters import (
    get_tokens,
    n_stop_words_segmentation,
    paragraph_segmentation,
    sentence_segmentation,
)

try:
    __version__ = version("rp-segmentation")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = (
    "__version__",
    "ensure_required_nltk_resources",
    "get_tokens",
    "sentence_segmentation",
    "paragraph_segmentation",
    "n_stop_words_segmentation",
)