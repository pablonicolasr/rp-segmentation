from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import Final

import nltk

from rp_segmentation.exceptions import NLTKResourceError

NLTK_RESOURCES: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
)


@cache
def ensure_nltk_resource(resource_name: str) -> None:
    """
    Ensures that a required NLTK resource is available.

    If the resource is missing, the function attempts to download it
    automatically.

    Parameters
    ----------
    resource_name:
        Name of the NLTK resource. Example: 'punkt_tab'.

    Raises
    ------
    NLTKResourceError
        If the resource is not supported by the package or cannot be found
        after attempting to download it.
    """
    resource_path = NLTK_RESOURCES.get(resource_name)

    if resource_path is None:
        supported_resources = ", ".join(sorted(NLTK_RESOURCES))

        raise NLTKResourceError(
            f"Unsupported NLTK resource for rp_segmentation: {resource_name}. "
            f"Supported resources are: {supported_resources}."
        )

    try:
        nltk.data.find(resource_path)
        return

    except LookupError:
        pass

    try:
        nltk.download(resource_name, quiet=True)
        nltk.data.find(resource_path)

    except Exception as exc:
        raise NLTKResourceError(
            f"Could not download or locate the NLTK resource: {resource_name}. "
            f"Try running: python -m nltk.downloader {resource_name}"
        ) from exc


def ensure_required_nltk_resources() -> None:
    """
    Ensures that all NLTK resources required by the package are available.
    """
    for resource_name in NLTK_RESOURCES:
        ensure_nltk_resource(resource_name)