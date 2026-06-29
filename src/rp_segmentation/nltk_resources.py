from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import Final, NoReturn
from zipfile import BadZipFile

import nltk

from rp_segmentation.exceptions import NLTKResourceError

NLTK_RESOURCES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
)


def _supported_resources_message() -> str:
    return ", ".join(sorted(NLTK_RESOURCES))


def _recovery_hint(resource_name: str) -> str:
    return f"Try running: python -m nltk.downloader {resource_name}"


def _raise_corrupted_resource_error(
    resource_name: str,
    exc: BadZipFile,
    *,
    after_download: bool = False,
) -> NoReturn:
    moment = " after download" if after_download else ""

    raise NLTKResourceError(
        f"The NLTK resource appears to be corrupted{moment}: {resource_name}. "
        f"Try deleting the local NLTK resource and running: "
        f"python -m nltk.downloader {resource_name}"
    ) from exc


def _resource_exists(resource_name: str, resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
        return True

    except LookupError:
        return False

    except BadZipFile as exc:
        _raise_corrupted_resource_error(resource_name, exc)


@cache
def ensure_nltk_resource(resource_name: str) -> None:
    """
    Ensures that a required NLTK resource is available.

    If the resource is missing, the function attempts to download it
    automatically. If the resource is corrupted or cannot be downloaded,
    a package-specific error is raised with a clear recovery message.

    Parameters
    ----------
    resource_name:
        Name of the NLTK resource. Example: 'punkt_tab'.

    Raises
    ------
    NLTKResourceError
        If the resource is unsupported, missing, corrupted, or cannot be
        downloaded.
    """
    resource_path = NLTK_RESOURCES.get(resource_name)

    if resource_path is None:
        raise NLTKResourceError(
            f"Unsupported NLTK resource for rp_segmentation: {resource_name}. "
            f"Supported resources are: {_supported_resources_message()}."
        )

    if _resource_exists(resource_name, resource_path):
        return

    try:
        downloaded = nltk.download(resource_name, quiet=True)

    except BadZipFile as exc:
        _raise_corrupted_resource_error(resource_name, exc, after_download=True)

    except Exception as exc:
        raise NLTKResourceError(
            f"Could not download the NLTK resource: {resource_name}. "
            f"{_recovery_hint(resource_name)}"
        ) from exc

    if not downloaded:
        raise NLTKResourceError(
            f"Could not download the NLTK resource: {resource_name}. "
            f"{_recovery_hint(resource_name)}"
        )

    try:
        nltk.data.find(resource_path)

    except BadZipFile as exc:
        _raise_corrupted_resource_error(resource_name, exc, after_download=True)

    except Exception as exc:
        raise NLTKResourceError(
            f"The NLTK resource was downloaded but could not be located: "
            f"{resource_name}. {_recovery_hint(resource_name)}"
        ) from exc


def ensure_required_nltk_resources() -> None:
    """
    Ensures that all NLTK resources required by the package are available.
    """
    for resource_name in NLTK_RESOURCES:
        ensure_nltk_resource(resource_name)
