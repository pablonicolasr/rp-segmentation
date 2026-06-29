from __future__ import annotations


class RPSegmentationError(Exception):
    """
    Base exception for all errors raised by the rp_segmentation package.

    This exception can be used to catch any package-specific error without
    catching unrelated built-in Python exceptions.
    """


class InvalidSegmentationParameterError(RPSegmentationError):
    """
    Raised when an invalid segmentation parameter is provided.

    Examples include empty language values, unsupported parameter values,
    non-positive segmentation thresholds, or texts that cannot be segmented
    according to the selected strategy.
    """


class NLTKResourceError(RPSegmentationError):
    """
    Raised when a required external NLTK resource is missing or cannot be loaded.

    This exception is used when the package cannot find, download, or access
    resources such as tokenizers or stopword corpora required for text
    processing.
    """


__all__ = (
    "RPSegmentationError",
    "InvalidSegmentationParameterError",
    "NLTKResourceError",
)