class RPSegmentationError(Exception):
    """
    Excepción base de la librería rp_segmentation.
    """


class InvalidSegmentationParameterError(RPSegmentationError):
    """
    Error para parámetros inválidos de segmentación.
    """


class NLTKResourceError(RPSegmentationError):
    """
    Error relacionado con recursos externos de NLTK.
    """