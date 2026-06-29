import pytest

from rp_segmentation.segmenters import (
    sentence_segmentation,
    paragraph_segmentation,
    n_stop_words_segmentation,
)
from rp_segmentation.exceptions import InvalidSegmentationParameterError


def test_sentence_segmentation():

    text = "Hola Pablo. Esto es una prueba. Funciona correctamente."

    result = sentence_segmentation(text)

    assert len(result) == 3
    assert result[0] == "hola pablo"
    assert result[1] == "esto es una prueba"
    assert result[2] == "funciona correctamente"



def test_paragraph_segmentation():

    text = "Primer párrafo.\n\nSegundo párrafo.\n\n\nTercer párrafo."

    result = paragraph_segmentation(text)

    assert len(result) == 3
    assert result[0] == "primer párrafo"
    assert result[1] == "segundo párrafo"
    assert result[2] == "tercer párrafo"



def test_n_stop_words_segmentation():

    text = '''
    
    
    '''

    result = n_stop_words_segmentation(text, tamanio=4, solapamiento=1)

    assert result == [
        "uno dos tres cuatro",
        "cuatro cinco seis siete",
        "siete ocho nueve diez",
    ]