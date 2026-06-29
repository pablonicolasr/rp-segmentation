from __future__ import annotations

from functools import lru_cache

import regex as re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from rp_segmentation.exceptions import InvalidSegmentationParameterError
from rp_segmentation.nltk_resources import ensure_nltk_resource


WORD_RE = re.compile(r"\p{L}+")


def _normalize_text(text: str) -> str:
    """
    Normalizes leading and trailing whitespace while preserving the internal
    structure of the text.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Normalized text.

    Raises
    ------
    TypeError
        If the input value is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("The input text must be a string.")

    return text.strip()


def _validate_language(language: str) -> str:
    """
    Validates and normalizes the language parameter.

    Parameters
    ----------
    language:
        Language identifier used by NLTK.

    Returns
    -------
    str
        Normalized language value.

    Raises
    ------
    InvalidSegmentationParameterError
        If the language value is invalid.
    """
    if not isinstance(language, str) or not language.strip():
        raise InvalidSegmentationParameterError(
            "The language parameter must be a non-empty string."
        )

    return language.strip().lower()


@lru_cache(maxsize=None)
def _get_stopwords(language: str) -> frozenset[str]:
    """
    Loads and caches stopwords for the selected language.

    Parameters
    ----------
    language:
        Language used to retrieve NLTK stopwords.

    Returns
    -------
    frozenset[str]
        Cached set of stopwords.

    Raises
    ------
    InvalidSegmentationParameterError
        If stopwords are not available for the selected language.
    """
    language = _validate_language(language)

    ensure_nltk_resource("stopwords")

    try:
        return frozenset(stopwords.words(language))

    except OSError as exc:
        raise InvalidSegmentationParameterError(
            f"Stopwords are not available for language: {language}."
        ) from exc


def get_tokens(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> list[str]:
    """
    Extracts lowercase word tokens from a text.

    The tokenizer keeps Unicode letter characters and discards numbers,
    punctuation marks, symbols, and empty values.

    Parameters
    ----------
    text:
        Input text.
    language:
        Language used when stopword removal is enabled.
        Default is 'english'.
    remove_stopwords:
        Whether to remove stopwords from the resulting tokens.
        Default is False.

    Returns
    -------
    list[str]
        List of lowercase word tokens.

    Examples
    --------
    >>> get_tokens("Hello, John. How are you?")
    ['hello', 'john', 'how', 'are', 'you']

    >>> get_tokens("Hello, John. How are you?", remove_stopwords=True)
    ['hello', 'john']
    """
    clean_text = _normalize_text(text)

    if not clean_text:
        return []

    tokens = WORD_RE.findall(clean_text.lower())

    if not remove_stopwords:
        return tokens

    stop_words = _get_stopwords(language)

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


def _clean_segment(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> str:
    """
    Applies the package's canonical text normalization strategy to a segment
    and returns the result as a single whitespace-normalized string.

    Parameters
    ----------
    text:
        Input sentence or paragraph.
    language:
        Language used when stopword removal is enabled.
        Default is 'english'.
    remove_stopwords:
        Whether to remove stopwords from the resulting segment.
        Default is False.

    Returns
    -------
    str
        Normalized segment.

    Examples
    --------
    >>> _clean_segment("Hello, John. How are you?")
    'hello john how are you'

    >>> _clean_segment("Hello, John. How are you?", remove_stopwords=True)
    'hello john'
    """
    return " ".join(
        get_tokens(
            text,
            language=language,
            remove_stopwords=remove_stopwords,
        )
    )


def sentence_segmentation(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> list[str]:
    """
    Segments a text into sentences and normalizes each resulting segment.

    The function validates and normalizes the input text, ensures that the
    required NLTK tokenizer resource is available, detects sentence boundaries
    using the selected language, and discards empty segments after
    normalization.

    Parameters
    ----------
    text:
        Input text to be segmented.
    language:
        Language used by NLTK for sentence boundary detection.
        Default is 'english'.
    remove_stopwords:
        Whether to remove stopwords from each normalized sentence.
        Default is False.

    Returns
    -------
    list[str]
        List of normalized sentence segments.

    Examples
    --------
    >>> segment_sentences("Hello, John. How are you?")
    ['hello john', 'how are you']

    >>> segment_sentences("Hello, John. How are you?", remove_stopwords=True)
    ['hello john']
    """
    language = _validate_language(language)

    clean_text = _normalize_text(text)

    if not clean_text:
        return []

    ensure_nltk_resource("punkt_tab")

    sentences = sent_tokenize(clean_text, language=language)

    return [
        cleaned_sentence
        for sentence in sentences
        if (
            cleaned_sentence := _clean_segment(
                sentence,
                language=language,
                remove_stopwords=remove_stopwords,
            )
        )
    ]


def paragraph_segmentation(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
) -> list[str]:
    """
    Segments a text into paragraphs and normalizes each resulting segment.

    Paragraph boundaries are detected using double or multiple line breaks.
    Each paragraph is normalized, and empty segments produced after
    normalization are discarded.

    Parameters
    ----------
    text:
        Input text to be segmented.
    language:
        Language used when stopword removal is enabled.
        Default is 'english'.
    remove_stopwords:
        Whether to remove stopwords from each normalized paragraph.
        Default is False.

    Returns
    -------
    list[str]
        List of normalized paragraph segments.

    Examples
    --------
    >>> segment_paragraphs("First paragraph.\\n\\nSecond paragraph.")
    ['first paragraph', 'second paragraph']

    >>> segment_paragraphs("This is the first paragraph.\\n\\nThis is another one.", remove_stopwords=True)
    ['first paragraph', 'another one']
    """
    language = _validate_language(language)

    clean_text = _normalize_text(text)

    if not clean_text:
        return []

    paragraphs = re.split(r"\n\s*\n+", clean_text)

    return [
        cleaned_paragraph
        for paragraph in paragraphs
        if (
            cleaned_paragraph := _clean_segment(
                paragraph,
                language=language,
                remove_stopwords=remove_stopwords,
            )
        )
    ]

def n_stop_words_segmentation(
        text: str, 
        language: str = "english", 
        n: int = 5,
        remove_stopwords: bool = False
) -> list[str]:

    """
    A text is segmented every N stop words.

    Parameters
    ----------
    text:
        Input text to be segmented.
    language:
        Language used when stopword removal is enabled.
        Default is 'english'.
    n:
        Number of stop words after which to segment the text.
        Default is 5.
    remove_stopwords:
        Whether to remove stopwords from each segment.
        Default is False.

    Returns
    -------
    list[str]
        List of segments created every N stop words.

    """

    clean_text = _normalize_text(text)

    if not clean_text:

        return []
    
    if n <= 0:
        raise InvalidSegmentationParameterError(
            "El número de palabras de parada debe ser mayor que cero."
        )
    
    tokens = get_tokens(text)

    n_tokens = len(tokens)

    if n_tokens <= 0:
        raise InvalidSegmentationParameterError(
            "El número de tokens debe ser mayor que cero."
        )
    
    stop = set(stopwords.words(language))

    stopword_count = sum(
        1
        for token in tokens
        if token in stop
    )

    if stopword_count / n_tokens < 0.2:

        raise InvalidSegmentationParameterError(
            "El texto no contiene suficientes palabras de parada para segmentar."
        ) 

    sub_tokens = []

    n_stop_words_grams = []

    i = 0       

    for token in tokens:

        if not remove_stopwords:

            sub_tokens.append(token)
        
        else:

            if token not in stop:

                sub_tokens.append(token)

        if token in stop:

            i += 1

            if i == n:       

                n_stop_words_grams.append(" ".join(sub_tokens).strip())

                sub_tokens = []

                i = 0
    
    if sub_tokens:

        n_stop_words_grams.append(" ".join(sub_tokens).strip())
            
    
    return n_stop_words_grams