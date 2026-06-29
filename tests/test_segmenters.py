import pytest

import rp_segmentation.segmenters as segmenters
from rp_segmentation.exceptions import InvalidSegmentationParameterError


@pytest.fixture(autouse=True)
def clear_stopwords_cache():
    """
    Prevents cached stopwords from leaking between tests.
    """
    cache_clear = getattr(segmenters._get_stopwords, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()

    yield

    cache_clear = getattr(segmenters._get_stopwords, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()


# ---------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------


def test_normalize_text_strips_leading_and_trailing_whitespace():
    assert segmenters._normalize_text("   Hello world   ") == "Hello world"


def test_normalize_text_preserves_internal_structure():
    text = "  Hello\n\nWorld\tinside  "

    assert segmenters._normalize_text(text) == "Hello\n\nWorld\tinside"


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        3.14,
        ["hello"],
        {"text": "hello"},
        ("hello",),
    ],
)
def test_normalize_text_rejects_non_string_values(value):
    with pytest.raises(TypeError, match="input text must be a string"):
        segmenters._normalize_text(value)


# ---------------------------------------------------------------------
# _validate_language
# ---------------------------------------------------------------------


def test_validate_language_normalizes_language():
    assert segmenters._validate_language("  ENGLISH  ") == "english"


def test_validate_language_accepts_spanish():
    assert segmenters._validate_language("Spanish") == "spanish"


@pytest.mark.parametrize(
    "language",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_validate_language_rejects_invalid_values(language):
    with pytest.raises(
        InvalidSegmentationParameterError,
        match="language parameter must be a non-empty string",
    ):
        segmenters._validate_language(language)


# ---------------------------------------------------------------------
# _get_stopwords
# ---------------------------------------------------------------------


def test_get_stopwords_returns_frozenset(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and", "is"],
    )

    result = segmenters._get_stopwords("english")

    assert isinstance(result, frozenset)
    assert result == frozenset({"the", "and", "is"})


def test_get_stopwords_normalizes_language(monkeypatch):
    received = {}

    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    def fake_stopwords(language):
        received["language"] = language
        return ["the", "and"]

    monkeypatch.setattr(segmenters.stopwords, "words", fake_stopwords)

    result = segmenters._get_stopwords("  ENGLISH  ")

    assert result == frozenset({"the", "and"})
    assert received["language"] == "english"


def test_get_stopwords_ensures_nltk_resource(monkeypatch):
    called_resources = []

    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: called_resources.append(resource_name),
    )

    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and"],
    )

    segmenters._get_stopwords("english")

    assert called_resources == ["stopwords"]


def test_get_stopwords_uses_cache(monkeypatch):
    calls = []

    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    def fake_stopwords(language):
        calls.append(language)
        return ["the", "and"]

    monkeypatch.setattr(segmenters.stopwords, "words", fake_stopwords)

    first_result = segmenters._get_stopwords("english")
    second_result = segmenters._get_stopwords("english")

    assert first_result == frozenset({"the", "and"})
    assert second_result == frozenset({"the", "and"})
    assert calls == ["english"]


def test_get_stopwords_rejects_invalid_language():
    with pytest.raises(InvalidSegmentationParameterError):
        segmenters._get_stopwords("")


def test_get_stopwords_raises_custom_error_when_language_is_not_available(
    monkeypatch,
):
    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    def raise_os_error(language):
        raise OSError("No such file or directory")

    monkeypatch.setattr(segmenters.stopwords, "words", raise_os_error)

    with pytest.raises(
        InvalidSegmentationParameterError,
        match="Stopwords are not available",
    ):
        segmenters._get_stopwords("unknown-language")


# ---------------------------------------------------------------------
# get_tokens
# ---------------------------------------------------------------------


def test_get_tokens_extracts_lowercase_words():
    text = "Hello, John. How are you?"

    assert segmenters.get_tokens(text) == [
        "hello",
        "john",
        "how",
        "are",
        "you",
    ]


def test_get_tokens_discards_numbers_punctuation_symbols_and_emojis():
    text = "Python 3.12 is great!!! Price: $100 #AI 🚀"

    assert segmenters.get_tokens(text) == [
        "python",
        "is",
        "great",
        "price",
        "ai",
    ]


def test_get_tokens_keeps_unicode_letters():
    text = "Niño, pingüino, café, São Paulo, über."

    assert segmenters.get_tokens(text) == [
        "niño",
        "pingüino",
        "café",
        "são",
        "paulo",
        "über",
    ]


def test_get_tokens_splits_contractions_by_apostrophe():
    text = "Don't stop believing."

    assert segmenters.get_tokens(text) == [
        "don",
        "t",
        "stop",
        "believing",
    ]


def test_get_tokens_splits_hyphenated_words():
    text = "State-of-the-art models."

    assert segmenters.get_tokens(text) == [
        "state",
        "of",
        "the",
        "art",
        "models",
    ]


def test_get_tokens_splits_underscore_words():
    text = "hello_world test_case"

    assert segmenters.get_tokens(text) == [
        "hello",
        "world",
        "test",
        "case",
    ]


def test_get_tokens_returns_empty_list_for_empty_text():
    assert segmenters.get_tokens("") == []


def test_get_tokens_returns_empty_list_for_blank_text():
    assert segmenters.get_tokens("   ") == []


def test_get_tokens_returns_empty_list_when_text_has_no_letters():
    assert segmenters.get_tokens("1234 !!! ??? $%&") == []


def test_get_tokens_rejects_non_string_input():
    with pytest.raises(TypeError):
        segmenters.get_tokens(123)


def test_get_tokens_removes_stopwords(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "_get_stopwords",
        lambda language: frozenset({"how", "are", "you"}),
    )

    text = "Hello, John. How are you?"

    assert segmenters.get_tokens(
        text,
        language="english",
        remove_stopwords=True,
    ) == ["hello", "john"]


def test_get_tokens_passes_language_to_get_stopwords(monkeypatch):
    received = {}

    def fake_get_stopwords(language):
        received["language"] = language
        return frozenset({"cómo", "estás"})

    monkeypatch.setattr(segmenters, "_get_stopwords", fake_get_stopwords)

    text = "Hola, Juan. Cómo estás?"

    assert segmenters.get_tokens(
        text,
        language="spanish",
        remove_stopwords=True,
    ) == ["hola", "juan"]

    assert received["language"] == "spanish"


# ---------------------------------------------------------------------
# _clean_segment
# ---------------------------------------------------------------------


def test_clean_segment_returns_normalized_string():
    text = "Hello, John. How are you?"

    assert segmenters._clean_segment(text) == "hello john how are you"


def test_clean_segment_returns_empty_string_for_non_word_segment():
    assert segmenters._clean_segment("!!! 123 ???") == ""


def test_clean_segment_removes_stopwords(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "_get_stopwords",
        lambda language: frozenset({"how", "are", "you"}),
    )

    text = "Hello, John. How are you?"

    assert segmenters._clean_segment(
        text,
        language="english",
        remove_stopwords=True,
    ) == "hello john"


def test_clean_segment_rejects_non_string_input():
    with pytest.raises(TypeError):
        segmenters._clean_segment(None)


# ---------------------------------------------------------------------
# sentence_segmentation
# ---------------------------------------------------------------------


def test_sentence_segmentation_segments_and_normalizes_sentences(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    monkeypatch.setattr(
        segmenters,
        "sent_tokenize",
        lambda text, language: ["Hello, John.", "How are you?"],
    )

    text = "Hello, John. How are you?"

    assert segmenters.sentence_segmentation(text) == [
        "hello john",
        "how are you",
    ]


def test_sentence_segmentation_ensures_punkt_resource(monkeypatch):
    called_resources = []

    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: called_resources.append(resource_name),
    )

    monkeypatch.setattr(
        segmenters,
        "sent_tokenize",
        lambda text, language: ["Hello."],
    )

    segmenters.sentence_segmentation("Hello.")

    assert called_resources == ["punkt_tab"]


def test_sentence_segmentation_passes_language_to_sent_tokenize(monkeypatch):
    received = {}

    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    def fake_sent_tokenize(text, language):
        received["text"] = text
        received["language"] = language
        return ["Hola, Juan.", "¿Cómo estás?"]

    monkeypatch.setattr(segmenters, "sent_tokenize", fake_sent_tokenize)

    segmenters.sentence_segmentation(
        "  Hola, Juan. ¿Cómo estás?  ",
        language="SPANISH",
    )

    assert received["text"] == "Hola, Juan. ¿Cómo estás?"
    assert received["language"] == "spanish"


def test_sentence_segmentation_discards_empty_segments(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    monkeypatch.setattr(
        segmenters,
        "sent_tokenize",
        lambda text, language: ["!!!", "Hello, John."],
    )

    assert segmenters.sentence_segmentation("!!! Hello, John.") == [
        "hello john",
    ]


def test_sentence_segmentation_returns_empty_list_for_empty_text():
    assert segmenters.sentence_segmentation("") == []


def test_sentence_segmentation_returns_empty_list_for_blank_text():
    assert segmenters.sentence_segmentation("   ") == []


def test_sentence_segmentation_rejects_invalid_language():
    with pytest.raises(InvalidSegmentationParameterError):
        segmenters.sentence_segmentation("Hello world.", language="")


def test_sentence_segmentation_rejects_non_string_text():
    with pytest.raises(TypeError):
        segmenters.sentence_segmentation(123)


def test_sentence_segmentation_removes_stopwords(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "ensure_nltk_resource",
        lambda resource_name: None,
    )

    monkeypatch.setattr(
        segmenters,
        "sent_tokenize",
        lambda text, language: ["Hello, John.", "How are you?"],
    )

    monkeypatch.setattr(
        segmenters,
        "_get_stopwords",
        lambda language: frozenset({"how", "are", "you"}),
    )

    text = "Hello, John. How are you?"

    assert segmenters.sentence_segmentation(
        text,
        language="english",
        remove_stopwords=True,
    ) == ["hello john"]


# ---------------------------------------------------------------------
# paragraph_segmentation
# ---------------------------------------------------------------------


def test_paragraph_segmentation_segments_and_normalizes_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."

    assert segmenters.paragraph_segmentation(text) == [
        "first paragraph",
        "second paragraph",
    ]


def test_paragraph_segmentation_supports_multiple_blank_lines():
    text = "First paragraph.\n\n\nSecond paragraph.\n   \nThird paragraph."

    assert segmenters.paragraph_segmentation(text) == [
        "first paragraph",
        "second paragraph",
        "third paragraph",
    ]


def test_paragraph_segmentation_discards_empty_segments():
    text = "!!!\n\nFirst paragraph."

    assert segmenters.paragraph_segmentation(text) == [
        "first paragraph",
    ]


def test_paragraph_segmentation_returns_empty_list_for_empty_text():
    assert segmenters.paragraph_segmentation("") == []


def test_paragraph_segmentation_returns_empty_list_for_blank_text():
    assert segmenters.paragraph_segmentation("   ") == []


def test_paragraph_segmentation_rejects_invalid_language():
    with pytest.raises(InvalidSegmentationParameterError):
        segmenters.paragraph_segmentation("First paragraph.", language="")


def test_paragraph_segmentation_rejects_non_string_text():
    with pytest.raises(TypeError):
        segmenters.paragraph_segmentation(None)


def test_paragraph_segmentation_removes_stopwords(monkeypatch):
    monkeypatch.setattr(
        segmenters,
        "_get_stopwords",
        lambda language: frozenset({"this", "is", "the"}),
    )

    text = "This is the first paragraph.\n\nThis is another one."

    assert segmenters.paragraph_segmentation(
        text,
        language="english",
        remove_stopwords=True,
    ) == [
        "first paragraph",
        "another one",
    ]


def test_paragraph_segmentation_passes_language_to_clean_segment(monkeypatch):
    received = []

    def fake_clean_segment(text, language="english", remove_stopwords=False):
        received.append((text, language, remove_stopwords))
        return "cleaned"

    monkeypatch.setattr(segmenters, "_clean_segment", fake_clean_segment)

    result = segmenters.paragraph_segmentation(
        "One.\n\nTwo.",
        language="SPANISH",
        remove_stopwords=True,
    )

    assert result == ["cleaned", "cleaned"]
    assert received == [
        ("One.", "spanish", True),
        ("Two.", "spanish", True),
    ]


# ---------------------------------------------------------------------
# n_stop_words_segmentation
# ---------------------------------------------------------------------


def test_n_stop_words_segmentation_returns_empty_list_for_empty_text():
    assert segmenters.n_stop_words_segmentation("") == []


def test_n_stop_words_segmentation_returns_empty_list_for_blank_text():
    assert segmenters.n_stop_words_segmentation("   ") == []


@pytest.mark.parametrize("n", [0, -1, -10])
def test_n_stop_words_segmentation_rejects_invalid_n(n):
    with pytest.raises(
        InvalidSegmentationParameterError,
        match="greater than zero",
    ):
        segmenters.n_stop_words_segmentation(
            "This is a valid text with enough stopwords.",
            n=n,
        )


def test_n_stop_words_segmentation_rejects_text_without_word_tokens():
    with pytest.raises(
        InvalidSegmentationParameterError,
        match="tokens must be greater than zero",
    ):
        segmenters.n_stop_words_segmentation("1234 !!! ???")


def test_n_stop_words_segmentation_segments_every_n_stopwords(monkeypatch):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and", "is", "of"],
    )

    text = "Alpha the beta and gamma is delta of omega."

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=2,
    ) == [
        "alpha the beta and",
        "gamma is delta of",
        "omega",
    ]


def test_n_stop_words_segmentation_segments_every_single_stopword(monkeypatch):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and"],
    )

    text = "Alpha the beta and gamma."

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=1,
    ) == [
        "alpha the",
        "beta and",
        "gamma",
    ]


def test_n_stop_words_segmentation_returns_single_segment_when_n_is_not_reached(
    monkeypatch,
):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and", "is", "of"],
    )

    text = "Alpha the beta and gamma is delta of omega."

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=10,
    ) == [
        "alpha the beta and gamma is delta of omega",
    ]


def test_n_stop_words_segmentation_removes_stopwords_from_segments(monkeypatch):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and", "is", "of"],
    )

    text = "Alpha the beta and gamma is delta of omega."

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=2,
        remove_stopwords=True,
    ) == [
        "alpha beta",
        "gamma delta",
        "omega",
    ]


def test_n_stop_words_segmentation_raises_when_stopword_ratio_is_too_low(
    monkeypatch,
):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the"],
    )

    text = "Alpha beta gamma delta epsilon the."

    with pytest.raises(
        InvalidSegmentationParameterError,
        match="enough stopwords",
    ):
        segmenters.n_stop_words_segmentation(
            text,
            language="english",
            n=1,
        )


def test_n_stop_words_segmentation_accepts_exact_minimum_stopword_ratio(
    monkeypatch,
):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the"],
    )

    text = "Alpha beta gamma delta the."

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=1,
    ) == [
        "alpha beta gamma delta the",
    ]


def test_n_stop_words_segmentation_passes_language_to_stopwords(monkeypatch):
    received = {}

    def fake_stopwords(language):
        received["language"] = language
        return ["el", "y"]

    monkeypatch.setattr(segmenters.stopwords, "words", fake_stopwords)

    text = "Casa el perro y gato."

    result = segmenters.n_stop_words_segmentation(
        text,
        language="spanish",
        n=1,
    )

    assert received["language"] == "spanish"
    assert result == [
        "casa el",
        "perro y",
        "gato",
    ]


def test_n_stop_words_segmentation_rejects_non_string_text():
    with pytest.raises(TypeError):
        segmenters.n_stop_words_segmentation(None)



def test_n_stop_words_segmentation_should_not_return_empty_segments(monkeypatch):
    monkeypatch.setattr(
        segmenters.stopwords,
        "words",
        lambda language: ["the", "and"],
    )

    text = "the and alpha"

    assert segmenters.n_stop_words_segmentation(
        text,
        language="english",
        n=2,
        remove_stopwords=True,
    ) == ["alpha"]



def test_n_stop_words_segmentation_should_wrap_unavailable_language(monkeypatch):
    def raise_os_error(language):
        raise OSError("No such file or directory")

    monkeypatch.setattr(segmenters.stopwords, "words", raise_os_error)

    with pytest.raises(InvalidSegmentationParameterError):
        segmenters.n_stop_words_segmentation(
            "This is a valid text.",
            language="unknown-language",
            n=1,
        )