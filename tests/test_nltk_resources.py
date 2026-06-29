from __future__ import annotations

from zipfile import BadZipFile

import pytest

import rp_segmentation.nltk_resources as nltk_resources
from rp_segmentation.exceptions import NLTKResourceError

STOPWORDS_RESOURCE = "stopwords"
STOPWORDS_PATH = "corpora/stopwords"

PUNKT_TAB_RESOURCE = "punkt_tab"
PUNKT_TAB_PATH = "tokenizers/punkt_tab"


def _clear_ensure_nltk_resource_cache() -> None:
    cache_clear = getattr(nltk_resources.ensure_nltk_resource, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def clear_nltk_resource_cache():
    """
    Clears the ensure_nltk_resource cache before and after each test.

    This prevents @cache from making tests order-dependent.
    """
    _clear_ensure_nltk_resource_cache()
    yield
    _clear_ensure_nltk_resource_cache()


def _error_message(exc_info: pytest.ExceptionInfo[NLTKResourceError]) -> str:
    return str(exc_info.value).lower()


def assert_error_mentions(
    exc_info: pytest.ExceptionInfo[NLTKResourceError],
    *expected_parts: str,
) -> None:
    message = _error_message(exc_info)

    for part in expected_parts:
        assert part.lower() in message


def assert_error_mentions_any(
    exc_info: pytest.ExceptionInfo[NLTKResourceError],
    *expected_parts: str,
) -> None:
    message = _error_message(exc_info)

    assert any(part.lower() in message for part in expected_parts)


def test_ensure_nltk_resource_rejects_unsupported_resource():
    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource("unknown_resource")

    assert_error_mentions(exc_info, "unsupported", "unknown_resource")

    for resource_name in nltk_resources.NLTK_RESOURCES:
        assert resource_name in str(exc_info.value)


def test_ensure_nltk_resource_does_not_download_when_resource_exists(monkeypatch):
    find_calls = []

    def fake_find(resource_path):
        find_calls.append(resource_path)
        return object()

    def fake_download(resource_name, quiet=True):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    nltk_resources.ensure_nltk_resource(STOPWORDS_RESOURCE)

    assert find_calls == [STOPWORDS_PATH]


def test_ensure_nltk_resource_downloads_missing_resource(monkeypatch):
    find_calls = []
    download_calls = []

    def fake_find(resource_path):
        find_calls.append(resource_path)

        if len(find_calls) == 1:
            raise LookupError("resource not found")

        return object()

    def fake_download(resource_name, quiet=True):
        download_calls.append((resource_name, quiet))
        return True

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    nltk_resources.ensure_nltk_resource(STOPWORDS_RESOURCE)

    assert find_calls == [STOPWORDS_PATH, STOPWORDS_PATH]
    assert download_calls == [(STOPWORDS_RESOURCE, True)]


def test_ensure_nltk_resource_uses_cache_after_success(monkeypatch):
    find_calls = []

    def fake_find(resource_path):
        find_calls.append(resource_path)
        return object()

    def fake_download(resource_name, quiet=True):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    nltk_resources.ensure_nltk_resource(STOPWORDS_RESOURCE)
    nltk_resources.ensure_nltk_resource(STOPWORDS_RESOURCE)

    assert find_calls == [STOPWORDS_PATH]


def test_ensure_nltk_resource_raises_when_existing_resource_is_corrupted(
    monkeypatch,
):
    def fake_find(resource_path):
        raise BadZipFile("corrupted resource")

    def fake_download(resource_name, quiet=True):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert_error_mentions(exc_info, "corrupt", PUNKT_TAB_RESOURCE)


def test_ensure_nltk_resource_raises_when_download_fails(monkeypatch):
    def fake_find(resource_path):
        raise LookupError("resource not found")

    def fake_download(resource_name, quiet=True):
        raise RuntimeError("download failed")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert_error_mentions(exc_info, "nltk", "resource", PUNKT_TAB_RESOURCE)
    assert_error_mentions_any(exc_info, "download", "locate", "located")


def test_ensure_nltk_resource_raises_when_download_returns_false(monkeypatch):
    def fake_find(resource_path):
        raise LookupError("resource not found")

    def fake_download(resource_name, quiet=True):
        return False

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert_error_mentions(exc_info, "nltk", "resource", PUNKT_TAB_RESOURCE)
    assert_error_mentions_any(exc_info, "download", "locate", "located")


def test_ensure_nltk_resource_raises_when_download_raises_bad_zip_file(
    monkeypatch,
):
    def fake_find(resource_path):
        raise LookupError("resource not found")

    def fake_download(resource_name, quiet=True):
        raise BadZipFile("corrupted download")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert_error_mentions(exc_info, "corrupt", PUNKT_TAB_RESOURCE)


def test_ensure_nltk_resource_raises_when_downloaded_resource_is_corrupted(
    monkeypatch,
):
    find_calls = []

    def fake_find(resource_path):
        find_calls.append(resource_path)

        if len(find_calls) == 1:
            raise LookupError("resource not found")

        raise BadZipFile("corrupted after download")

    def fake_download(resource_name, quiet=True):
        return True

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert find_calls == [PUNKT_TAB_PATH, PUNKT_TAB_PATH]
    assert_error_mentions(exc_info, "corrupt", PUNKT_TAB_RESOURCE)


def test_ensure_nltk_resource_raises_when_download_succeeds_but_lookup_fails(
    monkeypatch,
):
    def fake_find(resource_path):
        raise LookupError("resource still not found")

    def fake_download(resource_name, quiet=True):
        return True

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(NLTKResourceError) as exc_info:
        nltk_resources.ensure_nltk_resource(PUNKT_TAB_RESOURCE)

    assert_error_mentions(exc_info, "nltk", "resource", PUNKT_TAB_RESOURCE)
    assert_error_mentions_any(exc_info, "locate", "located", "found")


def test_ensure_required_nltk_resources_checks_all_supported_resources(monkeypatch):
    checked_resources = []

    def fake_ensure_nltk_resource(resource_name):
        checked_resources.append(resource_name)

    monkeypatch.setattr(
        nltk_resources,
        "ensure_nltk_resource",
        fake_ensure_nltk_resource,
    )

    nltk_resources.ensure_required_nltk_resources()

    assert checked_resources == list(nltk_resources.NLTK_RESOURCES)
