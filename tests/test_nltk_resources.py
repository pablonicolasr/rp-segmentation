import pytest

import rp_segmentation.nltk_resources as nltk_resources
from rp_segmentation.exceptions import NLTKResourceError


@pytest.fixture(autouse=True)
def clear_nltk_resource_cache():
    cache_clear = getattr(nltk_resources.ensure_nltk_resource, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()

    yield

    cache_clear = getattr(nltk_resources.ensure_nltk_resource, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()


def test_ensure_nltk_resource_rejects_unsupported_resource():
    with pytest.raises(
        NLTKResourceError,
        match="Unsupported NLTK resource",
    ):
        nltk_resources.ensure_nltk_resource("unknown_resource")


def test_ensure_nltk_resource_does_not_download_when_resource_exists(monkeypatch):
    find_calls = []

    def fake_find(resource_path):
        find_calls.append(resource_path)
        return object()

    def fake_download(resource_name, quiet=True):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    nltk_resources.ensure_nltk_resource("stopwords")

    assert find_calls == ["corpora/stopwords"]


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

    nltk_resources.ensure_nltk_resource("stopwords")

    assert find_calls == [
        "corpora/stopwords",
        "corpora/stopwords",
    ]

    assert download_calls == [
        ("stopwords", True),
    ]


def test_ensure_nltk_resource_raises_when_download_or_lookup_fails(monkeypatch):
    def fake_find(resource_path):
        raise LookupError("resource not found")

    def fake_download(resource_name, quiet=True):
        raise RuntimeError("download failed")

    monkeypatch.setattr(nltk_resources.nltk.data, "find", fake_find)
    monkeypatch.setattr(nltk_resources.nltk, "download", fake_download)

    with pytest.raises(
        NLTKResourceError,
        match="Could not download or locate the NLTK resource",
    ):
        nltk_resources.ensure_nltk_resource("punkt_tab")


def test_ensure_required_nltk_resources_checks_all_supported_resources(monkeypatch):
    checked_resources = []

    monkeypatch.setattr(
        nltk_resources,
        "ensure_nltk_resource",
        lambda resource_name: checked_resources.append(resource_name),
    )

    nltk_resources.ensure_required_nltk_resources()

    assert checked_resources == list(nltk_resources.NLTK_RESOURCES)