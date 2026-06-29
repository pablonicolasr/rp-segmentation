import importlib
import importlib.metadata
import sys


def test_public_api_exports_expected_names():
    import rp_segmentation

    assert "__version__" in rp_segmentation.__all__
    assert "ensure_required_nltk_resources" in rp_segmentation.__all__
    assert "sentence_segmentation" in rp_segmentation.__all__
    assert "paragraph_segmentation" in rp_segmentation.__all__
    assert "n_stop_words_segmentation" in rp_segmentation.__all__


def test_public_api_objects_are_importable():
    import rp_segmentation

    assert callable(rp_segmentation.ensure_required_nltk_resources)
    assert callable(rp_segmentation.sentence_segmentation)
    assert callable(rp_segmentation.paragraph_segmentation)
    assert callable(rp_segmentation.n_stop_words_segmentation)
    assert isinstance(rp_segmentation.__version__, str)


def test_version_fallback_when_package_metadata_is_not_found(monkeypatch):
    def fake_version(package_name):
        raise importlib.metadata.PackageNotFoundError(package_name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    sys.modules.pop("rp_segmentation", None)

    rp_segmentation = importlib.import_module("rp_segmentation")

    assert rp_segmentation.__version__ == "0.1.0"