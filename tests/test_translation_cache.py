import json
import tempfile
from pathlib import Path

import pytest

from app.utils.translation_cache import (
    FileTranslationCache,
    InMemoryTranslationCache,
)


class TestInMemoryTranslationCache:
    """Unit tests for the volatile in-memory cache."""

    def test_get_returns_none_for_missing_key(self):
        cache = InMemoryTranslationCache()
        assert cache.get("frango", True) is None

    def test_set_and_get(self):
        cache = InMemoryTranslationCache()
        cache.set("frango", True, "chicken")
        assert cache.get("frango", True) == "chicken"

    def test_generalize_flag_matters(self):
        cache = InMemoryTranslationCache()
        cache.set("peito de frango", True, "chicken")
        cache.set("peito de frango", False, "chicken breast")

        assert cache.get("peito de frango", True) == "chicken"
        assert cache.get("peito de frango", False) == "chicken breast"

    def test_case_insensitive_keys(self):
        cache = InMemoryTranslationCache()
        cache.set("FrAnGo", True, "chicken")
        assert cache.get("frango", True) == "chicken"
        assert cache.get("FRANGO", True) == "chicken"

    def test_overwrite(self):
        cache = InMemoryTranslationCache()
        cache.set("arroz", True, "rice")
        cache.set("arroz", True, "brown rice")
        assert cache.get("arroz", True) == "brown rice"

    def test_load_and_save_are_noops(self):
        cache = InMemoryTranslationCache()
        # Should not raise
        cache.load()
        cache.save()


class TestFileTranslationCache:
    """Unit tests for the persistent file-backed cache."""

    def test_persist_and_load(self, tmp_path: Path):
        cache_path = tmp_path / "translations.json"
        cache = FileTranslationCache(str(cache_path))
        cache.set("frango", True, "chicken")
        cache.set("arroz", False, "rice")
        cache.save()

        # Create a fresh instance pointing at the same file
        cache2 = FileTranslationCache(str(cache_path))
        cache2.load()

        assert cache2.get("frango", True) == "chicken"
        assert cache2.get("arroz", False) == "rice"
        assert cache2.get("frango", False) is None

    def test_load_missing_file_does_not_raise(self, tmp_path: Path):
        cache = FileTranslationCache(str(tmp_path / "nope.json"))
        cache.load()  # should not raise
        assert cache.get("anything", True) is None

    def test_load_malformed_json_logs_warning(self, tmp_path: Path, caplog):
        cache_path = tmp_path / "bad.json"
        cache_path.write_text("not json")
        cache = FileTranslationCache(str(cache_path))
        cache.load()
        assert "Failed to load translation cache" in caplog.text

    def test_load_non_dict_root_logs_warning(self, tmp_path: Path, caplog):
        cache_path = tmp_path / "list.json"
        cache_path.write_text(json.dumps([1, 2, 3]))
        cache = FileTranslationCache(str(cache_path))
        cache.load()
        assert "Translation cache root is not a dict" in caplog.text

    def test_save_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "cache.json"
        cache = FileTranslationCache(str(deep_path))
        cache.set("leite", True, "milk")
        cache.save()
        assert deep_path.exists()

    def test_save_noop_when_empty(self, tmp_path: Path):
        cache_path = tmp_path / "empty.json"
        cache = FileTranslationCache(str(cache_path))
        cache.save()
        # File should NOT be created if store is empty
        assert not cache_path.exists()

    def test_json_format(self, tmp_path: Path):
        cache_path = tmp_path / "fmt.json"
        cache = FileTranslationCache(str(cache_path))
        cache.set("ovo", True, "egg")
        cache.save()

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data == {"ovo|True": "egg"}

    def test_whitespace_stripping_in_keys(self, tmp_path: Path):
        cache = FileTranslationCache(str(tmp_path / "w.json"))
        cache.set("  peito de frango  ", True, "chicken")
        assert cache.get("peito de frango", True) == "chicken"
