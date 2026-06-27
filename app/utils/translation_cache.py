import json
import logging
import os
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class TranslationCache(Protocol):
    """
    Seam for ingredient translation caching.

    Implementations may be in-memory only or persisted to disk.
    The interface is intentionally minimal so callers get maximum leverage
    with minimum surface area.
    """

    def get(self, term: str, generalize: bool) -> str | None:
        """Return cached translation or None if absent."""
        ...

    def set(self, term: str, generalize: bool, translation: str) -> None:
        """Store a translation. Overwrites existing entries silently."""
        ...

    def load(self) -> None:
        """Hydrate the cache from external storage (no-op for pure memory)."""
        ...

    def save(self) -> None:
        """Persist the cache to external storage (no-op for pure memory)."""
        ...


class InMemoryTranslationCache:
    """
    Simple volatile cache backed by a dict.

    Key: (term.lower().strip(), generalize)
    Value: translation string
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, bool], str] = {}

    def get(self, term: str, generalize: bool) -> str | None:
        key = (term.strip().lower(), generalize)
        return self._store.get(key)

    def set(self, term: str, generalize: bool, translation: str) -> None:
        key = (term.strip().lower(), generalize)
        self._store[key] = translation

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass


class FileTranslationCache(InMemoryTranslationCache):
    """
    Persistent cache that stores translations as JSON on disk.

    Loads eagerly on ``load()`` and flushes eagerly on ``save()``.
    Safe for small-to-medium workloads; for very high concurrency
    consider a real KV store (Redis, etc.).
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = Path(path)

    def load(self) -> None:
        if not self._path.exists():
            logger.info(
                "Translation cache file not found at %s; starting empty.", self._path
            )
            return

        try:
            with self._path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load translation cache from %s: %s", self._path, exc
            )
            return

        if not isinstance(raw, dict):
            logger.warning("Translation cache root is not a dict; ignoring.")
            return

        count = 0
        for key_str, value in raw.items():
            # Key format: "term|generalize"  e.g. "frango|True"
            if "|" not in key_str:
                continue
            term_part, flag_part = key_str.rsplit("|", 1)
            try:
                flag = flag_part.lower() == "true"
            except Exception:
                continue
            if isinstance(value, str):
                self._store[(term_part, flag)] = value
                count += 1

        logger.info("Loaded %d entries from translation cache %s", count, self._path)

    def save(self) -> None:
        if not self._store:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                f"{term}|{generalize}": translation
                for (term, generalize), translation in self._store.items()
            }
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            logger.info(
                "Saved %d entries to translation cache %s", len(data), self._path
            )
        except OSError as exc:
            logger.error("Failed to save translation cache to %s: %s", self._path, exc)
