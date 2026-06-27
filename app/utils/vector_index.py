import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import faiss
except ModuleNotFoundError:  # type: ignore
    faiss = None  # type: ignore[assignment]

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Adapter that owns all FAISS-specific concerns.

    Interface:
        search(embedding, index_name, top_k) -> (distances, indices)

    The caller never touches faiss.Index directly; this module hides
    IVF/PQ/nprobe details and index lifecycle.
    """

    def __init__(self, index_dir: Path, nprobe: int = 16):
        if faiss is None:
            raise RuntimeError("faiss is not installed; VectorIndex cannot be used.")
        self._index_dir = Path(index_dir)
        self._nprobe = nprobe
        self._indexes: Dict[str, Any] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        for entry in self._index_dir.iterdir():
            if not entry.is_file() or not entry.suffix == ".index":
                continue
            index_name = entry.stem
            index = faiss.read_index(str(entry))  # type: ignore[union-attr]
            if hasattr(index, "nprobe"):
                index.nprobe = self._nprobe
                logger.info("Loaded %s with nprobe=%d", index_name, self._nprobe)
            self._indexes[index_name] = index
            logger.info("Loaded FAISS index: %s", index_name)

        if not self._indexes:
            raise FileNotFoundError(f"No .index files found in {self._index_dir}")

    def search(
        self, embedding: np.ndarray, index_name: str, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a single FAISS index.

        Returns:
            distances, indices — both shape (1, top_k)
        """
        index = self._indexes.get(index_name)
        if index is None:
            raise ValueError(f"Unknown index name: {index_name!r}")

        query_vector = np.array([embedding]).astype("float32")
        distances, indices = index.search(query_vector, top_k)
        valid_count = int(np.sum(indices[0] != -1))
        logger.debug(
            "index=%s top_k=%d valid_results=%d",
            index_name,
            top_k,
            valid_count,
        )
        return distances, indices

    def list_indexes(self) -> list[str]:
        return list(self._indexes.keys())
