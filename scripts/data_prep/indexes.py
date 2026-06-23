import logging
import os

import numpy as np
import faiss

logger = logging.getLogger(__name__)


def build_recipe_faiss_indexes(df, config):
    columns_to_embed = {
        "ingredients_cleaned": "ingredients_embedding",
        "ingredients_with_quantities": "ingredients_with_quantities_embedding",
        "name": "title_embedding",
    }

    index_dir = config["paths"]["faiss_index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    factory = config.get("faiss", {}).get("factory", "IVF4096,PQ48")

    for _, embed_col in columns_to_embed.items():
        logger.info("Building FAISS index for: %s (factory=%s)", embed_col, factory)
        embeddings = np.array(df[embed_col].tolist()).astype("float32")
        dim = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        if n_vectors < 50_000:
            logger.info("  Dataset small (%d), falling back to IndexFlatL2", n_vectors)
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.index_factory(dim, factory)
            logger.info("  Training on %d vectors...", n_vectors)
            assert not index.is_trained
            index.train(embeddings)
            assert index.is_trained

        index.add(embeddings)

        index_path = os.path.join(index_dir, f"{embed_col}.index")
        faiss.write_index(index, index_path)
        logger.info("  Saved FAISS index to: %s", index_path)
