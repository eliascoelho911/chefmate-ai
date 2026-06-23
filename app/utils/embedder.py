from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


def load_embedding_model(config):
    """
    Load SentenceTransformer model specified in the config.
    """
    model_name = config["embedding"]["model_name"]
    return SentenceTransformer(model_name)


def embed_text(text: str, model: SentenceTransformer) -> list:
    """
    Embed a single string input using the SentenceTransformer model.
    Returns a list (embedding vector).
    """
    embedding = model.encode([text], show_progress_bar=False)
    return embedding[0].tolist()


def embed_texts(
    texts: list[str], model: SentenceTransformer, batch_size: int = 32
) -> list[list[float]]:
    """
    Embed a list of texts in batches using the SentenceTransformer model.
    Returns a list of embedding vectors.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    return [emb.tolist() for emb in embeddings]


class SentenceTransformerEmbedder:
    """
    Adapter that satisfies the Embedder protocol using a loaded SentenceTransformer model.
    """

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()
