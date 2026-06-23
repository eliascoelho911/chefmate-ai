from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

from app.utils.embedder import load_embedding_model


def generate_recipe_embeddings(df, config):
    """
    Generate embeddings for recipe data columns (ingredients, title, etc.)
    and add them as new columns to the DataFrame.
    """
    embedding_model = load_embedding_model(config)
    batch_size = config["embedding"]["batch_size"]

    columns_to_embed = {
        "ingredients_cleaned": "ingredients_embedding",
        "ingredients_with_quantities": "ingredients_with_quantities_embedding",
        "name": "title_embedding",
    }

    for text_col, embed_col in columns_to_embed.items():
        print(f"Embedding column: {text_col}")
        texts = df[text_col].fillna("").astype(str).tolist()
        df[embed_col] = _embed_texts(texts, embedding_model, batch_size)

    return df


def _embed_texts(
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
