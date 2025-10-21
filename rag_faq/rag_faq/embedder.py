import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def embed_faqs(config, project_dir):
    """
    Generate sentence embeddings for each FAQ pair (question + answer).
    Saves the result in the project directory.
    """
    # Load embedding model
    model_name = config["embedding"]["model"]
    model = SentenceTransformer(model_name)

    # Load generated FAQ CSV
    faq_path = os.path.join(project_dir, "faq.csv")
    if not os.path.exists(faq_path):
        raise FileNotFoundError(f"FAQ file not found: {faq_path}")

    df = pd.read_csv(faq_path)

    texts = (df["question"]).tolist()

    # Generate embeddings
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings"):
        vec = model.encode(text, normalize_embeddings=True)
        embeddings.append(vec)

    # Save embeddings to .npy
    embeddings = np.array(embeddings)
    np.save(os.path.join(project_dir, "embeddings.npy"), embeddings)

    # Save enhanced CSV for debugging or visualization
    df["embedding_text"] = texts
    df.to_csv(os.path.join(project_dir, "faq_with_embeddings.csv"), index=False, encoding="utf-8")

    print(f"✅ Embeddings saved to: {project_dir}")
