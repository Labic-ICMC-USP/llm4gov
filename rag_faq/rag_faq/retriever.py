import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_similar_faqs(config, project_dir, user_question):
    """
    Retrieve top-k FAQ entries most similar to the user question.
    Returns a list of dictionaries with source_text, question, answer.
    """

    # Load embedding model
    model_name = config["embedding"]["model"]
    model = SentenceTransformer(model_name)

    # Load stored embeddings and corresponding data
    embeddings = np.load(os.path.join(project_dir, "embeddings.npy"))
    df = pd.read_csv(os.path.join(project_dir, "faq.csv"))

    # Encode the user question
    query_embedding = model.encode(user_question, normalize_embeddings=True).reshape(1, -1)

    # Compute cosine similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]
    df["score"] = scores

    # Select top-k
    top_k = config["retrieval"]["top_k"]
    top_results = df.nlargest(top_k, "score")

    return top_results[["source_text", "question", "answer", "score"]].to_dict(orient="records")
