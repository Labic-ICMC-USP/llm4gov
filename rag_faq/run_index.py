import pandas as pd
from rag_faq.config import load_config
from rag_faq.indexer import generate_faqs
from rag_faq.embedder import embed_faqs
import os

# Load LLM-generated texts
df = pd.read_csv("data/dataset_sample.csv")
texts = df["text"].tolist()

# Load config
config = load_config("config.yaml")

# Define project name
project_name = "llm_test_project"
project_dir = os.path.join(config["paths"]["projects_dir"], project_name)
os.makedirs(project_dir, exist_ok=True)

# Generate FAQs
generate_faqs(config, project_dir, texts)

# Generate embeddings
embed_faqs(config, project_dir)
