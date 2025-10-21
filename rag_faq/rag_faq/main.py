import os
import argparse
from rag_faq.config import load_config
from rag_faq.indexer import generate_faqs
from rag_faq.embedder import embed_faqs
from rag_faq.generator import run_rag

def main():
    # Argument parser for CLI interface
    parser = argparse.ArgumentParser(description="RAG with FAQ generation")
    parser.add_argument("--mode", choices=["index", "query"], required=True, help="Execution mode: index or query")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    project_dir = os.path.join(config["paths"]["projects_dir"], args.project)
    os.makedirs(project_dir, exist_ok=True)


    if args.mode == "index":
        generate_faqs(config, project_dir)
        embed_faqs(config, project_dir)

    elif args.mode == "query":
        run_rag(config, project_dir)

if __name__ == "__main__":
    main()
