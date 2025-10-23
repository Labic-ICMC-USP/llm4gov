import os
import argparse
from rag_faq.config import load_config
from rag_faq.indexer import generate_faqs
from rag_faq.embedder import embed_faqs
from rag_faq.generator import run_rag
from run_index import run_index, run_batch_indexing

def main():
    # Argument parser for CLI interface
    parser = argparse.ArgumentParser(description="RAG with FAQ generation")
    parser.add_argument("--mode", choices=["index", "query"], required=True, help="Execution mode: index or query")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML")
    
    # Index mode specific arguments
    parser.add_argument("--data-source", help="Data source name (from config) or CSV file path")
    parser.add_argument("--persona", choices=["aluno", "professor", "pesquisador"], help="Persona type for individual mode")
    parser.add_argument("--index-mode", choices=["individual", "unificado"], default="individual", help="Indexing mode: individual (single persona) or unificado (multi-persona)")
    parser.add_argument("--batch", action="store_true", help="Run batch indexing for all data sources")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    project_dir = os.path.join(config["paths"]["projects_dir"], args.project)
    os.makedirs(project_dir, exist_ok=True)

    if args.mode == "index":
        if args.batch:
            # Batch processing for multiple data sources
            run_batch_indexing(config, project_dir, mode=args.index_mode)
        else:
            # Single project indexing
            run_index(config, project_dir, 
                    data_source=args.data_source, 
                    persona_type=args.persona, 
                    mode=args.index_mode)

    elif args.mode == "query":
        run_rag(config, project_dir)

if __name__ == "__main__":
    main()
