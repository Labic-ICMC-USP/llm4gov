#!/usr/bin/env python3
"""
RAG FAQ Evaluation Runner

This script evaluates both the retrieval and generation performance of the RAG FAQ system.
It generates question variations, creates ground truth datasets, and evaluates using
retrieval metrics (Hit@k, MRR, Precision@k, Recall@k, NDCG@k) and
generation metrics (LLM score, correctness, completeness, relevance).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_faq.config import load_config
from rag_faq.evaluator import Evaluator, ProjectEvaluationResults

def run_evaluation_for_project(evaluator: Evaluator, project_dir: str, 
                              output_dir: str = "evaluation_results") -> ProjectEvaluationResults:
    """
    Run evaluation for a single project directory, including both retrieval and generation performance.

    Args:
        evaluator (Evaluator): The evaluator instance.
        project_dir (str): The path to the project directory to evaluate.
        output_dir (str): The directory to save evaluation results. Defaults to "evaluation_results".

    Returns:
        ProjectEvaluationResults: The aggregated evaluation results for the project.
    """

    project_name = os.path.basename(project_dir)
    print(f"\nINFO: Evaluating project: {project_name}")
    print(f"DIR: Project directory: {project_dir}")
    
    try:
        # Create evaluation dataset (shared for both retrieval and generation)
        print("WRITE: Creating evaluation dataset...")
        evaluation_dataset = evaluator.create_evaluation_dataset(project_dir)
        print(f"SUCCESS: Created {len(evaluation_dataset)} test questions")
        
        # Run retrieval evaluation
        print("INFO: Running retrieval evaluation...")
        retrieval_results = evaluator.evaluate_retrieval(project_dir, evaluation_dataset)
        
        # Run generation evaluation (using same questions)
        print("INFO: Running generation evaluation...")
        results = evaluator.evaluate_generation(project_dir, evaluation_dataset, retrieval_results)
        
        # Calculate project metrics
        print("CALC: Calculating metrics...")
        project_results = evaluator.calculate_project_metrics(results, project_dir)
        
        # Save results
        print("SAVE: Saving results...")
        evaluator.save_results(project_results, output_dir)
        
        # Print summary
        print(f"\nRESULTS: Results for {project_name}:")
        print(f"   Total questions evaluated (variations only): {project_results.total_questions}")
        print(f"   Note: Metrics exclude original questions to avoid bias")
        print(f"\n   Retrieval Metrics:")
        print(f"      Hit@1: {project_results.hit_at_k[1]:.3f}")
        print(f"      Hit@3: {project_results.hit_at_k[3]:.3f}")
        print(f"      Hit@5: {project_results.hit_at_k[5]:.3f}")
        print(f"      MRR: {project_results.mrr:.3f}")
        print(f"      Precision@1: {project_results.precision_at_k[1]:.3f}")
        print(f"      Precision@3: {project_results.precision_at_k[3]:.3f}")
        print(f"      Precision@5: {project_results.precision_at_k[5]:.3f}")
        print(f"      Recall@1: {project_results.recall_at_k[1]:.3f}")
        print(f"      Recall@3: {project_results.recall_at_k[3]:.3f}")
        print(f"      Recall@5: {project_results.recall_at_k[5]:.3f}")
        print(f"      NDCG@1: {project_results.ndcg_at_k[1]:.3f}")
        print(f"      NDCG@3: {project_results.ndcg_at_k[3]:.3f}")
        print(f"      NDCG@5: {project_results.ndcg_at_k[5]:.3f}")
        
        if project_results.avg_llm_score is not None:
            print(f"\n   Generation Metrics (LLM Judge):")
            print(f"      Average Score: {project_results.avg_llm_score:.2f}")
            print(f"      Average Correctness: {project_results.avg_correctness:.2f}")
            print(f"      Average Completeness: {project_results.avg_completeness:.2f}")
            print(f"      Average Relevance: {project_results.avg_relevance:.2f}")
        
        return project_results
        
    except Exception as e:
        print(f"ERROR: Error evaluating {project_name}: {e}")
        return None

def run_evaluation_all_projects(config: Dict, 
                               output_dir: str = "evaluation_results", 
                               specific_projects: List[str] = None) -> List[ProjectEvaluationResults]:
    """
    Run evaluation for all projects or specific projects.

    Args:
        config (Dict): The configuration file.
        output_dir (str): The directory to save evaluation results. Defaults to "evaluation_results".
        specific_projects (List[str]): A list of project paths to evaluate. If None, all projects will be evaluated.
    """
    evaluator = Evaluator(config)
    
    # Find project directories
    if specific_projects:
        project_dirs = evaluator.find_project_directories(specific_projects)
    else:
        print("ERROR: No project paths specified! Use --projects to specify which projects to evaluate.")
        return []
    
    if not project_dirs:
        print("ERROR: No project directories found!")
        return []
    
    print(f"TARGET: Found {len(project_dirs)} project directories to evaluate")
    
    all_results = []
    
    for project_dir in project_dirs:
        project_results = run_evaluation_for_project(
            evaluator, project_dir, output_dir
        )
        if project_results:
            all_results.append(project_results)
    
    return all_results

def get_parent_project_path(project_path: str) -> str:
    """
    Get the parent project path by removing the last folder (e.g., individual/unificado).
    Example: "projects/batch_proj/ppp_bcc/individual" -> "projects/batch_proj/ppp_bcc"
    """
    normalized_path = project_path.replace('\\', '/')
    path_parts = normalized_path.split('/')
    # Remove last part if it's individual/unificado, otherwise keep all
    if path_parts and path_parts[-1] in ['individual', 'unificado']:
        path_parts = path_parts[:-1]
    return '/'.join([p for p in path_parts if p])

def create_aggregated_summaries(all_results: List[ProjectEvaluationResults], output_dir: str):
    """
    Create aggregated summaries for projects that share the same parent path.
    Groups results by their parent path and creates aggregated metrics.

    Args:
        all_results (List[ProjectEvaluationResults]): A list of evaluation results.
        output_dir (str): The directory to save the aggregated summaries.
    """
    # Group results by parent project path
    grouped_results = {}
    for result in all_results:
        parent_path = get_parent_project_path(result.project_path)
        if parent_path not in grouped_results:
            grouped_results[parent_path] = []
        grouped_results[parent_path].append(result)
    
    # Create summary for each group
    for parent_path, grouped_result_list in grouped_results.items():
        # Calculate aggregated metrics
        total_questions = sum(r.total_questions for r in grouped_result_list)
        
        overall_hit_at_k = {}
        for k in [1, 3, 5]:
            overall_hit_at_k[k] = sum(r.hit_at_k[k] * r.total_questions for r in grouped_result_list) / total_questions
        
        overall_mrr = sum(r.mrr * r.total_questions for r in grouped_result_list) / total_questions
        
        overall_precision_at_k = {}
        overall_recall_at_k = {}
        overall_ndcg_at_k = {}
        
        for k in [1, 3, 5]:
            overall_precision_at_k[k] = sum(r.precision_at_k[k] * r.total_questions for r in grouped_result_list) / total_questions
            overall_recall_at_k[k] = sum(r.recall_at_k[k] * r.total_questions for r in grouped_result_list) / total_questions
            overall_ndcg_at_k[k] = sum(r.ndcg_at_k[k] * r.total_questions for r in grouped_result_list) / total_questions
        
        # Calculate generation metrics
        generation_results = [r for r in grouped_result_list if r.avg_llm_score is not None]
        generation_metrics = None
        if generation_results:
            total_gen_questions = sum(r.total_questions for r in generation_results)
            generation_metrics = {
                "avg_llm_score": sum(r.avg_llm_score * r.total_questions for r in generation_results) / total_gen_questions,
                "avg_correctness": sum(r.avg_correctness * r.total_questions for r in generation_results if r.avg_correctness is not None) / total_gen_questions,
                "avg_completeness": sum(r.avg_completeness * r.total_questions for r in generation_results if r.avg_completeness is not None) / total_gen_questions,
                "avg_relevance": sum(r.avg_relevance * r.total_questions for r in generation_results if r.avg_relevance is not None) / total_gen_questions,
                "total_evaluated": total_gen_questions
            }
        
        # Determine output folder path - use nested structure: batch_proj/ppp_bcc
        normalized_path = parent_path.replace('\\', '/')
        path_parts = normalized_path.split('/')
        relevant_parts = [part for part in path_parts if part and part not in ['projects']]
        
        if relevant_parts:
            # Create nested path: batch_proj/ppp_bcc
            project_output_dir = os.path.join(output_dir, *relevant_parts)
        else:
            project_output_dir = output_dir
        
        os.makedirs(project_output_dir, exist_ok=True)
        
        # Create aggregated summary
        aggregated_summary = {
            "project_path": parent_path,
            "evaluation_summary": {
                "total_subfolders": len(grouped_result_list),
                "total_questions": total_questions,
                "retrieval_metrics": {
                    "hit_at_1": overall_hit_at_k[1],
                    "hit_at_3": overall_hit_at_k[3],
                    "hit_at_5": overall_hit_at_k[5],
                    "mrr": overall_mrr,
                    "precision_at_1": overall_precision_at_k[1],
                    "precision_at_3": overall_precision_at_k[3],
                    "precision_at_5": overall_precision_at_k[5],
                    "recall_at_1": overall_recall_at_k[1],
                    "recall_at_3": overall_recall_at_k[3],
                    "recall_at_5": overall_recall_at_k[5],
                    "ndcg_at_1": overall_ndcg_at_k[1],
                    "ndcg_at_3": overall_ndcg_at_k[3],
                    "ndcg_at_5": overall_ndcg_at_k[5]
                },
                "generation_metrics": generation_metrics
            },
            "subfolder_results": [
                {
                    "project_path": r.project_path,
                    "total_questions": r.total_questions,
                    "hit_at_k": {str(k): v for k, v in r.hit_at_k.items()},
                    "mrr": r.mrr,
                    "precision_at_k": {str(k): v for k, v in r.precision_at_k.items()},
                    "recall_at_k": {str(k): v for k, v in r.recall_at_k.items()},
                    "ndcg_at_k": {str(k): v for k, v in r.ndcg_at_k.items()},
                    "generation_metrics": {
                        "avg_llm_score": r.avg_llm_score,
                        "avg_correctness": r.avg_correctness,
                        "avg_completeness": r.avg_completeness,
                        "avg_relevance": r.avg_relevance
                    } if r.avg_llm_score is not None else None
                }
                for r in grouped_result_list
            ]
        }
        
        # Save aggregated summary
        summary_path = os.path.join(project_output_dir, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_summary, f, indent=2, ensure_ascii=False)

def extract_project_folder(project_path: str) -> str:
    """
    Extract project folder name from project path.
    This is the first folder after 'projects' (if path starts with 'projects'),
    or the first meaningful folder in the path.
    Example: "projects/batch_proj/ppp_bcc/individual" -> "batch_proj"
    """
    normalized_path = project_path.replace('\\', '/')
    path_parts = [p for p in normalized_path.split('/') if p]  # Remove empty parts
    
    # If path starts with 'projects', return the next folder
    if path_parts and path_parts[0] == 'projects':
        if len(path_parts) > 1:
            return path_parts[1]
    
    # Otherwise, return the first meaningful folder
    if path_parts:
        return path_parts[0]
    
    return "unknown"

def generate_summary_report(all_results: List[ProjectEvaluationResults], output_dir: str):
    """
    Generate a summary report across all projects.
    Loads existing summary if present and merges with new results.
    Saves the summary inside the project folder (e.g., batch_proj/evaluation_summary.json)
    """
    if not all_results:
        print("ERROR: No results to summarize")
        return
    
    # Determine project folder from first result
    project_folder = extract_project_folder(all_results[0].project_path)
    project_folder_path = os.path.join(output_dir, project_folder)
    os.makedirs(project_folder_path, exist_ok=True)
    
    # Load existing summary if it exists
    summary_path = os.path.join(project_folder_path, "evaluation_summary.json")
    existing_results = []
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("project_results", [])
        except Exception as e:
            print(f"WARNING: Could not load existing summary: {e}")
    
    # Combine existing and new results
    # Create a set of project paths to avoid duplicates
    existing_paths = {r.get("project_path") if isinstance(r, dict) else r.project_path for r in existing_results}
    
    # Convert new results to dict format
    new_project_results = []
    for r in all_results:
        if r.project_path not in existing_paths:
            result_dict = {
                "project_path": r.project_path,
                "total_questions": r.total_questions,
                "hit_at_k": {str(k): v for k, v in r.hit_at_k.items()},
                "mrr": r.mrr,
                "precision_at_k": {str(k): v for k, v in r.precision_at_k.items()},
                "recall_at_k": {str(k): v for k, v in r.recall_at_k.items()},
                "ndcg_at_k": {str(k): v for k, v in r.ndcg_at_k.items()}
            }
            # Add generation metrics if available
            if r.avg_llm_score is not None:
                result_dict["generation_metrics"] = {
                    "avg_llm_score": r.avg_llm_score,
                    "avg_correctness": r.avg_correctness,
                    "avg_completeness": r.avg_completeness,
                    "avg_relevance": r.avg_relevance
                }
            new_project_results.append(result_dict)
    
    # Merge: keep existing, add new
    all_project_results = existing_results + new_project_results
    
    # Calculate overall metrics from all results
    total_questions = sum(r.get("total_questions", 0) for r in all_project_results)
    
    if total_questions == 0:
        print("WARNING: No valid results to summarize")
        return
    
    # Helper function to get metric value handling both int and string keys
    def get_metric_value(result_dict, metric_name, k):
        metric_dict = result_dict.get(metric_name, {})
        # Try both string and int keys
        return metric_dict.get(str(k), metric_dict.get(k, 0))
    
    overall_hit_at_k = {}
    for k in [1, 3, 5]:
        overall_hit_at_k[k] = sum(get_metric_value(r, "hit_at_k", k) * r.get("total_questions", 0) for r in all_project_results) / total_questions
    
    overall_mrr = sum(r.get("mrr", 0) * r.get("total_questions", 0) for r in all_project_results) / total_questions
    
    overall_precision_at_k = {}
    overall_recall_at_k = {}
    overall_ndcg_at_k = {}
    
    for k in [1, 3, 5]:
        overall_precision_at_k[k] = sum(get_metric_value(r, "precision_at_k", k) * r.get("total_questions", 0) for r in all_project_results) / total_questions
        overall_recall_at_k[k] = sum(get_metric_value(r, "recall_at_k", k) * r.get("total_questions", 0) for r in all_project_results) / total_questions
        overall_ndcg_at_k[k] = sum(get_metric_value(r, "ndcg_at_k", k) * r.get("total_questions", 0) for r in all_project_results) / total_questions
    
    # Calculate overall generation metrics
    generation_results = [r for r in all_project_results if r.get("generation_metrics") and r.get("generation_metrics", {}).get("avg_llm_score") is not None]
    generation_metrics = None
    if generation_results:
        total_gen_questions = sum(r.get("total_questions", 0) for r in generation_results)
        if total_gen_questions > 0:
            generation_metrics = {
                "avg_llm_score": sum(r.get("generation_metrics", {}).get("avg_llm_score", 0) * r.get("total_questions", 0) for r in generation_results) / total_gen_questions,
                "avg_correctness": sum(r.get("generation_metrics", {}).get("avg_correctness", 0) * r.get("total_questions", 0) for r in generation_results) / total_gen_questions,
                "avg_completeness": sum(r.get("generation_metrics", {}).get("avg_completeness", 0) * r.get("total_questions", 0) for r in generation_results) / total_gen_questions,
                "avg_relevance": sum(r.get("generation_metrics", {}).get("avg_relevance", 0) * r.get("total_questions", 0) for r in generation_results) / total_gen_questions,
                "total_evaluated": total_gen_questions
            }
    
    # Create summary
    summary = {
        "evaluation_summary": {
            "total_projects": len(all_project_results),
            "total_questions": total_questions,
            "retrieval_metrics": {
                "hit_at_1": overall_hit_at_k[1],
                "hit_at_3": overall_hit_at_k[3],
                "hit_at_5": overall_hit_at_k[5],
                "mrr": overall_mrr,
                "precision_at_1": overall_precision_at_k[1],
                "precision_at_3": overall_precision_at_k[3],
                "precision_at_5": overall_precision_at_k[5],
                "recall_at_1": overall_recall_at_k[1],
                "recall_at_3": overall_recall_at_k[3],
                "recall_at_5": overall_recall_at_k[5],
                "ndcg_at_1": overall_ndcg_at_k[1],
                "ndcg_at_3": overall_ndcg_at_k[3],
                "ndcg_at_5": overall_ndcg_at_k[5]
            },
            "generation_metrics": generation_metrics
        },
        "project_results": all_project_results
    }
    
    # Save summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nCALC: Overall Evaluation Summary:")
    print(f"   Total projects: {len(all_project_results)}")
    print(f"   Total questions: {total_questions}")
    print(f"   Overall Hit@1: {overall_hit_at_k[1]:.3f}")
    print(f"   Overall Hit@3: {overall_hit_at_k[3]:.3f}")
    print(f"   Overall Hit@5: {overall_hit_at_k[5]:.3f}")
    print(f"   Overall MRR: {overall_mrr:.3f}")
    print(f"   Overall Precision@1: {overall_precision_at_k[1]:.3f}")
    print(f"   Overall Precision@3: {overall_precision_at_k[3]:.3f}")
    print(f"   Overall Precision@5: {overall_precision_at_k[5]:.3f}")
    print(f"   Overall Recall@1: {overall_recall_at_k[1]:.3f}")
    print(f"   Overall Recall@3: {overall_recall_at_k[3]:.3f}")
    print(f"   Overall Recall@5: {overall_recall_at_k[5]:.3f}")
    print(f"   Overall NDCG@1: {overall_ndcg_at_k[1]:.3f}")
    print(f"   Overall NDCG@3: {overall_ndcg_at_k[3]:.3f}")
    print(f"   Overall NDCG@5: {overall_ndcg_at_k[5]:.3f}")
    print(f"\nSAVE: Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG FAQ retrieval performance")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--projects", nargs="+", required=True, help="Project base paths to evaluate (e.g., projects/batch_proj/ppp_bcc)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated without running")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Error loading config: {e}")
        return 1
    
    # Find project directories
    evaluator = Evaluator(config)
    
    if args.projects:
        project_dirs = evaluator.find_project_directories(args.projects)
    else:
        print("ERROR: No project paths specified! Use --projects to specify which projects to evaluate.")
        return 1
    
    if not project_dirs:
        print("ERROR: No project directories found!")
        return 1
    
    print(f"TARGET: Found {len(project_dirs)} project directories:")
    for project_dir in project_dirs:
        print(f"   - {project_dir}")
    
    if args.dry_run:
        print("\nINFO: Dry run complete. Use without --dry-run to execute evaluation.")
        return 0
    
    # Run evaluation
    try:
        all_results = run_evaluation_all_projects(
            config, 
            args.output_dir,
            args.projects
        )
        
        # Generate aggregated summaries (groups results by parent project path)
        create_aggregated_summaries(all_results, args.output_dir)
        
        # Generate overall summary report (merges with existing)
        generate_summary_report(all_results, args.output_dir)
        
        print(f"\nSUCCESS: Evaluation complete! Results saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())