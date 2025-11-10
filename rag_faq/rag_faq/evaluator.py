import os
import json
import re
import time
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Container for individual evaluation results"""
    question: str
    correct_chunk_id: str
    retrieved_chunks: List[str]
    correct_chunk_position: Optional[int]
    hit_at_k: Dict[int, bool]
    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    is_original: bool

    # Generation evaluation fields
    generated_answer: Optional[str] = None
    generation_context: Optional[List[Dict]] = None
    llm_judge_score: Optional[float] = None
    llm_judge_correctness: Optional[int] = None
    llm_judge_completeness: Optional[int] = None
    llm_judge_relevance: Optional[int] = None
    llm_judge_justification: Optional[str] = None

@dataclass
class ProjectEvaluationResults:
    """Container for project-level evaluation results"""
    project_path: str
    total_questions: int
    hit_at_k: Dict[int, float]
    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    detailed_results: List[EvaluationResult]

    # Generation evaluation fields
    avg_llm_score: Optional[float] = None
    avg_correctness: Optional[float] = None
    avg_completeness: Optional[float] = None
    avg_relevance: Optional[float] = None

class Evaluator:
    """Main class for evaluating retrieval and generation performance"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load evaluation configuration from config file
        eval_config = config.get("evaluation", {})
        self.num_questions = eval_config.get("num_questions", 10)
        self.num_variations = eval_config.get("num_variations", 3)
        self.top_k_values = eval_config.get("top_k_values", [1, 3, 5])
        
        # Initialize LLM for question variation generation and generation evaluation
        self.llm_cfg = config["llm"]["evaluator"]

        base_url = None
        if self.llm_cfg["provider"] == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif self.llm_cfg["provider"] == "openai":
            base_url = "https://api.openai.com/v1"

        self.model = ChatOpenAI(
            model=self.llm_cfg["model"],
            temperature=self.llm_cfg["temperature"],
            openai_api_key=self.llm_cfg["api_key"],
            base_url=base_url,
        )

        print(f"[Evaluator] Using model: {self.llm_cfg['model']} via {self.llm_cfg['provider']}")
        
        # Load generation judge prompt
        prompt_dir = config["paths"]["prompts_dir"]
        judge_prompt_path = os.path.join(prompt_dir, "generation_judge.txt")
        if os.path.exists(judge_prompt_path):
            with open(judge_prompt_path, 'r', encoding='utf-8') as f:
                self.judge_prompt = f.read()
        else:
            raise FileNotFoundError(f"Generation judge prompt not found: {judge_prompt_path}")
        
    def find_project_directories(self, base_paths: List[str]) -> List[str]:
        """
        Find all project directories that contain FAQ files within the specified base paths.

        Args:
            base_paths (List[str]): A list of base directories to search within.
        """
        project_dirs = []
        
        for base_path in base_paths:
            if not os.path.exists(base_path):
                print(f"WARNING: Path does not exist: {base_path}")
                continue
                
            for root, dirs, files in os.walk(base_path):                
                if "faq.csv" in files:
                    project_dirs.append(root)
        
        return sorted(project_dirs)
    
    def load_chunks_data(self, project_dir):
        """
        Load chunks data from the FAQ CSV file itself.
        The FAQ file contains source_text and chunk_id columns.

        Args:
            project_dir (str): The path to the project directory containing the faq.csv file.
        """
        faq_path = os.path.join(project_dir, "faq.csv")
        
        if not os.path.exists(faq_path):
            raise FileNotFoundError(f"FAQ file not found: {faq_path}")
        
        faq_df = pd.read_csv(faq_path)
        
        # Create chunks dataframe from FAQ data
        chunks_df = faq_df[['source_text', 'chunk_id']].drop_duplicates()
        chunks_df = chunks_df.rename(columns={'source_text': 'text'})
        
        return chunks_df
    
    def generate_question_variations(self, original_question: str) -> List[str]:
        """
        Generate variations of a given question using an LLM. These variations are specifically 
        for evaluation purposes, allowing the system to be tested against diverse phrasings beyond 
        the exact questions already present in the `faq.csv` file.

        Args:
            original_question (str): The original question to generate variations for.
        """

        prompt = f"""
            Gere {self.num_variations} formas diferentes de fazer a mesma pergunta.
            As variações devem:
            1. Ter o mesmo significado mas com palavras diferentes
            2. Utilizar diferentes estruturas de frase (por exemplo: o que / como / quando / por que).
            3. Utilizar vocabulário natural e realista, como em perguntas feitas por um usuário humano.
            
            Pergunta original: "{original_question}"
            
            Retorne apenas as variações, uma por linha, sem numeração, marcadores ou explicações adicionais.
        """
        
        try:
            response = self.model.invoke([
                SystemMessage(content="Você é um especialista em gerar variações naturais de perguntas."),
                HumanMessage(content=prompt)
            ])
            
            variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
            return variations[:self.num_variations]
            
        except Exception as e:
            print(f"Error generating variations for '{original_question}': {e}")
            return [original_question]  # Fallback to original
    
    def create_evaluation_dataset(self, project_dir: str) -> List[Dict]:
        """
        Create an evaluation dataset by sampling a subset of original questions from the faq.csv,
        and for each sampled question, generating multiple variations. This dataset is used
        to evaluate the retrieval and generation capabilities of the RAG system. 

        Args:
            project_dir (str): The directory of the project being evaluated.
        """

        faq_path = os.path.join(project_dir, "faq.csv")
        
        if not os.path.exists(faq_path):
            raise FileNotFoundError(f"FAQ file not found: {faq_path}")
        
        # Load FAQ data
        faq_df = pd.read_csv(faq_path)
        
        if 'chunk_id' not in faq_df.columns:
            raise ValueError("chunk_id column not found in FAQ file")
        
        # Sample random questions
        if len(faq_df) < self.num_questions:
            sampled_df = faq_df
        else:
            sampled_df = faq_df.sample(n=self.num_questions, random_state=42)
        
        evaluation_dataset = []
        
        for _, row in sampled_df.iterrows():
            original_question = row['question']
            correct_chunk_id = str(row['chunk_id'])
            
            # Generate variations
            variations = self.generate_question_variations(original_question)
            
            # Add original question and variations to dataset
            all_questions = [original_question] + variations
            
            for question in all_questions:
                evaluation_dataset.append({
                    'question': question,
                    'correct_chunk_id': correct_chunk_id,
                    'is_original': question == original_question,
                    'source_text': row['source_text'],
                    'answer': row['answer']
                })
        
        return evaluation_dataset
    
    def calculate_precision_at_k(self, retrieved_chunks: List[str], correct_chunk_id: str, k: int) -> float:
        """Calculate Precision@k"""
        if k == 0:
            return 0.0
        
        top_k_chunks = retrieved_chunks[:k]
        relevant_in_top_k = sum(1 for chunk_id in top_k_chunks if chunk_id == correct_chunk_id)
        return relevant_in_top_k / k
    
    def calculate_recall_at_k(self, retrieved_chunks: List[str], correct_chunk_id: str, k: int) -> float:
        """Calculate Recall@k (assuming only 1 relevant chunk per query)"""
        top_k_chunks = retrieved_chunks[:k]
        relevant_found = any(chunk_id == correct_chunk_id for chunk_id in top_k_chunks)
        return 1.0 if relevant_found else 0.0
    
    def calculate_ndcg_at_k(self, retrieved_chunks: List[str], correct_chunk_id: str, k: int) -> float:
        """Calculate NDCG@k"""
        if k == 0:
            return 0.0
        
        # For binary relevance (0 or 1), DCG@k = sum of (relevance_i / log2(i+1))
        # But we only count the first occurrence of the relevant chunk
        dcg = 0.0
        found_relevant = False
        for i, chunk_id in enumerate(retrieved_chunks[:k]):
            if chunk_id == correct_chunk_id and not found_relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                found_relevant = True
        
        # IDCG@k (Ideal DCG) - perfect ranking would have relevant chunk at position 1
        # For binary relevance with 1 relevant item, IDCG@k = 1.0 / log2(2) = 1.0
        idcg = 1.0 / np.log2(2)  # This equals 1.0
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(self, project_dir: str, evaluation_dataset: List[Dict]) -> List[EvaluationResult]:
        """
        Evaluate retrieval performance for a set of questions.

        Args:
            project_dir (str): The directory of the project being evaluated.
            evaluation_dataset (List[Dict]): A list of dictionaries, each representing a question and its correct chunk.
        """
        from rag_faq.retriever import retrieve_similar_faqs
        
        # Load chunks data for LLM evaluation
        chunks_df = self.load_chunks_data(project_dir)
        
        results = []
        
        for item in tqdm(evaluation_dataset, desc="Evaluating retrieval"):
            question = item['question']
            correct_chunk_id = item['correct_chunk_id']
            
            # Retrieve similar FAQs
            retrieved_faqs = retrieve_similar_faqs(self.config, project_dir, question)
            
            # Extract chunk IDs from retrieved results
            retrieved_chunk_ids = []
            for faq in retrieved_faqs:
                # Find chunk_id by matching source_text
                matching_chunks = chunks_df[chunks_df['text'] == faq['source_text']]
                if not matching_chunks.empty:
                    retrieved_chunk_ids.append(str(matching_chunks.iloc[0]['chunk_id']))
            
            # Find position of correct chunk
            correct_chunk_position = None
            if correct_chunk_id in retrieved_chunk_ids:
                correct_chunk_position = retrieved_chunk_ids.index(correct_chunk_id) + 1
            
            # Calculate Hit@k metrics
            hit_at_k = {}
            for k in self.top_k_values:
                hit_at_k[k] = correct_chunk_position is not None and correct_chunk_position <= k
            
            # Calculate MRR
            mrr = 1.0 / correct_chunk_position if correct_chunk_position else 0.0
            
            # Calculate Precision@k, Recall@k, and NDCG@k
            precision_at_k = {}
            recall_at_k = {}
            ndcg_at_k = {}
            
            for k in self.top_k_values:
                precision_at_k[k] = self.calculate_precision_at_k(retrieved_chunk_ids, correct_chunk_id, k)
                recall_at_k[k] = self.calculate_recall_at_k(retrieved_chunk_ids, correct_chunk_id, k)
                ndcg_at_k[k] = self.calculate_ndcg_at_k(retrieved_chunk_ids, correct_chunk_id, k)
            
            # Create result
            result = EvaluationResult(
                question=question,
                correct_chunk_id=correct_chunk_id,
                retrieved_chunks=retrieved_chunk_ids,
                correct_chunk_position=correct_chunk_position,
                hit_at_k=hit_at_k,
                mrr=mrr,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                ndcg_at_k=ndcg_at_k,
                is_original=item['is_original']
            )
            
            results.append(result)
        
        return results
    
    def evaluate_generation(self, project_dir: str, evaluation_dataset: List[Dict], 
                           retrieval_results: List[EvaluationResult]) -> List[EvaluationResult]:
        """
        Evaluate generation performance using LLM as judge.
        Updates the retrieval_results with generation evaluation data.

        Args:
            project_dir (str): The directory of the project being evaluated.
            evaluation_dataset (List[Dict]): The dataset containing questions and correct answers.
            retrieval_results (List[EvaluationResult]): A list of retrieval results to be updated with generation data.
        """
        from rag_faq.generator import generate_rag_answer
        
        # Create a mapping from question to retrieval result
        question_to_result = {r.question: r for r in retrieval_results}
        
        # Create mapping from evaluation dataset to get correct answer
        dataset_map = {item['question']: item for item in evaluation_dataset}
        
        # Update retrieval results in place with generation data
        for item in tqdm(evaluation_dataset, desc="Evaluating generation"):
            question = item['question']
            correct_answer = item['answer']
            
            # Get corresponding retrieval result
            result = question_to_result.get(question)
            if not result:
                continue
            
            try:
                # Generate answer using RAG
                rag_result = generate_rag_answer(self.config, project_dir, question, debug=False)
                generated_answer = rag_result['answer']
                generation_context = rag_result['context']
                
                # Evaluate using LLM judge
                judge_result = self._judge_generation(question, generated_answer, correct_answer)
                
                # Update result with generation data
                result.generated_answer = generated_answer
                result.generation_context = generation_context
                result.llm_judge_score = judge_result.get('overall_score')
                result.llm_judge_correctness = judge_result.get('correctness')
                result.llm_judge_completeness = judge_result.get('completeness')
                result.llm_judge_relevance = judge_result.get('relevance')
                result.llm_judge_justification = judge_result.get('justification')
                
            except Exception as e:
                print(f"Error evaluating generation for '{question}': {e}")
                # Keep the retrieval result even if generation fails
        
        # Return all results (updated in place)
        return list(question_to_result.values())

    def _judge_generation(self, question: str, generated_answer: str, correct_answer: str) -> Dict:
        """
        Use LLM as judge to evaluate generated answer against correct answer.
        Includes retry and timeout protection to handle slow or rate-limited APIs.
        """
        prompt = f"""Pergunta: {question}

            Resposta Gerada (RAG): {generated_answer}

            Resposta de Referência (Ground Truth): {correct_answer}

            Avalie a qualidade da resposta gerada comparando-a com a resposta de referência.
            Retorne um JSON com os campos:
            {{"overall_score": float, "correctness": int, "completeness": int, "relevance": int, "justification": string}}
        """

        max_retries = 3
        wait_time = 10  # seconds between retries
        timeout = 120   # seconds per request

        for attempt in range(max_retries):
            try:
                response = self.model.invoke(
                    [
                        SystemMessage(content=self.judge_prompt),
                        HumanMessage(content=prompt)
                    ],
                    config={"timeout": timeout}  # LangChain-style timeout
                )

                response_text = response.content.strip()

                # Try to extract JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    return json.loads(response_text)

            except Exception as e:
                print(f"[Warning] LLM judge attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time * (attempt + 1))
                else:
                    return {
                        "overall_score": 0.0,
                        "correctness": 0,
                        "completeness": 0,
                        "relevance": 0,
                        "justification": f"Erro após {max_retries} tentativas: {str(e)}"
                    }
    
    def calculate_project_metrics(self, results: List[EvaluationResult], project_path: str) -> ProjectEvaluationResults:
        """
        Calculate aggregated metrics for a project.
        Excludes original questions from metrics calculation to avoid bias.

        Args:
            results (List[EvaluationResult]): A list of evaluation results.
            project_path (str): The path to the project being evaluated.
        """
        # Filter out original questions for metrics calculation (they're already in FAQ)
        variation_results = [r for r in results if not r.is_original]
        total_questions = len(variation_results)
        
        if total_questions == 0:
            # Fallback: if no variations, use all results
            variation_results = results
            total_questions = len(results)
        
        # Calculate Hit@k rates (only on variations)
        hit_at_k = {}
        for k in self.top_k_values:
            hits = sum(1 for r in variation_results if r.hit_at_k.get(k, False))
            hit_at_k[k] = hits / total_questions if total_questions > 0 else 0.0
        
        # Calculate average MRR (only on variations)
        mrr = np.mean([r.mrr for r in variation_results]) if variation_results else 0.0
        
        # Calculate average Precision@k, Recall@k, and NDCG@k (only on variations)
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.top_k_values:
            precision_at_k[k] = np.mean([r.precision_at_k.get(k, 0.0) for r in variation_results])
            recall_at_k[k] = np.mean([r.recall_at_k.get(k, 0.0) for r in variation_results])
            ndcg_at_k[k] = np.mean([r.ndcg_at_k.get(k, 0.0) for r in variation_results])
        
        # Calculate generation metrics (only on variations with generation results)
        generation_results = [r for r in variation_results if r.llm_judge_score is not None]
        avg_llm_score = None
        avg_correctness = None
        avg_completeness = None
        avg_relevance = None
        
        if generation_results:
            avg_llm_score = np.mean([r.llm_judge_score for r in generation_results])
            avg_correctness = np.mean([r.llm_judge_correctness for r in generation_results if r.llm_judge_correctness is not None])
            avg_completeness = np.mean([r.llm_judge_completeness for r in generation_results if r.llm_judge_completeness is not None])
            avg_relevance = np.mean([r.llm_judge_relevance for r in generation_results if r.llm_judge_relevance is not None])
        
        return ProjectEvaluationResults(
            project_path=project_path,
            total_questions=total_questions,  # Only variations count for metrics
            hit_at_k=hit_at_k,
            mrr=mrr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            detailed_results=results,  # Keep all results (including originals) for reference
            avg_llm_score=avg_llm_score,
            avg_correctness=avg_correctness,
            avg_completeness=avg_completeness,
            avg_relevance=avg_relevance
        )
    
    def save_results(self, results: ProjectEvaluationResults, output_dir: str):
        """
        Save evaluation results to files with descriptive folder structure.

        Args:
            results (ProjectEvaluationResults): The evaluation results to save.
            output_dir (str): The directory to save the results to.
        """
        # Create descriptive folder structure
        # Convert path like "projects/batch_proj/ppp_bcc/individual" to "batch_proj/ppp_bcc/individual"
        normalized_path = results.project_path.replace('\\', '/')
        path_parts = normalized_path.split('/')
        
        # Remove 'projects' and create folder name
        relevant_parts = [part for part in path_parts if part and part not in ['projects']]
        
        if len(relevant_parts) >= 2:
            # Create nested folder structure: e.g batch_proj/ppp_bcc/individual
            subfolder = relevant_parts[-1]  # individual or unificado
            parent_parts = relevant_parts[:-1]  # batch_proj, ppp_bcc
            project_output_dir = os.path.join(output_dir, *parent_parts, subfolder)
        else:
            # Fallback for simple paths
            project_output_dir = os.path.join(output_dir, *relevant_parts) if relevant_parts else output_dir
        
        os.makedirs(project_output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_data = []
        for result in results.detailed_results:
            result_dict = {
                "question": result.question,
                "correct_chunk_id": result.correct_chunk_id,
                "retrieved_chunks": result.retrieved_chunks,
                "correct_chunk_position": result.correct_chunk_position,
                "hit_at_k": result.hit_at_k,
                "mrr": result.mrr,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "ndcg_at_k": result.ndcg_at_k,
                "is_original": result.is_original
            }
            
            # Add generation evaluation fields if available
            if result.generated_answer is not None:
                result_dict["generated_answer"] = result.generated_answer
                result_dict["generation_context"] = result.generation_context
                result_dict["llm_judge_score"] = result.llm_judge_score
                result_dict["llm_judge_correctness"] = result.llm_judge_correctness
                result_dict["llm_judge_completeness"] = result.llm_judge_completeness
                result_dict["llm_judge_relevance"] = result.llm_judge_relevance
                result_dict["llm_judge_justification"] = result.llm_judge_justification
            
            detailed_data.append(result_dict)
        
        detailed_path = os.path.join(project_output_dir, "evaluation_detailed.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"SUCCESS: Results saved to: {project_output_dir}")
        return detailed_path