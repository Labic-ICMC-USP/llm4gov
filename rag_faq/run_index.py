import pandas as pd
import os
from rag_faq.config import load_config
from rag_faq.indexer import generate_faqs
from rag_faq.embedder import embed_faqs

def run_index(config, project_dir, data_source=None, persona_type=None, mode="individual"):
    """
    Automated FAQ generation and embedding pipeline.
    
    Args:
        config: Configuration dictionary
        project_dir: Project directory path
        data_source: Data source name (from config) or CSV file path
        persona_type: Single persona type for individual mode
        mode: "individual" (single persona) or "unificado" (multi-persona)
    """
    
    # Get pipeline configuration
    pipeline_config = config.get("pipeline", {})
    available_personas = pipeline_config.get("personas", ["aluno", "professor", "pesquisador"])
    data_sources = pipeline_config.get("data_sources", [])
    
    # Determine data source
    if data_source:
        # Check if it's a configured data source
        source_config = next((s for s in data_sources if s["name"] == data_source), None)
        if source_config:
            csv_file = source_config["csv_file"]
            course_name = source_config["course_name"]
            course_dir_name = source_config["name"]  # Use the course directory name
        else:
            # Assume it's a direct file path
            csv_file = data_source
            course_name = "Unknown Course"
            course_dir_name = "unknown_course"
    else:
        # Default to first available data source
        if data_sources:
            source_config = data_sources[0]
            csv_file = source_config["csv_file"]
            course_name = source_config["course_name"]
            course_dir_name = source_config["name"]
        else:
            csv_file = "data/dataset_sample.csv"
            course_name = "Sample Dataset"
            course_dir_name = "sample_dataset"
    
    print(f"📊 Using data source: {csv_file}")
    print(f"🎓 Course: {course_name}")
    
    # Load texts
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Data file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    texts = df["text"].tolist()
    print(f"📝 Loaded {len(texts)} text chunks")
    
    # Create course-specific directory first
    course_dir = os.path.join(project_dir, course_dir_name)
    os.makedirs(course_dir, exist_ok=True)
    
    # Create subdirectories based on mode
    if mode == "individual":
        output_dir = os.path.join(course_dir, "individual")
        os.makedirs(output_dir, exist_ok=True)
        
        # Single persona generation
        if not persona_type:
            persona_type = "aluno"  # Default persona
        
        print(f"👤 Generating FAQs for persona: {persona_type}")
        generate_faqs(config, output_dir, texts, course_name=course_name, persona_type=persona_type)
        embed_faqs(config, output_dir)
        
    elif mode == "unificado":
        output_dir = os.path.join(course_dir, "unificado")
        os.makedirs(output_dir, exist_ok=True)
        
        # Multi-persona generation
        print(f"👥 Generating FAQs for all personas: {available_personas}")
        
        all_faqs = []
        
        for persona in available_personas:
            print(f"  🔄 Processing persona: {persona}")
            generate_faqs(config, output_dir, texts, course_name=course_name, 
                         persona_type=persona, multi_persona=True)
        
        # Merge all persona files
        merge_persona_files(output_dir, available_personas)
        
        # Generate embeddings for merged file
        embed_faqs(config, output_dir)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'individual' or 'unificado'")

def merge_persona_files(output_dir, personas):
    """Merge individual persona FAQ files into a single file."""
    import pandas as pd
    
    all_faqs = []
    
    for persona in personas:
        file_path = os.path.join(output_dir, f"faq_{persona}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_faqs.append(df)
            print(f"  ✅ Loaded {len(df)} FAQs from {persona}")
        else:
            print(f"  ⚠️  File not found: faq_{persona}.csv")
    
    if all_faqs:
        # Combine all DataFrames
        merged_df = pd.concat(all_faqs, ignore_index=True)
        
        # Save merged file
        merged_path = os.path.join(output_dir, "faq.csv")
        merged_df.to_csv(merged_path, index=False, encoding="utf-8")
        
        print(f"\n📊 Merge Summary:")
        print(f"📈 Total FAQs: {len(merged_df)}")
        
        # Show breakdown by persona
        persona_counts = merged_df['persona'].value_counts()
        print(f"👥 FAQs by persona:")
        for persona, count in persona_counts.items():
            print(f"   - {persona}: {count}")
        
        print(f"🎓 FAQs by course:")
        course_counts = merged_df['course'].value_counts()
        for course, count in course_counts.items():
            print(f"   - {course}: {count}")
            
        print(f"\n✅ Combined CSV saved to: {merged_path}")
    else:
        print("❌ No FAQ files found to merge!")

def run_batch_indexing(config, project_base_dir, data_sources=None, mode="individual"):
    """
    Run indexing for multiple data sources in batch.
    
    Args:
        config: Configuration dictionary
        project_base_dir: Base directory for projects
        data_sources: List of data source names (None = all sources)
        mode: "individual" or "unificado"
    """
    pipeline_config = config.get("pipeline", {})
    available_sources = pipeline_config.get("data_sources", [])
    
    if data_sources is None:
        data_sources = [source["name"] for source in available_sources]
    
    for source_name in data_sources:
        print(f"\n🚀 Processing data source: {source_name}")
        project_dir = os.path.join(project_base_dir, source_name)
        os.makedirs(project_dir, exist_ok=True)
        
        try:
            run_index(config, project_dir, data_source=source_name, mode=mode)
            print(f"✅ Completed: {source_name}")
        except Exception as e:
            print(f"❌ Failed: {source_name} - {str(e)}")
