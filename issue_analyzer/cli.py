import pandas as pd
from issue_analyzer.analyzer import IssueAnalyzer
import json

def run_batch_analysis(input_csv: str, output_json: str):
    analyzer = IssueAnalyzer()
    df = pd.read_csv(input_csv)

    results = []
    for _, row in df.iterrows():
        input_id = row['id']
        text = row['text']
        print(f"Analyzing ID: {input_id}...")
        try:
            result = analyzer.analyze(text)
            result_json = result.model_dump()
            result_json["id"] = input_id  # preserve original id
            results.append(result_json)
        except Exception as e:
            print(f"Error analyzing ID {input_id}: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_json}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python cli.py <input_csv> <output>")
    else:
        run_batch_analysis(sys.argv[1], sys.argv[2])
