import os
import argparse
from flask import Flask, request, jsonify, render_template_string
from rag_faq.config import load_config
from rag_faq.generator import generate_rag_answer

# Global vars set at runtime
config = None
project_dir = None

# Flask app
app = Flask(__name__)

# HTML template for browser interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG FAQ Assistant</title>
</head>
<body style="font-family: sans-serif; max-width: 700px; margin: 50px auto;">
    <h2>🤖 Ask the FAQ Assistant</h2>
    <form method="POST">
        <input type="text" name="question" placeholder="Enter your question..." style="width: 100%; padding: 10px;" required>
        <button type="submit" style="margin-top: 10px; padding: 10px 20px;">Submit</button>
    </form>

    {% if answer %}
        <hr>
        <h3>📘 Answer:</h3>
        <p>{{ answer }}</p>

        <h4>🔎 Retrieved Context:</h4>
        <ul>
            {% for item in context %}
                <li style="margin-bottom: 10px;">
                    <strong>Q:</strong> {{ item.question }}<br>
                    <strong>A:</strong> {{ item.answer }}<br>
                    <strong>Score:</strong> {{ "%.3f"|format(item.score) }}<br>
                    <details>
                      <summary><strong>📄 Source Text</strong></summary>
                      <p style="font-size: 90%; margin-top: 5px;">{{ item.source_text }}</p>
                    </details>
                </li>
            {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def html_interface():
    """
    Simple HTML page for entering a question and viewing the LLM-generated answer.
    """
    answer = None
    context = []

    if request.method == "POST":
        user_question = request.form.get("question")
        try:
            result = generate_rag_answer(config, project_dir, user_question, debug=False)
            answer = result["answer"]
            context = result["context"]
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, answer=answer, context=context)


@app.route("/rag", methods=["POST"])
def rag_api():
    """
    JSON API endpoint. Expects: { "question": "..." }
    Returns: { "answer": "...", "context": [...] }
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request"}), 400

    user_question = data["question"]
    try:
        result = generate_rag_answer(config, project_dir, user_question, debug=False)
        return jsonify({
            "answer": result["answer"],
            "context": result["context"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_server():
    global config, project_dir

    # Command-line args
    parser = argparse.ArgumentParser(description="Start Flask server for RAG FAQ")
    parser.add_argument("--port", type=int, required=True, help="Port for the Flask server")
    parser.add_argument("--project", type=str, required=True, help="Project name to load context from")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load config and check project dir
    config = load_config(args.config)
    project_dir = os.path.join(config["paths"]["projects_dir"], args.project)

    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    # Start server
    print(f"🚀 Flask server running at http://localhost:{args.port}/ (project: {args.project})")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    start_server()
