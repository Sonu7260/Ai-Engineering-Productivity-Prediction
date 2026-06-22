import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Determine path boundaries dynamically for Vercel Serverless container structures
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'adaboost_pkl.pkl')

# Load your pre-trained AdaBoost classifier securely
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Explicit schema mappings matching feature variables 
FEATURE_ORDER = [
    'Hours_Coding', 'AI_Usage_Hours', 'Lines_of_Code', 'Commits', 
    'Bugs_Reported', 'Sleep_Hours', 'Distractions', 'Cognitive_Load', 'Stress_Level'
]

MAP_COGNITIVE_LOAD = {'Low': 1, 'Medium': 2, 'High': 3}
MAP_STRESS_LEVEL = {'Low': 1, 'Medium': 2, 'High': 3}

# Embedded HTML & CSS Interface Layout Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Developer Performance Predictive Analytics Suite</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --border: #334155;
            --success: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .app-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .app-header {
            margin-bottom: 2.5rem;
            text-align: center;
        }

        .app-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .app-header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 968px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .input-group input, .input-group select {
            background-color: var(--bg-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            color: var(--text-primary);
            font-size: 1rem;
            transition: all 0.2s ease;
        }

        .input-group input:focus, .input-group select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        .btn-primary {
            background-color: var(--accent);
            color: var(--bg-main);
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 1rem 2rem;
            font-size: 1rem;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.2s ease;
        }

        .btn-primary:hover {
            background-color: var(--accent-hover);
        }

        .results-display {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .empty-state {
            text-align: center;
            color: var(--text-secondary);
            font-style: italic;
            padding: 3rem 0;
        }

        .alert {
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }

        .success-alert {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
        }

        .badge {
            background-color: var(--success);
            color: white;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 50px;
            display: inline-block;
            margin-bottom: 1rem;
        }

        .success-alert h3 {
            font-size: 1.6rem;
            color: #fff;
            margin-bottom: 0.5rem;
        }

        .confidence-indicator {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .error-alert {
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="app-header">
            <h1>AdaBoost Intelligence Engine</h1>
            <p>Predictive pipeline engine for professional workload and engineering output analysis</p>
        </header>

        <main class="dashboard-grid">
            <section class="card form-card">
                <h2>Predictive Feature Vector Parameters</h2>
                <form action="/predict" method="POST">
                    <div class="form-grid">
                        <div class="input-group">
                            <label for="Hours_Coding">Hours of Active Coding</label>
                            <input type="number" step="0.1" name="Hours_Coding" id="Hours_Coding" required value="{{ form_values.get('Hours_Coding', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="AI_Usage_Hours">AI Tool Interaction Time (Hours)</label>
                            <input type="number" step="0.1" name="AI_Usage_Hours" id="AI_Usage_Hours" required value="{{ form_values.get('AI_Usage_Hours', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Lines_of_Code">Lines of Written Code</label>
                            <input type="number" name="Lines_of_Code" id="Lines_of_Code" required value="{{ form_values.get('Lines_of_Code', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Commits">Version Control Commits</label>
                            <input type="number" name="Commits" id="Commits" required value="{{ form_values.get('Commits', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Bugs_Reported">Defects / Bugs Logged</label>
                            <input type="number" name="Bugs_Reported" id="Bugs_Reported" required value="{{ form_values.get('Bugs_Reported', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Sleep_Hours">Rest Cycle Period (Hours)</label>
                            <input type="number" step="0.1" name="Sleep_Hours" id="Sleep_Hours" required value="{{ form_values.get('Sleep_Hours', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Distractions">Distraction Index Counts</label>
                            <input type="number" name="Distractions" id="Distractions" required value="{{ form_values.get('Distractions', '') if form_values else '' }}">
                        </div>
                        <div class="input-group">
                            <label for="Cognitive_Load">Cognitive Burden Level</label>
                            <select name="Cognitive_Load" id="Cognitive_Load" required>
                                <option value="Low" {% if form_values and form_values.get('Cognitive_Load') == 'Low' %}selected{% endif %}>Low Load State</option>
                                <option value="Medium" {% if not form_values or form_values.get('Cognitive_Load') == 'Medium' %}selected{% endif %}>Moderate Load State</option>
                                <option value="High" {% if form_values and form_values.get('Cognitive_Load') == 'High' %}selected{% endif %}>Elevated Core Load State</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="Stress_Level">Symptomatic Stress Level</label>
                            <select name="Stress_Level" id="Stress_Level" required>
                                <option value="Low" {% if form_values and form_values.get('Stress_Level') == 'Low' %}selected{% endif %}>Low Stress State</option>
                                <option value="Medium" {% if not form_values or form_values.get('Stress_Level') == 'Medium' %}selected{% endif %}>Moderate Stress State</option>
                                <option value="High" {% if form_values and form_values.get('Stress_Level') == 'High' %}selected{% endif %}>Elevated Stress State</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn-primary">Execute Inference Routine</button>
                </form>
            </section>

            <section class="card results-card">
                <h2>Inference Metrics & Output</h2>
                <div class="results-display">
                    {% if prediction_text %}
                        <div class="alert success-alert">
                            <div class="badge">Model Class Output Verified</div>
                            <h3>{{ prediction_text }}</h3>
                            <p class="confidence-indicator">{{ confidence_text }}</p>
                        </div>
                    {% elif error_text %}
                        <div class="alert error-alert">
                            <h3>Pipeline Fault</h3>
                            <p>{{ error_text }}</p>
                        </div>
                    {% else %}
                        <div class="empty-state">
                            <p>Awaiting Parameter Initialization Vectors to Execute Predictions</p>
                        </div>
                    {% endif %}
                </div>
            </section>
        </main>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'Hours_Coding': float(request.form.get('Hours_Coding', 0)),
            'AI_Usage_Hours': float(request.form.get('AI_Usage_Hours', 0)),
            'Lines_of_Code': float(request.form.get('Lines_of_Code', 0)),
            'Commits': float(request.form.get('Commits', 0)),
            'Bugs_Reported': float(request.form.get('Bugs_Reported', 0)),
            'Sleep_Hours': float(request.form.get('Sleep_Hours', 0)),
            'Distractions': float(request.form.get('Distractions', 0)),
            'Cognitive_Load': MAP_COGNITIVE_LOAD.get(request.form.get('Cognitive_Load'), 2),
            'Stress_Level': MAP_STRESS_LEVEL.get(request.form.get('Stress_Level'), 2)
        }
        
        df_input = pd.DataFrame([form_data])[FEATURE_ORDER]
        
        prediction = int(model.predict(df_input)[0])
        probabilities = model.predict_proba(df_input)[0]
        confidence = round(float(probabilities[prediction]) * 100, 2)
        
        status_label = "High Productivity / Burnout Risk Detected" if prediction == 1 else "Normal Balance Maintained"
        
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f'Result: {status_label}',
            confidence_text=f'Confidence Match: {confidence}%',
            form_values=request.form
        )
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Execution Interrupted: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
