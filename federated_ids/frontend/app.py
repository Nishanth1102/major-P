import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler

# Support imports from the parent federated project directory seamlessly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import MLP
from data_utils import preprocess

app = Flask(__name__, static_folder='static')
CORS(app)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """Retrieve all available PyTorch checkpoint models available from the experiments"""
    if not os.path.exists(MODELS_DIR):
        return jsonify({"models": []})
    
    models = []
    for filepath in glob.glob(os.path.join(MODELS_DIR, '*.pth')):
        models.append(os.path.basename(filepath))
    return jsonify({"models": models})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predicts IDC (Intrusion Detection) across a CSV dataset"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    model_name = request.form.get('model')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not model_name:
        return jsonify({"error": "No model selected"}), 400
        
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({"error": "Target model checkpoint not found"}), 404
        
    # Save the CSV temporarily for processing
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    
    try:
        # Load dataset natively
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Strip identifiers and grab numeric vectors using the master data_utils logic
        X, y = preprocess(df)
        
        # Apply strict statistical scaling natively for the prediction batch
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        
        # Instantiate model structurally matching the provided feature space
        input_dim = X_scaled.shape[1]
        model = MLP(input_dim=input_dim)
        
        # Load the Federated Weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # Execute Evaluation
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
        
        with torch.no_grad():
            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=512, shuffle=False)
            
            correct = 0
            total = len(y)
            
            for X_batch, y_batch in loader:
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                
        wrong = total - correct
        accuracy = (correct / total) * 100
        
        # Safely clean up the processed CSV memory
        os.remove(filepath)
        
        return jsonify({
            "total": total,
            "correct": int(correct),
            "wrong": int(wrong),
            "accuracy": round(accuracy, 2)
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Evaluation crashed: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*55)
    print(" 🚀 Edge Node IDS Web UI Hosted at http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
