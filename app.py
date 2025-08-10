import torch
import torch.nn as nn
import joblib
from flask import Flask, request, render_template
import numpy as np
import random

app = Flask(__name__)

# Load models and preprocessors
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
action_enc = joblib.load('action_encoder.pkl')

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout), num_layers
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

transformer = TransformerModel()
transformer.load_state_dict(torch.load('transformer.pth'))
transformer.eval()

system_calls = ['open', 'read', 'write', 'close', 'rename']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    sample_sequence = None
    bottom_line = None
    if request.method == 'POST':
        if 'generate' in request.form:
            sample_sequence = ','.join(random.choices(system_calls, k=5))
        else:
            sample_sequence = request.form['sequence']
        
        # Preprocess for XGBoost
        seq_list = sample_sequence.split(',')
        ngram_features = [seq_list[i:i+3] for i in range(len(seq_list)-2)] if len(seq_list) >= 3 else []
        ngram_counts = [sum(1 for _ in ngram_features if _.count('write') >= 2)]
        features = scaler.transform([[len(seq_list), ngram_counts[0]]])
        xgb_pred = xgb_model.predict(features)[0]
        xgb_prob = xgb_model.predict_proba(features)[0][1]
        xgb_result = 'Malicious' if xgb_pred == 1 else 'Normal'
        
        # Preprocess for Transformer
        seq_encoded = [action_enc.transform([call])[0] for call in seq_list]
        seq_padded = np.pad(seq_encoded, (0, 10 - len(seq_encoded)), 'constant')[:10]
        seq_tensor = torch.tensor([seq_padded], dtype=torch.long)
        with torch.no_grad():
            trans_prob = transformer(seq_tensor).item()
        trans_result = 'Malicious' if trans_prob > 0.5 else 'Normal'
        
        prediction = f"Result: XGBoost: {xgb_result} ({xgb_prob:.2f}), Transformer: {trans_result} ({trans_prob:.2f})"
        # Determine bottom line based on model agreement
        if xgb_result == 'Malicious' or trans_result == 'Malicious':
            bottom_line = "Bottom Line: Potential Insider Threat - Investigate Further :( "
        else:
            bottom_line = "Bottom Line: No Suspicious Behaviour Detected :)"

    return render_template('index.html', prediction=prediction, sample_sequence=sample_sequence, bottom_line=bottom_line)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)