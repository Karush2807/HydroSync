from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from whatsapp_api_client_python import API

# Load environment variables
load_dotenv()

app = Flask(__name__)

greenAPI = API.GreenAPI("7105197606", "24f0fd5b279c45a28f1e2d38f8d1ec0dbcf26a571ea4482896")

# PyTorch Model Definition
class FloodPred(nn.Module):
    def __init__(self, input_size=20):
        super(FloodPred, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Model Loading
MODEL_PATHS = {
    'xgboost': 'models/floodXgBoostV1.pkl',
    'pytorch': 'models/floodNN.pt'
}

def load_models():
    try:
        xgboost_model = joblib.load(MODEL_PATHS['xgboost'])
        pytorch_model = FloodPred(input_size=20)
        pytorch_model.load_state_dict(
            torch.load(MODEL_PATHS['pytorch'], map_location=torch.device('cpu'))
        )
        pytorch_model.eval()
        return xgboost_model, pytorch_model
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

try:
    xgboost_model, pytorch_model = load_models()
except RuntimeError as e:
    app.logger.error(str(e))
    exit(1)

# Alert System
class DynamicThresholdCalculator:
    def __init__(self, data_path='dataset/train_data.csv'):
        try:
            self.data = pd.read_csv(data_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        except Exception as e:
            app.logger.error(f"Error loading threshold data: {str(e)}")
            self.data = pd.DataFrame()

    def calculate_threshold(self):
        if self.data.empty:
            return 0.45
        time_window = datetime.now() - timedelta(hours=6)
        recent_data = self.data[self.data['timestamp'] > time_window]
        if len(recent_data) > 50:
            return recent_data['risk_score'].quantile(0.75)
        return 0.5

threshold_calculator = DynamicThresholdCalculator()

def send_whatsapp_alert(result, phone_number):
    warning_threshold = 0.56  
    evacuation_threshold = 0.70
    if result >= evacuation_threshold:
        message = '''🚨 URGENT EVACUATION ALERT
⚠️ FLOOD RISK IS EXTREMELY HIGH!

Your area is at severe risk of flooding. Evacuate immediately to higher ground or a safe location.

1️⃣ Carry essentials (ID, cash, meds, food, water).
2️⃣ Turn off electricity and gas before leaving.
3️⃣ Avoid floodwaters—stay safe from strong currents.

DO NOT DELAY! ACT NOW!
For help, contact local authorities. Stay informed and stay safe!'''
        response = greenAPI.sending.sendMessage(phone_number, message)
        print(f"Evacuation Alert Sent: {response.data}")
    elif result >= warning_threshold:
        message = '''⚠️ FLOOD WARNING
There is a significant chance of flooding in your area. Stay alert and take these precautions:

1️⃣ Move to higher ground if you are in a low-lying area.
2️⃣ Prepare an emergency kit with essentials (ID, cash, meds, food, water).
3️⃣ Avoid walking or driving through floodwaters—stay safe from currents and debris.
4️⃣ Stay updated via official channels for further instructions.

Be prepared and stay safe!'''
        response = greenAPI.sending.sendMessage(phone_number, message)
        print(f"Flood Warning Sent: {response.data}")
    else:
        print("No alert sent. Flood probability is below the warning threshold.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features', [])
        phone_number = '919311570728@c.us'
        location = data.get('location', 'Unknown')
        if len(features) != 20:
            return jsonify({'error': 'Exactly 20 features required'}), 400
        STATIC_THRESHOLD = 0.65
        xgb_pred = xgboost_model.predict(np.array([features]))[0]
        with torch.no_grad():
            torch_pred = pytorch_model(torch.tensor([features], dtype=torch.float32)).item()
        combined_risk = (xgb_pred * 0.7 + torch_pred * 0.3)
        alert_status = "High Risk" if combined_risk > STATIC_THRESHOLD else "Low Risk"
        send_whatsapp_alert(combined_risk, phone_number)
        return jsonify({
            'risk': combined_risk,
            'threshold': STATIC_THRESHOLD,
            'alert': alert_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'operational',
        'models': list(MODEL_PATHS.keys()),
        'threshold_data': not threshold_calculator.data.empty
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
