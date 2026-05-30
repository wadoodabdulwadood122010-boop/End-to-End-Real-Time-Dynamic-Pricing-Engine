import os
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
import mlflow.pyfunc
from dotenv import load_dotenv
import dagshub

# --- Prometheus Imports ---
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

app = Flask(__name__)

# --- Prometheus Metrics Setup ---
# Track total requests by status
REQUEST_COUNT = Counter('predict_requests_total', 'Total number of requests to the predict endpoint', ['status'])
# Track prediction latency
PREDICT_LATENCY = Histogram('predict_latency_seconds', 'Time spent processing prediction requests')
# Track the actual predicted cost values (useful for monitoring model drift in Grafana)
PREDICTED_COST = Gauge('predicted_ride_cost_dollars', 'The latest predicted cost for a ride')
# Track internal errors
ERROR_COUNT = Counter('predict_errors_total', 'Total number of errors during prediction')


# --- MLflow Configuration ---

dagshub_token = os.getenv("DAGSHUB_TOCKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOCKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "wadoodabdulwadood122010"
repo_name = "End-to-End-Real-Time-Dynamic-Pricing-Engine"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

MODEL_NAME = "Dynamic_Pricing_Engine" 
STAGE = "Production"

print("✅ Loading local mathematical states & pulling staging model...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    
    # Loading the artifacts created by generate_artifacts.py
    encoders = joblib.load("artifacts/label_encoders.pkl")
    transformer = joblib.load("artifacts/power_transformer.pkl")
    mm_scaler = joblib.load("artifacts/minmax_scaler.pkl")
    qt_transformer = joblib.load("artifacts/quantile_transformer.pkl")
    target_scaler = joblib.load("artifacts/target_scaler.pkl")
    print("✅ Web UI backend ready.")
except Exception as e:
    print(f"⚠️ Initialization Error: {e}")


# --- Routes ---

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()  # Start timing for Prometheus latency metric
    
    try:
        user_input = request.json
        
        # 1. Feature Engineering (Ratio calculation)
        supply_demand_ratio = float(user_input['Number_of_Drivers']) / float(user_input['Number_of_Riders'])
        
        # 2. Structure live input into a DataFrame
        df_inf = pd.DataFrame({
            'Location_Category': [user_input['Location_Category']],
            'Customer_Loyalty_Status': [user_input['Customer_Loyalty_Status']],
            'Number_of_Past_Rides': [int(user_input['Number_of_Past_Rides'])],
            'Average_Ratings': [float(user_input['Average_Ratings'])],
            'Time_of_Booking': [user_input['Time_of_Booking']],
            'Vehicle_Type': [user_input['Vehicle_Type']],
            'Expected_Ride_Duration': [float(user_input['Expected_Ride_Duration'])],
            'Supply-Demand Ratio': [supply_demand_ratio]
        })

        # 3. Apply transformations using the loaded training states
        cat_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
        for col in cat_cols:
            df_inf[col] = encoders[col].transform(df_inf[col].astype(str))
            
        df_inf['Supply-Demand Ratio'] = transformer.transform(df_inf[['Supply-Demand Ratio']])
        df_inf['Average_Ratings'] = mm_scaler.transform(df_inf[['Average_Ratings']])
        df_inf['Expected_Ride_Duration_Gaussian'] = qt_transformer.transform(df_inf[['Expected_Ride_Duration']])
        
        # 4. Force strict column ordering to match processed.csv exactly
        feature_order = [
            'Location_Category', 
            'Customer_Loyalty_Status', 
            'Number_of_Past_Rides', 
            'Average_Ratings', 
            'Time_of_Booking', 
            'Vehicle_Type', 
            'Expected_Ride_Duration', 
            'Supply-Demand Ratio', 
            'Expected_Ride_Duration_Gaussian'
        ]
        df_inf = df_inf[feature_order]
        
        # 5. Compute Prediction
        scaled_prediction = model.predict(df_inf)
        if hasattr(scaled_prediction, 'iloc'):
            pred_val = scaled_prediction.iloc[0]
        elif isinstance(scaled_prediction, (list, np.ndarray)):
            pred_val = scaled_prediction[0]
        else:
            pred_val = scaled_prediction
            
        # 6. De-scale the prediction target to return the true dollar cost
        actual_cost = round(float(target_scaler.inverse_transform([[pred_val]])[0][0]), 2)
        
        # --- Update Prometheus Metrics (Success) ---
        REQUEST_COUNT.labels(status='success').inc()
        PREDICTED_COST.set(actual_cost)
        PREDICT_LATENCY.observe(time.time() - start_time)
        
        return jsonify({
            "status": "success",
            "predicted_historical_cost": actual_cost
        }), 200

    except Exception as e:
        # --- Update Prometheus Metrics (Error) ---
        REQUEST_COUNT.labels(status='error').inc()
        ERROR_COUNT.inc()
        PREDICT_LATENCY.observe(time.time() - start_time)
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)