import os
import shutil
import joblib
import mlflow
import mlflow.sklearn
import dagshub
from pathlib import Path
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
load_dotenv()
# ==========================================
# 1. DAGSHUB & CONFIGURATION SETUP
# ==========================================
# Use environment variables if they exist (crucial for CI/CD), otherwise fallback to init()
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

# Below code block is for local use
# -------------------------------------------------------------------------------------
#mlflow.set_tracking_uri('https://dagshub.com/wadoodabdulwadood122010/End-to-End-Real-Time-Dynamic-Pricing-Engine.mlflow')
#dagshub.init(repo_owner='wadoodabdulwadood122010', repo_name='End-to-End-Real-Time-Dynamic-Pricing-Engine', mlflow=True)

# -------------------------------------------------------------------------------------

# OS-agnostic paths using pathlib
LOCAL_MODEL_PATH = Path("./model/model.pkl").resolve()
LOCAL_MLFLOW_DIR = Path("./temp_mlflow_model").resolve() 
REGISTRY_MODEL_NAME = "Dynamic_Pricing_Engine"


def main():
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Could not find local model at: {LOCAL_MODEL_PATH}")

    tracking_uri = mlflow.get_tracking_uri()
    print(f"📡 Connected to MLflow Tracking URI: {tracking_uri}")
    client = MlflowClient()

    # ==========================================
    # 2. LOAD & LOCALLY SAVE MLFLOW MODEL
    # ==========================================
    print(f"📦 Loading local model file from: {LOCAL_MODEL_PATH}")
    model = joblib.load(LOCAL_MODEL_PATH)

    # Clean up the temp directory if it exists from a previous run
    if LOCAL_MLFLOW_DIR.exists():
        shutil.rmtree(LOCAL_MLFLOW_DIR)

    print("🛠️ Structuring model locally to bypass pathing issues...")
    mlflow.sklearn.save_model(
        sk_model=model, 
        path=str(LOCAL_MLFLOW_DIR), # MLflow requires a string representation of the path
        serialization_format="pickle"
    )

    # ==========================================
    # 3. LOG ARTIFACTS & REGISTER
    # ==========================================
    print("🚀 Connecting to DagsHub and starting a new MLflow tracking run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        print("📤 Uploading local model folder to DagsHub...")
        mlflow.log_artifacts(local_dir=str(LOCAL_MLFLOW_DIR), artifact_path="model")
        
        # --- VERIFICATION STEP ---
        artifacts = client.list_artifacts(run_id, path="model")
        uploaded_files = [a.path for a in artifacts]
        print(f"🔎 Files successfully received by DagsHub: {uploaded_files}")
        
        if not any("MLmodel" in file for file in uploaded_files):
            print("❌ UPLOAD FAILED: DagsHub did not receive the MLmodel metadata file.")
            return

        print(f"✅ Model logged successfully under Run ID: {run_id}")

    # ==========================================
    # 4. REGISTER MODEL IN DAGSHUB REGISTRY
    # ==========================================
    model_uri = f"runs:/{run_id}/model"
    print(f"🗂️ Registering model under the name: '{REGISTRY_MODEL_NAME}'...")
    
    try:
        model_version_details = mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
        version = model_version_details.version
        print(f"✅ Successfully registered. Assigned Version: {version}")
    except Exception as e:
        print(f"❌ Failed during registration phase: {e}")
        return

    # ==========================================
    # 5. TRANSITION VERSION TO STAGING STAGE
    # ==========================================
    print(f"🚦 Transitioning Version {version} to 'Staging' stage...")
    try:
        client.transition_model_version_stage(
            name=REGISTRY_MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False  
        )
        print(f"\n🎉 Success! Your model is now live on DagsHub in the STAGING environment.")
        
        # Clean up the temp folder so your workspace stays clean
        if LOCAL_MLFLOW_DIR.exists():
            shutil.rmtree(LOCAL_MLFLOW_DIR)
            
    except Exception as e:
        print(f"❌ Failed to transition stage to Staging: {e}")

if __name__ == "__main__":
    main()