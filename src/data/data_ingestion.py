import pandas as pd
import os
import joblib  # <-- Added for saving artifacts
# for outside cicd
from dotenv import load_dotenv
load_dotenv()

from src.Variables.variables import RAW_DATA_PATH, INTERIM_DATA_PATH
from sklearn.preprocessing import LabelEncoder

print('\n')
print('   🔴 data_ingestion.py   ')
print('✅ Getting AWS credentials ......')
Access_key = os.getenv("Access_key")
Secret_access_key = os.getenv("Secret_access_key")

def data_ingestion(url, path):
    try:
        df = pd.read_csv(
        url,
        storage_options={
            "key": Access_key,
            "secret": Secret_access_key
        }
        )
        return df
    except Exception as e:
        print('error: ', e)

def preprocessing(df):
    cols_to_encode = df.select_dtypes(include=['object', 'category']).columns
    encoders_dict = {} # <-- Dictionary to hold isolated column mappings
    
    for i in cols_to_encode:
        encoder = LabelEncoder() # Create a fresh encoder per column
        df[i] = encoder.fit_transform(df[i].astype(str))
        encoders_dict[i] = encoder # Store it in the dictionary
    
    # Save the dictionary so Flask can read 'Location_Category'
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(encoders_dict, 'artifacts/label_encoders.pkl')
    print('✅ Exported tracking map to artifacts/label_encoders.pkl')
    
    return df

def main():
    output_dir = os.path.dirname(RAW_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
    output_dir2 = os.path.dirname(INTERIM_DATA_PATH)
    os.makedirs(output_dir2, exist_ok=True)
    print('✅ Pulling data from S3')
    df = data_ingestion('s3://dynamic--pricing/dynamic_pricing.csv', 'C:\mlops\End-to-End-Real-Time-Dynamic-Pricing-Engine\data')
    df.to_csv(RAW_DATA_PATH)
    print('✅ Preprocessing data')
    data = preprocessing(df)
    print('✅ Saving Processed data ...')
    data.to_csv(INTERIM_DATA_PATH)
    print('✅ Done !')
    print('\n')

main()