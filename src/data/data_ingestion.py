import pandas as pd
import os
# for outside cicd
from dotenv import load_dotenv
load_dotenv()
from src.connections import s3_connection
from src.Variables.variables import RAW_DATA_PATH, INTERIM_DATA_PATH
from sklearn.preprocessing import LabelEncoder




print('\n')
print('   🔴 data_ingestion.py   ')
print('✅ Geting AWS crediancials ......')
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
    encoder = LabelEncoder()
    cols_to_encode = df.select_dtypes(include=['object', 'category']).columns
    encoder = LabelEncoder()
    for i in cols_to_encode:
        df[i] = encoder.fit_transform(df[i].astype(str))
    return df



def main():
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
    
