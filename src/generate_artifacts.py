import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer, StandardScaler

print("⚙️ Re-creating training states from raw data...")

# 1. Load the raw data that your ingestion script downloaded
if not os.path.exists('data/raw/raw.csv'):
    # Fallback if your path is different
    raw_path = 'raw.csv'
else:
    raw_path = 'data/raw/raw.csv'

df_raw = pd.read_csv(raw_path)
df_raw.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

os.makedirs('artifacts', exist_ok=True)

# 2. Recreate the exact Label Encoders
cat_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
encoders_dict = {}

for col in cat_cols:
    encoder = LabelEncoder()
    df_raw[col] = encoder.fit_transform(df_raw[col].astype(str))
    encoders_dict[col] = encoder

joblib.dump(encoders_dict, 'artifacts/label_encoders.pkl')
print("✅ Recreated and saved: artifacts/label_encoders.pkl")

# 3. Recreate the exact Mathematical Scalers
transformer = PowerTransformer(method='yeo-johnson')
mm = MinMaxScaler()
qt = QuantileTransformer(output_distribution='normal', n_quantiles=100)
ss = StandardScaler()

# Re-run the exact transformations to fit the scalers
supply_demand_ratio = df_raw['Number_of_Drivers'] / df_raw['Number_of_Riders']
transformer.fit(pd.DataFrame({'Supply-Demand Ratio': supply_demand_ratio}))
mm.fit(df_raw[['Average_Ratings']])
qt.fit(df_raw[['Expected_Ride_Duration']])
ss.fit(df_raw[['Historical_Cost_of_Ride']])

# Save them to disk
joblib.dump(transformer, 'artifacts/power_transformer.pkl')
joblib.dump(mm, 'artifacts/minmax_scaler.pkl')
joblib.dump(qt, 'artifacts/quantile_transformer.pkl')
joblib.dump(ss, 'artifacts/target_scaler.pkl')

print("✅ Recreated and saved all mathematical scaling curves to artifacts/")
print("\n🚀 All artifacts generated successfully! You can now run your Flask app.")