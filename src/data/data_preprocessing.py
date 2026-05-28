import pandas as pd
from src.Variables.variables import INTERIM_DATA_PATH, PROCESSED_DATA_PATH
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
ss = StandardScaler()
mm = MinMaxScaler()
qt = QuantileTransformer(output_distribution='normal', n_quantiles=100)

print('\n')
print('   🔴 data_preprocessing.py   ')



def load_data(path):
    df = pd.read_csv(path)
    return df



def preprocessing(df):
    df.drop(columns=['Unnamed: 0'], inplace = True)
    df['Supply-Demand Ratio'] = df['Number_of_Drivers']/df['Number_of_Riders']
    df['Supply-Demand Ratio'] = transformer.fit_transform(df[['Supply-Demand Ratio']])
    df.drop(columns=['Number_of_Riders', 'Number_of_Drivers'], inplace = True)
    df['Average_Ratings'] = mm.fit_transform(df[['Average_Ratings']])
    df['Historical_Cost_of_Ride'] = ss.fit_transform(df[['Historical_Cost_of_Ride']])
    df['Expected_Ride_Duration_Gaussian'] = qt.fit_transform(df[['Expected_Ride_Duration']])
    return df

def main():
    print('✅ loading interim data ...')
    df = load_data(INTERIM_DATA_PATH)
    print('✅ Data loaded !')
    print('✅ preprocessing ....')
    data = preprocessing(df)
    data.to_csv(PROCESSED_DATA_PATH)
    print('✅ Done !')


main()




