import pandas as pd
from sklearn.model_selection import train_test_split
from src.Variables.variables import PROCESSED_DATA_PATH, SPLITED_DATA_PATH, MODEL_PATH
from src.Variables.params import test_size, random_state, n_estimators , Random_state 
from sklearn.ensemble import RandomForestRegressor
import joblib
print('\n')
print('   🔴 build_model.py   ')

def load_data(path):
    df = pd.read_csv(path)
    return df

def splting_data(df, Test_size, Random_state,  path):
    x = df.drop(columns=['Historical_Cost_of_Ride'])
    y = df['Historical_Cost_of_Ride']
    x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=Test_size,random_state=Random_state)
    x_train.to_csv(path+'/x_train.csv', index=False)
    x_test.to_csv(path+'/x_test.csv', index=False)
    y_train.to_csv(path+'/y_train.csv', index=False)
    y_test.to_csv(path+'/y_test.csv', index=False)
    return x_train, x_test, y_train,y_test 

def build_model(x_train, y_train, N_estimator, RAndom_state):
    model = RandomForestRegressor(n_estimators=N_estimator, random_state=RAndom_state)
    model.fit(x_train, y_train)
    return model

def save_model(model, path_of_model):
    joblib.dump(model, path_of_model)


def main():
    print('✅ loading Data')
    df = load_data(PROCESSED_DATA_PATH)
    print('✅ spliting Data')
    x_train, x_test, y_train,y_test = splting_data(df, test_size, random_state, SPLITED_DATA_PATH)
    print('✅ building model')
    model = build_model(x_train, y_train, n_estimators, Random_state)
    print('✅ Saving model')
    save_model(model, MODEL_PATH)
    print('✅ Done !')

main()

