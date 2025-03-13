import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle as pickle

def create_model(data):
    X=data.iloc[:, 1:]
    y=data.iloc[:, 0]
    y = y.map({'B': 0, 'M': 1})
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    test_model(model,X_test,y_test)
    
    return model,scaler

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.iloc[:, 1:]
    return data

def test_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print("classification report\n",classification_report(y_test, y_pred))
    
    
def main():
    data = get_clean_data()
    model,scaler = create_model(data)
    
    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
        
    
    
if __name__ == '__main__':
    main()