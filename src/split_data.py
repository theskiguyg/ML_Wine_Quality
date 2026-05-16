import os
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/raw/WineQT.csv')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

os.makedirs('data/processed', exist_ok=True)

X_train.to_csv('data/processed/X_train.csv', index = False)
X_test.to_csv('data/processed/X_test.csv', index = False)
y_train.to_csv('data/processed/y_train.csv', index = False)
y_test.to_csv('data/processed/y_test.csv', index = False)

print("Data has been splitted and saved to data/processed!")