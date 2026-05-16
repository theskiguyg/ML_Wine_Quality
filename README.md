# Wine Quality Prediction

The project has been created by Zagrebin Egor, NUST MISIS AI "29. 

## 📦 Installation and launch


### 1. Creating a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```


### 2. Installing dependencies
```bash
pip install -r requirements.txt
```


### 3. Data preparation
Important: The split data is not stored in the repository. It must be generated locally. You can find the full dataset in `data/raw/WineQT.csv`. 

```bash
python src/split_data.py
```

This script does the following:

- Loads the original dataset from `data/raw/WineQT.csv`. 
- Splits the data into train/test sets (80/20)
- Saves the processed files to `data/processed/`:
    X_train.csv, X_test.csv — features
    y_train.csv, y_test.csv — target variable

### 4. Train the model
```bash
python src/train.py
```

### 5. Make predictions and evaluate 
```bash
python src/predict.py
```

