# Wine Quality Prediction

Created by Zagrebin Egor as part of the ML Engineering course based on the book "Hands-On Machine Learning" (Aurélien Géron, 1st edition, 2017). 

Goal: Building an end-to-end production-ready ML solution with clean code, configuration, and reproducibility. 


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
    - X_train.csv, X_test.csv — features
    - y_train.csv, y_test.csv — target variable

### 4. Train the model
```bash
python src/train.py
```

### 5. Make predictions and evaluate 
```bash
python src/predict.py
```


## Executive Summary

The regression task of predicting wine quality has been completed. Multiple models were trained and tested. The final result on the holdout test set is **Test R² = 0.4123**.

Analysis shows that predicting exact quality scores is challenging due to subjective expert ratings, noise, and limited dataset size. For this dataset, binary classification (good / bad wine) is a more suitable approach.

---

## Regression Results

### Final Model

**Model:** RandomForestRegressor

**Hyperparameters:**
- max_depth = 10
- min_samples_leaf = 5
- min_samples_split = 10
- n_estimators = 50
- max_features = "sqrt"

**Metrics:**
- Test R² = 0.4123
- Train R² = 0.6492
- CV mean (5-fold) = 0.4063
- CV std = 0.0617

---

### Other Models Tested

**LinearRegression**
CV mean ~0.38. Too simple, fails to capture non‑linear patterns.

**Ridge**
CV mean ~0.40. Slightly better than LinearRegression due to regularization.

**Lasso**
CV mean ~0.35. Strong regularization zeroed out many coefficients.

**GradientBoostingRegressor**
CV mean ~0.40. Showed overfitting and did not outperform RandomForest after tuning.

**RandomForestRegressor**
CV mean ~0.41. Achieved the best result among all tested models.

---

## Analysis of Results

### Why R² = 0.41?

**Noise in the data.** Quality scores are averaged from multiple expert ratings — inherently subjective.

**Small sample size.** Only ~1100 rows, limiting the model's ability to generalize.

**Incomplete information.** Chemical properties alone do not fully explain wine quality. Grape variety, production methods, and other factors also matter.

**Integer target.** The model predicts continuous values, while true labels are integers from 3 to 8.

### Conclusion on Regression

Predicting exact wine quality from chemical properties has an objective ceiling. R² ≈ 0.41 is a solid result in this context and aligns with best practices reported for this dataset.

---

## Next Step: Binary Classification

Regression has reached its practical limit. The logical next step is binary classification.

**New target:**
- Good wine = quality >= 7
- Bad wine = quality < 7


### Why Classification Will Perform Better

**Easier task.** The model does not need to hit an exact score — separating good from bad is simpler.

**Matches real‑world use.** In practice, one typically wants to know whether a wine is worth buying, not its precise rating.

**Less sensitive to noise.** A binary threshold smooths out subjective fluctuations in expert scores.

---

## Conclusion

### Regression Phase (completed)

- EDA and preprocessing — done
- Training multiple regression models — done
- Hyperparameter tuning with GridSearch — done
- Evaluation on test set — R² = 0.4123

### Planned

- Switch to binary classification to achieve higher metrics

The regression project is finished. The result meets expectations. Moving forward to classification.

