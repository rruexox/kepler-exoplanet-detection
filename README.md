# Kepler Exoplanet Classification

A machine learning project to classify exoplanet candidates from NASA's Kepler  
Space Observatory into three categories: FALSE POSITIVE, CANDIDATE, and CONFIRMED.

## Dataset
[Kepler Exoplanet Dataset](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset) — 
9,564 observations with 9 orbital and stellar features.

| Label | Class | Description |
|-------|-------|-------------|
| 0 | FALSE POSITIVE | Not a real exoplanet |
| 1 | CANDIDATE | Potential exoplanet, awaiting confirmation |
| 2 | CONFIRMED | Verified exoplanet |

## What I did
Started with basic preprocessing, then added feature engineering and iterative  
imputation to squeeze more signal out of the 9 features. Used SMOTE to handle  
the class imbalance, tuned XGBoost with cross-validation, and combined it with  
Random Forest in a soft voting ensemble.

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Decision Tree | 0.76 | 0.71 |
| XGBoost (tuned) | 0.80 | 0.76 |
| Ensemble | 0.81 | 0.77 |

> **98% accuracy** when classifying confirmed planets against false positives.

The CANDIDATE class was the hardest to classify — these are objects scientists  
haven't confirmed yet, so the ambiguity is real. Running binary classification  
between FALSE POSITIVE and CONFIRMED only hits  
**98% accuracy** and **0.97 macro F1**.
