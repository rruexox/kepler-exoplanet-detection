# Kepler Exoplanet Classification

A machine learning project to classify exoplanet candidates from NASA's Kepler Space Observatory into three categories: FALSE POSITIVE, CANDIDATE, and CONFIRMED.

## Dataset

[Kepler Exoplanet Dataset](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset) — 9,564 observations with 9 orbital and stellar features.

| Label | Class | Description |
|-------|-------|-------------|
| 0 | FALSE POSITIVE | Not a real exoplanet |
| 1 | CANDIDATE | Potential exoplanet, awaiting confirmation |
| 2 | CONFIRMED | Verified exoplanet |

## What I did

The dataset had 9 orbital and stellar features, so I started by engineering 4 new ones to capture relationships that the raw features couldn't express on their own:

- **score_period:** KOI score multiplied by orbital period, capturing how confidence scales with orbit length
- **radius_ratio:** planet radius divided by stellar radius, a direct measure of how large the planet is relative to its star
- **score_sq:** KOI score squared, to amplify the difference between high and low confidence signals
- **temp_ratio:** equilibrium temperature divided by stellar effective temperature, indicating how much heat the planet receives from its star

From there, I handled missing values with iterative imputation, scaled with RobustScaler (more stable than StandardScaler with outliers), and used SMOTE to fix the class imbalance since FALSE POSITIVEs heavily dominated the dataset.

I then ran XGBoost with RandomizedSearchCV over 40 iterations using stratified 5-fold cross-validation, and combined the tuned model with a Random Forest in a soft voting ensemble.

## Results

### Multiclass Classification (3 categories)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|----------|-----------------|--------------|----------|
| Decision Tree | 0.71 | 0.71 | 0.71 | 0.67 |
| XGBoost | 0.80 | 0.80 | 0.80 | 0.80 |
| Ensemble | 0.77 | 0.77 | 0.77 | 0.77 |

### Binary Classification (FALSE POSITIVE vs CONFIRMED)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| FALSE POSITIVE | 0.98 | 0.98 | 0.98 |
| CONFIRMED | 0.97 | 0.96 | 0.97 |
| **Macro Avg** | **0.97** | **0.97** | **0.97** |
| **Accuracy** | — | — | **0.98** |

The trickiest class was CANDIDATE, and that's not a model failure; that's just reality. These are objects scientists haven't confirmed yet, so no algorithm can reliably classify something that astronomers themselves are still unsure about.

To test this, I ran a binary classifier on just FALSE POSITIVE vs CONFIRMED, dropping CANDIDATEs entirely. The results jumped to **0.98 accuracy** and **0.97 macro precision/recall/F1**, which confirms the multiclass ambiguity was coming from the labels, not the model.

## Usage

### Prerequisites

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn jupyter
```

### Running the Notebook

```bash
jupyter notebook exoplanet_classification.ipynb
```

The notebook is fully self-contained and walks through:
1. Data loading and exploration
2. Feature engineering and preprocessing
3. Model training with hyperparameter tuning
4. Performance evaluation with confusion matrices and classification reports
5. Comparison of Decision Tree, XGBoost, and Ensemble models

## Files

- `exoplanet_classification.ipynb` — Main analysis notebook
- `data/` — Folder containing the dataset
  - `exoplanet_data.csv` — Raw Kepler exoplanet observations (9,564 rows, 9 features)
- `requirements.txt` — Python dependencies
- `README.md` — This file

## What I'd do next

The most natural next step is adding a feature importance analysis using SHAP values to understand which Kepler measurements actually drive the predictions. Features like radius_ratio and temp_ratio have real astronomical meaning, so it would be interesting to see if the model agrees with what astrophysics would predict.
