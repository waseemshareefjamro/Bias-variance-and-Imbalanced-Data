# Imbalanced Classification & Model Diagnostics: Medical Screening Case Study

A hands-on study of imbalanced classification using the Kaggle Stroke Prediction Dataset. A single end-to-end notebook explores model diagnostics, leakage-aware preprocessing, resampling, and noise sensitivity in a medical screening case study.

## Overview & Learning Objectives

Standard accuracy can hide poor performance on rare events. The notebook uses a dataset with about 4.9% positive stroke cases to demonstrate:

1. **Beyond Accuracy:** Demonstrating why accuracy is misleading under severe class imbalance and selecting task-appropriate metrics (Precision, Recall, F1, PR-AUC).
2. **Bias-Variance Diagnostics:** Identifying underfitting (high bias) and overfitting (high variance) using learning curves, validation curves, and direct variance measurements.
3. **Threshold & Curve Analysis:** Comparing ROC curves against Precision-Recall (PR) curves, understanding the limitations of AUC, and tuning decision thresholds for domain costs.
4. **Leakage-Aware Resampling:** Comparing class weighting, Random Undersampling, Random Oversampling, and SMOTE while keeping resampling away from the held-out test data.
5. **Noise Sensitivity:** Evaluating how label noise and feature noise degrade different model families—and why synthetic oversampling (SMOTE) can amplify noise.
6. **Medical Screening Case Study:** Combining the results into an interpretable model and threshold recommendation, with the limitations of a small positive test set made explicit.

## Repository Structure

```text
├── Bias_Variance_Imbalanced_Data.ipynb  # Complete analysis and experiments
└── README.md
```

## Key Takeaways

- **Choose Metrics for the Positive Class:** Precision-Recall curves help reveal minority-class performance that can be obscured by a large number of true negatives.
- **Keep Resampling Inside Training Data:** Applying SMOTE before a train/test split leaks information. Compare resampling against class weighting and threshold tuning rather than assuming it improves the model.
- **SMOTE + Noise Risk:** Interpolating between noisy minority samples creates synthetic outliers, hurting boundary-sensitive models.
- **Calibrate the Threshold to the Task:** A screening threshold trades false negatives against false alarms; choose it using training data and report both sides of that trade-off.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/waseemshareefjamro/Bias-variance-and-Imbalanced-Data.git
   cd Bias-variance-and-Imbalanced-Data
   ```

2. **Install the notebook dependencies** (Python 3.9 or newer recommended):
   ```bash
   python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn imbalanced-learn kagglehub
   ```

3. **Launch JupyterLab and open the notebook:**
   ```bash
   jupyter lab
   ```

4. **Provide the dataset.** The notebook looks for `healthcare-dataset-stroke-data.csv` in the current folder or a subfolder. If it is not present, it tries to download the dataset through `kagglehub`; download it from the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and place the CSV in the repository if automatic download is unavailable. The final manual-upload fallback uses Google Colab, so local users should place the CSV in the repository.

5. **Run all cells** from top to bottom. The notebook sets `SEED = 42` and splits the data before fitting preprocessing steps or comparing thresholds. It may install `imbalanced-learn` and `kagglehub` automatically when they are missing.

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`, `JupyterLab`
