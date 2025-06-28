# E-commerce Shopping Revenue Prediction: Binary Classification with Supervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![pandas](https://img.shields.io/badge/pandas-1.5.3-blueviolet)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## 📖 Project Overview

This project builds a **binary classification model** to predict whether an e-commerce user session will generate revenue. Using session-level behavioral data, technical environment metadata, and temporal features, the model assists business analysts in identifying high-value visits.

The pipeline includes:

* Data ingestion and feature engineering from raw session logs
* Encoding categorical and boolean variables
* Scaling numerical inputs for model compatibility
* Model benchmarking with multiple classification algorithms
* Hyperparameter optimization of the best-performing model
* Comprehensive evaluation using classification metrics and visual diagnostics

---

## 🗃️ Dataset Description

The input dataset (`shopping.csv`) contains session-level aggregated metrics.

---

## 🔧 Data Preprocessing & Feature Engineering

* **Categorical encoding:**

  * `Month` converted from string to integer (0 for January through 11 for December) using `pd.to_datetime` for temporal modeling.
  * `VisitorType` one-hot encoded into binary flags (`Visitor_Returning_Visitor`, `Visitor_New_Visitor`, `Visitor_Other`).
* **Boolean conversion:**

  * `Weekend` and `Revenue` converted from string TRUE/FALSE to integer 0/1.
* **Feature scaling:**

  * Numerical features standardized with `StandardScaler` to zero mean and unit variance to improve model convergence, particularly for KNN and SVM.

---

## ⚙️ Model Training and Evaluation

* **Train/Test Split:** Stratified 80/20 split preserving class proportions for target variable `Revenue`.
* **Cross-validation:** Stratified 4-fold cross-validation used during model comparison and hyperparameter tuning to prevent overfitting and estimate generalization performance.
* **Models evaluated:**

  * K-Nearest Neighbors (k=1)
  * Logistic Regression (balanced class weights)
  * Random Forest Classifier
  * Support Vector Machine (RBF kernel)

* **Metrics:**

  * Precision (macro-averaged) — how many predicted positives are true positives
  * Recall (weighted) — how many actual positives are detected
  * Weighted F1 score — harmonic mean of precision and recall balancing class imbalance
  * ROC-AUC for overall classification discrimination

---

## 🔍 Hyperparameter Optimization

* Performed grid search on Random Forest classifier:

  * `max_depth`: \[1, 8, 16]
  * `n_estimators`: \[1, 11]
* Best parameters: `max_depth=16`, `n_estimators=11` with weighted F1 score \~0.89
* Improved precision-recall balance and reduced overfitting.

---

## 🎯 Final Results (Test Set)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 88.44% |
| Precision | 90.43% |
| Recall    | 88.44% |
| F1 Score  | 89.22% |
| ROC-AUC   | 89.51% |

* Confusion matrix analysis confirms strong detection of revenue-generating sessions with minimal false negatives.
* Precision-Recall and ROC curves included to guide threshold tuning in deployment.

---

## Feature Importance Insights

* Extracted from Random Forest model’s `feature_importances_`.
* Top drivers of revenue prediction include:

  * **BounceRates**: session bounce likelihood
  * **ExitRates**: exit probability during session
  * **PageValues**: monetary value associated with pages viewed
* Feature importance visualized via gradient-colored horizontal bar chart for interpretability.

---

## 🚀 Usage Instructions

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the classifier script with dataset path:

```bash
python shopping.py shopping.csv
```

4. The script will output model evaluation statistics and generate visual performance plots.

---

## 📚 Tools & Libraries

* [pandas](https://pandas.pydata.org/) — data loading and manipulation
* [scikit-learn](https://scikit-learn.org/stable/) — classification algorithms, cross-validation, and metrics
* [matplotlib](https://matplotlib.org/) — visualization of metrics and feature importance

---

## ⚖️ License

Licensed under the **MIT License**.
