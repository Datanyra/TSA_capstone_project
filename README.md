# TSA_capstone_project(Telco Customer Churn Prediction)

**Imbalance Depth Track**

An end-to-end machine learning project that predicts which telecom customers are likely to churn, segments the customer base with clustering, and picks a deployment threshold from an explicit business cost rather than from a generic metric.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green.svg)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)

---

## Table of Contents

- [Overview](#overview)
- [Headline Results](#headline-results)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Customer Segments](#customer-segments)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Limitations and Next Steps](#limitations-and-next-steps)

---

## Overview

Customer churn is expensive: losing a subscriber costs future revenue plus the acquisition spend to replace them. This project answers three questions:

1. **Who is likely to leave?** Supervised classification with Logistic Regression, Random Forest and Gradient Boosting, benchmarked against a stratified dummy classifier.
2. **What kinds of customers do we have?** Unsupervised K-Means segmentation, with churn brought back in afterwards to test whether the segments are meaningful.
3. **How should the model be used?** A decision threshold chosen by minimising an assumed business cost (a missed churner costs 5x an unnecessary retention offer), with a sensitivity check on that assumption.

## Headline Results

| Metric | Stratified dummy | Final model |
|---|---|---|
| **F1 (churn class)** | 0.290 | **0.591** |
| **ROC AUC** | 0.516 | **0.843** |

**Final model:** Logistic Regression with the decision threshold tuned to **0.14** (cost-based, 5:1 FN:FP).

At this threshold, on the held-out test set of 1,409 customers (374 churners):

- Catches **345 of 374 churners** (recall 0.922), missing only 29
- Sends 449 unnecessary retention offers (precision 0.435)
- Cuts expected business cost from **1,002 to 594**, a **41% reduction**

## Dataset

**Telco Customer Churn** (IBM sample data), loaded through `kagglehub` from [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

| Property | Value |
|---|---|
| Rows / columns | 7,043 / 21 |
| Target | `Churn` (Yes / No) |
| Churn rate | about 26.5% (imbalanced, roughly 1 churner per 3 stayers) |
| Split | 80/20 stratified train/test |
| Data leakage | None found. `customerID` was dropped as a non-predictive identifier |

The data is downloaded at runtime, so no credentials or raw files need to be committed.

## Project Workflow

| Part | What it covers |
|---|---|
| **1. Data Foundation** | Loads the data, runs it through SQLite, and answers three business questions in SQL (churn by payment method, contract type and tenure band). EDA and cleaning follow. |
| **2. Feature Engineering & Encoding** | Three engineered features, label and one-hot encoding, and `StandardScaler`. |
| **3. Clustering** | K-Means with elbow and silhouette analysis (K = 2 to 10), cluster profiling, PCA visualisation. |
| **4. Classification** | Logistic Regression, Random Forest and Gradient Boosting plus a stratified dummy baseline. Accuracy, precision, recall, F1, ROC AUC and confusion matrices. |
| **5. Imbalance Handling** | Class weights, SMOTE and cost-based threshold tuning applied to all three models, compared on one test set. |
| **6. Tie-In** | Churn rate inside each K-Means cluster. |
| **7. Conclusion** | Deployment recommendation, next steps and reflections. |
| **Depth Track** | Precision-recall curves for every technique on one chart, with a final recommended operating point. |

### Engineered features

- **`tenure_bucket`**: tenure binned into 0-1yr, 1-2yr, 2-4yr, 4-5yr and 5yr+, since churn falls in bands rather than smoothly.
- **`num_addon_services`**: count of subscribed optional services, a proxy for how invested a customer is.
- **`avg_monthly_spend`**: `TotalCharges / tenure`, a historical average that can be compared to current `MonthlyCharges` to hint at price increases.

### Data cleaning

`TotalCharges` was stored as text because 11 rows were blank. All 11 belong to customers with a tenure of 0 months (new and not yet billed), so the blanks were filled with 0 rather than dropped or imputed with a mean.

## Key Findings

**Churn drivers from the SQL analysis**

| Factor | Finding |
|---|---|
| Contract type | Month-to-month churns at about 43%, one-year at about 11%, two-year at under 3% |
| Tenure | First-year customers churn at almost 47%, customers past four years at under 10% |
| Payment method | Electronic check users churn at a much higher rate than customers on automatic payments |

**Modelling insights**

- At the default 0.50 threshold every baseline catches only about half of real churners.
- ROC AUC stayed in the 0.826 to 0.845 range across every technique. Imbalance handling does not improve how well the model *ranks* customers, it only moves *where on the curve* you operate.
- The best F1 (0.625) came from Gradient Boosting with class weights, but the **lowest business cost** came from Logistic Regression with a tuned threshold. F1 treats both error types as equal, and the business does not.
- The simplest, most interpretable model came out on top.

## Handling Class Imbalance

Test-set results for **Logistic Regression** under each technique:

| Configuration | Precision | Recall | F1 | Test cost (5·FN + FP) |
|---|---|---|---|---|
| Baseline (t = 0.50) | 0.655 | 0.519 | 0.579 | 1,002 |
| Class weights (t = 0.50) | 0.501 | 0.791 | 0.613 | 685 |
| SMOTE (t = 0.50) | 0.510 | 0.786 | 0.619 | 682 |
| **Threshold tuned (t = 0.14)** | **0.435** | **0.922** | **0.591** | **594** |

### Business cost framing

| Error | Consequence | Assumed cost |
|---|---|---|
| False positive | Unnecessary retention offer (about one month's discount) | 1 unit |
| False negative | Lost customer: future revenue plus replacement cost | 5 units |

The threshold is chosen using **5-fold out-of-fold predictions on the training set only**, then applied once to the test set, so the test data never influences any decision.

### Sensitivity to the cost ratio

| FN:FP ratio | Approx. optimal threshold (Logistic Regression) |
|---|---|
| 3:1 | 0.24 |
| 5:1 | 0.14 |
| 10:1 | 0.07 |

The cost curve is fairly flat between about 0.10 and 0.25, so small threshold changes cost little. The 5:1 ratio is an assumption and should be validated with real offer costs and customer lifetime value before deployment.

### Leakage controls

- Scaling is fit on training data only (and refit inside each CV fold via pipelines).
- SMOTE is applied only to training data, inside an `imblearn` pipeline.
- The threshold is selected from out-of-fold training predictions.

## Customer Segments

K-Means with **K = 3** (the highest silhouette score). The clusters were built without ever seeing the churn label.

| Cluster | Persona | Size | Churn rate | Lift vs. overall | Share of all churners |
|---|---|---|---|---|---|
| 0 | Long-tenure, bundled premium | 1,526 (22%) | 7.4% | 0.28x | 6.0% |
| 1 | New, mid-spend, high-risk | 3,258 (46%) | **42.1%** | **1.59x** | **73.4%** |
| 2 | Long-tenure, low-spend basics | 2,259 (32%) | 17.0% | 0.64x | 20.6% |

Cluster 1 is under half the customer base but holds nearly three quarters of all churners. The fact that an unsupervised method recovers the same risk groups a supervised model relies on suggests churn here is largely driven by **customer lifecycle stage**: new, month-to-month and few add-ons. Cluster membership is a cheap, explainable proxy that non-technical teams can use in a dashboard.

> Cluster numbering can shift with column order, so check your own profile table rather than assuming the labels match.

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook, JupyterLab or Google Colab
- Internet access (the dataset is downloaded via `kagglehub`)

### Run on Google Colab

The notebook is designed for "Restart and run all" on a fresh Colab session and completes in **under 3 minutes**.

1. Open `Group_36_Oderinde_Suliha.ipynb` in Colab.
2. Select **Runtime → Run all**.

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install kagglehub imbalanced-learn pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch the notebook
jupyter notebook Group_36_Oderinde_Suliha.ipynb
```

No API token is hardcoded. `kagglehub` downloads the public dataset at runtime.

### Reproducibility

All randomness is seeded with `RANDOM_STATE = 42`.

## Repository Structure

```
.
├── Group_36_Oderinde_Suliha.ipynb   # Full analysis, from data loading to recommendations
└── README.md
```

A local `telco.db` SQLite file is created when the notebook runs.

## Limitations and Next Steps

- **Use SMOTENC instead of plain SMOTE.** Most features are one-hot dummies, and plain SMOTE can produce fractional values such as 0.4 for a binary column. It did not visibly hurt here, but SMOTENC handles categorical data correctly.
- **Calibrate probabilities** (Platt scaling or isotonic regression) before threshold tuning, since cost minimisation assumes the probabilities are meaningful and not just well ranked.
- **Validate the 5:1 cost ratio** with the retention and finance teams, then re-run the threshold procedure.
- **Add richer features** such as support-ticket counts, complaint history and local competitor promotions, to explain *why* customers inside the high-risk cluster leave.
- **Pilot before rollout.** Test the threshold on a slice of live traffic and monitor for drift as pricing and competitor offers change.

## Acknowledgements

- Dataset: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data)
- Built with pandas, scikit-learn, imbalanced-learn, matplotlib and seaborn
