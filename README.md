# COMP 9417: Air Quality Forecasting with Machine Learning

## Term 3 2025

### Group Members
| Name                    | zID        |
|-------------------------|------------|
| Jerry Zhou              | z5477946   |
| Omer Rizwan             | z5453492   |
| Mohammed Amaan          | z5629412   |
| Ritika Lakshminarayanan | z5610633   |
| Pallimulla Himasha      | z5536937   |

---

## Project Description

This project develops machine learning models to forecast air pollutant concentrations using the UCI Air Quality dataset. The system predicts CO, C6H6, NOx, and NO2 levels at multiple forecast horizons (1h, 6h, 12h, 24h ahead) using both regression and classification approaches.

**Key Features:**
- **Data Pipeline:** Automated preprocessing, feature engineering, and anomaly detection
- **Regression Models:** XGBoost, Random Forest, and SVM for concentration prediction
- **Classification Models:** XGBoost and Logistic Regression for pollution level categorization
- **Comprehensive EDA:** Time series analysis, correlation studies, and seasonal patterns

---

## Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or higher)
- **Jupyter Notebook** support (VS Code with Jupyter extension, or JupyterLab)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:jerryjlzhou/9417_Group_Project.git
cd 9417_Group_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Data Pipeline

Process the raw dataset and generate engineered features:

```bash
python src/data/data_pipeline.py
python src/data/feature_engineering.py
```

### 3. Run Models

**Exploratory Data Analysis:**
- Open and run `src/data/data_analysis.ipynb`

**Regression Models:**
- XGBoost: `src/regression/XGBoost/xgBoost.ipynb`
- Random Forest: `src/regression/Random_forest/random_forest.ipynb`
- SVM: `src/regression/SVM_model/SVM_Model.ipynb`

**Classification Models:**
- XGBoost: `python src/classification/XGBoost Classification/xgboost_classification.py`
- Logistic Regression: `python src/classification/Logistic regression/classification_logistic_regression.py`
- Baseline: `python src/classification/Classification Naive Baseline/baseline_classification.py`

---

## Dataset Reference
```
Vito, S. D. (2008). Air Quality [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
```



