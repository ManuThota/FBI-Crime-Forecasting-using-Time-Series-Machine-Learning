# FBI Crime Forecasting using Time Series & Machine Learning

A production-level Machine Learning project that predicts monthly crime incidents using historical FBI crime data.  
This project transforms raw crime data into actionable insights using time series analysis and advanced machine learning techniques.

---

## Project Overview

This project focuses on building a robust crime forecasting system to:

- Analyze historical crime patterns
- Predict future monthly crime incidents
- Enable proactive decision-making for public safety

The system leverages both:
- **Time Series Models** (ARIMA, SARIMA)
- **Machine Learning Model** (XGBoost - Final Model)

---

## Problem Statement

Urban crime rates are increasing, and traditional reactive approaches limit the ability of law enforcement agencies to act proactively.

This project addresses the problem by:

- Analyzing historical crime data (time + location + type)
- Capturing temporal and spatial patterns
- Forecasting future crime counts on a monthly basis

---

## Tech Stack

- **Python**
- **Pandas, NumPy** → Data Processing  
- **Matplotlib, Seaborn** → Visualization  
- **Scikit-learn** → ML Utilities  
- **Statsmodels** → Time Series Models  
- **XGBoost** → Final Prediction Model  
- **Joblib** → Model Saving  

---

## Project Structure
```

FBI-Crime-Forecasting-using-Time-Series-Machine-Learning/
│
├──data/
│  ├──raw/
│  └──processed/
|
├──notebooks/
|  ├──FBI_Time_Series_Project.ipynb
|  └──experiments.ipynb
|
├──reports/
|  ├──figures/
|  └──results.txt
|
├──src/
│  ├──config/
|  |  └──config.py
|  |
│  ├──data_ingestion/
|  |  ├──load_data.py
|  |  └──save_data.py
|  |   
│  ├──data_cleaning/
|  |  └──cleaning.py
|  |
│  ├──eda/
|  |  └──eda.py
|  |
│  ├──explainability/
|  |  └──explain.py
|  |
│  ├──feature_engineering/
|  |  └──features.py
|  |
│  ├──model_building/
|  |  ├──evaluate.py
|  |  └──train_model.py
|  |  
│  ├──pipeline/
|  |  └──train_pipeline.py
|  |
│  ├──preprocessing/
|  |  └──preprocess.py
|  |
│  └──__init__.py
|
├──models/
|  └──xgboost_model.pkl
|
├──main.py
|
├──requirements.txt
|
├──.gitignore
|
└──README.md
```

---

## Key Features

- Data Cleaning (missing values, duplicates, date reconstruction)
- Feature Engineering (aggregation + encoding)
- Time Series Processing (ADF test, differencing)
- Model Training (XGBoost, ARIMA, SARIMA)
- Model Evaluation (RMSE, MAE, R²)
- Feature Importance Analysis
- Automated Data & Report Saving
- End-to-End Pipeline Execution

---

## Model Performance Summary

| Model    | Performance |
|----------|------------|
| ARIMA    | Poor (R² < 0) |
| SARIMA   | Moderate |
| XGBoost  | Best (R² ≈ 0.87) |

**Final Model:** XGBoost Regressor

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/my-user_name/FBI-Crime-Forecasting.git
cd FBI-Crime-Forecasting
```
### 2. Create Virtual Environment
```
python -m venv venv
```
Activate it:
```
venv\Scripts\activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Add Dataset
```
data/raw/
├── Train.csv
└── Test.csv
```
### 5. Run the Pipeline
```
python main.py
```
---
### Output

After running the project:

### Processed Data
```
data/processed/
```
### Saved Model
```
models/xgboost_model.pkl
```
### Reports
```
reports/
├── figures/
└── results.txt
```
---
## EDA Visualizations

The project generates:

- Crime Trend Over Time
- Crime Type Distribution
- Top Crime Categories
- Monthly Crime Patterns

Saved in:

```
reports/figures/
```

---
## Key Learnings
- Importance of time-based data splitting
- Limitations of univariate time series models
- Power of feature engineering in ML
- Handling real-world messy datasets
- Building modular and scalable ML pipelines

---
## Future Improvements
- Deploy using FastAPI
- Build dashboard using Streamlit
- Add real-time data ingestion
- Use MLflow for experiment tracking

---
## Contributing

> Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to improve.