# Car-Price-Prediction

# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts used car prices based on various features. Built as a learning exercise to explore data science, ML algorithms, and web app deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Academic%20Project-green.svg)

## 📚 Project Overview

This is an academic/learning project that demonstrates the complete machine learning workflow - from data preprocessing to model deployment. The goal is to predict the selling price of used cars using various machine learning algorithms and compare their performance.

### 🎓 Learning Objectives

- Data cleaning and preprocessing techniques
- Feature engineering for better model performance
- Implementing multiple ML algorithms
- Model evaluation and comparison
- Hyperparameter tuning
- Building an interactive web application
- Understanding model interpretability

### 🔍 Problem Statement

Given various features of a used car (brand, age, mileage, specifications, etc.), can we build a model that accurately predicts its market value? This helps sellers price their cars correctly and buyers verify if the asking price is fair.

## Dataset

**Source**: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho - car_details_v3.csv

**Size**: ~8,000 car listings

**Features**:
- `name`: Car brand and model
- `year`: Year of manufacture
- `selling_price`: Target variable (price in INR)
- `km_driven`: Total kilometers driven
- `fuel`: Fuel type (Petrol/Diesel/CNG/LPG)
- `seller_type`: Individual/Dealer/Trustmark Dealer
- `transmission`: Manual/Automatic
- `owner`: First/Second/Third/Fourth Owner
- `mileage`: Fuel efficiency (kmpl)
- `engine`: Engine capacity (CC)
- `max_power`: Maximum power (bhp)
- `torque`: Torque specifications
- `seats`: Number of seats

### Programming & Libraries
Python 3.8+
├── Data Analysis
│   ├── Pandas
│   ├── NumPy
│   └── Matplotlib/Seaborn
├── Machine Learning
│   ├── Scikit-learn
│   ├── XGBoost
│   └── Scipy
└── Web App
    ├── Streamlit
    ├── Plotly
    └── Joblib

## Methodology

### 1. Data Preprocessing

**Challenges faced:**
- Missing values in mileage, engine, power columns
- Inconsistent data formats (torque had mixed units)
- Outliers in price and mileage


**Fix:**
```python
# Handle missing values
df.dropna(subset=['critical_columns'], inplace=True)

# Remove outliers using Z-score
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

# Log transform target variable
df['selling_price'] = np.log1p(df['selling_price'])
```

### 2. Feature Engineering

Created new features to improve model performance:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `car_age` | 2020 - year | Age is more relevant than year |
| `km_per_year` | km_driven / (age + 1) | Usage intensity |
| `power_to_engine` | power / engine_cc | Engine efficiency |
| `is_luxury` | brand in luxury_list | Premium brand flag |
| `power_age_interaction` | power × age | Combined effect |

### 3. Model Training
**Models Implemented:**

1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 regularization)
3. **Lasso Regression** (L1 regularization, feature selection)
4. **Random Forest** (Ensemble of decision trees)
5. **Gradient Boosting** (Sequential ensemble)
6. **XGBoost** (Optimized gradient boosting)
7. **Ensemble** (Voting regressor combining top models)

### 4. Hyperparameter Tuning

Used GridSearchCV with 5-fold cross-validation:
```python
# Example for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_cv = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
```

### 5. Model Evaluation

**Metrics Used:**
- **R² Score**: Measures proportion of variance explained
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold CV to check consistency

## 📈 Results

### Model Performance Comparison

| Model | Train R² | Test R² | RMSE | MAE | Training Time |
|-------|----------|---------|------|-----|---------------|
| Linear Regression | 0.9127 | 0.9052 | 0.0647 | 0.0491 | 0.1s |
| Ridge Regression | 0.9125 | 0.9056 | 0.0644 | 0.0489 | 0.1s |
| Lasso Regression | 0.8377 | 0.8392 | 0.1098 | 0.0842 | 0.2s |
| Random Forest | 0.9523 | 0.9312 | 0.0521 | 0.0382 | 12.3s |
| Gradient Boosting | 0.9445 | 0.9289 | 0.0548 | 0.0401 | 45.2s |
| XGBoost | 0.9612 | 0.9401 | 0.0487 | 0.0356 | 8.7s |
| **Ensemble (Best)** | **0.9598** | **0.9423** | **0.0465** | **0.0341** | **15.4s** |

### Key Findings

**Best Model**: Ensemble (XGBoost + Random Forest + Gradient Boosting)
- Achieved **94.23% R² score** on test data
- Average prediction error: **±5%**
- Stable performance across cross-validation folds

**Feature Importance** (Top 5):
1. Car Age (28%)
2. Max Power (22%)
3. Kilometers per Year (18%)
4. Brand (Premium brands) (15%)
5. Engine CC (10%)

**Insights**:
- Tree-based models significantly outperform linear models
- Feature engineering improved R² by ~3%
- Ensemble method provides best generalization
- Age and power are strongest predictors

## Web Application

Built an interactive Streamlit app for easy predictions:

**Features:**
- 📝 User-friendly form for car details
- 💰 Instant price prediction with range
- 📊 Visual analytics and comparisons
- 💡 Expert tips for maximizing sale price
- 🎨 Modern, responsive UI design

## Lessons Learned

### What Went Well
- Feature engineering significantly improved performance
- Ensemble methods provided best results
- Cross-validation prevented overfitting
- Streamlit made deployment easy and interactive

### Challenges Faced
- Handling inconsistent data formats (torque specifications)
- Dealing with missing values appropriately
- Balancing model complexity vs. training time
- Ensuring UI works across different screen sizes

### Future Improvements
- [ ] Add more data for better generalization
- [ ] Include car images for visual inspection
- [ ] Implement time-series analysis for price trends
- [ ] Add location-based pricing adjustments
- [ ] Deploy on cloud (Heroku/AWS/GCP)
- [ ] Add A/B testing for different feature combinations
- [ ] Create API endpoint for external integration

