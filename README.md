# Used Car Price Prediction - Random Forest Model

## Overview
Regression model to predict used car prices in India using Random Forest.
Includes interactive Gradio interface for easy predictions.

## Model Performance
- **R² Score: 0.9090** (91% variance explained)
- **MAE: 1.60** Lakhs INR
- **MAPE: 17.75%**

## Features Used
Mileage, Engine, Power, Year, Transmission Type, Fuel Type

## Interactive Interface (Gradio)

This project includes a user-friendly web interface built with **Gradio**.

### How to Run the Interface
```bash
pip install gradio joblib numpy scikit-learn
python app_random_forest.py
```

Then open the local URL in your browser to:
- Enter car specifications
- Get instant price predictions
- Beautiful, intuitive UI

### Features of the Interface
- Slider for year selection (1990-2024)
- Radio buttons for fuel type and transmission
- Automatic transmission selection for electric cars
- Real-time price predictions in INR
- Custom grayscale CSS styling

## Model Details

### Hyperparameter Optimization
Iterative tuning of:
- n_estimators: [50, 100, 200, 300, 400]
- max_depth: [None, 10, 20]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2']
