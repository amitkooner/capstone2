import pandas as pd
import numpy as np
random.seed(42)
np.random.seed(42)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import math

# Data loading
data = pd.read_excel('/Users/AKooner/Desktop/coding/capstone2/new_york_listings_2024.xls')

# Data preprocessing
# Drop columns that won't be used for modeling
data = data.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review', 'license'])

# Handle missing values
data['reviews_per_month'].fillna(0, inplace=True)  # Replace missing values with 0 for 'reviews_per_month'

# Convert 'No rating' to NaN and then handle NaN values in 'rating'
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')  # Convert 'No rating' to NaN
data['rating'].fillna(data['rating'].median(), inplace=True)  # Replace NaN with median or another statistic

# Convert 'Studio' to 0 in 'bedrooms', then convert the entire column to numeric
data['bedrooms'] = data['bedrooms'].replace('Studio', 0)
data['bedrooms'] = pd.to_numeric(data['bedrooms'])

# Handle 'Not specified' values (assuming it's in 'column_name')
data['baths'] = data['baths'].replace('Not specified', np.nan)  # Convert 'Not specified' to NaN
data['baths'].fillna(data['baths'].median(), inplace=True)  # Replace NaN with median or another statistic

# Identify categorical columns (Add or remove as per your dataset)
categorical_columns = ['neighbourhood_group', 'room_type', 'neighbourhood']  # Add other categorical columns here
# One-hot encoding for categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Log-transform the target variable 'price' and create 'price_log'
data['price_log'] = data['price'].apply(lambda x: 0 if x <= 0 else math.log1p(x))

# Define the models and their hyperparameter grids for tuning
models = [
    {
        'name': 'Linear Regression',
        'model': LinearRegression(),
        'param_grid': {}
    },
    {
        'name': 'Ridge Regression',
        'model': Ridge(),
        'param_grid': {'alpha': [0.1, 1.0, 10.0]}
    },
    {
        'name': 'Lasso Regression',
        'model': Lasso(),
        'param_grid': {'alpha': [0.1, 1.0, 10.0]}
    },
    {
        'name': 'Random Forest',
        'model': RandomForestRegressor(),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            # Add other hyperparameters to tune
        }
    },
    {
        'name': 'Gradient Boosting',
        'model': GradientBoostingRegressor(),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            # Add other hyperparameters to tune
        }
    },
    # Add more models to the list as needed
]

# Splitting the data into training and testing sets
X = data.drop(columns=['price', 'price_log'])
y = data['price_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search and model selection
best_model = None
best_rmse = float('inf')

for model_info in models:
    model_name = model_info['name']
    model = model_info['model']
    param_grid = model_info['param_grid']

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its performance
    best_estimator = grid_search.best_estimator_
    rmse = mean_squared_error(y_test, best_estimator.predict(X_test), squared=False)

    # Check if this model performed better than previous best
    if rmse < best_rmse:
        best_model = best_estimator
        best_rmse = rmse

# Save the best-performing model to a file
joblib.dump(best_model, 'best_model.joblib')  # or pickle.dump(best_model, open('best_model.pkl', 'wb'))

# Print the best model's performance
print(f'Best Model: {best_model}')
print(f'Best RMSE: {best_rmse}')