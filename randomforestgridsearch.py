import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar datos
data = pd.read_csv(r'data_cars.csv')

# Eliminar filas con valores NaN en la variable objetivo
data = data.dropna(subset=['Price'])

# Separar características y variable objetivo
features = [
    'Mileage', 'Engine', 'Power', 'Trans_Automatic', 'Trans_Manual', 'Year', 'Fuel_Diesel', 'Fuel_Electric', 'Fuel_Petrol',
    'Brand_HONDA', 'Brand_HYUNDAI', 'Brand_MARUTI', 'Brand_MERCEDES-BENZ', 'Brand_Otros', 'Brand_TOYOTA', 'Brand_VOLKSWAGEN'
]
X = data[features]
y = data['Price']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Definir el modelo base
rf = RandomForestRegressor(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200, 300, 340, 350, 360, 375, 400],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Configurar la búsqueda con validación cruzada
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  # Minimizar MAE
    cv=5,  # Validación cruzada con 5 folds
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=2
)

# Ajustar la búsqueda de hiperparámetros
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros encontrados
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Evaluar el mejor modelo en el conjunto de prueba
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Calcular métricas finales
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'R2 Score: {r2:.4f}')
