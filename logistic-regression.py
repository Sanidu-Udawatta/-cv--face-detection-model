import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
data = pd.read_csv('./dataset.csv')

# Features and target variable
X = data.drop(['ID', 'CVI_PHASE'], axis=1)  # Dropping 'ID' and target 'CVI_PHASE'
y = data['CVI_PHASE']

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% for training
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% validation, 20% testing

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Adjusting the param_grid for more balanced accuracy in the 94-96% range
param_grid_further = [
    {'C': [0.1, 0.5, 1, 2], 'solver': ['liblinear'], 'multi_class': ['ovr']},  # liblinear only supports ovr
    {'C': [0.1, 0.5, 1, 2], 'solver': ['lbfgs', 'newton-cg', 'saga'], 'multi_class': ['multinomial', 'ovr']}
]

# Initialize Logistic Regression
logistic_model_further = LogisticRegression(max_iter=1000)

# GridSearchCV to find the best parameters within the new param grid (using validation set)
grid_search_further = GridSearchCV(logistic_model_further, param_grid_further, cv=5, scoring='accuracy')
grid_search_further.fit(X_train_scaled, y_train)

# Best parameters from the further adjusted grid search
best_params_further = grid_search_further.best_params_

# Train the model with the best parameters from the further adjusted search
best_model_further = grid_search_further.best_estimator_
best_model_further.fit(X_train_scaled, y_train)

# Validate the model on the validation set
y_valid_pred = best_model_further.predict(X_valid_scaled)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f'Validation Accuracy: {valid_accuracy * 100:.2f}%')

# Make predictions on the test set
y_test_pred = best_model_further.predict(X_test_scaled)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Output best parameters and final accuracy
print(f'Best Parameters: {best_params_further}')

# Save the trained model to a file
joblib.dump(best_model_further, 'trained_model.pkl')

# Save the scaler used for data preprocessing
joblib.dump(scaler, 'scaler.pkl')