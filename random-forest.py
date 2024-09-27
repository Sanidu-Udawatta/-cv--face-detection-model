import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('./dataset.csv')

# Features and target variable
X = data.drop(['ID', 'CVI_PHASE'], axis=1)  # Dropping 'ID' and target 'CVI_PHASE'
y = data['CVI_PHASE']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Check an example scenario
example_index = 0  # First row of the test set
example_input = X_test.iloc[example_index].to_dict()  # Get the first test example as a dictionary
example_prediction = model.predict([list(example_input.values())])[0]  # Get the prediction for the example
actual_output = y_test.iloc[example_index]  # Actual value

print(f'Example input: {example_input}')
print(f'Predicted CVI Phase: {example_prediction}')
print(f'Actual CVI Phase: {actual_output}')
