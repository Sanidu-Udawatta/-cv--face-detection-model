import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('trained_model.pkl')

# Load the scaler used for preprocessing
loaded_scaler = joblib.load('scaler.pkl')

# Example new data
new_data = pd.DataFrame({
    'COLOR_PREFERENCE': [0.5, 0.75],
    'ATTENTION_TO_LIGHT': [0.25, 1.0],
    'ATTENTION_TO_MOVEMENT': [0.5, 1.0],
    'VISUAL_LATENCY': [0.75, 0.5],
    'PREFERRED_VISUAL_FIELD': [0.5, 0.75],
    'VISUAL_COMPLEXITY': [0.25, 0.75],
    'DIFFICULTY_IN_DISTANCE_VIEWING': [0.5, 0.75],
    'ATYPICAL_VISUAL_REFLEXES': [0.5, 1],
    'DIFFICULTY_IN_VISUAL_NOVELTY': [0.25, 1],
    'ABSENCE_OF_VISUAL_GUIDED_REACH': [0.5, 1]
})

# Scale the new data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)

# Predict CVI phase using the loaded model
predicted_cvi_phase = loaded_model.predict(new_data_scaled)
print("Predicted CVI Phases for new data:", predicted_cvi_phase)
