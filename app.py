from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="lstm_tomato_disease_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load training dataset for encoder and scaler
df_train = pd.read_csv("trainlstm.csv")  # Ensure this CSV is in your project directory

# Encode the Disease column
label_encoder = LabelEncoder()
df_train["Disease"] = label_encoder.fit_transform(df_train["Disease"])
last_recorded_disease = label_encoder.inverse_transform([df_train["Disease"].iloc[-1]])[0]

# Fit the scaler
scaler = MinMaxScaler()
df_train = df_train.drop(columns=["Day"], errors='ignore')  # Drop Day if present
scaler.fit(df_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expect JSON input with 'temperature', 'humidity', 'rainfall', 'soil_moisture', 'disease_name'

        # Extract today's inputs
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        rainfall = float(data['rainfall'])
        soil_moisture = float(data['soil_moisture'])
        disease_name = data['disease_name']

        if disease_name not in label_encoder.classes_:
            return jsonify({'error': f'Unknown disease name: {disease_name}'}), 400

        encoded_disease = label_encoder.transform([disease_name])[0]

        # Prepare today's row
        today_df = pd.DataFrame([[temperature, humidity, rainfall, soil_moisture, encoded_disease, 0]],
                                columns=["Temperature", "Humidity", "Rainfall", "Soil Moisture", "Disease", "Disease Severity (%)"])

        # Get last 9 records from training set
        recent_data = df_train.tail(9).copy()

        # Combine recent data and today's input
        df_seq = pd.concat([recent_data, today_df], ignore_index=True)

        # Normalize the full sequence
        df_seq_scaled = scaler.transform(df_seq)

        # Drop the target column (last column)
        df_seq_scaled = df_seq_scaled[:, :-1]

        # Reshape to (1, 10, 5)
        input_tensor = np.reshape(df_seq_scaled, (1, 10, 5)).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Reverse scale
        min_sev = df_train["Disease Severity (%)"].min()
        max_sev = df_train["Disease Severity (%)"].max()
        predicted_severity = prediction * (max_sev - min_sev) + min_sev
        predicted_severity = predicted_severity[0][0]

        # Adjust if disease changed
        if disease_name != last_recorded_disease:
            predicted_severity += random.uniform(-5, 10)

        # Clamp to [0, 100]
        predicted_severity = max(0, min(100, predicted_severity))

        return jsonify({'severity_prediction': round(predicted_severity, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✅ Flask API is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
