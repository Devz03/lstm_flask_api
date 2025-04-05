from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# 🔹 Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="lstm_tomato_disease_model.tflite")  # Make sure this filename matches your actual .tflite file
interpreter.allocate_tensors()

# 🔹 Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔹 Manual label map (same order as your training LabelEncoder)
DISEASE_LABELS = {
    "Tomato___Early_blight": 0,
    "Tomato___Late_blight": 1,
    "Tomato___Leaf_Mold": 2,
    "Tomato___Septoria_leaf_spot": 3,
    "Tomato___Spider_mites": 4,
    "Tomato___Target_Spot": 5,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 6,
    "Tomato___Tomato_mosaic_virus": 7,
    "Tomato___healthy": 8
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expect JSON input
        
        input_data = np.array(data['input'], dtype=object)  # Shape: [1, 10, 5], last value in each row is disease name string
        
        # Convert disease names to numbers
        for i in range(input_data.shape[1]):
            disease_name = input_data[0][i][-1]
            if disease_name not in DISEASE_LABELS:
                return jsonify({'error': f'Unknown disease name: {disease_name}'})
            input_data[0][i][-1] = DISEASE_LABELS[disease_name]

        # Convert everything to float32
        input_data = input_data.astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Return prediction as float
        severity = float(output[0][0])
        severity = max(0, min(100, severity))  # Clamp between 0 and 100

        return jsonify({'severity_prediction': round(severity, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("✅ Flask API is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
