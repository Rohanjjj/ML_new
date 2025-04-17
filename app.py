from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # Get input data from the request (Assuming JSON format)
    data = request.get_json()

    # Extract features
    flex1 = data.get('flex1')
    flex2 = data.get('flex2')
    flex3 = data.get('flex3')
    flex4 = data.get('flex4')
    accel_x = data.get('accel_x')
    accel_y = data.get('accel_y')
    accel_z = data.get('accel_z')
    gyro_x = data.get('gyro_x')
    gyro_y = data.get('gyro_y')
    gyro_z = data.get('gyro_z')

    # Prepare the feature array (same order as training data)
    features = np.array([[flex1, flex2, flex3, flex4, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]])

    # Predict the label
    prediction = model.predict(features)

    # Return the prediction as a response
    return jsonify({'gesture': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
