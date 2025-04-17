from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests

# Load the trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

# External server URL to send the prediction
external_server_url = "http://external-server-url.com/api"  # Replace with actual URL

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

    # Predict the label using the model
    prediction = model.predict(features)

    # Send only the predicted gesture text to the external server
    try:
        response = requests.post(external_server_url, json={'gesture': prediction[0]})
        
        if response.status_code == 200:
            return jsonify({'gesture': prediction[0]})
        else:
            # Log error in console and return message
            app.logger.error(f"Failed to send prediction. Status code: {response.status_code}")
            return jsonify({'gesture': prediction[0]}), 500

    except requests.exceptions.RequestException as e:
        # Log the exception in console
        app.logger.error(f"Error sending request to the external server: {str(e)}")
        return jsonify({'gesture': prediction[0]}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
