import serial
import numpy as np
from tensorflow.keras.models import load_model

port_name = "COM3"
baud_rate = 115200

data_format = "sensor1 sensor2 sensor3"
num_sensors = 3
data_points_per_sensor = 500

action_labels = ['Open', 'Close']

model_path = "path/to/model.h5"

# Threshold
threshold = 0.75
ser = serial.Serial(port_name, baud_rate)
data = []

# Load trained model
model = load_model(model_path)

while True:
    # Read data from serial port
    line = ser.readline().decode().strip()

    # Check if enough data is received
    if len(data) == data_points_per_sensor * num_sensors:

        features = np.array(data).reshape(1, data_points_per_sensor * num_sensors, 1)

        prediction = model.predict(features)

        # 1. Class with Highest Probability
        action_label_highest_prob = action_labels[np.argmax(prediction)]
        print(f"Predicted action (Highest Probability): {action_label_highest_prob}")

        # 2. Thresholding
        action_label_thresholded = None
        for i, prob in enumerate(prediction[0]):
            if prob >= threshold:
                action_label_thresholded = action_labels[i]
                break
        if action_label_thresholded:
            print(f"Predicted action (Threshold {threshold}): {action_label_thresholded}")
        else:
            print(f"Prediction uncertain (Probabilities below threshold)")

        # 3. Combining Probabilities (Example: Weighted Average)
        weighted_average_prob = np.dot(prediction[0], range(len(action_labels)))
        predicted_action_index = np.argmax(weighted_average_prob)
        action_label_weighted_avg = action_labels[predicted_action_index]
        print(f"Predicted action (Weighted Average): {action_label_weighted_avg}")
        print(f"  Individual probabilities: {prediction[0]}")

        # Reset data buffer
        data = []

    # Add sensor values to data buffer
    data.extend([float(value) for value in line.split()])
    ser.close()
