import RPi.GPIO as GPIO
import time
import os, sys
import serial
import time
import simplejson as json
import numpy as np
import tensorflow as tf
serial_port=serial.Serial('/dev/ttyUSB0',115200)

# Load the TFLite model for real-time classification
interpreter = tf.lite.Interpreter(model_path="jashemgAll.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = (1, 3000, 1)
output_shape = (1, 6)
interpreter.resize_tensor_input(input_details[0]['index'], input_shape)
interpreter.resize_tensor_input(output_details[0]['index'], output_shape)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the live function for real-time classification
def live(data):
    input_data = np.array([data], dtype=np.float32)
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label_index = np.argmax(predictions_data)
    actions = ["cylindrical", "hook", "lateral", "palmer", "spherical", "tip"]
    predicted_label = actions[predicted_label_index]
    return predicted_label

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(25, GPIO.OUT)
GPIO.setup(16, GPIO.IN)  # Set GPIO 16 to input

timeDelay = 3

# Define action functions
def Cyl_action():
    print('Cylindrical Action - 1 - 001')
    GPIO.output(23, 1)
    GPIO.output(24, 0)
    GPIO.output(25, 0)


def Sph_action():
    print('Spherical Action - 5  - 101')
    GPIO.output(23, 1)
    GPIO.output(24, 0)
    GPIO.output(25, 1)


def Palm_action():
    print('Palmer Action - 4 - 100')
    GPIO.output(23, 0)
    GPIO.output(24, 0)
    GPIO.output(25, 1)


def Lat_action():
    print('Lateral Action - 6 - 110')
    GPIO.output(23, 0)
    GPIO.output(24, 1)
    GPIO.output(25, 1)


def Hook_action():
    print('Hook Action - 3 011')
    GPIO.output(23, 1)
    GPIO.output(24, 1)
    GPIO.output(25, 0)


def Tip_action():
    print('Tip Action - 2 -010')
    GPIO.output(23, 0)
    GPIO.output(24, 1)
    GPIO.output(25, 0)

def Relax_action():
    print('Relax Action - 0 000')
    GPIO.output(23, 0)
    GPIO.output(24, 0)
    GPIO.output(25, 0)

def get_sensor_data(serial_port):
    try:
        line = serial_port.readline().decode().strip()
        values = line.split(',')
        if len(values) == 3:
            return [int(val) for val in values]
        else:
            return None
    except Exception as e:
        print("Error reading data:", e)
        return None

try:
    while True:
        startTime = time.time()

        Relax_action()  # Relax 000/111
        time.sleep(timeDelay)

        if GPIO.input(16) == 0:
            # Collect data and perform real-time classification
            sensor_data = get_sensor_data(serial_port)

            if sensor_data is not None:
                # Store data in respective lists
                data1_list.append(sensor_data[0])
                data2_list.append(sensor_data[1])
                data3_list.append(sensor_data[2])

                # Check if data collection is complete
                if len(data1_list) >= 1000 and len(data2_list) >= 1000 and len(data3_list) >= 1000:
                    collected_data = np.array([data1_list, data2_list, data3_list])
                    # Preprocessing steps similar to training data
                    collected_data = collected_data[:, :3000]  # Keep the relevant portion of the data
                    collected_data = np.asarray(collected_data).astype(np.float32)
                    collected_data = collected_data.reshape(1, collected_data.shape[0], collected_data.shape[1], 1)
                    print(collected_data)
                    live(collected_data)
                    print("Data collection complete.")
                    print("Data Collecting in 3 second")
                    time.sleep(3)
                    data1_list.clear()
                    data2_list.clear()
                    data3_list.clear()
                    print("")
                if predicted_label == "cylindrical":
                    Cyl_action()
                elif predicted_label == "spherical":
                    Sph_action()
                elif predicted_label == "palmer":
                    Palm_action()
                elif predicted_label == "lateral":
                    Lat_action()
                elif predicted_label == "tip":
                    Tip_action()
                elif predicted_label == "hook":
                    Hook_action()
                # Add more conditions for other actions

        endTime = time.time()
        elapTime = endTime - startTime
        print("Elapsed Time =", elapTime)
        print('Cycle Complete')

except KeyboardInterrupt:
    GPIO.cleanup()
