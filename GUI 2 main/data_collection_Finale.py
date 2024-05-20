# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import time
import serial
import pandas as pd
import tensorflow as tf
import numpy as np
# Serial port configuration
serial_port = serial.Serial("COM3", 115200)  # Replace "COM3" with your Arduino's port

def collect_data():
    global predicted_label
    data = []
    s1=[]
    s2=[]
    s3=[]
    try:
        while len(s3) < 1000:  # Collect 1000 data points
            line = serial_port.readline().decode().strip()
            if line:
                values = line.split(',')
                if len(values) == 3:
                    sensor1 = int(values[0])
                    sensor2 = int(values[1])
                    sensor3 = int(values[2])
                    s1.append(sensor1)
                    s2.append(sensor2)
                    s3.append(sensor3)
    except KeyboardInterrupt:
        pass
    #l=input("enter label for data ")
    data=s1+s2+s3
    return data

while True:
    data = collect_data()
    # Save data to CSV
    df = pd.DataFrame(data)
    df=df.transpose()
    df.to_csv("emg_data.csv", mode='a', header=False, index=False)  # Append to the CSV file
        # Close the serial port

    serial_port.close()
    print("starting in 3..")
    time.sleep(1)
    print("Starting in 2..")
    time.sleep(1)
    print("Starting in 1..")
    time.sleep(1)
    # Reopen the serial port
    serial_port.open()