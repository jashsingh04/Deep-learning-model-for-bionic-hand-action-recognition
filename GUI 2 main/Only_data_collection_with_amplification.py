import time
import serial
import pandas as pd
import tensorflow as tf
import numpy as np
# Serial port configuration
serial_port = serial.Serial("COM3", 115200)  # Replace "COM3" with your Arduino's port
trial=0
def collect_data():
    global predicted_label
    data = []
    s1=[]
    s2=[]
    s3=[]
    c=0
    try:

        while len(s3) < 500:  # Collect 1000 data points
            line = serial_port.readline().decode().strip()
            if line and c>5:
                if c==6:
                    print("startiing in 1")
                    time.sleep(1)
                print(line)
                values = line.split(' ')
                if len(values) == 3:
                    sensor1 = (values[0])
                    sensor2 = (values[1])
                    sensor3 = (values[2])
                    s1.append(sensor1)
                    s2.append(sensor2)
                    s3.append(sensor3)
            c=c+1
    except KeyboardInterrupt:
        pass
    # l=input("enter label for data ")
    data=s1+s2+s3
    print(data)
    return data

while True:
    data = collect_data()
    # Save data to CSV
    df = pd.DataFrame(data)
    print(df)
    df=df.transpose()
    df.to_csv("emg_data.csv", mode='a', header=False, index=False)  # Append to the CSV file
        # Close the serial port
    serial_port.close()

    print("starting in 1")
    time.sleep(1)
    trial=trial+1
    print(trial)
    # Reopen the serial port
    serial_port.open()