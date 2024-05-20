import serial

# Adjust the serial port and baud rate based on your configuration
ser = serial.Serial('COM4', 9600)

# Function to parse serial data
def parse_serial_data(serial_data):
    data_list = serial_data.strip().split(',')
    return [int(data) for data in data_list]

try:
    while True:
        # Read the data from the serial port
        serial_data = ser.readline().decode().strip()

        # Parse and display the data
        emg_data = parse_serial_data(serial_data)
        print("Sensor 1:", emg_data[0])
        print("Sensor 2:", emg_data[1])
        print("Sensor 3:", emg_data[2])
        print("-----------------------")

except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
