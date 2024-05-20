import tkinter as tk
import os
from tkinter import *
from tkinter import filedialog as fdcd, filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from matplotlib import pyplot as plt
from tkinter.ttk import Progressbar
from prettytable import PrettyTable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import time
import tensorflow_model_optimization as tfmot

# Global variables
# Datasets definition
dataset = None
x, y = None, None
ex, ey = None, None
Results = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
model = None
unique_labels_emg1 = None
unique_labels_emg2 = None
unique_labels_emg3 = None
TrainAcc = None
TestAcc = None
Trainloss = None
Testloss = None
cmodel = None


# Function Definitions
def Importdata():
    global dataset
    filename = fdcd.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    dataset = pd.read_csv(filename)

    # Get unique labels and their counts for each EMG sensor
    global unique_labels_emg1
    global unique_labels_emg2
    global unique_labels_emg3
    unique_labels_emg1 = dataset.loc[dataset['EMG'] == 'EMG1', 'label'].value_counts()
    unique_labels_emg2 = dataset.loc[dataset['EMG'] == 'EMG2', 'label'].value_counts()
    unique_labels_emg3 = dataset.loc[dataset['EMG'] == 'EMG3', 'label'].value_counts()

    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = ["", "EMG1", "EMG2", "EMG3"]

    # Get unique labels from all EMG sensors
    labels = set(unique_labels_emg1.index) | set(unique_labels_emg2.index) | set(unique_labels_emg3.index)

    # Add rows to the table
    for label in labels:
        row = [label, unique_labels_emg1.get(label, 0), unique_labels_emg2.get(label, 0),
               unique_labels_emg3.get(label, 0)]
        table.add_row(row)

    # Display the label information as a table
    label_info = str(table)

    label_info_label.config(text=label_info)


def create_model():
    model = Sequential()
    model.add(tfmot.quantization.keras.quantize_annotate_layer(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x.shape[1], 1))))
    model.add(tfmot.quantization.keras.quantize_annotate_layer(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def get_emg_data(sensor_name):
    global x, y
    dataset_features_ch1 = dataset[dataset['EMG'] == sensor_name]
    dataset_labels = dataset_features_ch1['label']
    dataset_features = dataset_features_ch1.drop(columns=['label', 'EMG'], axis=1)
    encoder = LabelEncoder()
    encoder.fit(dataset_labels)
    encoded_y = encoder.transform(dataset_labels)
    y = np_utils.to_categorical(encoded_y)
    x = np.array(dataset_features[:])
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = np.asarray(x).astype(np.float32)

    # displaying that sensors stats only
    # Create a PrettyTable object
    table = PrettyTable()
    table.field_names = ["", sensor_name]
    labels = set(unique_labels_emg1.index)
    sensor_stats = None
    if sensor_name == "EMG1":
        sensor_stats = unique_labels_emg1
    elif sensor_name == "EMG2":
        sensor_stats = unique_labels_emg2
    elif sensor_name == "EMG3":
        sensor_stats = unique_labels_emg3

    for label in labels:
        row = [label, sensor_stats.get(label, 0)]
        table.add_row(row)

    label_selected_sensor = str(table)
    sensor_info_label.config(text=label_selected_sensor)


def get_multi_emg_data(sensor1, sensor2):
    global x, y
    dataset_features_multi = dataset[(dataset['EMG'] == sensor1) | (dataset['EMG'] == sensor2)]
    dataset_labels = dataset_features_multi['label']
    dataset_features = dataset_features_multi.drop(columns=['label', 'EMG'], axis=1)
    encoder = LabelEncoder()
    encoder.fit(dataset_labels)
    encoded_y = encoder.transform(dataset_labels)
    y = np_utils.to_categorical(encoded_y)
    x = np.array(dataset_features[:])
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = np.asarray(x).astype(np.float32)

    table = PrettyTable()
    table.field_names = ["", sensor1, sensor2]
    labels = set(unique_labels_emg1.index) | set(unique_labels_emg2.index)
    sensor1_stats = unique_labels_emg1
    sensor2_stats = unique_labels_emg2
    for label in labels:
        row = [label, sensor1_stats.get(label, 0), sensor2_stats.get(label, 0)]
        table.add_row(row)

    label_selected_sensor = str(table)
    sensor_info_label.config(text=label_selected_sensor)


def on_select_option(*args):
    selected_option = click_action.get()
    if selected_option == "Only EMG1":
        get_emg_data('EMG1')
    elif selected_option == "Only EMG2":
        get_emg_data('EMG2')
    elif selected_option == "Only EMG3":
        get_emg_data('EMG3')
    elif selected_option == "Only EMG1 and EMG2":
        get_multi_emg_data('EMG1', 'EMG2')
    else:
        pass


def display_empty_progress_bar():
    progress_bar = Progressbar(root, length=400, mode='determinate')
    progress_bar.place(x=290, y=260)
    progress_bar['value'] = 0
    progress_bar.update()
    return progress_bar


def train_data():
    global Results
    global x, y
    global TrainAcc, TestAcc, Trainloss, Testloss
    kfold = KFold(n_splits=3, shuffle=True)
    model = Sequential()
    model.add(tfmot.quantization.keras.quantize_annotate_layer(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x.shape[1], 1))))
    model.add(tfmot.quantization.keras.quantize_annotate_layer(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Quantization-aware model conversion
    # Create a quantization-aware training configuration
    quantize_config = tfmot.quantization.keras.QuantizeConfig()
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model, quantize_config)

    # Convert the annotated model to a quantization-aware model 
    quantize_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    quantize_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Enable quantization-aware training
    quantize_model = tfmot.quantization.keras.quantize_model(quantize_model)

    total_folds = kfold.get_n_splits()
    total_epochs = 100
    total_iterations = total_epochs * total_folds
    iteration = 0

    # Display empty progress bar
    progress_bar = display_empty_progress_bar()

    # Training starts
    fold_no = 1
    for train, test in kfold.split(x, y):
        print('------------------------------------------------------------------------')
        print(f"Training for fold {fold_no} ...")
        history = quantize_model.fit(x[train], y[train], epochs=100, batch_size=100, verbose=1,
                                     validation_data=(x[test], y[test]))
        fold_iterations = total_epochs

        # Update progress bar for each epoch
        for epoch in range(fold_iterations):
            print(f"Epoch {epoch + 1}/{fold_iterations}")
            iteration += 1
            progress_value = int(iteration * 100 / total_iterations)
            progress_bar['value'] = progress_value
            progress_bar.update()

        Results['accuracy'].append(history.history['accuracy'])
        Results['loss'].append(history.history['loss'])
        Results['val_accuracy'].append(history.history['val_accuracy'])
        Results['val_loss'].append(history.history['val_loss'])

        # Increase fold number
        fold_no += 1

    progress_bar.destroy()
    quantize_model.save('Quantized-aware-model_GUI.h5')
    messagebox.showinfo("Training Complete", "The data has been trained successfully!")

    A = Results['accuracy']
    B = Results['val_accuracy']
    C = Results['loss']
    D = Results['val_loss']

    TrainAcc = np.concatenate((A[0], A[1], A[2]), axis=0)
    TestAcc = np.concatenate((B[0], B[1], B[2]), axis=0)
    Trainloss = np.concatenate((C[0], C[1], C[2]), axis=0)
    Testloss = np.concatenate((D[0], D[1], D[2]), axis=0)
    plot_training_graph()


def plot_training_graph():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(3)
    fig.set_figwidth(6)
    ax1.plot(TrainAcc, 'b')
    ax1.plot(TestAcc, 'r-')
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_title('Model Accuracy')
    ax1.set(xlabel='Epochs', ylabel='Accuracy')
    ax1.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
    ax2.plot(Trainloss, 'b')
    ax2.plot(Testloss, 'r-')
    ax2.set_title('Model Loss')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(['Train Loss', 'Test Loss'], loc='upper right')

    for widget in frm3.winfo_children():
        widget.destroy()

    # Embedding the matplotlib figure in the tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=frm3)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Optional: Add a toolbar for the plot
    toolbar = NavigationToolbar2Tk(canvas, frm3)
    toolbar.update()
    canvas.get_tk_widget().pack()

    # Optional: Add a button to save the plot
    save_button = Button(frm3, text="Save Plot", command=save_plot)
    save_button.pack()


def save_plot():
    # Save the plot as an image file
    filetypes = [('PNG Image', '*.png'), ('JPEG Image', '*.jpg'), ('All Files', '*.*')]
    filename = fdcd.asksaveasfilename(filetypes=filetypes, defaultextension='.png')
    if filename:
        plt.savefig(filename)
        messagebox.showinfo("Save Plot", "Plot saved successfully!")


def import_model():
    global model
    filename = fdcd.askopenfilename(filetypes=[("H5 Files", "*.h5")])
    model = filename
    file_size = os.path.getsize(filename)
    size_in_mb = file_size / (1024 * 1024)  # Convert to MB
    messagebox.showinfo("Loaded Model", "Model has been loaded successfully!")
    load_model_label.config(text=f"File Name: {os.path.basename(filename)} \n File size: {size_in_mb:.2f} mb")


def evaluate_model():
    global model
    if model is not None:
        start_time = time.time()
        loaded_model = keras.models.load_model(model)
        loss, acc = loaded_model.evaluate(x, y, verbose=2)
        end_time = time.time()
        execution_time = end_time - start_time
        evaluate_model_label.config(
            text=f"Model Evaluation:\nModel Accuracy: {acc * 100:.2f}% \n Execution Time: {execution_time} sec")
    else:
        messagebox.showwarning("No Model", "Please import a model first!")


# Convert to tflite
def load_cmodel():
    global cmodel
    filename = filedialog.askopenfilename(filetypes=[("H5 Files", "*.h5")])
    loaded_model = tf.keras.models.load_model(filename)

    # Convert the model to TensorFlow Lite format with post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the quantized model to a file
    tflite_filename = "quantized_model.tflite"
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    file_size = os.path.getsize(tflite_filename)
    size_in_mb = file_size / (1024 * 1024)
    load_cmodel_label.config(text=f"Converted h5 to tflite model \n "
                                  f"File Name: {tflite_filename} \n"
                                  f"File Size: {size_in_mb:.2f}mb")

    cmodel = tflite_filename


# Evaluating tflite model
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=cmodel)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    input_shape = input_details[0]['shape']
    input_data = np.array(x[:input_shape[0]]).astype(np.float32)

    # Run inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    execution_time = end_time - start_time

    # Evaluate accuracy
    predicted_labels = np.argmax(output_data, axis=1)
    # true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predicted_labels == y) * 100

    load_lite_model_label.config(
        text=f"Model Evaluation:\nModel Accuracy: {accuracy:.2f}% \n Execution Time: {execution_time} sec"
    )


# GUI Setup
root = tk.Tk()
root.geometry("1500x700")
root.resizable(False, False)
root.title("Train EMG Data")

click_action = StringVar()
click_action.set("Choose EMG")
click_action.trace("w", on_select_option)

Action_menu = OptionMenu(root, click_action, "Only EMG1", "Only EMG2", "Only EMG3", "Only EMG1 and EMG2")
Action_menu.place(x=30, y=300)
Action_menu.config(width=22, height=2)

frm1 = Frame(root, bg='black', width=2, height=650)
frm1.place(x=800, y=20)

frm2 = Frame(root, bg='black', width=2, height=650)
frm2.place(x=280, y=20)

frm3 = Frame(root, bg='grey', width=500, height=570)
frm3.place(x=290, y=100)

Import_button = Button(root, text="Import Data", height=2, width=12, command=Importdata)
Import_button.place(x=30, y=20)

# Label Information Section
label_info_frame = Frame(root, bg='white', width=200, height=100)
label_info_frame.place(x=30, y=80)

label_info_label = Label(label_info_frame, text="Shape info", font=("Arial", 9, "bold"))
label_info_label.pack()

# sensor label information section
sensor_info_frame = Frame(root, bg='white', width=200, height=100)
sensor_info_frame.place(x=30, y=350)

sensor_info_label = Label(sensor_info_frame, text="Shape Info", font=("Arial", 9, "bold"))
sensor_info_label.pack()

# load label
load_model_label = Label(root, text="Model: ", wraplength=150, font=("Arial", 9, "bold"))
load_model_label.place(x=820, y=90)

# evaluate label
evaluate_model_label = Label(root, text="Model Evaluation: ", wraplength=200, font=("Arial", 10, "bold"))
evaluate_model_label.place(x=1100, y=90)

# cmodel label to display info
load_cmodel_label = Label(root, text="Converted Model: \n", wraplength=200, font=("Arial", 9, "bold"))
load_cmodel_label.place(x=820, y=250)

# tflite model label to display evaluation
load_lite_model_label = Label(root, text="Evaluated Model", wraplength=150, font=("Arial", 9, "bold"))
load_lite_model_label.place(x=1100, y=250)

#  **********************************************************************************

Import_button = Button(root, text="Import Data", height=2, width=12, command=Importdata)
Import_button.place(x=30, y=20)

Train_button = Button(root, text="Train Data", height=2, width=70, command=train_data)
Train_button.place(x=290, y=20)

Load_button = Button(root, text="Load Model", height=2, width=12, command=import_model)
Load_button.place(x=810, y=20)

Load_cmodel_button = Button(root, text="Import model [tflite conv]", height=2, width=12, command=load_cmodel,
                            wraplength=100)
Load_cmodel_button.place(x=820, y=200)

# load tflite model
Load_tflite_button = Button(root, text="Load TFLite Model", height=2, width=12, command=load_tflite_model,
                            wraplength=80)
Load_tflite_button.place(x=1100, y=200)

Eval_button = Button(root, text="Evaluate Model", height=2, width=12, command=evaluate_model)
Eval_button.place(x=1100, y=20)

Close_button = Button(root, text="Close", height=4, width=18, command=root.destroy)
Close_button.place(x=1300, y=500)

root.mainloop()
