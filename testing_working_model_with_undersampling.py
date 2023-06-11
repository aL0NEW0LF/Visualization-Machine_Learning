import numpy as np
import pandas as pd
import graphviz
import pydot
import pylab
from keras.models import load_model
from keras.utils import pad_sequences, plot_model, model_to_dot
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_10col = load_model('model_SimpleRNN_10col.h5')
model_20col = load_model('model_SimpleRNN_20col.h5')

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

file_paths1 = ['D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1.xlsx']

file_paths = ['D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2.xlsx', 'D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3.xlsx']

# Load and preprocess the new data
new_data = []
new_data1 = []
true_labels_10col = ['Defaut bague externe',
                     'Sans defaut',
                     'Sans defaut',
                     'Sans defaut',
                     'Sans defaut',
                     'Sans defaut',
                     'Defaut bague externe',
                     'Sans defaut']

true_labels_20col = ['Sans defaut',
                     'Sans defaut',
                     'Defaut bague intern',
                     'Defaut billes']

true_labels_10col = np.array(true_labels_10col)
true_labels_20col = np.array(true_labels_20col)

# Iterate over each Excel file
for file_path in file_paths1:
    # Load the Excel file
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # Iterate over each sheet in the Excel file
    for sheet_name, sheet_data in excel_data.items():
        # Convert the sheet data to a numpy array
        sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()

        # Add the sequence to the data list
        new_data1.append(sequence)

# Iterate over each Excel file
for file_path in file_paths:
    # Load the Excel file
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # Iterate over each sheet in the Excel file
    for sheet_name, sheet_data in excel_data.items():
        # Convert the sheet data to a numpy array
        sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()

        # Add the sequence to the data list
        new_data.append(sequence)

        # Add the label corresponding to the sheet name
        # labels.append(sheet_name['Failure_type'])

# Convert data and labels to numpy arrays
new_data = np.array(new_data)

# Pad the new data sequences to a fixed length
padded_new_data = pad_sequences(new_data, dtype='float32', maxlen=10000)

# Convert data and labels to numpy arrays
new_data1 = np.array(new_data1)

# Pad the new data sequences to a fixed length
padded_new_data1 = pad_sequences(new_data1, dtype='float32', maxlen=10000)

data_list = [padded_new_data, padded_new_data1]


for j in data_list:
    if j.shape[2] == 10:
        # Make predictions
        predicted_probabilities = model_10col.predict(j)
        print(f"model_10col:")

        print(f"predicted_probabilities:\n"
              f"{predicted_probabilities}")

        # Convert predicted probabilities to class labels
        predictions_10col = np.argmax(predicted_probabilities, axis=1)

        print(f"predictions_10col:"
              f"{predictions_10col}")
        predictions_10col = label_encoder.inverse_transform(predictions_10col)

        # Calculate accuracy
        accuracy_10col= np.mean(predictions_10col == true_labels_10col)

        # Calculate precision
        precision_10col = precision_score(true_labels_10col, predictions_10col, average=None)
        recall_10col = recall_score(true_labels_10col, predictions_10col, average=None)
        f1_10col = f1_score(true_labels_10col, predictions_10col, average=None)


        # Print the predictions
        for i, label in enumerate(predictions_10col):
            print(f"Prediction for sequence {i + 1}: {label}")
        print(f"Accuracy: {accuracy_10col}\nPrecision:{precision_10col}\nRappel: {recall_10col}\nF1-score: {f1_10col}")

    elif j.shape[2] == 20:
        # Make predictions
        predicted_probabilities = model_20col.predict(j)
        print(f"model_20col:")

        print(f"predicted_probabilities:\n"
              f"{predicted_probabilities}")
        # Convert predicted probabilities to class labels
        predictions_20col = np.argmax(predicted_probabilities, axis=1)
        print(f"predictions_20col:"
              f"{predictions_20col}")

        predictions_20col = label_encoder.inverse_transform(predictions_20col)

        # Calculate accuracy
        accuracy_20col = np.mean(predictions_20col == true_labels_20col)

        # Calculate precision
        precision_20col = precision_score(true_labels_20col, predictions_20col, average=None)
        recall_20col = recall_score(true_labels_20col, predictions_20col, average=None)
        f1_20col = f1_score(true_labels_20col, predictions_20col, average=None)

        # Print the predictions
        for i, label in enumerate(predictions_20col):
            print(f"Prediction for sequence {i + 1}: {label}")
        print(f"Accuracy: {accuracy_20col}\nPrecision:{precision_20col}\nRappel: {recall_20col}\nF1-score: {f1_20col}")

# Calculate overall performance metrics for the combined models
true_labels_combined = np.concatenate([true_labels_10col, true_labels_20col])

accuracy_combined = np.mean(np.concatenate([predictions_10col, predictions_20col]) == true_labels_combined)
precision_combined = precision_score(true_labels_combined, np.concatenate([predictions_10col, predictions_20col]), average=None)
recall_combined = recall_score(true_labels_combined, np.concatenate([predictions_10col, predictions_20col]), average=None)
f1_combined = f1_score(true_labels_combined, np.concatenate([predictions_10col, predictions_20col]), average=None)

print(f"Overall Accuracy (combined): {accuracy_combined}")
print(f"Overall Precision (combined): {precision_combined}")
print(f"Overall Recall (combined): {recall_combined}")
print(f"Overall F1-score (combined): {f1_combined}")