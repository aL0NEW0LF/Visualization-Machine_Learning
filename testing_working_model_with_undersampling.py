import numpy as np
import pandas as pd
import graphviz
import pydot
from keras.models import load_model
from keras.utils import pad_sequences, plot_model
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_10col = load_model('working_model_with_undersampling_10col.h5')
model_20col = load_model('working_model_with_undersampling_20col.h5')

# dot_img_file = 'D:/PFE/Statistical_Indicators/model_1.png'
#
# plot_model(model_10col, to_file=dot_img_file, show_shapes=True, dpi=300, show_layer_activations=True, show_trainable=True)

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

        # # Add the label corresponding to the sheet name
        # labels.append(sheet_name['Failure_type'])

# Convert data and labels to numpy arrays
new_data = np.array(new_data)

# Pad the new data sequences to a fixed length
padded_new_data = pad_sequences(new_data, dtype='float32')

# Convert data and labels to numpy arrays
new_data1 = np.array(new_data1)

# Pad the new data sequences to a fixed length
padded_new_data1 = pad_sequences(new_data1, dtype='float32')

data_list = [padded_new_data, padded_new_data1]

for j in data_list:
    if j.shape[2] == 10:
        # Make predictions
        predicted_probabilities = model_10col.predict(j)

        print(predicted_probabilities)
        # Convert predicted probabilities to class labels
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        print(predicted_labels)
        predicted_labels = label_encoder.inverse_transform(predicted_labels)

        # Calculate accuracy
        accuracy_10col= np.mean(predicted_labels == true_labels_10col)

        # Calculate precision
        precision_10col = precision_score(true_labels_10col, predicted_labels, average=None)

        print(accuracy_10col)
        print(precision_10col)
        # Print the predictions
        for i, label in enumerate(predicted_labels):
            print(f"Prediction for sequence {i + 1}: {label}")
    elif j.shape[2] == 20:
        # Make predictions
        predicted_probabilities = model_20col.predict(j)

        print(predicted_probabilities)
        # Convert predicted probabilities to class labels
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        print(predicted_labels)
        predicted_labels = label_encoder.inverse_transform(predicted_labels)

        # Calculate accuracy
        accuracy_20col = np.mean(predicted_labels == true_labels_20col)

        # Calculate precision
        precision_20col = precision_score(true_labels_20col, predicted_labels, average=None)

        print(accuracy_20col)
        print(precision_20col)
        # Print the predictions
        for i, label in enumerate(predicted_labels):
            print(f"Prediction for sequence {i + 1}: {label}")


# # Make predictions
# predicted_probabilities = model_10col.predict(padded_new_data)
#
# print(predicted_probabilities)
# # Convert predicted probabilities to class labels
# predicted_labels = np.argmax(predicted_probabilities, axis=1)
#
# print(predicted_labels)
# predicted_labels = label_encoder.inverse_transform(predicted_labels)
#
# # Print the predictions
# for i, label in enumerate(predicted_labels):
#     print(f"Prediction for sequence {i + 1}: {label}")
# # # Decode the predicted labels
# # predicted_labels = label_encoder.inverse_transform(predicted_labels)
# #
# # # Print the predictions
# # for i, label in enumerate(predicted_labels):
# #     print(f"Prediction for sequence {i + 1}: {label}")
#
# # Make predictions
# predicted_probabilities1 = model_20col.predict(padded_new_data1)
#
# print(predicted_probabilities1)
# # Convert predicted probabilities to class labels
# predicted_labels1 = np.argmax(predicted_probabilities1, axis=1)
#
# print(predicted_labels1)
# predicted_labels1 = label_encoder.inverse_transform(predicted_labels1)
#
# # Print the predictions
# for i, label in enumerate(predicted_labels1):
#     print(f"Prediction for sequence {i + 1}: {label}")