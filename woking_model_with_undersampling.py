import itertools
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.optimizers import Adam

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3.xlsx', sheet_name=None)

# Load and preprocess the data
SansDefaut_data_20col = []
data_20col = []
labels_20col = []
SansDefaut_data_10col = []
data_10col = []
labels_10col = []

# # Iterate over each Excel file
# for file_path in file_paths:
#     # Load the Excel file
#     excel_data = pd.read_excel(file_path, sheet_name=None)
#
#     # Iterate over each sheet in the Excel file
#     for sheet_name, sheet_data in excel_data.items():
#         # Convert the sheet data to a numpy array
#         sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()
#
#         # Add the sequence to the data list
#         data.append(sequence)
#
#         # # Add the label corresponding to the sheet name
#         # labels.append(sheet_name['Failure_type'])

sequence = file1['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data_20col.append(sequence)
sequence = file1['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data_20col.append(sequence)

sequence = file1['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_20col.append(sequence)
sequence = file1['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_20col.append(sequence)

sequence = file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data_10col.append(sequence)
# data_10col.append(sequence)
sequence = file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data_10col.append(sequence)

sequence = file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
sequence = file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
sequence = file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
sequence = file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
sequence = file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
sequence = file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
SansDefaut_data_10col.append(sequence)
# data_10col.append(sequence)

labels_20col = ['Sans defaut',
                'Defaut bague intern',
                'Defaut billes']

labels_10col = ['Defaut bague externe',
                'Defaut bague externe',
                'Sans defaut',
                'Sans defaut']

# Convert data and labels to numpy arrays
# data_10col = np.array(data_10col)
labels_10col = np.array(labels_10col)
SansDefaut_data_10col = np.array(SansDefaut_data_10col)

# Use the label_encoder to encode the labels we have
encoded_labels_10col = label_encoder.transform(labels_10col)

# Convert data and labels to numpy arrays
# data_20col = np.array(data_20col)
labels_20col = np.array(labels_20col)
SansDefaut_data_20col = np.array(SansDefaut_data_20col)

# Use the label_encoder to encode the labels we have
encoded_labels_20col = label_encoder.transform(labels_20col)

# Select a subset of 'Sans defaut' data randomly for each combination
best_accuracy_10col = 0.0
best_model_10col = None

# Generate all possible combinations of 'Sans defaut' indices
combinations_10col = list(itertools.combinations(SansDefaut_data_10col, 2))

for combination_10col in combinations_10col:
    selected_data_10col = data_10col.copy()
    for i in combination_10col:
        selected_data_10col.append(i)
    selected_data_10col = np.array(selected_data_10col)

    # Pad sequences to a fixed length
    padded_data_10col = pad_sequences(selected_data_10col, dtype='float32')

    # Build the model
    model_10col = Sequential()
    model_10col.add(LSTM(64, input_shape=(None, 10), return_sequences=True))
    model_10col.add(SimpleRNN(64))
    model_10col.add(Dense(4, activation='softmax'))

    # Compile the model_10col
    model_10col.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model_10col
    model_10col.fit(padded_data_10col, encoded_labels_10col, batch_size=32, epochs=10)

    # Evaluate the model_10col
    _, accuracy_10col = model_10col.evaluate(padded_data_10col, encoded_labels_10col)

    if accuracy_10col > best_accuracy_10col:
        best_accuracy_10col = accuracy_10col
        best_model_10col = model_10col

# Save the best model
best_model_10col.save('best_model_10col.h5')


# Select a subset of 'Sans defaut' data randomly for each combination
best_accuracy_20col = 0.0
best_model_20col = None

# Generate all possible combinations of 'Sans defaut' indices
combinations_20col = list(itertools.combinations(SansDefaut_data_20col, 1))

for combination_20col in combinations_20col:
    selected_data_20col = data_20col.copy()
    for i in combination_20col:
        selected_data_20col.append(i)
    selected_data_20col = np.array(selected_data_20col)

    # Pad sequences to a fixed length
    padded_data_20col = pad_sequences(selected_data_20col, dtype='float32')

    # Build the model
    model_20col = Sequential()
    model_20col.add(LSTM(64, input_shape=(None, 20), return_sequences=True))
    model_20col.add(SimpleRNN(64))
    model_20col.add(Dense(4, activation='softmax'))

    # Compile the model_20col
    model_20col.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model_20col
    model_20col.fit(padded_data_20col, encoded_labels_20col, batch_size=32, epochs=10)

    # Evaluate the model_20col
    _, accuracy_20col = model_20col.evaluate(padded_data_20col, encoded_labels_20col)

    if accuracy_20col > best_accuracy_20col:
        best_accuracy_20col = accuracy_20col
        best_model_20col = model_20col

# Save the best model
best_model_20col.save('best_model_20col.h5')



# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Build the model
# model_10col = Sequential()
# model_10col.add(LSTM(64, input_shape=(None, 10), return_sequences=True))
# model_10col.add(SimpleRNN(64))
# model_10col.add(Dense(4, activation='softmax'))  # Assuming num_classes is the number of unique labels
#
# # Compile the model_10col
# model_10col.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model_10col
# model_10col.fit(padded_data_10col, encoded_labels_10col, batch_size=32, epochs=10)
#
# # Save the trained model_10col
# model_10col.save('working_model_with_undersampling_10col_2.h5')

# # Build the model
# model_20col = Sequential()
# # model_20col.add(LSTM(64, input_shape=(None, 20)))  # Assuming num_features is the number of columns in each sheet
# # model_20col.add(SimpleRNN(64, input_shape=(None, 20)))
# model_20col.add(LSTM(64, input_shape=(None, 20), return_sequences=True))
# model_20col.add(SimpleRNN(64))
# model_20col.add(Dense(4, activation='softmax'))  # Assuming num_classes is the number of unique labels

# # Compile the model_20col
# model_20col.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model_20col
# model_20col.fit(padded_data_20col, encoded_labels_20col, batch_size=32, epochs=10)
#
# # Save the trained model_20col
# model_20col.save('working_model_with_undersampling_20col_2.h5')