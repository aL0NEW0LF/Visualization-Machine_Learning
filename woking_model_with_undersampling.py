import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing.xlsx', sheet_name=None)

# Load and preprocess the data
data = []
labels = []

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

sequence = file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data.append(sequence)
# sequence = file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
# data.append(sequence)
sequence = file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data.append(sequence)
sequence = file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data.append(sequence)
sequence = file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).to_numpy()
data.append(sequence)

labels = ['Defaut bague externe',
          'Sans defaut',
          'Defaut bague externe',
          'Sans defaut']

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Pad sequences to a fixed length
padded_data = pad_sequences(data, dtype='float32')

# Use the label_encoder to encode the labels we have
encoded_labels = label_encoder.transform(labels)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(64, input_shape=(None, 10)))  # Assuming num_features is the number of columns in each sheet
model.add(Dense(4, activation='softmax'))  # Assuming num_classes is the number of unique labels

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_data, encoded_labels, batch_size=32, epochs=10)

# Save the trained model
model.save('working_model_with_undersampling.h5')