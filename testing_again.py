import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = load_model('another_model_test_2.h5')

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

# Load and preprocess the new data
new_data = []

sheet_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing.xlsx', sheet_name='Bearing_3')
# excel_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing.xlsx', sheet_name=None)
#
# # Iterate over each sheet in the Excel file
# for sheet_name, sheet_data in excel_data.items():
#     # Convert the sheet data to a numpy array
#     sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()
#
#     # Add the sequence to the new data list
#     new_data.append(sequence)

# Convert the sheet data to a numpy array
sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()

# Add the sequence to the new data list
new_data.append(sequence)

# Pad the new data sequences to a fixed length
padded_new_data = pad_sequences(new_data, dtype='float32')

# Make predictions
predicted_probabilities = model.predict(padded_new_data)

print(predicted_probabilities)
# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predicted_probabilities, axis=1)

print(predicted_labels)
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Print the predictions
for i, label in enumerate(predicted_labels):
    print(f"Prediction for sequence {i + 1}: {label}")
# # Decode the predicted labels
# predicted_labels = label_encoder.inverse_transform(predicted_labels)
#
# # Print the predictions
# for i, label in enumerate(predicted_labels):
#     print(f"Prediction for sequence {i + 1}: {label}")