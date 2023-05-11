import os
import pandas as pd
import glob
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder

# # Set the folder path where the CSV files are located
# folder_path = "D:\PFE\\2nd_test\\2nd_test"
#
# # Get a list of all CSV files in the folder
# csv_files = [f for f in os.listdir(folder_path)]
#
# # Create an empty dictionary to store dataframes
# df_dict = {}
#
# # Loop through the CSV files and add them to the dictionary with their filename as key
# for file in csv_files:
#     df_dict[file] = pd.read_csv(os.path.join(folder_path, file))
#
# # Create a writer object to write to a single CSV file with multiple sheets
# writer = pd.ExcelWriter(os.path.join(folder_path, 'combined.xlsx'), engine='xlsxwriter')
#
# # Loop through the dictionary and write each dataframe to a separate sheet
# for key, value in df_dict.items():
#     value.to_excel(writer, sheet_name=key[:-4], index=False)
#
# # Save the workbook
# writer._save()


# folder_path = "D:/PFE/2nd_test/2nd_test"  # Replace with the actual folder path
#
# # Use glob to find all PDF files in the folder
# pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
#
# # Loop through the PDF files and delete each one
# for pdf_file in pdf_files:
#     os.remove(pdf_file)

# # Load the saved model
# model = load_model('model_test_4.h5')
#
# # Load the statistical indicators files
# file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing - Copy.xlsx', sheet_name='Bearing_1')
#
# # # Combine all sheets into one dataframe
# # data = pd.concat([file1['Bearing_1'], file1['Bearing_2'], file1['Bearing_3'], file1['Bearing_4']])
#
# X_test = file1.drop(['File Name', 'Failure_type'], axis=1)
#
# # Make predictions using the loaded model
# predictions = model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
# print(predictions)
#
# # Take the mean of the predicted values for each row in the sequence
# mean_predictions = predictions.mean(axis=0)
#
# # Use argmax to get the index of the highest predicted value
# predicted_index = round(mean_predictions.arg())
#
# print(predicted_index)

# # Decode the predicted labels to string values
# label_encoder = LabelEncoder()
# label_encoder.fit(['Sans défaut', 'Défaut bague interne', 'Défaut bague externe', 'Défaut billes'])
# predicted_labels = label_encoder.inverse_transform(predictions.ravel().round().astype(int))
#
# # Print the predicted failure types for each data point in the sequence
# for i, label in enumerate(predicted_labels):
#     print(f"Data point {i+1}: {label}")

# # Map the index to the corresponding label
# label_encoder = LabelEncoder()
# label_encoder.fit(['Sans défaut', 'Défaut bague interne', 'Défaut bague externe', 'Défaut billes'])
# predicted_label = label_encoder.inverse_transform([predicted_index])[0]
#
# # Print the predicted label for the entire sequence
# print(f"Predicted failure type for sequence: {predicted_label}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import load_model

# Load the new data for prediction
new_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing - Copy.xlsx', sheet_name='Bearing_1')

# Combine the data for each bearing into a single dataframe
# new_combined = pd.concat(new_data.values(), ignore_index=True)

# Convert the features to numpy array
X_new = new_data.drop(['File Name', 'Failure_type'], axis=1).values

# Normalize the input features using the same scaler used for training
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X_new)

# Reshape the input data for the RNN model
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

# Load the trained model
model = load_model('another_model_test.h5')

# Make predictions
predictions = model.predict(X_new)
predicted_labels = np.argmax(predictions, axis=1)

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod.npy', allow_pickle=True)

predicted_labels = label_encoder.inverse_transform(predicted_labels)

for i in predicted_labels:
    print(i)