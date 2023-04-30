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

# Load the saved model
model = load_model('model_test.h5')

# Load the statistical indicators files
file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing - Copy.xlsx', sheet_name=None)

# Combine all sheets into one dataframe
data = pd.concat([file1['Bearing_1'], file1['Bearing_2'], file1['Bearing_3'], file1['Bearing_4']])

X_test = data.drop(['File Name', 'Failure_type'], axis=1)

X_test.info()

# Make predictions using the loaded model
predictions = model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))

# Decode the predicted labels to string values
label_encoder = LabelEncoder()
label_encoder.fit(['Sans défaut', 'Défaut bague interne', 'Défaut bague externe', 'Défaut billes'])
predicted_labels = label_encoder.inverse_transform(predictions.ravel().round().astype(int))

# Print the predicted failure types for each data point in the sequence
for i, label in enumerate(predicted_labels):
    print(f"Data point {i+1}: {label}")