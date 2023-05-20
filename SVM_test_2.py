import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import pad_sequences
import pickle
from sklearn.preprocessing import LabelEncoder

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

excel_data = pd.read_excel('D:/PFE/Statistical_Indicators/Summary_Stats_Tests__.xlsx', sheet_name=None)
# Load and preprocess the data
data2 = []
data = []
labels = []

# Iterate over each sheet in the Excel file
for sheet_name, sheet_data in excel_data.items():
    # Convert the sheet data to a numpy array
    sequence = sheet_data.drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()

    # Add the sequence to the data list
    data.append(sequence)

sequence = excel_data['Test2_Bearing1'].drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()
data2.append(sequence)
sequence = excel_data['Test2_Bearing2'].drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()
data2.append(sequence)
# sequence = excel_data['Test3_Bearing1'].drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()
# data2.append(sequence)
sequence = excel_data['Test3_Bearing3'].drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()
data2.append(sequence)
sequence = excel_data['Test3_Bearing4'].drop(['File Name', 'Sheet Name', 'Column Name', 'Count'], axis=1).to_numpy()
data2.append(sequence)

labels = ['Defaut bague externe',
          'Sans defaut',
          'Defaut bague externe',
          'Sans defaut']

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)
data2 = np.array(data2)

# Use the label_encoder to encode the labels we have
encoded_labels = label_encoder.transform(labels)

# Reshape the data to have two dimensions
reshaped_data = data.reshape(data.shape[0], -1)

# Reshape the data to have two dimensions
reshaped_data2 = data2.reshape(data2.shape[0], -1)

# Train SVM model
model = svm.SVC()
model.fit(reshaped_data2, encoded_labels)

# Print the summary
print(model)

y_pred = model.predict(reshaped_data)

y_pred = label_encoder.inverse_transform(y_pred)

print(y_pred)