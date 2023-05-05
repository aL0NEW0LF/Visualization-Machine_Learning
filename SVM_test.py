import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import pad_sequences
import pickle

# Load the statistical indicators files
file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing - Copy.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing - Copy.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing - Copy.xlsx', sheet_name=None)

# # Combine all data
# dataframes = [test1, test2, test3]
# data = pd.concat(dataframepip inss, axis=1)

# # Extract features and labels
# X = np.array([file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1),
#               file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1),
#               file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1),
#               file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1),
#               file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1),
#               file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1),
#               file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1),
#               file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1)])

# Extract features and labels
bearing1 = file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1)
bearing2 = file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1)
bearing3 = file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1)
bearing4 = file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1)
bearing5 = file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1)
bearing6 = file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1)
bearing7 = file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1)
bearing8 = file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1)

X = np.concatenate([bearing1, bearing2, bearing3, bearing4, bearing5, bearing6, bearing7, bearing8])
Y = np.array(['Défaut bague externe', 'Sans défaut', 'Sans défaut','Sans défaut', 'Sans défaut', 'Sans défaut', 'Défaut bague externe', 'Sans défaut'])

# Pad sequences to same length
X_padded = pad_sequences(X, dtype='float32', padding='post')

# Reshape input data
X_reshaped = X_padded.reshape(X_padded.shape[0], -1)

print("Shape of X:", X_reshaped.shape)
print("Shape of Y:", Y.shape)

print(X_reshaped)
# # Train SVM model
# model = svm.SVC()
# model.fit(X_reshaped, Y)
#
# # Extract features for prediction
# data = file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1)
# data_padded = pad_sequences([data], dtype='float32', padding='post')
# data_reshaped = data_padded.reshape(1, -1)
#
# # Evaluate accuracy
# y_pred = model.predict(data_reshaped)
# acc = accuracy_score(Y, y_pred)
# print('Accuracy:', acc)
#
# # Save model to file
# with open('svm_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
