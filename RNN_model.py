import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Load the statistical indicators files
file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing.xlsx', sheet_name=None)

# # Extract features and labels
# X = np.array([file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).astype(object),
#               file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).astype(object)])
#
# data = {'features': [file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).values,
#                      file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).values],
#         'output': ['Défaut bague externe', 'Sans défaut', 'Sans défaut','Sans défaut', 'Sans défaut', 'Sans défaut', 'Défaut bague externe', 'Sans défaut']}

data_tensor = tf.ragged.constant([
     file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).values,
     file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).values,
     file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).values,
     file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).values,
     file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1).values,
     file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1).values,
     file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1).values,
     file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1).values])

Y = tf.constant([['Defaut bague externe'],
                 ['Sans defaut'],
                 ['Sans defaut'],
                 ['Sans defaut'],
                 ['Sans defaut'],
                 ['Sans defaut'],
                 ['Defaut bague externe'],
                 ['Sans defaut']])

Y_nums = tf.constant([[2],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [2],
                      [0]])

# df = pd.DataFrame(data)
# Y = np.array(['Défaut bague externe', 'Sans défaut', 'Sans défaut','Sans défaut', 'Sans défaut', 'Sans défaut', 'Défaut bague externe', 'Sans défaut']).astype(str)
#
# # create a dictionary to map class labels to integers
# class_dict = {'Sans défaut': 0, 'Défaut bague interne': 1, 'Défaut bague externe': 2, 'Défaut billes': 3}
#
# # encode class labels as integers
# Y = [class_dict[label] for label in Y]

# # Convert labels to one-hot encoding
# Y = to_categorical(Y)

#to check our data
print(data_tensor.shape)
print(data_tensor)
print(Y)
print(Y.shape)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(None, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(Y.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(data_tensor, Y_nums, epochs=100)

# Save model to file
model.save('rnn_model.h5')
