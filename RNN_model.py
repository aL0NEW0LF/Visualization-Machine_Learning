import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Load the statistical indicators files
file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing.xlsx', sheet_name=None)

# Extract features and labels
X = np.array([file2['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1),
              file2['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1),
              file2['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1),
              file2['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1),
              file3['Bearing_1'].drop(['File Name', 'Failure_type'], axis=1),
              file3['Bearing_2'].drop(['File Name', 'Failure_type'], axis=1),
              file3['Bearing_3'].drop(['File Name', 'Failure_type'], axis=1),
              file3['Bearing_4'].drop(['File Name', 'Failure_type'], axis=1)])

Y = np.array(['Défaut bague externe', 'Sans défaut', 'Sans défaut','Sans défaut', 'Sans défaut', 'Sans défaut', 'Défaut bague externe', 'Sans défaut'])

# create a dictionary to map class labels to integers
class_dict = {'Sans défaut': 0, 'Défaut bague interne': 1, 'Défaut bague externe': 2, 'Défaut billes': 3}

# encode class labels as integers
Y = [class_dict[label] for label in Y]

# Convert labels to one-hot encoding
Y = to_categorical(Y)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(None, 1), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, Y, epochs=100, validation_split=0.2)

# Save model to file
model.save('rnn_model.h5')
