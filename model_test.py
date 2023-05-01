import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense

# Load the statistical indicators files
file1 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing - Copy.xlsx', sheet_name=None)
file2 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing - Copy.xlsx', sheet_name=None)
file3 = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing - Copy.xlsx', sheet_name=None)

# Combine all sheets into one dataframe
data = pd.concat([file1['Bearing_1'], file1['Bearing_2'], file1['Bearing_3'], file1['Bearing_4'],
                  file2['Bearing_1'], file2['Bearing_2'], file2['Bearing_3'], file2['Bearing_4'],
                  file3['Bearing_1'], file3['Bearing_2'], file3['Bearing_3'], file3['Bearing_4']])

# Split the data into input (X) and output (Y) variables
X = data.drop(['File Name', 'Failure_type'], axis=1)
Y = data['Failure_type']

# encode string labels to numerical values
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)


# Y = Y.astype('string')

# Build the LSTM/GRU model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(None, 1)))
# model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1))) # Use this line for GRU
model.add(LSTM(units=50))
# model.add(GRU(units=50)) # Use this line for GRU
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X.values.reshape(X.shape[0], X.shape[1], 1), Y, epochs=50, batch_size=32)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('model_test.h5')