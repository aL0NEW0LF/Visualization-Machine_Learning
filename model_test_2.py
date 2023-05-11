import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Load the statistical indicators files
test1_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1_testing - Copy.xlsx', sheet_name=None)
test2_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing - Copy.xlsx', sheet_name=None)
test3_data = pd.read_excel('D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing - Copy.xlsx', sheet_name=None)

# Combine the data for each bearing into a single dataframe
test2_combined = pd.concat(test2_data.values(), ignore_index=True)
test3_combined = pd.concat(test3_data.values(), ignore_index=True)

# Concatenate all test sets
combined_data = pd.concat([test2_combined, test3_combined], ignore_index=True)

# Convert the features and labels to numpy arrays
X = combined_data.drop(['File Name', 'Failure_type'], axis=1).values
y = combined_data['Failure_type'].values

# Normalize the input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert string labels to numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data for the RNN model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Define the RNN model architecture
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

np.save('encod.npy', label_encoder.classes_)
# Save the trained model
model.save('model_test_5.h5')