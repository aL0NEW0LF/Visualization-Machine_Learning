import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
data = []
labels = []

file_paths = ['D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2_testing.xlsx', 'D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3_testing.xlsx']

# Decode the predicted labels to get the original string labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('encod2.npy', allow_pickle=True)

# Iterate over each Excel file
for file_path in file_paths:
    # Load the Excel file
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # Iterate over each sheet in the Excel file
    for sheet_name, sheet_data in excel_data.items():
        # Convert the sheet data to a numpy array
        sequence = sheet_data.drop(['File Name', 'Failure_type'], axis=1).to_numpy()

        # Add the sequence to the data list
        data.append(sequence)

        # # Add the label corresponding to the sheet name
        # labels.append(sheet_name['Failure_type'])

labels = ['Defaut bague externe',
          'Sans defaut',
          'Sans defaut',
          'Sans defaut',
          'Sans defaut',
          'Sans defaut',
          'Defaut bague externe',
          'Sans defaut']

default_labels = ['Defaut bague externe',
                  'Defaut bague intern',
                  'Defaut billes',
                  'Sans defaut']

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Determine the maximum sequence length
max_sequence_length = max(len(sequence) for sequence in data)

# Pad the sequences
padded_data = pad_sequences(data, padding='post')

labels = label_encoder.transform(labels)

# for i,j in zip(padded_data, labels):
#     print(i, '\n')
#     print(j, '\n')

# Reshape the data
reshaped_data = padded_data.reshape(-1, padded_data.shape[-1])

# # Reshape the padded data to have two dimensions
# reshaped_data = padded_data.reshape(-1, max_sequence_length)

# Check the class distribution
class_counts = np.bincount(labels)
minority_class_label = np.argmin(class_counts)

print("Class Distribution:")
for class_label, count in enumerate(class_counts):
    print(f"Class {class_label}: {count}")

print(reshaped_data.shape, labels.shape)
# Perform random undersampling
rus = RandomUnderSampler()
data_balanced, labels_balanced = rus.fit_resample(reshaped_data, labels)

# Check the class distribution
unique_labels, counts = np.unique(labels_balanced, return_counts=True)
class_distribution = dict(zip(unique_labels, counts))
print("Balanced Class Distribution:")
for label, count in class_distribution.items():
    print(f"Class {label}: {count}")

# # Create a mask to identify the majority class sequences
# mask_majority = labels != minority_class_label
#
# # Separate majority and minority class sequences
# data_majority = padded_data[mask_majority]
# data_minority = padded_data[~mask_majority]
# labels_majority = labels[mask_majority]
# labels_minority = labels[~mask_majority]
#
# # Perform under-sampling on majority class sequences
# rus = RandomUnderSampler()
# data_balanced, labels_balanced = rus.fit_resample(data_majority, labels_majority)
#
# # Concatenate the balanced data with the minority class sequences
# balanced_data = np.concatenate((data_balanced, data_minority), axis=0)
# balanced_labels = np.concatenate((labels_balanced, labels_minority), axis=0)
#
# # Calculate the balanced class distribution
# class_counts_balanced = np.bincount(balanced_labels)
# for i, count in enumerate(class_counts_balanced):
#     print(f"Class {i}: {count}")

# # Randomly sample sequences from the majority class
# random_indices_majority = np.random.choice(np.where(mask_majority)[0], size=len(data) - np.sum(mask_majority), replace=False)
#
# # Combine the minority class sequences with the randomly sampled majority class sequences
# resampled_data = np.concatenate((data[~mask_majority], data[random_indices_majority]), axis=0)
# resampled_labels = np.concatenate((labels[~mask_majority], labels[random_indices_majority]), axis=0)

# # Perform random undersampling to balance the data
# rus = RandomUnderSampler(random_state=42)
# data_balanced, labels_balanced = rus.fit_resample(reshaped_data, labels)

## Check the balanced class distribution
# class_counts_balanced = np.bincount(labels_balanced)
# print("\nBalanced Class Distribution:")
# for class_label, count in enumerate(class_counts_balanced):
#     print(f"Class {class_label}: {count}")

# # Check the balanced class distribution
# class_counts_balanced = np.bincount(resampled_labels)
# print("\nBalanced Class Distribution:")
# for class_label, count in enumerate(class_counts_balanced):
#     print(f"Class {class_label}: {count}")