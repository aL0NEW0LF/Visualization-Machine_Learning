import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from math import sqrt

# specify the folder path where the CSV files are located
folder_path = "D:/PFE/1st_test/all_files_1.xlsx"

# specify the path for the output Excel file
output_file_path = "D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1.xlsx"
# specify the column names for the first test
columns_names = ["B1X", "B1Y", "B2X", "B2Y", "B3X", "B3Y", "B4X", "B4Y"]

# create an empty dictionary to store the statistical indicators
result_dict = {}

# iterate through each file in the folder
dfs = pd.read_excel(folder_path, sheet_name=None)

for sheet_name, df in dfs.items():
    print(f"Processing sheet: {sheet_name}")
    file_stats = {'File Name': sheet_name}
    for i, col in zip(columns_names, df.columns):
        # calculate statistical indicators
        max_val = df[col].max()
        min_val = df[col].min()
        mean_val = df[col].mean()
        std_val = df[col].std()
        rms_val = sqrt(np.mean(np.square(df[col])))
        skewness_val = skew(df[col])
        kurtosis_val = kurtosis(df[col])
        crest_factor = max_val / rms_val
        form_factor = rms_val / abs(mean_val)
        # add the statistical indicators to the dictionary with column names as numbers
        file_stats[i + '_Max'] = max_val
        file_stats[i + '_Min'] = min_val
        file_stats[i + '_Mean'] = mean_val
        file_stats[i + '_Std'] = std_val
        file_stats[i + '_RMS'] = rms_val
        file_stats[i + '_Skewness'] = skewness_val
        file_stats[i + '_Kurtosis'] = kurtosis_val
        file_stats[i + '_Crest Factor'] = crest_factor
        file_stats[i + '_Form Factor'] = form_factor

    # add the file statistics to the result dictionary
    result_dict[sheet_name] = file_stats

# convert the result dictionary to a dataframe
result_df = pd.DataFrame.from_dict(result_dict, orient='index')

# save the result dataframe to an Excel file
result_df.to_excel(output_file_path, index=False)
print(f"Statistical indicators saved to '{output_file_path}'.")
