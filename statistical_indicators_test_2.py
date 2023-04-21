import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from math import sqrt

# specify the folder path where the CSV files are located
folder_path = "D:/PFE/2nd_test/all_files_2.xlsx"

# specify the path for the output Excel file
output_file_path = "D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2.xlsx"

# create an empty dataframe to store the statistical indicators
result_dict = {}
# iterate through each file in the folder
dfs = pd.read_excel(folder_path, sheet_name=None)

# iterate through each sheet in the dictionary
for sheet_name, df in dfs.items():
    print(f"Processing sheet: {sheet_name}")
    file_stats = {'File Name': sheet_name}
    for i, col in enumerate(df.columns):
        # calculate statistical indicators
        max_val = df[col].max()
        min_val = df[col].min()
        mean_val = df[col].mean()
        var_val = df[col].var()
        std_val = df[col].std()
        rms_val = sqrt(np.mean(np.square(df[col])))
        skewness_val = skew(df[col])
        kurtosis_val = kurtosis(df[col])
        crest_factor = max_val / rms_val
        form_factor = rms_val / abs(mean_val)
        # add the statistical indicators to the dictionary
        file_stats['B' + str(i+1) + '_Max'] = max_val
        file_stats['B' + str(i+1) + '_Min'] = min_val
        file_stats['B' + str(i+1) + '_Mean'] = mean_val
        file_stats['B' + str(i+1) + '_Var'] = var_val
        file_stats['B' + str(i+1) + '_Std'] = std_val
        file_stats['B' + str(i+1) + '_RMS'] = rms_val
        file_stats['B' + str(i+1) + '_Skewness'] = skewness_val
        file_stats['B' + str(i+1) + '_Kurtosis'] = kurtosis_val
        file_stats['B' + str(i+1) + '_Crest Factor'] = crest_factor
        file_stats['B' + str(i+1) + '_Form Factor'] = form_factor

    # add the file statistics to the result dictionary
    result_dict[sheet_name] = file_stats

# convert the result dictionary to a dataframe
result_df = pd.DataFrame.from_dict(result_dict, orient='index')

# save the result dataframe to an Excel file
result_df.to_excel(output_file_path, index=False)
print(f"Statistical indicators saved to '{output_file_path}'.")
