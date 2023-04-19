import os

import pandas as pd

# specify the path to the input Excel file
input_file_path = "D:/PFE/2nd_test/test.xlsx"

# specify the path for the output Excel file
output_folder_path = "D:/PFE/Preprocessed_Data/Test 2/"

# read the Excel file into a dictionary of dataframes, with sheet names as keys
dfs = pd.read_excel(input_file_path, sheet_name=None)

# create a new dictionary to store the processed data
processed_dfs = {}

# iterate through each sheet in the dictionary
for sheet_name, df in dfs.items():
    print(f"Processing sheet: {sheet_name}")

    # handling missing values
    df.dropna(inplace=True)

    # removing duplicate rows
    df.drop_duplicates(inplace=True)

    # removing extreme values (outliers)
    # you can use your own logic to identify and remove extreme values here
    # for example, you can use z-score or percentile-based approach
    # below is an example of using percentile-based approach to remove extreme values
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        # Count number of rows before processing
        num_rows_before = df.shape[0]

        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]


        # # Count number of rows after processing
        # num_rows_after = df.shape[0]
        #
        # # Calculate number of deleted rows
        # num_deleted_rows = num_rows_before - num_rows_after
        #
        # if num_deleted_rows > 0:
        #     print(f'{num_deleted_rows} rows deleted in column: {col}')


    # do further data preprocessing tasks for each sheet as needed
    # e.g., data normalization, feature engineering, etc.

    # add the processed dataframe to the dictionary of processed dataframes without column labels
    processed_dfs[sheet_name] = df.values

    # save the processed sheet to a CSV file
    output_file_path = f"{output_folder_path}/{sheet_name}.csv"
    pd.DataFrame(df).to_csv(output_file_path, index=False, header=False)

    print(f"Sheet '{sheet_name}' processed and saved to '{output_file_path}'.")

print("All sheets processed and saved to CSV files.")