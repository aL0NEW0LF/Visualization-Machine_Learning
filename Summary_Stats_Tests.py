import os
import pandas as pd

# specify the folder path where the Excel files are located
folder_path = "C:/PFE/Statistical_Indicators"

# specify the path for the output Excel file
output_file_path = "C:/PFE/Summary_statistical_Indicators.xlsx"

# create a dictionary to store the dataframes for each test
test_dfs = {}

# iterate through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsx"):
        print(f"Processing file: {file_name}")
        file_path = os.path.join(folder_path, file_name)

        # extract the test name from the file name
        test_name = file_name.split(".")[0]

        # read the Excel file
        dfs = pd.read_excel(file_path, sheet_name=None)

        # create an empty dataframe to store the statistical indicators for this test
        combined_df = pd.DataFrame()

        for sheet_name, df in dfs.items():
            print(f"Processing sheet: {sheet_name}")
            for col in df.columns:
                # calculate summary statistics
                count = df[col].count()
                mean_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                std_val = df[col].std()
                percentile_25 = df[col].quantile(0.25)
                percentile_50 = df[col].quantile(0.5)
                percentile_75 = df[col].quantile(0.75)

                # create a dictionary with the summary statistics
                col_stats = {
                    'File Name': file_name,
                    'Sheet Name': sheet_name,
                    'Column Name': col,
                    'Count': count,
                    'Mean': mean_val,
                    'Max': max_val,
                    'Min': min_val,
                    'Std': std_val,
                    '25%': percentile_25,
                    '50%': percentile_50,
                    '75%': percentile_75
                }

                # add the dictionary to the combined dataframe
                combined_df = pd.concat([combined_df, pd.DataFrame(col_stats, index=[0])], ignore_index=True)

        # store the dataframe for this test in the dictionary
        test_dfs[test_name] = combined_df

# write the dataframes for each test to separate sheets in the output Excel file
writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')
for test_name, test_df in test_dfs.items():
    test_df.to_excel(writer, sheet_name=test_name, index=False)
writer.close()

print(f"Summary statistics saved to '{output_file_path}'.")
