import pandas as pd
from scipy.stats import zscore

# Define the paths to the input Excel files
input_files = ["D:/PFE/Statistical_Indicators/Statistical_Indicators_10col.xlsx", "D:/PFE/Statistical_Indicators/Statistical_Indicators_20col.xlsx"]

# Define the path to the output Excel files
output_files = ["D:/PFE/Statistical_Indicators/normalized_10col.xlsx", "D:/PFE/Statistical_Indicators/normalized_20col.xlsx"]

# Iterate over the input files
for i in range(len(input_files)):
    input_file = input_files[i]
    output_file = output_files[i]

    # Load the Excel file
    xls = pd.ExcelFile(input_file)

    # Get the sheet names
    sheet_names = xls.sheet_names

    # Create a new Excel writer
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    # Iterate over the sheets
    for sheet_name in sheet_names:
        # Read the sheet data into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Select only the numeric columns
        numeric_columns = df.select_dtypes(include='number')

        # Normalize the data using z-score normalization
        normalized_data = numeric_columns.apply(zscore)

        # Write the normalized data to the new Excel file
        normalized_data.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the new Excel file
    writer._save()

    print(f"Normalized data saved to {output_file}")
