import os
import pandas as pd
import glob

# # Set the folder path where the CSV files are located
# folder_path = "D:\PFE\\2nd_test\\2nd_test"
#
# # Get a list of all CSV files in the folder
# csv_files = [f for f in os.listdir(folder_path)]
#
# # Create an empty dictionary to store dataframes
# df_dict = {}
#
# # Loop through the CSV files and add them to the dictionary with their filename as key
# for file in csv_files:
#     df_dict[file] = pd.read_csv(os.path.join(folder_path, file))
#
# # Create a writer object to write to a single CSV file with multiple sheets
# writer = pd.ExcelWriter(os.path.join(folder_path, 'combined.xlsx'), engine='xlsxwriter')
#
# # Loop through the dictionary and write each dataframe to a separate sheet
# for key, value in df_dict.items():
#     value.to_excel(writer, sheet_name=key[:-4], index=False)
#
# # Save the workbook
# writer._save()


folder_path = "D:/PFE/2nd_test/2nd_test"  # Replace with the actual folder path

# Use glob to find all PDF files in the folder
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

# Loop through the PDF files and delete each one
for pdf_file in pdf_files:
    os.remove(pdf_file)