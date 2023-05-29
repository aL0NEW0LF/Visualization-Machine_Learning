import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
columns_names = ['Skewness', 'Kurtosis', 'Crest Factor', 'Form Factor']
columns_names_test1 = [['X_Skewness', 'X_Kurtosis', 'X_Crest Factor', 'X_Form Factor'], ['Y_Skewness', 'Y_Kurtosis', 'Y_Crest Factor', 'Y_Form Factor']]

# path to the folder containing the data files
data_folder = ["D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2.xlsx", "D:/PFE/Statistical_Indicators/Statistical_Indicators_Test3.xlsx"]
data_test1 = "D:/PFE/Statistical_Indicators/Statistical_Indicators_Test1.xlsx"
# path to the folder where we will be saving the visuals
visuals_folder = "D:/PFE/Statistical_Indicators/"
resolution = (76.80 , 76.80)


for i, data_file in enumerate(data_folder, 2):
    # load the data from the file
    data = pd.read_excel(data_file, sheet_name=None)

    # get the time vector (assuming it's in the first column)
    time = [j / len(data['Bearing_1']) for j in range(len(data['Bearing_1']))]
    print(f"Test: {i}")
    for sheet_name, df in data.items():
        print(f"    Processing sheet: {sheet_name}")

        column_data = df[columns_names]

        # Plot data
        fig, ax = plt.subplots(figsize=resolution)
        ax.plot(time, column_data, linewidth=1, label=columns_names)
        ax.set_xlabel('Temps')
        ax.set_ylabel(columns_names)
        ax.set_title(columns_names)
        ax.legend()
        plt.savefig(os.path.join(visuals_folder, "Statistical_indicators_plot_test" + f"{i}" + "_" + f"{sheet_name}" + ".pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(visuals_folder, "Statistical_indicators_plot_test" + f"{i}" + "_" + f"{sheet_name}" + ".png"), dpi=300, bbox_inches='tight')

        # clear plot for next file
        plt.close()

data = pd.read_excel(data_test1, sheet_name=None)

# get the time vector (assuming it's in the first column)
time = [j / len(data['Bearing_1']) for j in range(len(data['Bearing_1']))]

print("Test: 1")

for sheet_name, df in data.items():
    print(f"    Processing sheet: {sheet_name}")

    for k, columns_name in zip(['X', 'Y'], columns_names_test1):
        column_data = df[columns_name]

        # Plot data
        fig, ax = plt.subplots(figsize=resolution)
        ax.plot(time, column_data, linewidth=1, label=columns_name)
        ax.set_xlabel('Temps')
        ax.set_ylabel(columns_name)
        ax.set_title(columns_name)
        ax.legend()
        plt.savefig(os.path.join(visuals_folder, "Statistical_indicators_plot_test1_" + f"{sheet_name}" + f"{k}" + ".pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(visuals_folder, "Statistical_indicators_plot_test1_" + f"{sheet_name}" + f"{k}" + ".png"), dpi=300, bbox_inches='tight')

        # clear plot for next file
        plt.close()