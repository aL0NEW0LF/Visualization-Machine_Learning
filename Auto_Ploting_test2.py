import os
import matplotlib.pyplot as plt
import numpy as np

# path to the folder containing the data files
data_folder = "D:/PFE/2nd_test/2nd_test"

# path to the folder where we will be saving the visuals
visuals_folder = "D:/PFE/Visuals/Test 2"
resolution = (76.80 , 21.60)

# loop through all files in the folder
for filename in os.listdir(data_folder):
    # load the data from the file
    data = np.loadtxt(os.path.join(data_folder, filename))

    # get the time vector (assuming it's in the first column)
    time = [i / 20000 for i in range(len(data))]

    # get the number of columns in the data
    num_columns = data.shape[1]

    # plot each column as a line plot
    for i in range(0, num_columns):
        column_data = data[:, i]

        # Plot data
        fig, ax = plt.subplots(figsize=resolution)
        ax.plot(time, column_data, label=f"Column {i + 1}")
        ax.set_xlabel('Temps')
        ax.set_ylabel('Acceleration')
        ax.set_title(filename)
        ax.legend()
        plt.savefig(os.path.join(visuals_folder, filename + f"CH{i + 1}" + ".pdf"), dpi=300, bbox_inches='tight')

        # clear plot for next file
        plt.close()