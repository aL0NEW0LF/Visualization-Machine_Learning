import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import plot_model

# # path to the folder containing the data files
# data_folder = "D:/PFE/2nd_test/2nd_test"
# columns_names = ['Skewness', 'Kurtosis', 'Crest Factor', 'Form Factor']
#
# # path to the folder where we will be saving the visuals
# visuals_folder = "D:/PFE/"
#
# # filename = "2004.02.18.13.12.39"
# filename = "D:/PFE/Statistical_Indicators/Statistical_Indicators_Test2.xlsx"
# resolution = (76.80, 30)

# Load the saved model
# model_LSTM_10col = load_model('model_LSTM_10col.h5')
# model_LSTM_20col = load_model('model_LSTM_20col.h5')
# model_SimpleRNN_10col = load_model('model_SimpleRNN_10col.h5')
model_SimpleRNN_20col = load_model('model_SimpleRNN_20col.h5')
dot_img_file = 'D:/PFE/Statistical_Indicators/Graphs/model_SimpleRNN_20col.png'

plot_model(model_SimpleRNN_20col, to_file=dot_img_file, show_shapes=True, dpi=300, show_layer_activations=True,
           show_trainable=True)

# # load the data from the file
# # data = np.loadtxt(os.path.join(data_folder, filename))
# data = pd.read_excel(filename, sheet_name='Bearing_1')
#
# # get the time vector (assuming it's in the first column)
# time = [i / 20000 for i in range(len(data))]
#
# # # get the number of columns in the data
# # num_columns = data.shape[1]
#
# # column_data = data[:, 1]
#
# # # Plot data
# # fig, ax = plt.subplots(figsize=resolution)
# # ax.plot(time, column_data, linewidth=1, label="Column 2", color="black")
# # ax.set_xlabel('Temps')
# # ax.set_ylabel('Acceleration')
# # ax.set_title(filename)
# # ax.legend()
# # plt.savefig(os.path.join(visuals_folder, filename + "CH2" + ".png"), dpi=300, bbox_inches='tight')
# #
# # # clear plot for next file
# # plt.close()
# column_data = data[columns_names]
#
# # Plot data
# fig, ax = plt.subplots(figsize=resolution)
# ax.plot(time, column_data, linewidth=1, label=columns_names)
# ax.set_xlabel('Temps')
# ax.set_ylabel(columns_names)
# ax.set_title(columns_names)
# ax.legend()
# plt.savefig(os.path.join(visuals_folder, "Statistical_indicators_plot_test.png"), dpi=300, bbox_inches='tight')
#
# # clear plot for next file
# plt.close()