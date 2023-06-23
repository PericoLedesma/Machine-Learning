## Let´s start!
print('*****Start of the  main script')

# Tratamiento de datos
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

from PIL import Image

# Function in other files
# from read_script import read_dataset_mnist_funct
from read_script import read_simple_dataset_funct
from split_dataset import split_dataset_funct
# from preprocesing_script import preprocesing_funct
# from plot_number import plot_digit_funct

# -------------------------------------------------------------------------------------------------
# Reading the MNIST dataset
# X_train, y_train, X_test, y_test = read_dataset()

# Reading simple dataset

dataset = read_simple_dataset_funct()
print(type(dataset))

dataset = split_dataset_funct(dataset)




# print(dataset[train_index])

# plt.title(y[i])

# Convert NumPy array back to Pillow image
# img = Image.fromarray(train_index[0])
# print(img)

# # image = np.reshape(train_index[0],[15, 16])
# plt.imshow(image, cmap='binary')
# plt.show()








#Preprocesing script
# X_train, X_test = preprocesing_funct(X_train, X_test, y_train, y_test)

# for i in range(3):
#     plot_digit_funct(i, X_test, y_test)



# END
print("*****End of the script")