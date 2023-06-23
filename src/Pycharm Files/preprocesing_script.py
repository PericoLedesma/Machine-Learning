# Tratamiento de datos
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
# -------------------------------------------------------------------------------------------------


def preprocesing_funct(X_train, X_test, y_train, y_test):

    # Scaling values between 1 and 0

    # Changing the format to store decimal values
    X_train = X_train.astype(np.float64)/255
    X_test = X_test.astype(np.float64)/255

    # We need to expand the dimention of images to (28,28,1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # Convert to hot encoding vector the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # # Let´s see the hot encoding
    # print(y_test)

    return X_train, X_test, y_train, y_test