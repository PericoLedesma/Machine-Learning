import numpy as np


def split_dataset_funct(dataset):

    dataset = np.resize(dataset, (200, 2400))
    dataset = np.array[np.resize(row, (100, 240)) for row in dataset]

    X_train = dataset[::2]
    X_test = dataset[1::2]

    return X_train , X_test, y_train, y_test