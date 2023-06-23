import numpy as np
# from mlxtend.data import loadlocal_mnist


def read_simple_dataset_funct():

    with open('mfeat-pix.txt') as f:
        raw_dataset = f.readlines()

    with open('mfeat-pix.txt') as f:
        raw_dataset = f.readlines()

    for i in range(len(raw_dataset)):
        array = raw_dataset[0].split(' ')
        a = np.array('')
        # We do an array with the items
        for item in array:
            a = np.append(a, item)

        # We delete empty items
        a = a[a != ""]

        # We change it from string to float
        a = a.astype('float64')

        if i == 0:
            dataset = a
        else:
            dataset = np.vstack([dataset, a])

    dataset = dataset.astype(int)

    # Shape dataset (2000, 240)
    # Type

    return dataset
















# def read_dataset_mnist_funct():
#     print("**Running read dataset function.")
#
#     X_train, y_train = loadlocal_mnist(images_path='train-images.idx3-ubyte', labels_path='train-labels.idx1-ubyte')
#     X_test, y_test = loadlocal_mnist(images_path='t10k-images.idx3-ubyte', labels_path='t10k-labels.idx1-ubyte')
#
#     print("Shape of train datasets: ", X_train.shape, y_train.shape)
#     print("Shape of test datasets: ", X_test.shape, y_test.shape)
#
#     # Resize the arrays
#     X_train.resize(60000, 28, 28)
#     X_test.resize(10000, 28, 28)
#
#
#     return X_train, y_train, X_test, y_test





