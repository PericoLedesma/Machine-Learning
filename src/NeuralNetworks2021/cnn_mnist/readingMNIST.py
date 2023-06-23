"""
Read MNIST data set from txt file.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_train_std_MNIST():
    X_train = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    return X_train

def load_test_std_MNIST():
    X_test = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    return X_test

def show_number(display_number):
    display_number = np.resize(display_number, (16, 15))

    plt.imshow(display_number, cmap='binary')
    plt.show()

def load_loaders(X_train, X_test):
    loaders = {
        'train' : torch.utils.data.DataLoader(X_train, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(X_test, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders

def make_dataset():
    with open('mfeat-pix.txt') as f:
        raw_dataset = f.readlines()

    for i in range(len(raw_dataset)):
        array = raw_dataset[i].split(' ')
        a = np.array('')
  
        for item in array:
            a = np.append(a, item)

        a = a[a != ""]

        # a = a.astype('float64')

        if i == 0:
            dataset = a
        else:
            dataset = np.vstack([dataset, a])
    
    dataset = dataset.astype(int)

    # Visualizing the two hundred and third datapoint which is the third example of number 1 - the first 200 datapoints will be of the number 0
    # show_number(dataset[203])

    # No need for these confusing transformations ...
    # dataset = np.resize(dataset, (20, 24000))
    # dataset = np.array([np.resize(row, (100, 240))for row in dataset])

    return dataset

def build_datasets(dataset):
    X_train = []
    X_test = []
    
    # Construct training and testing datasets
    hundred_counter = 0
    for i in range(len(dataset)):
        if hundred_counter > 199:
            hundred_counter = 0
        if hundred_counter < 100:
            X_train.append(dataset[i])         
        else:
            X_test.append(dataset[i])  
        hundred_counter += 1

    # Construct the labels for training and testing datasets
    Y_train = [0]
    number = 0

    for i in range(1,1000):
        if i%100 == 0 and i!=0:
            number+=1
            
        Y_train = np.vstack([Y_train, number])
    Y_test = Y_train
    return X_train, X_test, Y_train, Y_test

