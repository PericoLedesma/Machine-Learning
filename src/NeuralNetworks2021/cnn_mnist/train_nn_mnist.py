
"""
Train Neural Network on MNIST dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from readingMNIST import *
from torch.utils.data import DataLoader
from CNN import *
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor

# REGISTER PATH HERE FOR NEW TRAIN
# savePath = "/home/goncalo/Documents/RUG/MSc_1st_year/block_1b/machineLearning/NeuralNetworks2021/cnn_mnist/models/model_cnn_id0001.pt"

def train(cnn, loaders):
    # Training parameters
    num_epochs = 10
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   

    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()                # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        pass
    pass

def main():
    # Classic loading 50/50 as explained in the lecture
    # df = pd.read_csv('mfeat-pix.txt', header=None, delimiter='\n')
    # dataset = make_dataset()
    # print(dataset.shape)
    # X_train, X_test, Y_train, Y_test = build_datasets(dataset)
    # X_test = np.array(X_test)
    # X_train = np.array(X_train)
    # show_number(X_train[203])
    # show_number(X_test[203])

    # print(Y_test[203])
    # print(Y_train[203])

    # Loading as preferred by pytorch
    X_train = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    X_test = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

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

    cnn = CNN()
    print(cnn, loaders)
    # print(X_test.shape)
    # print(X_train.shape)

    train(cnn, loaders)

    sample = next(iter(loaders['test']))
    imgs, lbls = sample

    actual_number = lbls[:10].numpy()
    print(actual_number)

    test_output, last_layer = cnn(imgs[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(f'Prediction number: {pred_y}')
    print(f'Actual number: {actual_number}')
    torch.save(cnn, savePath)

if __name__ == '__main__':
    main()