#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from scipy import sparse

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import numpy as np

import pdb

batch_size=20

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(97, 300)
        self.fc1 = nn.Linear(300, 500)
        self.fc2 = nn.Linear(500, 800)
        self.fcout = nn.Linear(800, 8092)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fcout(x)

def convert_x(scipy_sparse, max_y):
    return scipy_sparse.data/max_y

def convert_y(sparse, adj, max_y):
    return np.array(sparse.todense()[adj.todense().astype(bool)])[0]/max_y

def train_test(model, X_train, X_test, y_train, y_test, adj):
    max_y = max([samp.max() for samp in y_test])

    print("Reformatting data")
    train_zipped = [(convert_x(x_samp, max_y), convert_y(y_samp, adj, max_y)) for x_samp, y_samp in zip(X_train, y_train)]
    test_zipped = [(convert_x(x_samp, max_y), convert_y(y_samp, adj, max_y)) for x_samp, y_samp in zip(X_test, y_test)]

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #, momentum=0.9)

    print("Setting up data loader")

    loader = DataLoader(train_zipped, batch_size=batch_size,
                        shuffle=True)
    
    test_loader = DataLoader(test_zipped, batch_size=batch_size,
                                         shuffle=False)

    train_losses = []
    test_losses = []
    for epoch in range(10):  # loop over the dataset multiple times
        print("epoch")
        running_loss = 0.0
        for i, batch in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            once_loss = loss.item() / len(outputs)
            running_loss += once_loss
            if i % 1 == 1 or True:
                print('[%d, %3d]      loss: %.3f' %
                      (epoch + 1, i + 1, once_loss*outputs.shape[1]))

            
            test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = model(images)
                    test_once_loss = (criterion(outputs, labels).item() / len(outputs)) * outputs.shape[1]
                    test_loss += test_once_loss/len(test_loader)
            print('[%d, %3d] test loss: %.3f' %
                      (epoch + 1, i + 1, test_loss))
            test_losses.append(test_loss)
            train_losses.append(once_loss*outputs.shape[1])

            loss.backward()
            optimizer.step()

    test_loader = DataLoader(test_zipped, batch_size=1,
                                         shuffle=False)
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.title("Loss for normalized data, dense network")
    plt.show()
    true_test = []
    pred_test = []
    test_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image, label = data
            output = model(image).squeeze()
            label = label.squeeze()

            truth = np.zeros(adj.shape)
            pred = np.zeros(adj.shape)

            truth[adj.todense().astype(bool)] = label*max_y
            pred[adj.todense().astype(bool)] = output*max_y
            true_test += [sparse.csr_matrix(truth)]
            pred_test += [sparse.csr_matrix(pred)]

            test_once_loss = criterion(output, label).item() / len(test_loader)
            test_loss += test_once_loss

    print("Finished training")
    return model,true_test, pred_test
