#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import numpy as np

from gcnn import GraphConvolution

import pdb

batch_size=20


class MLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(194, 150*150)
        self.fc1 = nn.Linear(6*146*146, 500)
        self.fc2 = nn.Linear(500, 13447)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(194, 150*150)
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.upsample = nn.Upsample(scale_factor=1)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(6*146*146, 500)
        self.fc2 = nn.Linear(500, 13447)

    def forward(self, x):
        pdb.set_trace()
        #x = F.relu(self.fc0(x))
        #x = F.relu(self.conv1(x.to_dense().unsqueeze(1)))
        x = F.relu(self.conv1(x.to_dense().unsqueeze(1)))
        pdb.set_trace()
        #x = self.upsample(x)
        #x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.fc1(x.view(-1, 6*146*146)))
        x = self.fc2(x)
        return x

def convert(scipy_sparse):
    scipy_sparse = scipy_sparse.tocoo()
    #pdb.set_trace()
    return torch.sparse.FloatTensor(torch.LongTensor([scipy_sparse.row.tolist(), scipy_sparse.col.tolist()]),
                                    torch.FloatTensor(scipy_sparse.data.astype(np.float32)), size=scipy_sparse.shape)


def train_test(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    print("Reformatting data")
    train_zipped = [(convert(x_samp), convert(y_samp)) for x_samp, y_samp in zip(X_train, y_train)]
    test_zipped = [(convert(x_samp), convert(y_samp)) for x_samp, y_samp in zip(X_test, y_test)]

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #, momentum=0.9)

    print("Setting up data loader")

    loader = DataLoader(train_zipped, batch_size=batch_size,
                        shuffle=True)
    
    test_loader = DataLoader(test_zipped, batch_size=batch_size,
                                         shuffle=True)

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
                      (epoch + 1, i + 1, once_loss))

            
            test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = model(images)
                    test_once_loss = criterion(outputs, labels).item() / len(outputs)
                    test_loss += test_once_loss
            print('[%d, %3d] test loss: %.3f' %
                      (epoch + 1, i + 1, test_loss/len(test_loader)))

            loss.backward()
            optimizer.step()


    print("Finished training")
    return model
