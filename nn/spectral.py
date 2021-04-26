#!/usr/bin/env python3

import pdb
from pprint import pprint as pp
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data import make_data_cached

from conv_model import train_test, ConvNet, MLNet
import dmlc
import torch.nn.functional as F
import dgl

def dcgn_model(X, y, adj):
    import pdb
    pdb.set_trace()
    model = dmlc.MWE_DGCN(1, 100, X[0].shape[1], 3, F.relu, 0)
    dmlc.train_test(model, X, y, adj)

def keras_model(X, y):
    pdb.set_trace()
    net = ConvNet()
    train_test(net, X, y)

def sklearn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    model = MLPRegressor(
        hidden_layer_sizes=(25, 100),
        warm_start=True,
        early_stopping=True,
        learning_rate="adaptive",
        max_iter=99999,
        n_iter_no_change=100,
        verbose=True)

    model.fit(X_train, y_train)

    score = mean_squared_error(y_test, model.predict(X_test))

    print(f"Score: {score}")


if __name__ == "__main__":
    X, y, adj = make_data_cached("data", force_reload=False)
    #X = make_input("data/multiple")
    #y = make_output("data/multiple_output/")
    dcgn_model(X, y, adj)
    #keras_model(X, y)
    #sklearn_model(X, y)
    pdb.set_trace()
