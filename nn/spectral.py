#!/usr/bin/env python3

import pdb
from pprint import pprint as pp
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import click
from os import path

from data import make_data_cached, output_results

import dmlc
import dense
import torch.nn.functional as F
import dgl

import pandas as pd

def dcgn_model(X_train, X_test, y_train, y_test, adj, epochs=2):
    model = dmlc.MWE_DGCN(1, 100, X_train[0].iloc[0].shape[1], 3, F.relu, 0)
    model, pred_test, true_test = dmlc.train_test(model, X_train[0], X_test[0], y_train[0], y_test[0], adj, epochs=epochs)

    return model, pred_test, true_test

def dense_model(X_train, X_test, y_train, y_test, adj, epochs=2):
    net = dense.DenseNet()
    return dense.train_test(net, X_train[0], X_test[0], y_train[0], y_test[0], adj)

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


@click.command()
@click.option('--data-folder', help='Path to QP input/output', default="data")
@click.option('--epochs', help="Number of epochs to run", default=5)
def main(data_folder, epochs):
    X, y, adj, node_maps,files = make_data_cached(data_folder, force_reload=False)
    files = np.array(files)
    print(X[0].shape)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=16)
    print(X_test.index)
    if 0 not in X_test.index:
        import sys
        sys.exit(0)
    #model, pred_test, true_test = dense_model(X_train, X_test, y_train, y_test, adj, epochs=epochs)
    model, pred_test, true_test = dcgn_model(X_train, X_test, y_train, y_test, adj, epochs=epochs)

    output_results(data_folder, pred_test, true_test, node_maps, files[X_test.index])
    pdb.set_trace()


if __name__ == "__main__":
    main()
