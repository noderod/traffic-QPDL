#!/usr/bin/env python3
import json
import numpy as np
from glob import glob
from os import path

from preprocess import preprocess_x, preprocess_y, remap_nodes

import pdb

def load_file(filename):
    with open(filename) as fi:
        return json.load(fi)


def load_data(folder):
    data = []
    for filename in glob(path.join(folder, "*")):
        data.append(load_file(filename))

    return data

def preprocess_aadt(input_dict, node_maps, matrix=False, join=False):
    if matrix:
        return preprocess_x(input_dict, node_maps)[0]
    
    values = sorted(input_dict["AADT roads"].items(), key=lambda x: int(x[0]))

    means, sigmas = list(zip(*[(value["μ"], value["σ"]) for _,(value) in values]))

    if join:
        return np.array(means + sigmas)
    else:
        return np.array(means), np.array(sigmas)

def preprocess_calc_aadt(output_dict, node_maps, matrix=False):
    if matrix:
        return preprocess_y(output_dict, node_maps)


    values = sorted(output_dict["roads"].items(), key=lambda x: int(x[0]))

    trues = np.array([value["calculated AADT"] for _,value in values])
    return trues

def make_input(input_filepath, matrix=False):
    print("Loading input data...")
    input_files = glob(path.join(input_filepath, "*"))
    
    node_maps, adj = remap_nodes(load_file(input_files[0])["roads"])
    return node_maps, adj, np.array([preprocess_aadt(load_file(input_file), node_maps, matrix=matrix, join=True) for input_file in input_files])

def make_output(output_filepath, node_maps, matrix=False):
    print("Loading output data...")
    output_files = glob(path.join(output_filepath, "*"))
    return np.array([preprocess_calc_aadt(load_file(output_file), node_maps, matrix=matrix) for output_file in output_files])

def make_data_cached(folder, matrix=False, force_reload=False):
    matrix = True
    X_path = path.join(folder, "X.npy")
    y_path = path.join(folder, "y.npy")
    adj_path = path.join(folder, "adj.npy")

    if force_reload:
        node_maps, adj, X = make_input(path.join(folder, "multiple"), matrix=matrix)
        y = make_output(path.join(folder, "multiple_output"), node_maps, matrix=matrix)

        np.save(X_path, X)
        np.save(y_path, y)
        np.save(adj_path, np.array([adj]))

        return X, y, adj

    if path.exists(X_path) and path.exists(y_path) and path.exists(adj_path):
        print("Loading data from cache...")
        return np.load(X_path, allow_pickle=True), np.load(y_path, allow_pickle=True), np.load(adj_path, allow_pickle=True)[0]
    else:
        print("Loading...")
        return make_data_cached(folder, force_reload=True)
