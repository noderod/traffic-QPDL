#!/usr/bin/env python3
import json
import numpy as np
from glob import glob
from os import path
import os

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
    input_files = sorted(input_files, key=lambda x: int(path.basename(x)[path.basename(x).rfind("_")+1:-5]))
    node_maps, adj = remap_nodes(load_file(input_files[0])["roads"])
    return node_maps, input_files, adj, np.array([preprocess_aadt(load_file(input_file), node_maps, matrix=matrix, join=True) for input_file in input_files])

def make_output(output_filepath, node_maps, matrix=False):
    print("Loading output data...")
    output_files = glob(path.join(output_filepath, "*"))
    output_files = sorted(output_files, key=lambda x: int(path.basename(x)[path.basename(x).rfind("_")+1:-5]))
    return np.array([preprocess_calc_aadt(load_file(output_file), node_maps, matrix=matrix) for output_file in output_files])

def make_data_cached(folder, matrix=False, force_reload=False):
    matrix = True
    X_path = path.join(folder, "X.npy")
    y_path = path.join(folder, "y.npy")
    adj_path = path.join(folder, "adj.npy")
    nodemaps_path = path.join(folder, "node_maps.json")
    filenames_path = path.join(folder, "files.json")

    if force_reload:
        node_maps, filenames, adj, X = make_input(path.join(folder, "multiple"), matrix=matrix)
        y = make_output(path.join(folder, "multiple_output"), node_maps, matrix=matrix)

        np.save(X_path, X)
        np.save(y_path, y)
        np.save(adj_path, np.array([adj]))
        with open(nodemaps_path, "w") as out:
            json.dump(node_maps, out)
        with open(filenames_path, "w") as out:
            json.dump(filenames, out)

        return X, y, adj, node_maps, filenames

    if path.exists(X_path) and path.exists(y_path) and path.exists(adj_path) and path.exists(nodemaps_path) and path.exists(filenames_path):
        print("Loading data from cache...")
        with open(nodemaps_path, "r") as in_file:
            node_maps = json.load(in_file)
        with open(filenames_path, "r") as in_file:
            filenames = json.load(in_file)

        return (np.load(X_path, allow_pickle=True),
                np.load(y_path, allow_pickle=True),
                np.load(adj_path, allow_pickle=True)[0],
                node_maps, filenames)
    else:
        print("Loading...")
        return make_data_cached(folder, force_reload=True)

def invert_dict(dict_):
    inverted = {}
    for key in dict_:
        inverted[dict_[key]] = key

    return inverted

def output_results(folder, preds, trues, node_maps, filenames):
    input_filepath = path.join(folder, "multiple")
    sample_input = glob(path.join(input_filepath, "*"))[0]
    with open(sample_input, "r") as input_fi:
        data = json.load(input_fi)

    output_preds_folder = path.join(folder, "nn_preds")
    output_trues_folder = path.join(folder, "nn_truths")
    if not path.exists(output_preds_folder):
        os.mkdir(output_preds_folder)
    if not path.exists(output_trues_folder):
        os.mkdir(output_trues_folder)

    #unmaps = invert_dict(node_maps)
    counter = 0
    for pred, true in zip(preds, trues):
        filename = path.basename(filenames[counter])
        output_pred_path = path.join(output_preds_folder, filename)
        output_true_path = path.join(output_trues_folder, filename)
        out_pred_dict = {}

        out_pred_dict["roads"] = data["roads"]
        out_pred_dict["nodes"] = data["nodes"]
        out_true_dict = out_pred_dict.copy()
        out_pred_dict["AADT roads"] = dict()
        out_true_dict["AADT roads"] = dict()

        #true_true = preprocess_aadt(load_file(filenames[counter]), node_maps, matrix=True, join=True)

        sse = 0
        sum_ = 0
        for road,road_data in data["roads"].items():
            #out_dict["AADT_roads"][road]["μ"]
            x = node_maps[road_data['start node']]
            y = node_maps[road_data['end node']]
            out_pred_dict["AADT roads"][road] = dict()
            out_true_dict["AADT roads"][road] = dict()
            out_pred_dict["AADT roads"][road]["μ"] = float(pred[x,y])
            out_true_dict["AADT roads"][road]["μ"] = float(true[x,y])
            sse += (float(pred[x,y]) - float(true[x,y]))**2
            sum_ += float(true[x,y])
        #import pdb
        #pdb.set_trace()

        print(f"sse: {sse} | sum: {sum_}")
        with open(output_pred_path, "w") as out:
            json.dump(out_pred_dict, out, ensure_ascii=False, indent=2)
        with open(output_true_path, "w") as out:
            json.dump(out_true_dict, out, ensure_ascii=False, indent=2)
        print(f"File {counter} outputted")
        counter += 1
