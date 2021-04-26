#!/usr/bin/env python3
#
import numpy as np
from scipy import sparse
import pdb

def remap_nodes(road_info):
    counter = 0
    remap_dict = {}
    for road in road_info:
        #pdb.set_trace()
        start_node = road_info[road]["start node"]
        end_node = road_info[road]["end node"]
        if start_node not in remap_dict:
            remap_dict[start_node] = counter
            counter += 1
        if end_node not in remap_dict:
            remap_dict[end_node] = counter
            counter += 1

    num_roads = len(remap_dict)
    adj = np.matrix(np.zeros((num_roads, num_roads)), dtype=np.float32)
    for road in road_info:
        start_node = remap_dict[road_info[road]["start node"]]
        end_node = remap_dict[road_info[road]["end node"]]
        if adj[start_node,end_node] == 1:
            pass
        adj[start_node,end_node] = 1
    
    return remap_dict, sparse.csr_matrix(adj)

def to_adjacency(road_info, AADT_info, node_maps=None, reduce_mat=False, sparse_mat=False):
    if type(node_maps) == type(None):
        print("Warning: regenerating node map")
        node_maps, _ = remap_nodes(road_info)
        
    num_roads = len(node_maps)

    adj_mu = np.matrix(np.zeros((num_roads, num_roads)), dtype=np.float32)
    adj_sig = np.matrix(np.zeros((num_roads, num_roads)), dtype=np.float32)

    for aadt_road in AADT_info:
        from_road = road_info[aadt_road]["start node"]
        to_road = road_info[aadt_road]["end node"]

        from_road = node_maps[from_road]
        to_road = node_maps[to_road]

        adj_mu[from_road,to_road] = AADT_info[aadt_road]['μ']
        adj_sig[from_road,to_road] = AADT_info[aadt_road]['σ']

    if reduce_mat:
        adj_mu = adj_mu[:, np.array(np.sum(adj_mu, axis=0)>0)[0]]
        adj_mu = adj_mu[np.array(np.sum(adj_mu, axis=1)>0)[:,0],:]
        adj_sig = adj_sig[:, np.array(np.sum(adj_sig, axis=0)>0)[0]]
        adj_sig = adj_sig[np.array(np.sum(adj_sig, axis=1)>0)[:,0],:]
    if sparse_mat:
        adj_mu = sparse.csr_matrix(adj_mu)
        adj_sig = sparse.csr_matrix(adj_sig)

    return adj_mu, adj_sig

def preprocess_x(x, node_maps):
    # Roads
    road_info = x["roads"]

    # AADT values
    AADT_info = x["AADT roads"]

    mus, sigs = to_adjacency(road_info, AADT_info, node_maps=node_maps, reduce_mat=False, sparse_mat=True)

    return mus, sigs

def preprocess_y(road_info, node_maps):
    AADT_info = {}
    for road in road_info["roads"]:
        AADT_info[road] = {
            "μ": road_info["roads"][road]["calculated AADT"],
            "σ": -1,
        }
    
    mus, _ = to_adjacency(road_info["roads"], AADT_info, node_maps=node_maps, reduce_mat=False, sparse_mat=True)
    return mus

def preprocess_x2(x):
    # Gets and stores node values
    node_info = x["nodes"]

    # Roads
    road_info = x["roads"]

    # AADT values
    AADT_info = x["AADT roads"]


    # Intersection node and which roads enter and exit
    # {node_ID:{"in":[road ID 1, ...], "out":[road ID 1, ...]}, ...}
    intersection_topology = {}
    num_junctions = len(list(node_info.keys()))

    # Gets a list of all roads, sorted so that the order is repeatable
    road_IDs = sorted(list(road_info.keys()))

    # Assigns each road ID a number ID starting at zero, and viceversa
    road_ID_to_num = {}
    num_to_road_ID = {}

    lr = len(road_IDs)

    for i in range(0, lr):

        current_road_ID = road_IDs[i]

        road_ID_to_num[current_road_ID] = i
        num_to_road_ID[i] = current_road_ID

        # Handles the intersections
        current_road_info = road_info[current_road_ID]
        road_start_node = current_road_info["start node"]
        road_end_node = current_road_info["end node"]


        # Exiting intersection for start
        if road_start_node not in intersection_topology:
            intersection_topology[road_start_node] = {"in":[], "out":[current_road_ID]}
        else:
            intersection_topology[road_start_node]["out"].append(current_road_ID)

        if road_end_node not in intersection_topology:
            intersection_topology[road_end_node] = {"in":[current_road_ID], "out":[]}
        else:
            intersection_topology[road_end_node]["in"].append(current_road_ID)
        

    junction_nodes_ordered = sorted(list(intersection_topology.keys()))

    ###########################
    # MATRIX GENERATION
    ###########################

    # -> A matrix, b vector


    # Start with all inputs being zero
    A = np.zeros((lr, lr))
    b = np.zeros((lr))


    # Keeps track of the max AADT observed
    observed_max_ADDT = -10**8

    for an_AADT_road in AADT_info:
        pdb.set_trace()

        road_index = road_ID_to_num[an_AADT_road]

        AADT_road_info = AADT_info[an_AADT_road]

        AADT_μ = AADT_road_info["μ"]
        AADT_σ = AADT_road_info["σ"]

        A[road_index][road_index] = 1/(AADT_σ**2)
        b[road_index] = -AADT_μ/(AADT_σ**2)

        # Updates the maximum observed AADT
        observed_max_ADDT = max(observed_max_ADDT, AADT_μ)

    return sparse.csc_matrix(A)

