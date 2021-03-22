"""
SUMMARY

Quadratic Programming optimizer.
It outputs a JSON file containing a list of nodes and roads. Each road will have a calculated AADT value.
"""


import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import osqp
from scipy import sparse

import aux_preprocessing as aux


parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--input",required=True,  help="Filepath to preprocessing output filepath, taken as input to be read", type=str)
required_flags.add_argument("--output", required=True, help="Filepath to output JSON", type=str)
parser.add_argument("--show", help="Show nodes and ways in a map", action="store_true")
parser.add_argument("--verbose", help="Show nodes and ways in a map", action="store_true")
args = parser.parse_args()


# Verbose
verbosity = args.verbose



###########################
# INPUT HANDLING
###########################


# Reads JSON data into dict
with open(args.input, "r") as jf:
    road_raw = json.load(jf)


# Gets and stores node values
node_info = road_raw["nodes"]

# Roads
road_info = road_raw["roads"]

# AADT values
AADT_info = road_raw["AADT roads"]


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


# Fills the matrix
for an_AADT_road in AADT_info:

    road_index = road_ID_to_num[an_AADT_road]

    AADT_road_info = AADT_info[an_AADT_road]

    AADT_μ = AADT_road_info["μ"]
    AADT_σ = AADT_road_info["σ"]

    A[road_index][road_index] = 1/(AADT_σ**2)
    b[road_index] = -AADT_μ/(AADT_σ**2)

    # Updates the maximum observed AADT
    observed_max_ADDT = max(observed_max_ADDT, AADT_μ)

A = sparse.csc_matrix(A)
b = np.array(b)


# -> l vector, always zero
l = np.zeros((lr + num_junctions))


# -> u vector, always zero except for the first entrances, which are the number of roads
u = np.zeros((lr + num_junctions))

twice_observed_max_ADDT = 2*observed_max_ADDT

for r in range(0, lr):
    u[r] = twice_observed_max_ADDT


# -> C matrix

C = np.zeros((lr + num_junctions, lr))

# Traffic intervals
for r in range(0, lr):
    C[r][r] = 1

# Junctions
for j in range(lr, lr + num_junctions):

    junction_node = junction_nodes_ordered[j - lr]
    junction_roads = intersection_topology[junction_node]
    roads_IDs_in = [road_ID_to_num[a_road] for a_road in junction_roads["in"]]
    roads_IDs_out = [road_ID_to_num[a_road] for a_road in junction_roads["out"]]

    # 1 if incoming road
    for a_road_ID_in in roads_IDs_in:
        C[j][a_road_ID_in] = 1

    # -1 if exiting road
    for a_road_ID_out in roads_IDs_out:
        C[j][roads_IDs_out] = -1

C = sparse.csc_matrix(C)


###########################
# QP OPTIMIZATION
###########################

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(A, b, C, l, u, alpha=1.0)

# Solve problem
res = prob.solve()
AADT_solutions = res.x



###########################
# OUTPUT
###########################

# Assigns each solution to its corresponding road
for r in range(0, lr):
    road_ID = num_to_road_ID[r]
    road_info[road_ID]["calculated AADT"] = AADT_solutions[r]


output_dict = {
    "nodes":node_info,
    "roads":road_info
}


# Writes the output to the specified JSON filepath
with open(args.output, "w") as jf:
    jf.write(json.dumps(output_dict, indent = 2))



###########################
# SHOW AADT MAP
###########################

# Shows the output map if needed
if not verbosity:
    sys.exit()

