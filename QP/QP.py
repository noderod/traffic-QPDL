"""
SUMMARY

Quadratic Programming optimizer.
It outputs a JSON file containing a list of nodes and roads. Each road will have a calculated AADT value.
"""


import argparse
import json
import pickle
import sys

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import osqp
from scipy import sparse

import aux_preprocessing as aux


parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--input",required=True,  help="Filepath to preprocessing output filepath, taken as input to be read", type=str)
required_flags.add_argument("--output", required=True, help="Filepath to output JSON", type=str)
required_flags.add_argument("--QP-matrices-outputs", help="Pickle filepath to output the QP matrices (A, b, l, C, u) as outputs", type=str)
parser.add_argument("--show", help="Show nodes and ways in a map (not designed for use within a batch solve via generate_multiple_QP_solutions.sh)", action="store_true")
parser.add_argument("--verbose", help="OSQP solver verbosity, notes when the solver returns a negative value", action="store_true")
args = parser.parse_args()


# Pickle output file
matrices_qp_pickle_location = args.QP_matrices_outputs

# Verbose
verbosity = args.verbose
show_map = args.show



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
prob.setup(A, b, C, l, u, alpha=1.0, verbose=verbosity)

# Solve problem
res = prob.solve()
AADT_solutions = res.x


# Stores the matrices if needed
# https://wiki.python.org/moin/UsingPickle
if (matrices_qp_pickle_location != None) and (matrices_qp_pickle_location != ""):
    pickle.dump({"A":A, "b":b, "l":l, "C":C, "u":u}, open(matrices_qp_pickle_location, "wb"))


# If an AADT value is lower than zero for any value, correct it by minimizing the squared error
found_negative = False
for a_road_traffic in AADT_solutions:
    if a_road_traffic < 0:
        found_negative = True
        break


if found_negative:

    if verbosity:
        print("A negative AADT value was found for at least one road, correcting it by attempting to minimize SSE under the same junction constraints")


    # Attempts to minimze the SSE with a second quadratic optimizer

    # Attempt to minimize \sum (t_r1 - t_r)**2, where t_r is a constant

    # -> b vector (always zero)
    b = np.zeros((lr))

    for r in range(0, lr):
        b[r] = -2*AADT_solutions[r]


    # -> A matrix

    # Start with all inputs being zero
    A = np.zeros((lr, lr))

    for r in range(0, lr):
        # 2 because OSQP minimizes 1/2 x^T A x + bx
        A[r][r] = 2
    A = sparse.csc_matrix(A)


    # -> l vector, u vector, C matrix
    # No change


    SSE_minimzer = osqp.OSQP()

    # Setup workspace and change alpha parameter
    SSE_minimzer.setup(A, b, C, l, u, alpha=1.0, verbose=verbosity)

    # Solve problem
    res = SSE_minimzer.solve()
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
if not show_map:
    sys.exit()


# Min, max AADT values found
AADT_min = min(AADT_solutions)
AADT_max = max(AADT_solutions)


# Colors obtained from https://www.rapidtables.com/web/color/RGB_Color.html
# From lowest AADT to highest AADT
utilized_colors = [
    [0, 0, 255/255],
    [0, 128/255, 255/255],
    [0, 255/255, 255/255],
    [0, 255/255, 128/255],
    [0, 255/255, 0],
    [128/255, 255/255, 0],
    [255/255, 255/255, 0],
    [255/255, 128/255, 0],
    [255/255, 0, 0]
]


# Ordered AADT values
AADT_solutions_sorted = sorted(AADT_solutions)


# Corresponding AADT values
AADT_color_limits = []
# There must be the same number of AADT separator values as colors
luc_1 = len(utilized_colors) - 1
l_AADT = len(AADT_solutions_sorted)

for qq in range(0, luc_1):
    AADT_color_limits.append(AADT_solutions_sorted[int(l_AADT*qq/luc_1)])
else:
    AADT_color_limits.append(AADT_max)



plt.figure()


# Checks each road for its coordinates
for r in range(0, lr):
    road_ID = num_to_road_ID[r]

    current_road_info = road_info[road_ID]
    road_start_node = current_road_info["start node"]
    road_end_node = current_road_info["end node"]

    # Gets coordinates of the start and end nodes
    start_x = node_info[road_start_node]["lon"]
    start_y = node_info[road_start_node]["lat"]

    end_x = node_info[road_end_node]["lon"]
    end_y = node_info[road_end_node]["lat"]

    calculated_AADT = AADT_solutions[r]

    # Finds the location
    # Cannot be larger 
    location = np.searchsorted(AADT_color_limits, calculated_AADT)

    # Assigns the appropriate road color
    r_equivalent = aux.interpolate(calculated_AADT, utilized_colors[location - 1][0], utilized_colors[location][0], AADT_color_limits[location - 1], AADT_color_limits[location])

    g_equivalent = aux.interpolate(calculated_AADT, utilized_colors[location - 1][1], utilized_colors[location][1], AADT_color_limits[location - 1], AADT_color_limits[location])
    b_equivalent = aux.interpolate(calculated_AADT, utilized_colors[location - 1][2], utilized_colors[location][2], AADT_color_limits[location - 1], AADT_color_limits[location])

    plt.plot([start_x, end_x], [start_y, end_y], ls="-", lw=3, color=[r_equivalent, g_equivalent, b_equivalent])


# Sets the axes and bounds
bound_lat_min, bound_lat_max, bound_lon_min, bound_lon_max = road_raw["bounds"]
plt.xlim(bound_lon_min, bound_lon_max)
plt.ylim(bound_lat_min, bound_lat_max)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Shows the colormap
cmap11 = LinearSegmentedColormap.from_list('custom', utilized_colors)
img = plt.imshow(np.array([[0, AADT_max]]), cmap=cmap11) # Dummy image that is not plotted
img.set_visible(False)
utilized_colorbar = plt.colorbar(orientation="vertical", label = 'AADT (calculated), scale is linear between each mark', ticks = np.linspace(AADT_min, AADT_max, len(utilized_colors)))
utilized_colorbar.set_ticklabels(AADT_color_limits)

plt.title("AADT calculated per road")

plt.show()
