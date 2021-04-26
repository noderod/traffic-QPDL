"""
SUMMARY

Enforces constraints on the results obtained from the neural network
"""

import argparse
import copy
import json
import pickle
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import osqp
from scipy import sparse

import aux_preprocessing as aux


parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--NN-outputs",required=True,  help="Path to directory containing all Neural Network outputs (including the ending '/')", type=str)
required_flags.add_argument("--QP-outputs", required=True, help="Path to directory containing all QP outputs (including the ending '/')", type=str)
required_flags.add_argument("--QP-matrices", help="Pickle filepath to storing the QP matrices (A, b, l, C, u) as outputs", type=str)
required_flags.add_argument("--constrained-outputs", required=True, help="Path to directory to store the constrained outputs (including the ending '/')", type=str)
parser.add_argument("--mse", help="Show the MSE", action="store_true")
parser.add_argument("--show", help="Show nodes and ways in a map for the first result", action="store_true")
args = parser.parse_args()


NN_outputs_dir = args.NN_outputs
QP_outputs_dir = args.QP_outputs
constrained_outputs_dir = args.constrained_outputs


# Creates a constrained output directory if it does not exist already
if not os.path.isdir(constrained_outputs_dir):
    os.mkdir(constrained_outputs_dir)


QP_matrices_filepath = args.QP_matrices



#######################################
# INPUT HANDLING
#######################################

# Gets the list of files to process, sorted to maintain order
NN_files_to_process = sorted(os.listdir(NN_outputs_dir))

# Ensures that the same file exists in both directories
for a_file in NN_files_to_process:
    assert os.path.isfile(QP_outputs_dir + a_file), a_file + "is not present in the QP outputs directory"


# Gets the number of samples (x), equal to the number of files
n_samples = len(NN_files_to_process)


# Gets the number of roads
with open(NN_outputs_dir + NN_files_to_process[0], "r") as NN_jf:
    first_NN_data = json.load(NN_jf)
    n_roads = len(first_NN_data["roads"])


# Retrives the QP matrices
# https://wiki.python.org/moin/UsingPickle
retrieved_matrices = pickle.load(open(QP_matrices_filepath, "rb" ))

# Only the l, C, u matrices are useful
l = retrieved_matrices["l"]
C = retrieved_matrices["C"]
u = retrieved_matrices["u"]



#######################################
# OPTIMIZATION
#######################################

# Goes through each file
for a_file in NN_files_to_process:

    NN_filepath = NN_outputs_dir + a_file
    QP_filepath = QP_outputs_dir + a_file
    constrained_filepath = constrained_outputs_dir + a_file

    # Data always stored as:
    # [μ0, μ1, ...]
    # Where each index is the corresponding calculated AADT value
    NN_AADT = np.zeros((n_roads))
    QP_AADT = np.zeros((n_roads))

    # Loads the NN outputs
    with open(NN_filepath, "r") as NN_jf:
        NN_data = json.load(NN_jf)["AADT roads"]


    # Loads the QP outputs
    with open(QP_filepath, "r") as QP_jf:
        QP_complete_data = json.load(QP_jf)
        QP_data = QP_complete_data["roads"]
        QP_nodes = QP_complete_data["nodes"]

    # Stores the outputs into the corresponding array
    for a_road_ID in range(0, n_roads):
        a_road_ID_str = str(a_road_ID)
        NN_AADT[a_road_ID] = NN_data[a_road_ID_str]["μ"]
        QP_AADT[a_road_ID] = QP_data[a_road_ID_str]["calculated AADT"]


    # Creates the A, b QP matrices
    # Attempt to minimize \sum (t_r1 - t_r)**2, where t_r is a constant

    # -> b vector
    b = np.zeros((n_roads))

    for r in range(0, n_roads):
        b[r] = -2*NN_AADT[r]


    # -> A matrix

    # Start with all inputs being zero
    A = np.zeros((n_roads, n_roads))

    for r in range(0, n_roads):
        # 2 because OSQP minimizes 1/2 x^T A x + bx
        A[r][r] = 2
    A = sparse.csc_matrix(A)


    # Solves the QP
    SSE_minimzer = osqp.OSQP()

    # Setup workspace and change alpha parameter
    SSE_minimzer.setup(A, b, C, l, u, alpha=1.0, verbose=False)

    # Solve problem
    res = SSE_minimzer.solve()
    constrained_AADT_solutions = res.x


    # If there is a single negative result, run it again
    found_negative = False
    for a_road_traffic in constrained_AADT_solutions:
        if a_road_traffic < 0:
            found_negative = True
            break

    if found_negative:

        # Same A as before

        # Updated with the obtained values
        b = np.zeros((n_roads))

        for r in range(0, n_roads):
            b[r] = -2*constrained_AADT_solutions[r]

        second_SSE_minimzer = osqp.OSQP()

        # Setup workspace and change alpha parameter
        second_SSE_minimzer.setup(A, b, C, l, u, alpha=1.0, verbose=False)

        # Solve problem
        res = second_SSE_minimzer.solve()
        constrained_AADT_solutions = res.x



    # Saves the results to a format identical to the quadratic optimizer
    constrained_outputs = copy.deepcopy(QP_data)

    for r in range(0, n_roads):
        constrained_outputs[str(r)]["calculated AADT"] = constrained_AADT_solutions[r]

    # Stores the results
    with open(constrained_filepath, "w") as con_jf:
        con_jf.write(json.dumps({"roads":constrained_outputs, "nodes":QP_nodes}, indent = 2))



#######################################
# SHOWS MSE IF REQUESTED
#######################################
if args.mse:

    total_sq_sum = 0

    for a_file in NN_files_to_process:
        QP_filepath = QP_outputs_dir + a_file
        constrained_filepath = constrained_outputs_dir + a_file

        # Loads the info for both
        with open(QP_filepath, "r") as QP_jf:
            QP_data = json.load(QP_jf)["roads"]

        with open(constrained_filepath, "r") as con_jf:
            con_data = json.load(con_jf)["roads"]

        # Adds their squared difference
        for r in range(0, n_roads):
            r_str = str(r)
            total_sq_sum += (con_data[r_str]["calculated AADT"] - QP_data[r_str]["calculated AADT"])**2


    MSE = total_sq_sum/(n_roads*len(NN_files_to_process))
    RMS = MSE**0.5
    print("MSE = " + str(MSE))
    print("RMS = " + str(RMS))





#######################################
# PLOTS THE FIRST RESULT IF REQUESTED
#######################################
if not args.show:
    sys.exit()

with open(constrained_outputs_dir + NN_files_to_process[0], "r") as jf:
    first_constrained_data = json.load(jf)

node_info = first_constrained_data["nodes"]
road_info = first_constrained_data["roads"]

# Gets the AADT values
AADT_solutions = np.zeros((n_roads))

for r in range(0, n_roads):
    AADT_solutions[r] = road_info[str(r)]["calculated AADT"]


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
for r in range(0, n_roads):
    road_ID = str(r)

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
# Fixed in this case, since only Niles, MI was used
bound_lat_min, bound_lat_max, bound_lon_min, bound_lon_max = [
    41.7903,
    41.8751,
    -86.3211,
    -86.1709
  ]

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
