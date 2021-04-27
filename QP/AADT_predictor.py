"""
SUMMARY

Attempts to predict the original AADT values from the results obtained from the quadratic optimizer.
Designed for batch use after executing ../generate_multiple_QP_solutions.sh
Expected to fail due to singular matrix, since all streets receive cars from others, making them linear combinations of one another
"""

import argparse
import json
import os

import numpy as np
import numpy.linalg as LA


parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--AADT-inputs",required=True,  help="Path to directory containing all AADT inputs (including the ending '/')", type=str)
required_flags.add_argument("--QP-outputs", required=True, help="Path to directory containing all QP outputs from the AADT inputs (including the ending '/')", type=str)
args = parser.parse_args()


AADT_inputs_dir = args.AADT_inputs
QP_outputs_dir = args.QP_outputs

# Gets the list of files to process, sorted to maintain order
AADT_files_to_process = sorted(os.listdir(AADT_inputs_dir))

# Ensures that the same file exists in both directories
for a_file in AADT_files_to_process:
    assert os.path.isfile(AADT_inputs_dir + a_file), a_file + "is not present in the QP outputs directory"


# Gets the number of samples (x), equal to the number of files
n_samples = len(AADT_files_to_process)

# Gets the number of features, roads
# Gets the number of output columns (AADT roads)
with open(AADT_inputs_dir + AADT_files_to_process[0], "r") as AADT_jf:
    first_AADT_data = json.load(AADT_jf)
    n_features = len(first_AADT_data["roads"])
    n_outputs = len(first_AADT_data["AADT roads"])
    observed_AADT_roads = list(first_AADT_data["AADT roads"].keys())


# Creates the X, Y matrices
X = np.zeros((n_samples, n_features))
Y = np.zeros((n_samples, n_outputs))


# Fills X, Y
sample_counter = 0
for a_file in AADT_files_to_process:

    AADT_filepath = AADT_inputs_dir + a_file
    QP_output_filepath = QP_outputs_dir + a_file

    with open(AADT_filepath, "r") as AADT_jf:
        AADT_data = json.load(AADT_jf)["AADT roads"]

    with open(QP_output_filepath, "r") as QP_jf:
        QP_output_data = json.load(QP_jf)["roads"]


    # X
    for a_road_ID in range(0, n_samples):

        a_road_ID_str = str(a_road_ID)
        X[sample_counter][a_road_ID] = QP_output_data[a_road_ID_str]["calculated AADT"]

    # Y
    for an_AADT_road_ID, an_observed_ADDT_road in zip(range(0, n_outputs), observed_AADT_roads):

        Y[sample_counter][an_AADT_road_ID] = AADT_data[an_observed_ADDT_road]["μ"]


    sample_counter += 0




# Solves for the outputs
X_T = X.transpose()

first = LA.inv(np.dot(X_T, X))
second = np.dot(X_T, Y)

W = np.dot(first, second)


# Calculates the prediction
predicted_Y = np.dot(X, W)


# Calculates the MSE
sq_error_sum = 0

for row in range(0, n_samples):
    for col in range(0, n_outputs):
        sq_error_sum += (predicted_Y[row][col] - Y[row][col])**2



MSE = sq_error_sum/(n_samples*n_outputs)

print("MSE = " + str(MSE))
