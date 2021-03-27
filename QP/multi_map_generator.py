"""
SUMMARY

Generates as many copies as desired for an already processed road map in JSON form (preprocessing.py output).
"""

import argparse
import json
import os
import sys

import numpy as np


parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--preprocessed_input",required=True,  help="Filepath to preprocessed (JSON) input to be read", type=str)
required_flags.add_argument("--multiples",required=True,  help="Times to generate a new map, including the original file", type=int)
required_flags.add_argument("--output_dir",required=True,  help="Directory where to store the output (JSON) files, with the ending '/'", type=str)
args = parser.parse_args()



preprocessed_input_filepath = args.preprocessed_input
num_multiples = args.multiples
output_dir = args.output_dir



###########################
# INPUT HANDLING
###########################

# Loads data
with open(preprocessed_input_filepath, "r") as jf:
    original_data = json.load(jf)


# There must be at least one copycannot be 0 multiples
assert num_multiples >= 0, "There cannot be zero multiples of a file"


# If 0 copies, do nothing
if num_multiples == 0:
    sys.exit()


# Generates the output directory if it does not exist already
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


################################
# GENERATE THE NECESSARY COPIES
################################


# Gets the original name of the file, which is everything before the last .json
# Assumed that the file-name ends in .json
original_filename = preprocessed_input_filepath.split("/")[-1][:len(preprocessed_input_filepath) - 5]


# Generates the appropriate filepath, given an end number
def generate_filepath(index):
    return output_dir + original_filename + "_" + str(index) + ".json"



# Copies the original data
with open(generate_filepath(0), "w") as jf:
    jf.write(json.dumps(original_data, indent = 2))


# Generates as many other multiples as needed
original_AADT_data = original_data["AADT roads"]

AADT_road_IDs = list(original_AADT_data.keys())


for an_index in range(1, num_multiples):

    # Updates the AADT values
    updated_AADT = {}

    new_map_info = original_data


    for a_road_ID in AADT_road_IDs:

        original_AADT_road = original_AADT_data[a_road_ID]

        original_μ = original_AADT_road["μ"]
        original_σ = original_AADT_road["σ"]

        # Calculates a new mean as a random normal
        # But enforcing a minimum of 0
        new_AADT_μ = max(0, np.random.normal(original_μ, original_σ))

        updated_AADT[a_road_ID] = {"μ":new_AADT_μ, "σ":original_σ}

    new_map_info["AADT roads"] = updated_AADT

    # Writes the data to an output file
    with open(generate_filepath(an_index), "w") as jf:
        jf.write(json.dumps(new_map_info, indent = 2))


