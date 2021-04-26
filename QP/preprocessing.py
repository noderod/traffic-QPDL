"""
SUMMARY

Preprocesses an OpenStreetMap and AADT data into a summary JSON file to be used by the quadratic optimizer as input.
"""


import argparse
import json
import sys
from xml.dom import minidom

import matplotlib.pyplot as plt

import aux_preprocessing as aux



# Obtained from another project assignment
parser = argparse.ArgumentParser()
required_flags = parser.add_argument_group(title="Required")
required_flags.add_argument("--osm",required=True,  help="Filepath to OSM (XML) input to be read", type=str)
required_flags.add_argument("--aadt",required=True,  help="Filepath to AADT (JSON) input to be read", type=str)
required_flags.add_argument("--output", required=True, help="Filepath to output JSON", type=str)
parser.add_argument("--show", help="Show nodes and ways in a map", action="store_true")
parser.add_argument("--verbose", help="Show counts of nodes and ways", action="store_true")
args = parser.parse_args()


# Verbose
verbosity = args.verbose



# Reads XML file
osm_data = minidom.parse(args.osm)


# Gets the bounds
bounds_provided = osm_data.getElementsByTagName("bounds")

for bounds_xml in bounds_provided:
    bound_lat_min = float(bounds_xml.attributes["minlat"].value)
    bound_lat_max = float(bounds_xml.attributes["maxlat"].value)
    bound_lon_min = float(bounds_xml.attributes["minlon"].value)
    bound_lon_max = float(bounds_xml.attributes["maxlon"].value)





# Gets the list of nodes
nodes_provided = osm_data.getElementsByTagName("node")

# Enters the node information
nodes_latlon = {}

for a_node in nodes_provided:
    nodes_latlon[a_node.attributes["id"].value] = {"lat":float(a_node.attributes["lat"].value), "lon":float(a_node.attributes["lon"].value)}


# Gets the list of ways
ways_provided = osm_data.getElementsByTagName("way")

# Nodes viewed {"node id":[way id 1, way id 2]
nodes_used = {}


# Ways: {"wid":{"nodes":[]}}
ways_roads = {}

# Only roads are considered
invalid_ways = {
    "railway":True,
    "cycleway":True,
    "footway":True,
    "aeroway":True,
    "waterway":True,
    "amenity":True,
    "leisure":True,
    "building":True
}

# Invalid services
invalid_services = {
    "driveway":True,
    "parking_aisle":True
}





for a_way in ways_provided:

    wid_osm = a_way.attributes["id"].value

    # Checks if one way
    tags_in_way = a_way.getElementsByTagName("tag")
    way_is_twoway = True

    # Only roads are considered
    way_not_meant_for_cars = False
    way_is_not_highway = True

    # Specialized conditions for non highways
    way_is_short_range = True


    for an_extra_tag in tags_in_way:

        k = an_extra_tag.attributes["k"].value.lower()

        if k == "highway":
            way_is_not_highway = False
            way_not_meant_for_cars = False


        if way_is_not_highway and (k in invalid_ways):
            way_not_meant_for_cars = True
            continue

        # Skips driveways
        if k == "service":
            v = an_extra_tag.attributes["v"].value

            if v in invalid_services:
                way_not_meant_for_cars = True
                break

        if k == "oneway":
            way_is_twoway = False


    # Avoids non-roads
    if way_not_meant_for_cars:
        continue


    # If two-way, create two roads
    wid_0 = wid_osm + "-0"
    wid_1 = wid_osm + "-1"


    # Gets the nodes
    nodes_in_way = a_way.getElementsByTagName("nd")

    nodes_within = []
    num_nodes_in_way = 0

    for a_node_in_way in nodes_in_way:

        node_id = a_node_in_way.attributes["ref"].value

        # If the node was not seen before, ignore
        if node_id not in nodes_latlon:
            continue

        nodes_within.append(node_id)
        num_nodes_in_way += 1

        # If the node has not been seen yet
        if node_id not in nodes_used:
            nodes_used[node_id] = []


    if num_nodes_in_way == 0:
        continue

    # Processes the nodes for a one way

    # First node has the way leaving from it
    nodes_used[nodes_within[0]].append(["out", wid_0])

    # All other nodes except the last one are entered and exited by the road
    for a_node_within in nodes_within[1:(num_nodes_in_way-1)]:
        nodes_used[a_node_within].append(["in", wid_0])
        nodes_used[a_node_within].append(["out", wid_0])

    # Last node is only entered by the road
    nodes_used[nodes_within[num_nodes_in_way-1]].append(["in", wid_0])

    ways_roads[wid_0] = {"nodes":nodes_within}


    # If the road is two way, do the same as above, but in the opposite direction
    if way_is_twoway:

        # Reverses the node order
        nodes_within = nodes_within[::-1]

        # First node has the way leaving from it
        nodes_used[nodes_within[0]].append(["out", wid_1])

        # All other nodes except the last one are entered and exited by the road
        for a_node_within in nodes_within[1:(num_nodes_in_way-1)]:
            nodes_used[a_node_within].append(["in", wid_1])
            nodes_used[a_node_within].append(["out", wid_1])

        # Last node is only entered by the road
        nodes_used[nodes_within[num_nodes_in_way-1]].append(["in", wid_1])

        ways_roads[wid_1] = {"nodes":nodes_within}



# Shows the number of nodes and ways
if verbosity:
    print("Nodes: %d" % (len(nodes_latlon.keys()), ))
    print("Ways:  %d" % (len(ways_roads.keys()), ))
    print()



# Keeps track of which nodes are never used
nodes_to_be_disregarded = []

for a_provided_node in nodes_latlon:
    if a_provided_node not in nodes_used:
        nodes_to_be_disregarded.append(a_provided_node)


for node_to_be_deleted in nodes_to_be_disregarded:
    nodes_latlon.pop(node_to_be_deleted)


if verbosity:
    print("Remaining nodes: %d" % (len(nodes_latlon.keys()), ))
    print("Deleted nodes:   %d" % (len(nodes_to_be_disregarded), ))
    print()




###########################
# PROCESSING INTERSECTIONS
###########################


# An intersection is a node which is not merely a road
# ["node_id", ...]

valid_intersections = {}


for a_node in nodes_latlon:

    # Gets the ways passing through this node
    ways_passing_through = nodes_used[a_node]
    num_ways_passing_through = len(ways_passing_through)


    # Removes any intersection with zero or just one way, since there would not be any traffic going through it
    if num_ways_passing_through < 2:
        continue

    if num_ways_passing_through == 2:
        # Close any intersection with 2 ways with both of them being the same
        if ways_passing_through[0][1] == ways_passing_through[1][1]:
            continue
        # Closes any intersection acting as a source or a sink
        if ways_passing_through[0][0] == ways_passing_through[1][0]:
            continue

    # Close any intersection with 4 ways, 2 of them being the same just going in opposing directions
    #"""
    if num_ways_passing_through == 4:

        # Gets the unique ways
        unique_ways = list(set([w[1] for w in ways_passing_through]))

        if len(unique_ways) == 2:
            if unique_ways[0].split("-")[0] == unique_ways[1].split("-")[0]:
                continue
    #"""


    # Valid intersection
    valid_intersections[a_node] = nodes_latlon[a_node]


if verbosity:
    print("Valid intersections:   %d" % (len(valid_intersections), ))
    print()



###########################
# NORMALIZING LAT, LON
###########################

# Obtains min, max lat, lot

lat_min, lon_min =  10000000000,  10000000000
lat_max, lon_max = -10000000000, -10000000000

for a_node in valid_intersections:

    lat = valid_intersections[a_node]["lat"]
    lon = valid_intersections[a_node]["lon"]

    lat_min = min(lat_min, lat)
    lat_max = max(lat_max, lat)

    lon_min = min(lon_min, lon)
    lon_max = max(lon_max, lon)



# Normalizes each factor

lat_denominator = lat_max - lat_min
lon_denominator = lon_max - lon_min

for a_node in valid_intersections:

    lat = valid_intersections[a_node]["lat"]
    lon = valid_intersections[a_node]["lon"]

    valid_intersections[a_node]["x"] = (lon - lon_min)/lon_denominator
    valid_intersections[a_node]["y"] = (lat - lat_min)/lat_denominator



###########################
# PROCESSING ROADS
###########################

# Each road stores:
# {"road id":{"start node": intersection node ID, "end node": intersection node ID, "original way ID"}, ...}
roads = {}


# Easy node-node road check
# {"node id -node id":True, ...}
node_node_edges = {}



current_way_id = 0


# For each valid intersection node
for a_node in valid_intersections:

    # Gets the way IDs passing through this node
    ways_passing_through = [w[1] for w in nodes_used[a_node]]

    # Checks each way
    for a_way_passing_through in ways_passing_through:

        # Gets the nodes from this way
        nodes_in_way = ways_roads[a_way_passing_through]["nodes"]
        num_nodes_in_way = len(nodes_in_way)

        # Goes for each node until it finds the current node
        # Excludes the last node because a way needs at least two valid intersections
        not_found_intersection_before_last = True
        for node_index in range(0, num_nodes_in_way - 1):

            if nodes_in_way[node_index] == a_node:
                current_intersection_index = node_index
                not_found_intersection_before_last = False
                break

        # If the intersection were the last point, skip this way
        if not_found_intersection_before_last:
            continue

        # Finds the first node on the list
        for node_index in range(current_intersection_index + 1, num_nodes_in_way):

            # Ignores node if it is the current node. I.e. no self-pointing roads
            if nodes_in_way[node_index] == a_node:
                continue


            # If this node is a valid intersection, stop the road here
            if nodes_in_way[node_index] in valid_intersections:
                roads[current_way_id] = {"start node":a_node, "end node":nodes_in_way[node_index], "original way ID":a_way_passing_through}
                node_node_edges[a_node + "-" + nodes_in_way[node_index]] = current_way_id

                # Increase road counnter
                current_way_id += 1
                break




if verbosity:
    print("Roads:   %d" % (len(roads.keys()), ))
    print()



###########################
# AADT ASSOCIATIONS
###########################


# Reads JSON data into dict
with open(args.aadt, "r") as jf:
    AADT_raw = json.load(jf)


# Gets the actual AADT values
AADT_provided = AADT_raw["AADT"]


if verbosity:
    print("AADT values provided:   %d" % (len(AADT_provided), ))
    print()



# Calculates the middle point of each road, radius from middle-length, and x vector representing it
# {"road ID":{"x_m":x_m, "y_m":y_m, "r**2":r**2, "w_0":w_0, "w_1":w_1, "w_2":w_2, "||w||**2":||w||**2}
road_aux = {}

for a_road_id in roads:
    aux.calculate_road_secondary(a_road_id, roads, road_aux, valid_intersections)


# Assigns each AADT value to a road
# {"Road ID":{"μ":μ, "σ":σ}}
road_AADT = {}


road_IDs = [a_road_ID for a_road_ID in roads]


# Goes through each AADT value, assigning it to the road which is the closest
# Assumed at most one AADT value per road, if more that one is provided, it overrides the previous one
for AADT_info in AADT_provided:

    lat, lon = AADT_info["latlon"]
    μ, σ = AADT_info["AADT μ"], AADT_info["AADT σ"]

    # Normalizes lat, lon values
    xlon = (lon - lon_min)/lon_denominator
    ylat = (lat - lat_min)/lat_denominator


    distance_sq_min_so_far = 10**6

    P = [xlon, ylat]


    # Goes through each road
    for a_road_id in road_IDs:

        distance_sq = aux.calculate_distance_to_road(xlon, ylat, a_road_id, road_aux)

        if distance_sq <= distance_sq_min_so_far:
            distance_sq_min_so_far = distance_sq
            closest_road_ID = a_road_id

    # Marks this road
    road_AADT[closest_road_ID] = {"μ":μ, "σ":σ}


original_AADT_roads = list(road_AADT.keys())



# Handles two-way roads, also assigning the same AADT to the oppsoite direction
for an_AADT_road in original_AADT_roads:

    road_info = roads[an_AADT_road]
    start_node, end_node = road_info["start node"], road_info["end node"]

    # Finds if there is a road which goes the opposite way
    if (end_node + "-" + start_node) in node_node_edges:
        opposite_road_ID = node_node_edges[end_node + "-" + start_node]
        AADT_info = road_AADT[an_AADT_road]
        road_AADT[opposite_road_ID] = AADT_info




if verbosity:
    print("Roads with AADT:       %d" % (len(road_AADT.keys()), ))
    print("Roads with AADT (%%):   %f%%" % (100*len(road_AADT.keys())/len(roads.keys()), ))
    print()



###########################
# STORES THE OUTPUT AS A JSON FILE
###########################

output_dict = {
    "nodes":valid_intersections,
    "roads":roads,
    "AADT roads":road_AADT
}


with open(args.output, "w") as jf:
    jf.write(json.dumps(output_dict, indent = 2))





###########################
# SHOWS THE MAP
###########################

if not args.show:
    sys.exit()


plt.figure(0)


# latitude, longitude coordinates
lat, lon = [], []


# Shows the intersections
for a_node in valid_intersections:

    node_info = valid_intersections[a_node]

    lat.append(node_info["lat"])
    lon.append(node_info["lon"])

plt.plot(lon, lat, "ko")


# Shows the roads
node_connection_already_seen = {}



# Plots AADT roads first
for an_AADT_road in original_AADT_roads:

    road_info = roads[an_AADT_road]
    start_node, end_node = road_info["start node"], road_info["end node"]

    road_connection_tracker = "-".join(sorted([start_node, end_node]))
    node_connection_already_seen[road_connection_tracker] = True

    start_node_info = valid_intersections[start_node]
    end_node_info = valid_intersections[end_node]

    X = [start_node_info["lon"], end_node_info["lon"]]
    Y = [start_node_info["lat"], end_node_info["lat"]]

    plt.plot(X, Y, "g-", lw=4)



for a_road in roads:

    start_node = roads[a_road]["start node"]
    end_node = roads[a_road]["end node"]

    # Skips showing roads which opposite has already been shown
    road_connection_tracker = "-".join(sorted([start_node, end_node]))
    if road_connection_tracker in node_connection_already_seen:
        continue
    else:
        node_connection_already_seen[road_connection_tracker] = True

    start_node_info = valid_intersections[start_node]
    end_node_info = valid_intersections[end_node]

    X = [start_node_info["lon"], end_node_info["lon"]]
    Y = [start_node_info["lat"], end_node_info["lat"]]

    plt.plot(X, Y, "k-", lw=1)





plt.xlabel("Longitude")
plt.ylabel("Latitude")

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylim.html
plt.xlim([bound_lon_min, bound_lon_max])
plt.ylim([bound_lat_min, bound_lat_max])

plt.show()
