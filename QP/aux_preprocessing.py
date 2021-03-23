"""
SUMMARY

Auxiliary functions for preprocessing.py and QP.py
"""


# Calculates which side of the plane (W = [w0, w1, w2]) a point (P = [x, y]) is
# True: Above or at the plane
# False: Below theplane
def calculate_side_plane(P, W):

    x, y = P
    w0, w1, w2 = W

    tmp = w0 + w1*x + w2*y

    return (w0 + w1*x + w2*y) >= 0



# Computes the equation of a line given P1 = [x1, x2], P2 = [x2, y2]
# Line: w0 + w1*x + w2*y
def compute_line_equation(P1, P2):
    x1, y1 = P1
    x2, y2 = P2

    # Ensures that x1 <= x2
    # Swaps if opposite
    if x2 < x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    # If vertical line
    if x1 == x2:
        w0 = -x1
        w1 = 1
        w2 = 0
    else:
        m = (y2 - y1)/(x2 - x1)

        w1 = -m
        w2 = 1
        w0 = -m*x1 + y1

    return [w0, w1, w2]



# Calculates the middle point of each road, radius from middle-length, and x vector representing it
# road_aux = {"road ID":{"x_m":x_m, "y_m":y_m, "w_0":w_0, "w_1":w_1, "w_2":w_2, "||w||**2":||w||**2,
#               "opposite first":[w_0, w_1, w_2], "opposite second":[w_0, w_1, w_2]}, ...}
def calculate_road_secondary(road_ID, roads, road_aux, valid_intersections):

    road_info = roads[road_ID]
    start_node, end_node = road_info["start node"], road_info["end node"]

    start_node_info = valid_intersections[start_node]
    end_node_info   = valid_intersections[end_node]

    x1, y1 = start_node_info["x"], start_node_info["y"]
    x2, y2 = end_node_info["x"], end_node_info["y"]

    # Half-point
    x_m = (x1 + x2)/2
    y_m = (y1 + y2)/2

    # Calculates its radius squared from the half-point to the edge
    r_sq = (x_m - x1)**2 + (x_m - y1)**2

    # Calculates line equation
    w_0, w_1, w_2 = compute_line_equation([x1, y1], [x2, y2])


    # Generates the opposite lines

    # Finds which point goes first
    if w_1 == 0:
        opposite_first = [-x1, 1, 0]
        opposite_second = [-x2, 1, 0]
    elif w_2 == 0:
        opposite_first = [-y1, 0, 1]
        opposite_second = [-y2, 0, 1]
    else:
        opposite_first = [x1/w_1 - y1, -1/w_1, 1]
        opposite_second = [x2/w_1 - y2, -1/w_1, 1]



    road_aux[road_ID] = {"x_m":x_m, "y_m":y_m, "w_0":w_0, "w_1":w_1, "w_2":w_2, "||w||**2":(w_1**2) + (w_2**2),
        "opposite first":opposite_first,
        "opposite second":opposite_second,
        "S":[x1, y1], "E":[x2, y2]
    }



# Checks if a point (P = [x1, y2]) is within a line segment defined by road id (road_ID)
# A point is wihin a line if it is above the start line and below the right line, or viceversae
def point_within_segment_margins(P, road_ID, road_aux):

    road_auxiliary_info = road_aux[road_ID]

    opp1 = road_auxiliary_info["opposite first"]
    opp2 = road_auxiliary_info["opposite second"]

    # Side must be different on both planes
    return calculate_side_plane(P, opp1) != calculate_side_plane(P, opp2)



# Calculates the distance squared from a point (P = [x, y]) to the line segment represented by a road (road_ID)
def calculate_distance_to_road(x, y, road_ID, road_aux):

    road_aux_info = road_aux[road_ID]

    # If within the segment, calculate distance to line
    if point_within_segment_margins([x, y], road_ID, road_aux):
        distance_sq_top = (road_aux_info["w_0"] + road_aux_info["w_1"]*x + road_aux_info["w_2"]*y)**2
        return distance_sq_top/road_aux_info["||w||**2"]

    else:

        xs, ys = road_aux_info["S"][0], road_aux_info["S"][1]
        xe, ye = road_aux_info["E"][0], road_aux_info["E"][1]

        # If outside the segment boundaries, compute the squared distance to both ends, and return the minimum
        ds_sq = (x - xs)**2 + (y - ys)**2
        de_sq = (x - xe)**2 + (y - ye)**2

        return min(ds_sq, de_sq)



# Interpolates a value
def interpolate(x, y_min, y_max, x_min, x_max):

    if x >= x_max:
        return y_max

    if x <= x_min:
        return y_min


    Δ = y_max - y_min

    return y_min + Δ*(x - x_min)/(x_max - x_min)


