#!/usr/bin/env python3

from scipy import sparse as sp
from scipy.sparse import linalg

import pandas as pd
import numpy as np
import pdb
from matplotlib import pyplot as plt

def clean(ratings):
    user_map = {uid:idx for idx,uid in enumerate(set(ratings["userId"]))}
    movie_map = {movid:idx for idx,movid in enumerate(set(ratings["movieId"]))}
    ratings["userId"] = ratings["userId"].apply(lambda x: user_map[x])
    ratings["movieId"] = ratings["movieId"].apply(lambda x: movie_map[x])

    rating_matrix = np.zeros((len(user_map), len(movie_map)))
    indicator_matrix = np.zeros((len(user_map), len(movie_map)))

    rating_matrix[ratings["userId"],ratings["movieId"]] = ratings["rating"]
    indicator_matrix[ratings["userId"],ratings["movieId"]] = 1
    indicator_matrix = indicator_matrix.astype(bool)

    return rating_matrix, indicator_matrix

def eye(k):
    return np.matrix(np.eye(k))

def run_UV_algorithm(ratings, indic, k, lam=.01, stop_thresh=.1):
    num_users, num_movies = ratings.shape
    delta = 1e9

    U = sp.random(num_users, k, density=1).tocsr()
    V = sp.random(num_movies, k, density=1).tocsr()
    ratings = sp.csr_matrix(ratings)

    indic = indic.astype(int)

    while delta > stop_thresh:
        U_prev = U.copy()
        V_prev = V.copy()
        for i in range(num_users):
            # In practice instead of multiplying by the indicator diagonal matrix
            # We use it to index, thus getting rid of all the associated 0*0 multiplications
            if i%300==0:
                print(f"U_{i}/{num_users}")
            indicator = sp.diags(np.array(indic[i])[0])#indicator = sp.diags(indic[i])
            #import pdb
            #pdb.set_trace()
            inv = sp.csr_matrix(linalg.inv((V.transpose()*indicator*V) + lam*sp.eye(k)))

            U[i] = ((inv*(V.transpose()*indicator))*ratings[i].transpose()).transpose()
        for j in range(num_movies):
            if j%300==0:
                print(f"V_{j}/{num_movies}")
            indicator =  sp.diags(np.array(indic[:,j]).transpose()[0])
            inv = sp.csr_matrix(linalg.inv((U.transpose()*indicator*U) + lam*sp.eye(k)))
            V[j] = (inv*(U.transpose()*(indicator*ratings[:,j]))).transpose()

        delta =  np.mean(abs(U-U_prev)) + np.mean(abs(V - V_prev))
        print(f"Mean difference delta: {delta}")

    return U,V

def drop(matrix, proportion):
    """
    Sets given proportion of matrix elements to 0
    """
    return np.multiply(matrix, np.random.random(matrix.shape) > proportion)

def evaluate(ratings, predictions, indic, indic_drop):
    """
    Returns MSE between real and predicted ratings using an indicator matrix
    """
    return np.mean(np.square(ratings[indic^indic_drop] - predictions[indic^indic_drop]))

def run_averaging_algorithm(ratings, indic):
    ratings_mean = np.zeros(ratings.shape)
    for i in range(ratings.shape[1]):
        if len(ratings[indic[:,i],i]) > 0:
            ratings_mean[:,i] = np.mean(ratings[indic[:,i],i])
        else:
            ratings_mean[:,i] = 0

    return ratings_mean

def main():
    #movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    ratings, indic = clean(ratings)

    # Indicator matrix with 25% of the values dropped
    indic_drop = drop(indic, .25)

    filled_dumb = run_averaging_algorithm(ratings, indic_drop)
    score = evaluate(ratings, filled_dumb, indic, indic_drop)
    print(f"Averaging-model MSE: {score}")

    # Running model with k and lambda values varying
    ks = [1, 2, 4, 6]#, 10, 15, 20, 25, 30]#, 10, 15]
    lambdas = [.001, .01, .1, 1, 5]#, .01, .1, 1, 10, 100]#, 10, 100]
    scores = []
    for k in ks:
        for lam in lambdas:
            U,V = run_UV_algorithm(ratings, indic_drop, k=k, lam=lam, stop_thresh=.3)
            predictions = U*V.transpose()
            score = evaluate(ratings, predictions, indic, indic_drop)
            scores.append(score)
            print(f"K: {k}, lam: {lam}, MSE: {score}")

    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(len(ks)))
    ax.set_xticks(np.arange(len(lambdas)))

    ax.set_yticklabels(ks)
    ax.set_xticklabels(lambdas)

    im = ax.imshow(np.array(scores).reshape(len(ks), -1), cmap="jet")
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.xlabel("Lambda")
    plt.ylabel("k")
    plt.title("Heatmap of k vs lambda with MSE values")

    plt.show()


    print(scores)
    pdb.set_trace()

if __name__ == "__main__":
    main()
