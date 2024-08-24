
# %%
# import modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from pathlib import Path

# if using dark mode
import matplotlib as mpl
# mpl.rcParams['figure.facecolor'] = 'white'
from matplotlib.ticker import MaxNLocator

from tqdm.notebook import tqdm
import time
import pandas as pd
import seaborn as sns

import numpy as np
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

from grammodels.treegram_methods import linkage2treegram

# %%
#

def create_random_similar_trees(num_taxa=10,
                                num_trees=10,
                                repetitions=10,
                                max_repetitions=100,
                                random_integers=50,
                                method='complete'):
    """
    Generate a list of random similar trees.
    Parameters:
        num_taxa (int): The number of taxa in the trees. Default is 10.
        num_trees (int): The number of trees to generate. Default is 10.
        repetitions (int): The number of repetitions before increasing the noise level. Default is 10.
        max_repetitions (int): The maximum number of repetitions before raising an error. Default is 100.
        random_integers (int): The maximum value for random integers in the distance matrix. Default is 0.
        method (str): The linkage method to use. Default is 'complete'.
    Returns:
        treegram_list (list): A list of treegrams representing the generated trees.
        treedistance (list): A list of distances corresponding to each treegram. (maximum 1)
    Raises:
        ValueError: If enough different trees cannot be generated within the maximum 
                    number of repetitions.
    """
    treegram_list = []
    treedistance = []

    # Generate a random distance matrix
    if random_integers == 0:
        distmat = np.random.random((num_taxa, num_taxa))
        distmat = (distmat + distmat.T) / 2
    else:
        distmat = np.random.randint(0, random_integers, (num_taxa, num_taxa))
        distmat = (distmat + distmat.T) / 2
    np.fill_diagonal(distmat, 0)
    distmat = MinMaxScaler().fit_transform(distmat.reshape(-1,1)).reshape(np.shape(distmat))

    # Generate a random linkage matrix
    linkage = hierarchy.linkage(squareform(distmat), method=method)
    treegrm, tdist = linkage2treegram(linkage, distmat)

    treegram_list.append(treegrm)
    treedistance.append(tdist)

    if random_integers == 0:
        increase = 0.05
    else:
        increase = 1
    rep_number = 0
    while len(treegram_list) < num_trees:
        if rep_number == repetitions:
            if random_integers == 0:
                increase += 0.05
            else:
                increase = max(increase / 1.05, 0)
            rep_number = 0

        if random_integers == 0:
            distmatnoise = distmat + np.random.normal(loc=0,
                                                    scale=increase,
                                                    size=distmat.shape)
            distmatnoise = (distmatnoise + distmatnoise.T) / 2
        else:
            distmatnoise = distmat + (np.random.exponential(scale=1, size=distmat.shape) / increase).astype(np.int64)
            distmatnoise = np.ceil((distmatnoise + distmatnoise.T) / 2)
        distmatnoise = MinMaxScaler().fit_transform(distmatnoise.reshape(-1,1)).reshape(np.shape(distmat))
        np.fill_diagonal(distmatnoise, 0)
        distmatnoise = MinMaxScaler().fit_transform(distmatnoise.reshape(-1,1)).reshape(np.shape(distmat))

        linkage2 = hierarchy.linkage(squareform(distmatnoise), method='complete')
        treegrm2, tdist2 = linkage2treegram(linkage2, distmatnoise)

        if np.all([len(set(tg.keys()).symmetric_difference(set(treegrm2.keys()))) > 0 for tg in treegram_list]):
            treegram_list.append(treegrm2)
            treedistance.append(tdist2)
            rep_number = 0
        else:
            rep_number += 1

        if rep_number == max_repetitions:
            raise ValueError("Could not generate enough different trees")
    return(treegram_list, np.array(treedistance))



def create_random_trees(num_taxa=10,
                        num_trees=10,
                        random_integers=0,
                        method='complete'):
    """Generate a list of random trees.
    Parameters:
        num_taxa (int): The number of taxa in the trees. Default is 10.
        num_trees (int): The number of trees to generate. Default is 10.
        random_integers (int): The maximum value for random integers in the distance matrix. Default is 0.
        method (str): The linkage method to use. Default is 'complete'.
    Returns:
        treegram_list (list): A list of treegrams representing the generated trees.
        treedistance (list): A list of distances corresponding to each treegram. (maximum 1)
    """
    treegram_list = []
    treedistance = []

    # Generate a random distance matrix
    for _ in range(num_trees):
        if random_integers == 0:
            distmat = np.random.random((num_taxa, num_taxa))
            distmat = (distmat + distmat.T) / 2
        else:
            distmat = np.random.randint(0, random_integers, (num_taxa, num_taxa))
            distmat = (distmat + distmat.T) / 2
        np.fill_diagonal(distmat, 0)
        distmat = MinMaxScaler().fit_transform(distmat.reshape(-1,1)).reshape(np.shape(distmat))

        # Generate a random linkage matrix
        linkage = hierarchy.linkage(squareform(distmat), method=method)
        treegrm, tdist = linkage2treegram(linkage, distmat)

        treegram_list.append(treegrm)
        treedistance.append(tdist)
    
    return(treegram_list, np.array(treedistance))


def create_random_trees_from_blobs(num_taxa=20, num_trees=10,
                                   num_blob_points=10, num_blob_centers=4,
                                   method='complete'):
    if num_blob_points > num_taxa:
        raise ValueError("The number of blob points must be less than the number of taxa")
    
    treegram_list = []
    treedistance = []
    for _ in range(num_trees):
        X, y = make_blobs(n_samples=num_blob_points, centers=num_blob_centers)
        X = X[np.argsort(y), :]
        if num_taxa > num_blob_points:
            X = np.vstack([np.random.random([num_taxa - num_blob_points, 2]),
                        MinMaxScaler().fit_transform(X)])
        dist_mat = pairwise_distances(X)
        dist_mat = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(dist_mat, 0)
        dist_mat = dist_mat / np.max(dist_mat)

        linkage = hierarchy.linkage(squareform(dist_mat), method=method)
        treegrm, tdist = linkage2treegram(linkage, dist_mat)

        treegram_list.append(treegrm)
        treedistance.append(tdist)
    return(treegram_list, np.array(treedistance))
