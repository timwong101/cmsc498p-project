"""
ranked_medoids contains the class RankedMedoids, which is used to run the ranked algorithm based on this paper:
    Ranked k-medoids: A fast and accurate rank-based partitioning algorithm for clustering large datasets

RankedMedoids can be initialized with desired arguments from GAM

The fit function is called from GAM

The fit function calls cluster, which calls main

main is the method that holds the ranked algorithm

The algorithm creates a rank and similarity matrix once at the beginning, and then uses those tables for all future
iterations in the update step, minimizing calculations needed
"""

import sklearn
import vaex
from gam_package.preprocessor.preprocessor import load_data, setArguments
import csv
import logging
from collections import Counter
import random

import matplotlib.pylab as plt
import numpy as np
from numpy import sqrt, sum, matmul
from math import exp
from scipy.sparse.linalg import svds
from timeit import default_timer
import math
from numpy.linalg import svd

from gam_package.distance_functions.kendall_tau_distance import mergeSortDistance
from gam_package.distance_functions.spearman_distance import spearman_squared_distance
from gam_package.distance_functions.euclidean_distance import euclidean_distance

from gam_package.medoids_algorithms.parallel_medoids import ParallelMedoids
from gam_package.plot_functions.plot import parallelPlot, radarPlot, facetedRadarPlot, silhouetteAnalysis
from gam_package.medoids_algorithms.ranked_medoids import RankedMedoids
from gam_package.medoids_algorithms.bandit_pam import BanditPAM

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


class RankedMedoids:
    def __init__(self, m = 10, n_clusters=1, distance=euclidean_distance, max_iter=1000,
                 tol=0.0001, dataset=None):  # conf = SparkConf().setAppName("project-gam")
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance = distance
        self.dataset = dataset

    # Randomly select k data points as starter k medoids.
    def randomMedoids(self):
        sample = random.sample(range(self.n), self.k)
        return sample

    def loadData(self, args):

        total_data, total_labels, sigma, feature_labels = load_data(args)

        self.total_data = total_data
        x = total_data
        self.sigma = sigma
        self.feature_labels = feature_labels

        if total_labels is None:
            y = np.array(range(x.shape[0]))
        else:
            y = np.array(total_labels)
        label_vector_rdd = vaex.from_arrays(x=x, y=y)

        self.data = (total_data, total_labels, sigma, feature_labels)

    def cluster(self, n_clusters):
        self.args = setArguments(self.dataset)
        dataset = self.loadData(self.args)
        R, sortedIndex = self.buildTables(dataset)
        self.sortedIndex = sortedIndex; self.R = R


    def buildTables(self, data):

        # data = np.array([[1], [3], [4], [6]])
        n = len(data)
        self.n = n

        sortedIndex = np.array([[0] * n] * n)
        R = np.array([[0] * n] * n)

        def printTable(clusters):
            for i in range(len(clusters)):
                print(clusters[i] + 1)

        def distance(data1, data2):
            return np.sqrt(np.sum((data1 - data2) ** 2))

        print("sortedIndex: ", sortedIndex)
        for i, datapoint in enumerate(data):
            print("\nrow ", i)
            rankDictionary = dict()
            for j, datapoint in enumerate(data):
                print("i: ", i, ", j: ", j)
                rankDictionary[j] = distance(data[i], data[j])
            print("rankDictionary: ", rankDictionary)
            sortedRank = sorted(rankDictionary.items(), key=lambda x: x[1])

            print("sortedRank: ", sortedRank)
            for index, kv in enumerate(sortedRank):
                sortedIndex[i][index] = kv[0]
            print("sortedIndex: ", sortedIndex)

        print("")
        printTable(sortedIndex)

        for i, datapoint in enumerate(sortedIndex):
            for j, datapoint in enumerate(sortedIndex):
                k = sortedIndex[i][j]
                R[i][k] = j

        print("")
        printTable(R)

        return R, sortedIndex



    def selectMMostSimilar(self):
        sortedIndex = self.sortedIndex



    def fit(self, X=None, plotit=False, verbose=True, n_clusters=3):
        """
        Fits kmedoids with the option for plotting
        """

        start = default_timer()
        medoids, clusters, n = self.cluster(n_clusters)
        duration = default_timer() - start

        self.centers = medoids
        self.members = clusters
        if plotit:
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.k > len(colors):
                raise ValueError("we need more colors")

            for i in range(len(self.centers)):
                X_c = X[self.members == i, :]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
                ax.scatter(
                    X[self.centers[i], 0],
                    X[self.centers[i], 1],
                    c=colors[i],
                    alpha=1.0,
                    s=250,
                    marker="*",
                )
        return n, duration


if __name__ == '__main__':
    # rankedMedoids = RankedMedoids()
    # rankedMedoids.cluster()






