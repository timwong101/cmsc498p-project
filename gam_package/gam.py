
import csv
import logging
import math
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from timeit import default_timer

from gam_package.clustering import KMedoids
from gam_package.kendall_tau_distance import mergeSortDistance
from gam_package.spearman_distance import spearman_squared_distance
from gam_package.parallel_medoids import ParallelMedoids
from gam_package.plot import parallelPlot, radarPlot, facetedRadarPlot, silhouetteAnalysis
from gam_package.ranked_medoids import RankedMedoids
from gam_package.ucb_pam import BanditPAM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
)


class GAM:
    """Generates global attributions

    Args:
        k (int): number of clusters and centroids to form, default=2
        attributions_path (str): path for csv containing local attributions
        cluster_method: None, or callable, default=None
            None - use GAM library routines for k-medoids clustering
            callable - user provided external function to perform clustering
        distance: {â€˜spearmanâ€™, â€˜kendallâ€™}  distance metric used to compare attributions, default='spearman'
        use_normalized (boolean): whether to use normalized attributions in clustering, default='True'
        scoring_method (callable) function to calculate scalar representing goodness of fit for a given k, default=None
        max_iter (int): maximum number of iteration in k-medoids, default=100
        tol (float): tolerance denoting minimal acceptable amount of improvement, controls early stopping, default=1e-3
    """

    def __init__(
        self,
        k=2,
        attributions_path='data/mushroom-attributions-200-samples.csv',
        cluster_method="banditPAM",
        distance="euclidean",
        use_normalized=True,
        scoring_method=None,
        max_iter=100,
        tol=1e-3,
    ):
        self.attributions_path = attributions_path # file path for csv dataset
        self.cluster_method = cluster_method # string representing appropriate k-medoids algorithm

        self.distance = distance # string specifying appropriate dissimilarity metric
        if self.distance == "euclidean":
            self.distance_function = self.euclidean_distance
        elif self.distance == "spearman":
            self.distance_function = spearman_squared_distance
        elif self.distance == "kendall":
            self.distance_function = mergeSortDistance
        else:
            self.distance_function = (
                distance
            )  # assume this is metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS

        self.scoring_method = scoring_method

        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        self.attributions = None # later initialized to pandas dataframe holding csv data
        self.use_normalized = use_normalized # (boolean): whether to use normalized attributions in clustering, default='True'
        self.clustering_attributions = None #
        self.feature_labels = None

        self.subpopulations = None
        self.subpopulation_sizes = None
        self.explanations = None
        self.score = None

    def euclidean_distance(self, data1, data2):
        """example distance function"""
        return np.sqrt(np.sum((data1 - data2) ** 2))

    @staticmethod
    def normalize(attributions):
        """
        Normalizes attributions by via absolute value
            normalized = abs(a) / sum(abs(a))

        Args:
            attributions (numpy.ndarray): for example, [(2, 8), (1, 9)]

        Returns: normalized attributions (numpy.ndarray). For example, [(.2, .8), (.1, .9)]
        """
        # keepdims for division broadcasting
        total = np.abs(attributions).sum(axis=1, keepdims=True)

        return np.abs(attributions) / total

    def _read_local(self):
        """
        Reads attribution values and feature labels from csv

        Returns:
            attributions (numpy.ndarray): for example, [(.2, .8), (.1, .9)]
            feature labels (tuple of labels): ("height", "weight")
        """

        # use numpy to process data from csv file, the header labels are discarded
        self.attributions = np.genfromtxt(
            self.attributions_path, dtype=float, delimiter=",", skip_header=1
        )

        # extract the feature labels
        with open(self.attributions_path) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))

        # TODO: utilize appropriate encoding for categorical, non-numerical data
        df = pd.DataFrame(self.attributions, columns=self.feature_labels)
        #pd.get_dummies(obj_df, columns=["drive_wheels"]).head()

    @staticmethod
    def get_subpopulation_sizes(subpopulations):
        """Computes the sizes of the subpopulations using membership array
        Args:
            subpopulations (list): contains index of cluster each sample belongs to.
                Example, [0, 1, 0, 0].
        Returns:
            list: size of each subpopulation ordered by index. Example: [3, 1]
        """
        index_to_size = Counter(subpopulations)
        sizes = [index_to_size[i] for i in sorted(index_to_size)]

        return sizes

    @staticmethod
    def get_subpopulation_sizes_lol(n, subpopulations):
        """Computes the sizes of the subpopulations using membership array

        Let list of medoid indices = [medoidIndex_{1}, medoidIndex2_{2}, ..., medoidIndex_{n}],
        where medoidIndex_{i} represents a datapoint in the dataframe.

        Args:
            subpopulations (list of lists):
            a list of lists of cluster members
            where each list of cluster members contains row indices of cluster each sample belongs to.
            We assume the dataset comes unordered and but our gam.py assigns row indices anyway.
                Example, [[rowIndex_1, rowIndex_2, ..., rowIndex_1], ..., [..., ...,rowIndex_i]],
                where rowIndex_i < rowIndex_n = last row index of dataset.

        Returns:
            list: size of each subpopulation ordered by index.
            Example: [medoidIndex_{i_1}, medoidIndex2_{i_2}, medoidIndex_{i_n}]
            where medoidIndex_{i_n} <= medoidIndex_{n}.
            medoidIndex refers to the index in the array of final medoids calculated.
        """
        subpop = [None]*n # create empty array to fill with medoid indices
        for medoidIndex, members in enumerate(subpopulations):
            for rowIndex in members:
                subpop[rowIndex] = medoidIndex # assign medoidIndex for each rowIndex
        index_to_size = Counter(subpop)
        sizes = [index_to_size[i] for i in sorted(index_to_size)]

        return sizes

    def _get_explanations(self, centers):
        """Converts subpopulation centers into explanations using feature_labels

        Args:
            centers (list): index of subpopulation centers. Example: [21, 105, 3]

        Returns: explanations (list).
            Example: [[('height', 0.2), ('weight', 0.8)], [('height', 0.5), ('weight', 0.5)]].
        """
        explanations = []

        for center_index in centers:
            explanation_weights = self.clustering_attributions[center_index]
            explanations.append(list(zip(self.feature_labels, explanation_weights)))
        return explanations


    def plot(self, num_features=5, output_path_base=None, display=True):
        """Shows bar graph of feature importance per global explanation

        Args:
            num_features: number of top features to plot, int
            output_path_base: path to store plots
            display: option to display plot after generation, bool
        """
        if not hasattr(self, "explanations"):
            self.generate()

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(self.explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey=True)

            explanations_sorted = sorted(
                explanations, key=lambda x: x[-1], reverse=False
            )[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0, 1])
            axs.set_title("Explanation {}".format(idx + 1), size=10)
            axs.set_xlabel("Importance", size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = "{}_explanation_{}.png".format(output_path_base, idx + 1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches="tight")

            if display:
                plt.show()

    def plot_explanations(self, gam_explanations, num_features=5, output_path_base=None, display=True):
        """Shows bar graph of feature importance per global explanation

        Args:
            num_features: number of top features to plot, int
            output_path_base: path to store plots
            display: option to display plot after generation, bool
        """

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(gam_explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey=True)

            explanations_sorted = sorted(
                explanations, key=lambda x: x[-1], reverse=False
            )[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0, 1])
            axs.set_title("Explanation {}".format(idx + 1), size=10)
            axs.set_xlabel("Importance", size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = "{}_explanation_{}.png".format(output_path_base, idx + 1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches="tight")

            if display:
                plt.show()


    def generate(self, distance_function=None, max_iter=1000, tol=0.0001):
        """
        Clusters local attributions into subpopulations with global explanations
        """
        self._read_local() # read in the data
        if self.use_normalized: # normalize clustering attributions
            self.clustering_attributions = GAM.normalize(self.attributions)
        else: # attributions non-normalized
            self.clustering_attributions = self.attributions
        print("self.attributions: ",str(self.attributions))


        # Cluster according to appropriate algorithm, distance metric
        # Calls local kmedoids module to group attributions
        # Use the distance metric and k-medoids algorithm specified
        # max_iter = maximum number of k-medoid updates
        # tol = minimum error for medoid optimum
        if self.cluster_method is None: # use regular k-medoids
            clusters = KMedoids(
                self.k,
                dist_func=self.distance_function,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            clusters.fit(self.clustering_attributions, verbose=False)

            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes(clusters.members)
            self.explanations = self._get_explanations(clusters.centers)

        elif self.cluster_method == "parallel medoids":
            clusters = ParallelMedoids()
            n, dfp, mlist, duration = clusters.fit(X=self.clustering_attributions, verbose=False)
            self.duration = duration
            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, clusters.members)
            self.explanations = self._get_explanations(clusters.centers)
            parallelPlot(dfp)
            radarPlot(dfp, mlist)
            facetedRadarPlot(dfp, mlist)
            silhouetteAnalysis(dfp, mlist)

        elif self.cluster_method == "ranked medoids":
            clusters = RankedMedoids()

            n, duration = clusters.fit(X=self.clustering_attributions, verbose=False)
            self.duration = duration
            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, clusters.members)
            self.explanations = self._get_explanations(clusters.centers)
            #parallelPlot(dfp)
            #radarPlot(dfp, mlist)
            #facetedRadarPlot(dfp, mlist)
        elif self.cluster_method == "spectral clustering":
            pass
        elif self.cluster_method == "banditPAM":
            banditPAM = BanditPAM()
            n, imgs, feature_labels, duration = banditPAM.fit(X=self.clustering_attributions, verbose=False)
            self.duration = duration

            self.clustering_attributions = imgs
            self.attributions = imgs
            self.feature_labels = feature_labels
            #self.feature_labels = range(1, len(imgs[0]+1))

            self.subpopulations = banditPAM.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes_lol(n, banditPAM.members)
            self.explanations = self._get_explanations(banditPAM.centers)
        else: # use passed in cluster_method and pass in GAM itself
            self.cluster_method(self)

        # Use scoring method if one is provided at initialization
        if self.scoring_method:
            self.score = self.scoring_method(self)



if __name__ == '__main__':
    local_attribution_path = 'data/mushroom-attributions-200-samples.csv' # the pathway to the data file
    g = GAM(attributions_path = local_attribution_path, k=3) # initialize GAM with filename, k=number of clusters
    g.generate() # generate GAM using k-medoids algorithm with number of features specified
    g.plot(num_features=7) # plot the GAM
    g.subpopulation_sizes # generate subpopulation sizes
    g.explanations # generate explanations