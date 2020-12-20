import csv
import os
import numpy as np
import mnist
import argparse
import pandas as pd
import pickle
import shlex

import csv
import logging
from collections import Counter

import matplotlib.pylab as plt
import numpy as np

from gam_package.medoids_algorithms.k_medoids import KMedoids
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

def setArguments(datasetFilePath):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--verbose', help='print debugging output', action='count', default=0, required=False)
    parser.add_argument('-k', '--num_medoids', help='Number of medoids', type=int, default=3, required=False)
    parser.add_argument('-N', '--sample_size', help='Sampling size of dataset', type=int, default=700,
                        required=False)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42, required=False)
    parser.add_argument('-d', '--dataset', help='Dataset to use', type=str, default='MNIST', required=False)
    parser.add_argument('-c', '--cache_computed', help='Cache computed', default=None, required=False)
    parser.add_argument('-m', '--metric', help='Metric to use (L1 or L2)', type=str, required=False)
    parser.add_argument('-f', '--force', help='Recompute Experiments', action='store_true', required=False)
    parser.add_argument('-p', '--fast_pam1', help='Use FastPAM1 optimization', action='store_true', required=False)
    parser.add_argument('-r', '--fast_pam2', help='Use FastPAM2 optimization', action='store_true', required=False)
    parser.add_argument('-w', '--warm_start_medoids', help='Initial medoids to start with', type=str, default='',
                        required=False)
    parser.add_argument('-B', '--build_ao_swap', help='Build or Swap, B = just build, S = just swap, BS = both',
                        type=str, default='BS', required=False)
    parser.add_argument('-e', '--exp_config', help='Experiment configuration file to use', required=False)

    cmdline = "-k 5 -N 1000 -s 42 -d MNIST -m L2 -p"
    args = parser.parse_args(shlex.split(cmdline))


    args.dataset = datasetFilePath
    args.metric = 'L2'
    args.fast_pam1 = True
    args.num_medoids = 3

    if args.dataset == 'MNIST':
        pass
    elif args.dataset == "SCRNA":
        pass
    elif args.dataset == "SCRNAPCA":
        pass
    elif args.dataset == 'HOC4':
        pass
    elif args.dataset == 'mushrooms':
        args.sample_size = 30
    elif args.dataset == 'data/mushrooms.csv':
        args.sample_size = 30
    elif args.dataset == 'data/wine.csv':
        args.sample_size = 30
    elif args.dataset == 'data/mice_protein.csv':
        args.sample_size = 100
    else:
        raise Exception("Didn't specify a valid dataset")


    return args

def load_data(args):
    print("data_utils -> load_data")
    '''
    Load the different datasets, as a numpy matrix if possible. In the case of
    HOC4, load the datasets as a list of trees.
    '''
    if args.dataset == 'MNIST':
        N = 70000
        m = 28
        sigma = 0.7
        train_data = mnist.train_images()
        train_labels = mnist.train_labels()
        test_data = mnist.test_images()
        test_labels = mnist.test_labels()
        total_data = np.append(train_data, test_data, axis=0)
        total_labels = np.append(train_labels, test_labels, axis=0)
        return total_data.reshape(N, m * m) / 255, total_labels, sigma
    elif args.dataset == "SCRNA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/NUMPY_OUT/np_data.npy'
        data_ = np.load(file)
        sigma = 25
        return data_, None, sigma
    elif args.dataset == "SCRNAPCA":
        file = 'person1/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices/analysis_csv/pca/projection.csv'
        df = pd.read_csv(file, sep=',', index_col=0)
        np_arr = df.to_numpy()
        sigma = 0.01
        return np_arr, None, sigma
    elif args.dataset == 'HOC4':
        dir_ = 'hoc_data/hoc4/trees/'
        tree_files = [dir_ + tree for tree in os.listdir(dir_) if tree != ".DS_Store"]
        trees = []
        for tree_f in sorted(tree_files):
            with open(tree_f, 'rb') as fin:
                tree = pickle.load(fin)
                trees.append(tree)

        if args.verbose >= 1:
            print("NUM TREES:", len(trees))

        return trees, None, 0.0
    elif args.dataset == 'mushrooms':
        filepath = self.attributions_path
        self.total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        return self.total_data, self.feature_labels, sigma

    elif args.dataset == 'wine':
        filepath = self.attributions_path
        self.total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        return self.total_data, self.feature_labels, sigma

    elif args.dataset == 'data/mice_protein.csv':

        # import os
        #
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))

        filepath = self.attributions_path
        self.total_data = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)
        with open(filepath) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))
        sigma = 0.01
        self.total_data = np.nan_to_num(self.total_data)
        return self.total_data, self.feature_labels, sigma

    else:
        raise Exception("Didn't specify a valid dataset")