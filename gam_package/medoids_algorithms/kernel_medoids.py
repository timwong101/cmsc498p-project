
import vaex
from gam_package.preprocessor.preprocessor import load_data, setArguments
import csv
import logging
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
from timeit import default_timer

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

class KernelMedoids :

    def __init__(self, n_clusters=1, max_iter=1000, tol=0.0001, attributions_path="../data/mushrooms.csv",
                 CLUSTER_NUM=3, TARGET_DIM=6, SKETCH_SIZE=60, SIGMA=1):

        self.attributions_path = attributions_path
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None

        ## Parse parameters from command line arguments
        self.CLUSTER_NUM = CLUSTER_NUM
        self.TARGET_DIM = CLUSTER_NUM * 2
        self.SKETCH_SIZE = TARGET_DIM * 10
        self.SIGMA = SIGMA
        
        print("####################################")
        print("cluster number = ", CLUSTER_NUM)
        print("target dimension = ", TARGET_DIM)
        print("sketch size = ", SKETCH_SIZE)
        print("sigma = ", SIGMA)

        # import os
        # cwd = os.getcwd()  # Get the current working directory (cwd)
        # files = os.listdir(cwd)  # Get all the files in that directory
        # print("Files in %r: %s" % (cwd, files))


    def fit(self, X=None, plotit=False, verbose=True, attributions_path=None, datasetName='MNIST'):

        ## Loads data
        # self.data = vaex.from_csv(self.attributions_path, copy_index = True)

        datasetName = datasetName
        args = setArguments(datasetName)
        total_data, total_labels, sigma = load_data(args)
        total_data = total_data[np.random.choice(range(len(total_data)), size=args.sample_size, replace=False)]

        ## Parse the data to get labels and features
        # RDD[(Int, Array[Double])]
        # label_vector_rdd = df.rdd.map(pair= > (pair[0], pair[1]) )
        x = total_data
        y = total_labels
        label_vector_rdd = vaex.from_arrays(x=x, y=y)

        ## Perform kernel k-means with Nystrom approximation
        ## (Array[String], Array[String])
        # result = kernel_kmeans(sc, label_vector_rdd, CLUSTER_NUM, TARGET_DIM, SKETCH_SIZE, SIGMA)
        print("")

    '''
     * Compute the RBF kernel function of two vectors.
     def rbf(x1: Array[Double], x2: Array[Double], sigma: Double): Double 
     '''

    def rbf(self, x1, x2, sigma):
        ## squared l2 distance between x1 and x2
        dist = x1.zip(x2).map(pair= > pair._1 - pair._2).map(x= > x * x).sum
        ## RBF kernel function is exp( - ||x1-x2||^2 / 2 / sigma^2 )
        math.exp(- dist / (2 * sigma * sigma))


    '''
     * Compute the RBF kernel functions of a vector and a collection of vectors.
     def rbf(x1: Array[Double], x2: Array[Array[Double]], sigma: Double): Array[Double] 
     '''

    def rbf(x1, x2, sigma):
        n = x2.length
        kernel_arr = [0] * n
        sigma_sq = 2 * sigma * sigma
        dist = 0.0
        # for (i <- 0 until n) {
        for i in range(0, n):
            ## squared l2 distance between x1 and x2(i)
            dist = x1.zip(x2(i)).map(pair= > pair._1 - pair._2).map(x= > x * x).sum
            ## RBF kernel function
            kernel_arr(i) = math.exp(- dist / sigma_sq)
        return kernel_arr

    '''
         * The Nystrom method for approximating the RBF kernel matrix.
         * Let K be n-by-n kernel matrix. The Nystrom method approximates 
         * K by C * W^{-1}_l * C^T, where we set l = c/2.
         * The Nystrom feature vectors are the rows of C * W^{-1/2}_l.
         *
         * Input
         *  sc: SparkContext
         *  label_vector_rdd: RDD of labels and raw input feature vectors
         *  c: sketch size (k < s < c/2)
         *  sigma: kernel width parameter
         *
         * Output
         *  RDD of (lable, Nystrom feature vector) pairs
         *  the dimension of Nystrom feature vector is (c/
         return RDD[(Int, Vector)]
         '''

    def nystrom(self, label_vector_rdd, c, sigma):
        n = label_vector_rdd.count

        ## Randomly sample about c points from the dataset
        frac = c / n
        # data_samples = label_vector_rdd.sample(false, frac).map(pair= > pair._2).collect
        # broadcast_samples = sc.broadcast(data_samples)
        # RDD[(Int, Array[Double])]
        data_samples = label_vector_rdd.sample(frac=frac)

        ## Compute the C matrix of the Nystrom method
        # rbf_fun = (x1: Array[Double], x2: Array[Array[Double]])

        # c_mat_rdd = label_vector_rdd
        # .map(pair= > (pair._1, broadcast_rbf.value(pair._2, broadcast_samples.value)))
        # .map(pair= > (pair._1, new DenseVector(pair._2)))
        c_mat_rdd = label_vector_rdd
        .map(lambda pair: self.rbf.value(pair._2, data_samples.value) )
        .map(pair= > (pair._1, new DenseVector(pair._2)))

        ## Compute the W matrix of the Nystrom method
        ## decompose W as: U * U^T \approx W^{-1}
        u_mat = nystrom_w_mat(data_samples, sigma)
        broadcast_u_mat = sc.broadcast(u_mat)

        ## Compute the features extracted by Nystrom
        ## feature matrix is C * W^{-1/2}_{c/2}
        nystrom_rdd = c_mat_rdd
        .map(pair= > (pair._1, pair._2.t * broadcast_u_mat.value))
        .map(pair= > (pair._1, Vectors.dense(pair._2.t.toArray)))

        return nystrom_rdd


    '''
     Kernel k-means clustering with Nystrom approximation.
     Input
     *  label_vector_rdd: RDD of labels and raw input feature vectors
     *  k: cluster number
     *  s: target dimension of extracted feature vectors
     *  c: sketch size (k < s < c/2)
     *  sigma: kernel width parameter
     *
     * Output
     *  labels: Array of (true label, predicted label) pairs
     *  time: Array of the elapsed time of Nystrom, PCA, and k-means, respectively
     return (Array[String], Array[String])
    '''
    def kernel_kmeans(self, label_vector_rdd, k, s, c, sigma):
        ## max number of iterations (can be tuned)
        MAX_ITER = 100

        ## Extract features by the Nystrom method
        t0 = default_timer()

        # nystrom_rdd: RDD[(Int, Vector)]
        nystrom_rdd = nystrom(label_vector_rdd, c, sigma)
        t1 = default_timer() - t0

        ##label_vector_rdd.unpersist()
        print("####################################")
        print("Nystrom method costs  " + t1 + "  seconds.")
        print("####################################")
        
        ## Extract s principal components from the Nystrom features
        ## The V matrix stored in a local dense matrix
        t2 = default_timer()
        mat = new RowMatrix(nystrom_rdd.map(pair => pair._2))
        svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(s, computeU = false)
        v_mat: Matrix = svd.V.transpose
        broadcast_v_mat = sc.broadcast(v_mat)
        nystrom_pca_rdd: RDD[(Int, Vector)] = nystrom_rdd
                .map(pair => (pair._1, broadcast_v_mat.value.multiply(pair._2)))
                .map(pair => (pair._1, Vectors.dense(pair._2.toArray)))
                .persist()
        nystrom_pca_rdd.count
        t3 = default_timer() - t2
        ##broadcast_v_mat.destroy()
        ##nystrom_rdd.unpersist()
        print("####################################")
        print("PCA costs  " + t3 + "  seconds.")
        print("####################################")

        ## K-means clustering over the extracted features
        t4 = System.nanoTime()
        feature_rdd: RDD[Vector] = nystrom_pca_rdd.map(pair => pair._2)
        clusters = KMeans.train(feature_rdd, k, MAX_ITER)
        t5 = System.nanoTime()
        time(2) = ((t5 - t4) * 1.0E-9).toString
        print("####################################")
        print("K-means clustering costs  " + time(2) + "  seconds.")
        print(" ")
        print("getExecutorMemoryStatus:")
        print(sc.getExecutorMemoryStatus.toString())
        print("####################################")
        
        ## Predict labels
        broadcast_clusters = sc.broadcast(clusters)
        labels: Array[String] = nystrom_pca_rdd
                .map(pair => (pair._1, broadcast_clusters.value.predict(pair._2)))
                .map(pair => pair._1.toString + " " + pair._2.toString)
                .collect()
        
        return (labels, time)
    
    '''
     * Locally compute the W matrix of Nystrom and its factorization W_{l}^{-1} = U * U^T.
     def nystrom_w_mat(data_samples: Array[Array[Double]], sigma: Double): DenseMatrix[Double] 
     '''
    def nystrom_w_mat(data_samples, sigma):
        c = data_samples.length
        l = math.ceil(c * 0.5).toInt
        
        ## Compute the W matrix of Nystrom method
        w_mat = DenseMatrix.zeros[Double](c, c)
        for (j <- 0 until c) {
            for (i <- 0 until c) {
                w_mat(i, j) = rbf(data_samples(i), data_samples(j), sigma)
            }
        }
        
        ## Compute matrix U such that U * U^T = W_{l}^{-1}
        usv = svd.reduced(w_mat)
        u_mat = usv.U(::, 0 until l)
        s_arr = usv.S(0 until l).toArray.map(s => 1 / math.sqrt(s))
        for (j <- 0 until l) {
            u_mat(::, j) :*= s_arr(j)
        }
        
        u_mat
    }
    
    #/mnt/c/Users/charm/PycharmProjects/SparkKernelKMeans/data
#'''

if __name__ == '__main__':

    kernelMedoids = KernelMedoids()
    kernelMedoids.fit()