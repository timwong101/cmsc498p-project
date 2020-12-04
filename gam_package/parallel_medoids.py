"""
Code up using Apache Spark
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
import copy
import multiprocessing as mp
from multiprocessing import  Pool
import matplotlib.pyplot as plt


class DisplayRDD:
    def __init__(self, rdd):
        self.rdd = rdd

    def _repr_html_(self):
        x = self.rdd.mapPartitionsWithIndex(lambda i, x: [(i, [y for y in x])])
        l = x.collect()
        s = "<table><tr>{}</tr><tr><td>".format("".join(["<th>Partition {}".format(str(j)) for (j, r) in l]))
        s += '</td><td valign="bottom" halignt="left">'.join(["<ul><li>{}</ul>".format("<li>".join([str(rr) for rr in r])) for (j, r) in l])
        s += "</td></table>"
        return s


class ParallelMedoids:
    def _apply_df(self, args):
        df, func, num, kwargs = args
        return num, df.apply(func, **kwargs)

    def apply_by_multiprocessing(self, df, func, **kwargs):
        workers = kwargs.pop('workers')
        pool = mp.Pool(processes=workers)
        result = pool.map(self._apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])
        pool.close()
        result = sorted(result, key=lambda x: x[0])
        return pd.concat([i[1] for i in result])

    def _get_distance(data1, data2):
        """example distance function"""
        return np.sqrt(np.sum((data1 - data2) ** 2))

    def nearestCluster(self, row):
        nearestClustersMap = {}
        row_df = row.to_frame()
        row_df = row_df.transpose()
        for m in self.medoids:
            sum = 0
        for i in range(0, len(m.values[0])):
            m_value = m.values[0][i]
            row_df_value = row_df.values[0][i]
            sum += math.sqrt((m_value - row_df_value) ** 2)
        nearestClustersMap[m.index[0]] = sum

        return min(nearestClustersMap, key=nearestClustersMap.get)

    def exactMedoidUpdate(self, patternsInClusters, verbose=True):
        if len(patternsInClusters) == 0:
            return None
        patterns = np.asmatrix(patternsInClusters)
        distanceMatrix = pdist(patterns, 'euclidean')
        sumRows = squareform(distanceMatrix).sum(axis=1)
        minIndex = np.argmin(sumRows)
        if verbose:
            print("number of costs calculated", sumRows.shape)
            # print("Costs - ", sumRows)
            print("Total cost - ", sumRows[minIndex])
        newMedoid = patternsInClusters.iloc[minIndex].to_frame().transpose()
        return newMedoid

    def distance(self,row):
        vector = []
        row_df = row.to_frame()
        row_df = row_df.transpose()
        for m in self.medoids:
            sum = 0
            for i in range(0, len(m.values[0])):
                m_value = m.values[0][i]
                row_df_value = row_df.values[0][i]
                sum += math.sqrt((m_value - row_df_value) ** 2)
            vector.append(sum)
        # print("vector: ", vector)
        return min(vector)

    def nearestDist(self,df):
        N = df.count()
        dfp = df.toPandas()

        for iter in range(0, N):
            distances_series = dfp.apply(self.distance, axis=1)  # distancesRDD = datasetRDD.map(d(pattern,medoids)
            distancesdf = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

            distances_array = np.concatenate(distancesdf.values,
                                             axis=0)  # distances = nearestDisRDD.values().collect();
            sum = distancesdf[0].sum()

        return self.medoids
    def initialSeeds(self, df, k):
        N = df.count()
        dfp = df.toPandas()
        sample = dfp.sample()
        # global self.medoids
        self.medoids = []
        self.medoids.append(sample)

        for iter in range(1, k):
            distances_series = dfp.apply(self.distance, axis=1)  # distancesRDD = datasetRDD.map(d(pattern,medoids)
            distancesdf = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

            distances_array = np.concatenate(distancesdf.values,
                                             axis=0)  # distances = nearestDisRDD.values().collect();
            sum = distancesdf[0].sum()
            # distances_list = sqlContext.createDataFrame(distancesdf).collect()
            distances_array = [distance / sum for distance in distances_array]  # distances = distances/sum(distances)

            newMedoidId = np.random.choice(N, 1,
                                           p=distances_array)  # newMedoidID = random.choice.range(1:N), prob=distances)
            newMedoid = dfp.iloc[
                newMedoidId[0]]  # medoids = medoids.append(datasetRDD.filter(ID==newMedoidID).collect());
            newMedoid = newMedoid.to_frame().transpose()
            self.medoids.append(newMedoid)
        return self.medoids

    def _cluster(self, max_iter=1000, tol=0.001, verbose=True):
        print("Max Iterations: ", max_iter)

        sc = SparkContext()
        sc.setLogLevel("OFF")
        sqlContext = SQLContext(sc)
        df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
            self.filename)

        max_iter = 10; k = 3; WCSoD1 = float("inf")
        approximateTrackingFlag = False
        self.medoids = self.initialSeeds(df, k)
        if verbose:
            print("Initial centers are ", self.medoids)
        self.members = [0]*len(self.medoids)
        for iter in range(0,max_iter):
            print("\n\n------------------------------------------------------------------------------------------------")
            print("Starting Iteration: ", iter+1)
            previousMedoids = copy.deepcopy(self.medoids)
            print("------------------------------------------------------------------------------------------------")
            print("calculate distance based on these self.medoids: ", previousMedoids)
            dfp = df.toPandas()

            distances_series  = self.apply_by_multiprocessing(dfp, self.distance, axis=1, workers=4)
            nearestDistancesPDF = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

            # nearest_clusters = dfp.apply(self.nearestCluster, axis=1)  # nearestClusterRDD = distancesRDD.map(argmin(distanceVector));
            nearest_clusters  = self.apply_by_multiprocessing(dfp, self.nearestCluster, axis=1, workers=4)
            nearestClustersPDF = nearest_clusters.to_frame()

            for mindex, m in enumerate(self.medoids):
                clID = m.index[0]
                patternsIndex = []
                nearestClustersPDF_flat = nearestClustersPDF.values.flatten()
                for i in range(len(nearestClustersPDF_flat)):
                    if nearestClustersPDF_flat[i] == clID:
                        patternsIndex.append(i)

                # if len(patternsIndex) > 0:
                patternsInClusterPDF = dfp.iloc[patternsIndex, :]
                if verbose:
                    print("Members - ", patternsInClusterPDF.shape)
                newMedoid = self.exactMedoidUpdate(patternsInClusterPDF)

                if len(patternsInClusterPDF) > 0:
                    self.medoids[mindex] = newMedoid

                self.members[mindex] = patternsInClusterPDF
            if verbose:
                print("Change centers to ", self.medoids)
            if verbose and iter+1 > max_iter:
                print("End Searching by reaching maximum iteration", max_iter)
                break
            WCSoD2 = nearestDistancesPDF.sum(axis = 0)[0]
            print("previous sum of distances: ", WCSoD1, "current sum of distances: ", WCSoD2)
            if abs(WCSoD1 - WCSoD2) == 0: #if abs(WCSoD1 - WCSoD2) < .000000000001:
                print("--------------------------------")
                print("final self.medoids: ", self.medoids)
                self.medoids = previousMedoids
                if verbose:
                    print("End Searching by no swaps")
                break
            else:
                WCSoD1 = WCSoD2


        print('here')
        return self.medoids, self.members

    def fit(self, X, plotit=False, verbose=True):
        """
        Fits kmedoids with the option for plotting
        """
        # centers, members = self.kmedoids_run(X, self.n_clusters, self.dist_func,
        #                                      max_iter=self.max_iter, tol=self.tol, verbose=verbose)

        centers, members = self._cluster(max_iter=self.max_iter, tol=self.tol, verbose=verbose)

        # set centers as instance attributes
        self.centers = centers
        self.members = members

        if plotit:
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.n_clusters > len(colors):
                raise ValueError("we need more colors")

            for i in range(len(centers)):
                X_c = X[members == i, :]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
                ax.scatter(
                    X[centers[i], 0],
                    X[centers[i], 1],
                    c=colors[i],
                    alpha=1.0,
                    s=250,
                    marker="*",
                )


    def __init__(self, n_clusters = 1, dist_func=_get_distance, max_iter=1000, tol=0.0001, f = "mushroom-attributions-200-samples.csv"):        # conf = SparkConf().setAppName("project-gam")
        self.filename = f
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None
        self.medoids = []


'''
    def kmedoids_run(self, X, n_clusters, dist_func, max_iter=1000, tol=0.001, verbose=True):
        """Runs kmedoids algorithm with custom dist_func.
        Returns: centers, members, costs, tot_cost, dist_mat
        """
        # Get initial centers
        n_samples, _ = X.shape
        init_ids = _get_init_centers(n_clusters, n_samples)
        if verbose:
            print("Initial centers are ", init_ids)
        centers = init_ids
        members, costs, tot_cost, dist_mat = _get_cost(X, init_ids, dist_func)
        if verbose:
            print("Members - ", members.shape)
            print("Costs - ", costs.shape)
            print("Total cost - ", tot_cost)
        cc, swaped = 0, True
        print("Max Iterations: ", max_iter)
        while True:
            swaped = False
            for i in range(n_samples):
                if i not in centers:
                    for j in range(len(centers)):
                        centers_ = deepcopy(centers)
                        centers_[j] = i
                        members_, costs_, tot_cost_, dist_mat_ = _get_cost(
                            X, centers_, dist_func
                        )
                        if tot_cost_ - tot_cost < tol:
                            members, costs, tot_cost, dist_mat = (
                                members_,
                                costs_,
                                tot_cost_,
                                dist_mat_,
                            )
                            centers = centers_
                            swaped = True
                            if verbose:
                                print("Change centers to ", centers)
                            self.centers = centers
                            self.members = members
            if cc > max_iter:
                if verbose:
                    print("End Searching by reaching maximum iteration", max_iter)
                break
            if not swaped:
                if verbose:
                    print("End Searching by no swaps")
                break
            cc += 1
            print("Starting Iteration: ", cc)
        return centers, members, costs, tot_cost, dist_mat
'''
    # def predict(self, X):
    #     raise NotImplementedError()



