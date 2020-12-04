
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
import copy
import multiprocessing as mp
from multiprocessing import Pool
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from math import pi




class ParallelMedoids:
    def __init__(self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,
                 f="mushroom-attributions-200-samples.csv"):  # conf = SparkConf().setAppName("project-gam")
        self.filename = f
        self.k = n_clusters
        if dist_func == 'euclidean':
            self.dist = self.distance
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None

    # creates a vector with distances from the current sample to each medoid
    # returns the smallest distance
    def distance(self, row):
        vector = []
        row_df = row.to_frame()
        row_df = row_df.transpose()
        for m in medoidsList:
            sum = 0
            for i in range(0, len(m.values[0])):
                m_value = m.values[0][i]
                row_df_value = row_df.values[0][i]
                sum += math.sqrt((m_value - row_df_value) ** 2)
            vector.append(sum)
        return min(vector)

    # helper function for multiprocessing
    def _apply_df(self, args):
        df, func, num, kwargs = args
        return num, df.apply(func, **kwargs)

    # sets up workers for multiprocessing on apply functions
    def apply_by_multiprocessing(self, df, func, **kwargs):
        workers = kwargs.pop('workers')
        pool = mp.Pool(processes=workers)
        result = pool.map(self._apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])
        pool.close()
        result = sorted(result, key=lambda x: x[0])
        return pd.concat([i[1] for i in result])

    # chooses the first selection of medoids using a k-means++ style algorithm
    # uses a probability distribution to select better initial medoids than randomly choosing them
    def initialSeeds(self, df, k):
      N = df.count()
      dfp = df.toPandas()
      sample = dfp.sample()
      global medoidsList
      medoidsList = []
      # chooses the first medoid at random
      medoidsList.append(sample)
      # uses k-means++ to choose the remaining initial medoids
      for iter in range(1,k):
        distances_series = self.apply_by_multiprocessing(dfp, self.distance, axis=1, workers=4)
        #distances_series = dfp.apply(distance, axis = 1) # distancesRDD = datasetRDD.map(d(pattern,medoids)
        distancesdf = distances_series.to_frame()    # nearestRDD = distancesRDD.map(min(distanceVector))

        distances_array = np.concatenate(distancesdf.values, axis = 0) # distances = nearestDisRDD.values().collect();
        sum = distancesdf[0].sum()
        #distances_list = sqlContext.createDataFrame(distancesdf).collect()
        distances_array = [distance / sum for distance in distances_array]  # distances = distances/sum(distances)

        newMedoidId = np.random.choice(N, 1, p = distances_array)   # newMedoidID = random.choice.range(1:N), prob=distances)
        newMedoid = dfp.iloc[newMedoidId[0]]                        # medoids = medoids.append(datasetRDD.filter(ID==newMedoidID).collect());
        newMedoid = newMedoid.to_frame().transpose()
        medoidsList.append(newMedoid)
      return medoidsList

    # returns the index of the nearest medoid for the given sample
    def nearestCluster(self, row):
      nearestClustersMap = {}
      row_df = row.to_frame()
      row_df = row_df.transpose()
      for m in medoidsList:
        sum = 0
        for i in range(0, len(m.values[0])):
            m_value = m.values[0][i]
            row_df_value = row_df.values[0][i]
            sum += math.sqrt((m_value - row_df_value)**2)
        nearestClustersMap[m.index[0]] = sum

      return min(nearestClustersMap, key=nearestClustersMap.get)

    # finds a new medoid given a cluster of samples
    def exactMedoidUpdate(self, patternsInClusters):
        patterns = np.asmatrix(patternsInClusters)
        distanceMatrix = pdist(patterns, 'euclidean')
        sumRows = squareform(distanceMatrix).sum(axis = 1)
        minIndex = np.argmin(sumRows)
        newMedoid = patternsInClusters.iloc[minIndex].to_frame().transpose()
        return newMedoid
    """
    def parallelize_dataframe(df, func, n_cores=4):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        #df = pd.concat(pool.map(func, df_split))
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df
    """

    def fit(self, X = None, plotit=False, verbose=True):
        """
        Fits kmedoids with the option for plotting
        """
        _,_, n = self._cluster()

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
        return n

    def _cluster(self):
        sc = SparkContext()
        # Set new loglevel: "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"
        sc.setLogLevel("OFF")
        sqlContext = SQLContext(sc)
        # sets up the initial df and initializes variables
        df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
            "../mushroom-attributions-200-samples.csv")
        max_iter = 10; k = 3; WCSoD1 = float("inf")
        approximateTrackingFlag = False
        # (self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,f="mushroom-attributions-200-samples.csv")
        medoids = self.initialSeeds(df, k)
        global nearestClustersGlobal
        dfp = df.toPandas()
        # updates medoids until criteria is reached or the maximum number of iterations is reached
        for iter in range(0, max_iter):
            previousMedoids = copy.deepcopy(medoids)

            distances_series = self.apply_by_multiprocessing(dfp, self.distance, axis=1, workers=4)
            # distances_series = dfp.apply(distance, axis=1)  # distancesRDD = datasetRDD.map(d(pattern,medoids)
            nearestDistancesPDF = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

            nearest_clusters = self.apply_by_multiprocessing(dfp, self.nearestCluster, axis=1,
                                                                        workers=4)
            # nearest_clusters = dfp.apply(nearestCluster, axis=1)  # nearestClusterRDD = distancesRDD.map(argmin(distanceVector));
            nearestClustersPDF = nearest_clusters.to_frame()
            nearestClustersGlobal = nearestClustersPDF
            # updates each medoid for the given cluster
            for mindex, m in enumerate(medoids):
                clID = m.index[0]
                patternsIndex = []
                nearestClustersPDF_flat = nearestClustersPDF.values.flatten()
                for i in range(len(nearestClustersPDF_flat)):
                    if nearestClustersPDF_flat[i] == clID:
                        patternsIndex.append(i)
                patternsInClusterPDF = dfp.iloc[patternsIndex, :]
                newMedoid = self.exactMedoidUpdate(patternsInClusterPDF)
                medoids[mindex] = newMedoid
            WCSoD2 = nearestDistancesPDF.sum(axis=0)[0]
            if abs(WCSoD1 - WCSoD2) == 0:
                medoids = previousMedoids
                break
            else:
                WCSoD1 = WCSoD2

        # groupsDict is a dictionary with the (key:value) as (medoid:list of samples in cluster)
        groupsDict = {}
        for i in range(len(medoidsList)):
            groupsDict[medoidsList[i].index[0]] = nearestClustersGlobal[
                nearestClustersGlobal[0] == medoidsList[i].index[0]].index.tolist()
        groupsList = []
        for i in list(groupsDict.keys()):
            groupsList.append(groupsDict[i])
        print(groupsDict)
        print(medoidsList)
        # adding medoid column for plotting purposes
        dfp['medoid'] = 0
        for key, value in groupsDict.items():
            dfp.loc[value, 'medoid'] = key

        graph = 'parallel'
        # parallel plot of all points
        if graph == 'parallel':
            parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
            plt.show()
        # radar plot of medoids
        elif graph == 'radar':
            # create background
            col = dfp.pop('medoid')
            dfp.insert(0, 'medoid', col)
            categories = list(dfp)[1:]
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories)
            ax.set_rlabel_position(0)
            plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
            plt.ylim(0, .3)
            # add plots
            for i in range(len(medoidsList)):
                medoidname = medoidsList[i].index[0]
                values = medoidsList[i].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
                ax.fill(angles, values, 'b', alpha=0.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.show()
        # faceted radar plot of medoids
        elif graph == 'faceted radar':
            def make_spider(row, title, color):
                col = dfp.pop('medoid')
                dfp.insert(0, 'medoid', col)
                categories = list(dfp)[1:]
                N = len(categories)
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]
                ax = plt.subplot(2, 2, row + 1, polar=True, )
                ax.set_theta_offset(pi / 2)
                ax.set_theta_direction(-1)
                plt.xticks(angles[:-1], categories, color='grey', size=8)
                ax.set_rlabel_position(0)
                plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
                plt.ylim(0, .3)
                medoidname = medoidsList[row].index[0]
                values = medoidsList[row].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
                ax.fill(angles, values, 'b', alpha=0.1)
                plt.title(title, size=11, color=color, y=1.1)

            my_dpi = 96
            plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
            my_palette = get_cmap("Set2", len(medoidsList))
            for row in range(len(medoidsList)):
                make_spider(row=row, title=medoidsList[row].index[0], color=my_palette(row))
            plt.show()
        self.centers = []
        self.members = []
        for medoid, members in groupsDict.items():
            self.centers.append(medoid)
            self.members.append(members)
        return self.centers, self.members, len(dfp)









if __name__ == '__main__':
    parallelMedoids = ParallelMedoids()
    parallelMedoids.fit()



    """
    sc = SparkContext()
    # Set new loglevel: "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"
    sc.setLogLevel("OFF")
    sqlContext = SQLContext(sc)
    # sets up the initial df and initializes variables
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        "../mushroom-attributions-200-samples.csv")
    max_iter = 10; k = 3; WCSoD1 = float("inf")
    approximateTrackingFlag = False
    #(self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,f="mushroom-attributions-200-samples.csv")
    parallelMedoids = ParallelMedoids()
    medoids = parallelMedoids.initialSeeds(df, k)
    global nearestClustersGlobal
    dfp = df.toPandas()
    # updates medoids until criteria is reached or the maximum number of iterations is reached
    for iter in range(0,max_iter):
        previousMedoids = copy.deepcopy(medoids)

        distances_series  = parallelMedoids.apply_by_multiprocessing(dfp, parallelMedoids.distance, axis=1, workers=4)
        #distances_series = dfp.apply(distance, axis=1)  # distancesRDD = datasetRDD.map(d(pattern,medoids)
        nearestDistancesPDF = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

        nearest_clusters = parallelMedoids.apply_by_multiprocessing(dfp, parallelMedoids.nearestCluster, axis=1, workers=4)
        #nearest_clusters = dfp.apply(nearestCluster, axis=1)  # nearestClusterRDD = distancesRDD.map(argmin(distanceVector));
        nearestClustersPDF = nearest_clusters.to_frame()
        nearestClustersGlobal = nearestClustersPDF
        # updates each medoid for the given cluster
        for mindex, m in enumerate(medoids):
            clID = m.index[0]
            patternsIndex = []
            nearestClustersPDF_flat = nearestClustersPDF.values.flatten()
            for i in range(len(nearestClustersPDF_flat)):
                if nearestClustersPDF_flat[i] == clID:
                    patternsIndex.append(i)
            patternsInClusterPDF = dfp.iloc[patternsIndex,:]
            newMedoid = parallelMedoids.exactMedoidUpdate(patternsInClusterPDF)
            medoids[mindex] = newMedoid
        WCSoD2 = nearestDistancesPDF.sum(axis = 0)[0]
        if abs(WCSoD1 - WCSoD2) == 0:
            medoids = previousMedoids
            break
        else:
            WCSoD1 = WCSoD2

    # groupsDict is a dictionary with the (key:value) as (medoid:list of samples in cluster)
    groupsDict = {}
    for i in range(len(medoidsList)):
        groupsDict[medoidsList[i].index[0]] = nearestClustersGlobal[nearestClustersGlobal[0] == medoidsList[i].index[0]].index.tolist()
    groupsList = []
    for i in list(groupsDict.keys()):
        groupsList.append(groupsDict[i])
    print(groupsDict)
    print(medoidsList)
    # adding medoid column for plotting purposes
    dfp['medoid'] = 0
    for key, value in groupsDict.items():
        dfp.loc[value, 'medoid'] = key

    graph = 'parallel'
    # parallel plot of all points
    if graph == 'parallel':
        parallel_coordinates(dfp, class_column='medoid', colormap=get_cmap("Set1"))
        plt.show()
    # radar plot of medoids
    elif graph == 'radar':
        # create background
        col = dfp.pop('medoid')
        dfp.insert(0, 'medoid', col)
        categories = list(dfp)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
        plt.ylim(0, .3)
        # add plots
        for i in range(len(medoidsList)):
            medoidname = medoidsList[i].index[0]
            values = medoidsList[i].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()
    # faceted radar plot of medoids
    elif graph == 'faceted radar':
        def make_spider(row, title, color):
            col = dfp.pop('medoid')
            dfp.insert(0, 'medoid', col)
            categories = list(dfp)[1:]
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(2, 2, row + 1, polar=True, )
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            ax.set_rlabel_position(0)
            plt.yticks([.1, .2, .3], [".1", ".2", ".3"], color="grey", size=7)
            plt.ylim(0, .3)
            medoidname = medoidsList[row].index[0]
            values = medoidsList[row].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=medoidname)
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title(title, size = 11, color = color, y = 1.1)
        my_dpi = 96
        plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
        my_palette = get_cmap("Set2", len(medoidsList))
        for row in range(len(medoidsList)):
            make_spider(row = row, title = medoidsList[row].index[0], color = my_palette(row))
        plt.show()
    """