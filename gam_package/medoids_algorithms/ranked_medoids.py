import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from timeit import default_timer

from gam_package.distance_functions.euclidean_distance import euclidean_distance
from gam_package.distance_functions.kendall_tau_distance import mergeSortDistance
from gam_package.distance_functions.spearman_distance import spearman_squared_distance
import numpy as np

class RankedMedoids:
    def __init__(self, n_clusters=3, dist_func=None, dist_func_type='euclidean', max_iter=1000, tol=0.0001,
                 attributions_path="data/mushroom-attributions-200-samples.csv"):  # conf = SparkConf().setAppName("project-gam")

        print("RankedMedoids(n_clusters=1, dist_func='euclidean', max_iter=, tol=, f='')")

        self.attributions_path = attributions_path
        self.n_clusters = n_clusters

        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None


        self.dist_func_type = dist_func_type

        if self.dist_func_type == "euclidean":
            self.dist_func = euclidean_distance
        elif self.dist_func_type == "spearman":
            self.dist_func = spearman_squared_distance
        elif self.dist_func_type == "kendall":
            self.dist_func = mergeSortDistance

        if dist_func is not None:
            self.dist_func = dist_func

            # Attain the original dataset for the analysis
    def getDataset(self):
        print("RankedMedoids.getDataset()")

        data = pd.read_csv(self.attributions_path)
        data = data[['odor', 'bruises']]  # Reduce dims of dataset in order to visualize
        dataset = []
        l = len(data)
        # l = 100
        for i in range(l):
            dataset.append(data.loc[i])
        return dataset

    # Generate 2-d test data
    def generate2dData(self):
        print("RankedMedoids.generate2dData()")

        xs = []
        ys = []
        n = 500
        for i in range(n):
            xs.append(random.random())
            ys.append(random.random())
            if i > .9 * n:
                xs.append(random.random() + .5)
                ys.append(random.random() + .5)
        data = []
        for i in range(n):
            data.append([xs[i], ys[i]])
        return data


    # Build Rank and Similarity tables. This function produces the crucial tables this algorithm requires
    def buildRankTable(self, data):
        print("RankedMedoids.buildRankTable(data)")

        n = len(data)
        rank = [[(0, 0) for i in range(n)] for j in range(n)]
        similarityMetrix = [[(0, 0) for i in range(n)] for j in range(n)]

        data = np.array(data)

        # Compute the distance and record them in a tuple
        for i in range(n):
            for j in range(i, n):
                d = self.dist_func(data[i], data[j])
                rank[i][j] = (d, j)
                rank[j][i] = (d, i)

        for i in range(n):
            rank[i].sort(key=lambda x: x[0])  # Sort based on distances
            r = 0
            rankings = []
            for j in range(n):
                rankings.append(r)
                if j <= n - 2 and rank[i][j][0] == rank[i][j + 1][0]:
                    r -= 1
                r += 1
                similarityMetrix[i][j] = rank[i][j][1]  # Store the sample points
                rank[i][j] = rank[i][j][1]

            newList = [0] * n
            for j in range(n):
                newList[rank[i][j]] = rankings[j]

            rank[i] = newList

        return rank, similarityMetrix

    # Compute the hostility value of a data point
    # tableS: a nxn matrix containing all Sij values
    # setG: set of integers - indexes of data points
    # n: size of dataset
    def getHv(self, index, m, n, rankTable, setG):
        # print("RankedMedoids.getHv(index, m, n, rankTable, setG)")

        hv = m * (m + 1) / 2
        for j in range(n):
            if not j in setG:
                for r in rankTable[index]:
                    if r < rankTable[index][j]:
                        hv += 1
        return hv

    # Randomly select k data points as our starting k medoids.
    def randomMedoids(self, k, data):
        print("RankedMedoids.randomMedoids(k, data)")

        medoids = set()
        for i in range(k):
            oldLen = len(medoids)
            while (len(medoids) == oldLen):
                medoids.add(random.randint(0, len(data) - 1))
        return medoids

    # Assign each points to closest medoid, using Rank table
    def assignToClusters(self, k, n, medoids, rankTable):
        print("RankedMedoids.assignToClusters(k, n, medoids, rankTable)")

        groups = []  # list of Set
        for i in range(k):
            groups.append([])  # Make k empty groups
        medoidsList = list(medoids)

        for i in range(n):
            rankRowI = rankTable[i]
            min = n
            for j in range(k):
                m = medoidsList[j]
                if rankRowI[m] < min:
                    min = rankRowI[m]
            for j in range(k):
                m = medoidsList[j]
                if rankRowI[m] == min:
                    groups[j].append(i)

        return groups, medoids


    def fit(self, X=None, plotit=False, verbose=True, attributions_path = ''):
        print("RankedMedoids.fit(X=None, plotit=False, verbose=True, attributions_path = '')")

        """
        Fits kmedoids with the option for plotting
        """
        self.attributions_path = attributions_path
        start = default_timer()
        medoids, clusters, n = self.cluster()
        duration = default_timer() - start
        self.centers = medoids
        self.members = clusters
        if plotit:
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.n_clusters > len(colors):
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

    # Print out the table nicely
    def printClusters(self, clusters):
        print("RankedMedoids.printClusters(clusters)")

        for i in range(len(clusters)):
            print(clusters[i])

    def testPrintTables(self, data):
        print("RankedMedoids.testPrintTables(data)")

        r, s = self.buildRankTable(data)
        print("Rank Table: ")
        self.printClusters(r)
        print()
        print("Similarity Table: ")
        self.printClusters(s)

    def test2D(self):
        print("RankedMedoids.test2D()")

        self.main(self.generate2dData())

    # Cluster the data
    def cluster(self):
        print("RankedMedoids.cluster()")

        df = pd.read_csv(self.attributions_path)
        dataset = []
        for i in range(len(df)):
            dataset.append(df.loc[i])
        medoids, clusters = self.main(dataset)
        return medoids, clusters, len(df)

    # This method checks whether there exists same data points in different clusters
    def checkClustersContainDifferent(self, clusters):
        print("RankedMedoids.checkClustersContainDifferent(clusters)")

        for i in range(len(clusters)):
            for d in clusters[i]:
                for j in range(len(clusters)):
                    if i != j:
                        if d in clusters[j]:
                            return False
        return True

    # Standardize data before the PCA analysis
    def standardizeData(self, df):
        print("RankedMedoids.standardizeData(df)")

        features = df.columns
        # Separating out the features
        x = df.loc[:, features].values
        # Separating out the target

        # Standardizing the features
        x = StandardScaler().fit_transform(x)

    # Print out the table nicely
    def printMedoidsAndClusters(self, medoids, clusters):
        print("RankedMedoids.printMedoidsAndClusters(clusters)")

        for i in range(len(clusters)):
            print(medoids[i], " -> ", clusters[i])

    # Update medoids: use the two tables to calculate the point with highest hv value in each cluster and set it as the new medoid
    def updateMedoids(self, k, m, n, medoids, similarityTable, rankTable):
        # print("RankedMedoids.updateMedoids(k, m, n, medoids, similarityTable, rankTable)")

        newMedoids = []
        for med in medoids:
            mostSimilar = similarityTable[med][:m]
            maxHv = (-1, -1)
            for simi in mostSimilar:
                hv = self.getHv(simi, m, n, rankTable, set(mostSimilar))
                if hv > maxHv[1]:
                    maxHv = (simi, hv)
            newMedoids.append(maxHv[0])

        return newMedoids

    # Do the main job of clustering. Define values for k, m, numOfLoops. When looping, update medoids accordingly
    def main(self, data):
        print("RankedMedoids.main(data)")
        n = len(data)
        numOfLoops = 10
        k = 5
        m = 12
        medoids = list(self.randomMedoids(k, data))
        print("inital medoids: ", medoids)
        rankTable, similarityMetrix = self.buildRankTable(data)
        for i in range(numOfLoops):
            medoids = self.updateMedoids(k, m, n, medoids, similarityMetrix, rankTable)
            print("current medoids: ", medoids)
        clusters, medoids = self.assignToClusters(k, n, medoids, rankTable)

        # self.printClusters(clusters)
        self.printMedoidsAndClusters(medoids, clusters)

        # print(self.checkClustersContainDifferent(clusters))
        # self.plotParralelAndRadar(clusters, data)
        return medoids, clusters


if __name__ == '__main__':
    # data = getDataset()
    # main(data)

    # test2D()

    rankedMedoids = RankedMedoids()
    rankedMedoids.cluster()