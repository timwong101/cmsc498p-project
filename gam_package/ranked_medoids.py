import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates


class RankedMedoids:
    def __init__(self, n_clusters=1, dist_func='euclidean', max_iter=1000, tol=0.0001,
                 f="mushroom-attributions-200-samples.csv"):  # conf = SparkConf().setAppName("project-gam")
        self.filename = f
        self.k = n_clusters

        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.members = None

    # Attain the original dataset for the analysis
    def getDataset(self):
        data = pd.read_csv("mushroom-attributions-200-samples.csv")
        data = data[['odor', 'bruises']]  # Reduce dims of dataset in order to visualize
        dataset = []
        l = len(data)
        # l = 100
        for i in range(l):
            dataset.append(data.loc[i])
        return dataset

    # Generate 2-d test data
    def generate2dData(self):
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

    # Compute distance between each pair of data points
    # data: the actual data we use for this algorithm, it is a collection of data points, each point has dimension dims
    def computeDistance(self, i, j, data):
        dims = len(data[0])
        dis = 0
        for x in range(dims):
            dis += (data[i][x] - data[j][x]) ** 2
        return dis ** .5

    # Build Rank and Similarity tables. This function produces the crucial tables this algorithm requires
    def buildRankTable(self, data):
        n = len(data)
        rank = [[(0, 0) for i in range(n)] for j in range(n)]
        similarityMetrix = [[(0, 0) for i in range(n)] for j in range(n)]

        # Compute the distance and record them in a tuple
        for i in range(n):
            for j in range(i, n):
                d = self.computeDistance(i, j, data)
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
        hv = m * (m + 1) / 2
        for j in range(n):
            if not j in setG:
                for r in rankTable[index]:
                    if r < rankTable[index][j]:
                        hv += 1
        return hv

    # Randomly select k data points as our starting k medoids.
    def randomMedoids(self, k, data):
        medoids = set()
        for i in range(k):
            oldLen = len(medoids)
            while (len(medoids) == oldLen):
                medoids.add(random.randint(0, len(data) - 1))
        return medoids

    # Assign each points to closest medoid, using Rank table
    def assignToClusters(self, k, n, medoids, rankTable):
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

    # Update medoids: use the two tables to calculate the point with highest hv value in each cluster and set it as the new medoid
    def updateMedoids(self, k, m, n, medoids, similarityTable, rankTable):
        newMedoids = set()
        for med in medoids:
            mostSimilar = similarityTable[med][:m]
            maxHv = (-1, -1)
            for simi in mostSimilar:
                hv = self.getHv(simi, m, n, rankTable, set(mostSimilar))
                if hv > maxHv[1]:
                    maxHv = (simi, hv)
            newMedoids.add(maxHv[0])



    def fit(self, X=None, plotit=False, verbose=True):
        """
        Fits kmedoids with the option for plotting
        """
        medoids, clusters, n = self.cluster()
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
        return n



    def plotParralelAndRadar(self, clusters, data):
        allData = []
        for c in clusters:
            clusterData = []
            for index in c:
                clusterData.append(data[index])
            allData.append(clusterData)
        parallel_coordinates(data, color=['g', 'm'])

    def printNice(self, clusters, medoids):
        for i in range(len(clusters)):
            print(medoids[i], ' ', clusters[i])

    def testPrintTables(self, data):
        r, s = self.buildRankTable(data)
        print("Rank Table: ")
        self.printNice(r)
        print()
        print("Similarity Table: ")
        self.printNice(s)

    def test2D(self):
        self.main(self.generate2dData())

    def cluster(self):
        df = pd.read_csv('mushroom-attributions-200-samples.csv')
        dataset = []
        for i in range(len(df)):
            dataset.append(df.loc[i])
        medoids, clusters = self.main(dataset)
        return medoids, clusters, len(df)

    # This method checks whether there exists some data points in different clusters
    def checkClustersContainDifferent(self, clusters):
        for i in range(len(clusters)):
            for d in clusters[i]:
                for j in range(len(clusters)):
                    if i != j:
                        if d in clusters[j]:
                            return False
        return True

    # Standardize data before the PCA analysis
    def standardizeData(self, df):
        features = df.columns
        # Separating out the features
        x = df.loc[:, features].values
        # Separating out the target

        # Standardizing the features
        x = StandardScaler().fit_transform(x)

    # Perform PCA and plot
    def pcaDataAndPlot(self, x, clusters):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

        colors = ['blue', 'black', 'yellow', 'green', 'purple']
        plt.figure(figsize=(10, 8))
        plt.xlim(-3, 3)
        for i in clusters[0]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[0])
        for i in clusters[1]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[1])
        for i in clusters[2]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[2])
        for i in clusters[3]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[3])
        for i in clusters[4]:
            plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[4])

    def main(self, data):
        n = len(data)
        numOfLoops = 100
        k = 5
        m = 12
        medoids = self.randomMedoids(k, data)
        rankTable, similarityMetrix = self.buildRankTable(data)
        for i in range(numOfLoops):
            self.updateMedoids(k, m, n, medoids, similarityMetrix, rankTable)
        clusters, medoids = self.assignToClusters(k, n, medoids, rankTable)

        # self.printNice(clusters, medoids)
        # print(self.checkClustersContainDifferent(clusters))
        # self.plotParralelAndRadar(clusters, data)
        return medoids, clusters


if __name__ == '__main__':
    # data = getDataset()
    # main(data)

    # test2D()

    rankedMedoids = RankedMedoids()
    rankedMedoids.testMushroomDataset()
