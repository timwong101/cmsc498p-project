"""
Code up using Apache Spark
"""

import pyspark
from pyspark import SparkContext, SparkConf
import csv
import random
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, LongType
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window
import pandas as pd
import numpy as np
import math


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

def distance(row):
  vector = []
  row_df = row.to_frame()
  row_df = row_df.transpose()
  for m in medoidsList:
    sum = 0
    for i in range(0, len(m.values[0])):
        m_value = m.values[0][i]
        row_df_value = row_df.values[0][i]
        sum += math.sqrt((m_value - row_df_value)**2)
    vector.append(sum)
  # print("vector: ", vector)
  return min(vector)

def initialSeeds(df, k):
  N = df.count()
  dfp = df.toPandas()
  sample = dfp.sample()
  global medoidsList
  medoidsList = []
  medoidsList.append(sample)

  for iter in range(1,k):
    distances_series = dfp.apply(distance, axis = 1) # distancesRDD = datasetRDD.map(d(pattern,medoids)
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

def nearestDist(df):
  N = df.count()
  dfp = df.toPandas()

  for iter in range(0,N):
    distances_series = dfp.apply(distance, axis = 1) # distancesRDD = datasetRDD.map(d(pattern,medoids)
    distancesdf = distances_series.to_frame()    # nearestRDD = distancesRDD.map(min(distanceVector))

    distances_array = np.concatenate(distancesdf.values, axis = 0) # distances = nearestDisRDD.values().collect();
    sum = distancesdf[0].sum()

  return medoidsList

def nearestCluster(row):
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # conf = SparkConf().setAppName("project-gam")
    # sc = SparkContext(conf=conf)
    sc = SparkContext()
    # Set new loglevel: "ALL", "DEBUG", "ERROR", "FATAL", "INFO", "OFF", "TRACE", "WARN"
    sc.setLogLevel("OFF")
    sqlContext = SQLContext(sc)
    #spark = SparkSession.builder.getOrCreate()
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        "mushroom-attributions-200-samples.csv")

    max_iter = 10; k = 5
    approximateTrackingFlag = False
    medoids = initialSeeds(df, k)
    for iter in range(0,max_iter):
        previousMedoids = medoids
        dfp = df.toPandas()


        distances_series = dfp.apply(distance, axis=1)  # distancesRDD = datasetRDD.map(d(pattern,medoids)
        nearestDistancesPDF = distances_series.to_frame()  # nearestRDD = distancesRDD.map(min(distanceVector))

        nearest_clusters = dfp.apply(nearestCluster, axis=1)  # nearestClusterRDD = distancesRDD.map(argmin(distanceVector));
        nearestClustersPDF = nearest_clusters.to_frame()

        for m in medoids:
            clID = m.index[0]
            patternsIndex = []
            nearestClustersPDF_flat = nearestClustersPDF.values.flatten()
            for i in range(len(nearestClustersPDF_flat)):
                if nearestClustersPDF_flat[i] == clID:
                    patternsIndex.append(i)
            patternsInClusterPDF = dfp.iloc[patternsIndex,:]
            display = DisplayRDD(sc.parallelize(dfp, 5))
    print(medoids)

