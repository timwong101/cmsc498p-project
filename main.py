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

def dist(row):
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
  print("vector: ", vector)
  return min(vector)

def initialSeeds(df, k):
  N = df.count()
  dfp = df.toPandas()
  sample = dfp.sample()
  global medoidsList
  medoidsList = []
  medoidsList.append(sample)

  for iter in range(1,k):
    distances_series = dfp.apply(dist, axis = 1)
    distancesdf = distances_series.to_frame()
    distances_array = np.concatenate(distancesdf.values, axis = 0)
    sum = distancesdf[0].sum()
    #distances_list = sqlContext.createDataFrame(distancesdf).collect()
    distances_array = [distance / sum for distance in distances_array]
    newMedoidId = np.random.choice(N, 1, p = distances_array)
    newMedoid = dfp.iloc[newMedoidId[0]]
    newMedoid = newMedoid.to_frame().transpose()
    medoidsList.append(newMedoid)
  return medoidsList



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conf = SparkConf().setAppName("project-gam")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    #spark = SparkSession.builder.getOrCreate()
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        "mushroom-attributions-200-samples.csv")

    max_iter = 10
    k = 5

    medoids = initialSeeds(df, k)
    print(medoids)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
