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
  """
  print("row_df type: \n", type(row_df))
  print("row_df: \n", row_df)
  print("row_df index: \n", row_df.index)
  print("row_df values shape: \n", row_df.values.reshape(-1, 1).shape)
  print("row_df shape: \n", row_df.shape)
  print("row_df transpose: \n", row_df.transpose())
  """
  row_df = row_df.transpose()
  for m in medoidsList:
    row_df_sum = row_df.to_numpy().sum()
    m_sum = m.to_numpy().sum()
    sum = row_df_sum - m_sum
    sum = sum**2
    print("row_df: \n", row_df)
    print("row_df_sum: ", row_df_sum)
    print("m_sum: ", m_sum)
    print("sum: ", sum)
    vector.append(sum)
  print("vector: ", vector)
  return vector

def initialSeeds(df, k):
  dfp = df.toPandas()
  sample = dfp.sample()
  global medoidsList
  medoidsList = []
  medoidsList.append(sample)

  distances_series = dfp.apply(dist, axis = 1)
  distancesdf = distances_series.to_frame()
  return distancesdf




#print(type(medoids[0]), str(medoids[0]))
#DisplayRDD(medoids)
#print("Type: ", (medoids.head()))
#print(medoids)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conf = SparkConf().setAppName("project-gam")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        "mushroom-attributions-200-samples.csv")

    max_iter = 10
    k = 5

    medoids = initialSeeds(df, k)
    print(medoids)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
