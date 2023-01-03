import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import countDistinct, count, avg

def init_spark():
    conf = SparkConf().setMaster("local").setAppName("News analyser")
    spark = SparkContext(conf=conf)
    return SQLContext(spark)

def read_dataset(sc: SQLContext):
    dataset_schema = StructType([
        StructField("crt", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("published_date", StringType(), True),
        StructField("authors",  StringType(), True),
        StructField("description", StringType(), True),
        StructField("section", StringType(), True),
        StructField("content", StringType(), True),
        StructField("link", StringType(), True)
    ])

    ds = sc.read.format("csv") \
        .option("header", "true") \
        .option("delimiter", ",") \
        .schema(dataset_schema) \
        .load("dataset.csv")
    
    ds.registerTempTable("dataset")

    return ds

def add_text_topic(ds: DataFrame):
    # here we will add the implementation for subject finder
    print(ds.rdd.count())

if __name__ == "__main__":
    sc = init_spark()
    ds = read_dataset(sc)
    add_text_topic(ds)