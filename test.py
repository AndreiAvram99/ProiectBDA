# To find out the path where pyspark is installed
import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import countDistinct, count, avg

conf = SparkConf().setMaster("local").setAppName("BDA-Lab7-using DataFrame DSL")
spark = SparkContext(conf=conf)
sql_context = SQLContext(spark)

employees_df = sql_context.read.format("csv") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .load("dataset.csv")

print(employees_df.filter())