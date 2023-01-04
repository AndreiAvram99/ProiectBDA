import findspark
findspark.init()
import re

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import countDistinct, count, avg

import topic_extractor
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases
from nltk.corpus import stopwords

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

def get_topic(data_words, data_words_nostops):
    topic_extractor.get_topic_modeling_result(data_words, data_words_nostops)

def add_text_topic(ds: DataFrame):
    # here we will add the implementation for subject finder
    contents = ds.limit(10).rdd.map(lambda row: row['content']) \
        .map(lambda content: re.sub('[,\.!?]', '', content)) \
        .map(lambda content: content.lower()) \
        .map(lambda sentence: simple_preprocess(str(sentence), deacc=True)) \
        .cache()
    
    data_words = list(contents.collect())

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'told', 'make', 'edu', 'use',
                       'could', 'many', 'said', 'mr', 'bbc', 'also', 'ms', 'one',
                       'two', 'three', 'year', 'would', 'says', 'government',
                       'people', 'minister', 'may', 'need', 'see'])

    data_words_nostops = list(contents.map(lambda sentence: [word for word in sentence if word not in stop_words]).collect())

    get_topic(data_words, data_words_nostops)

if __name__ == "__main__":
    sc = init_spark()
    ds = read_dataset(sc)
    add_text_topic(ds)