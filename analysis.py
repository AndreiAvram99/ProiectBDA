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
import spacy
import gensim.corpora as corpora
import person_extractor

def init():
    global ner
    ner = spacy.load("en_core_web_lg")

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

def get_topic(corpus, id2word):
    topic_extractor.get_topic_modeling_result(corpus, id2word)


import gensim.models
def make_grams(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return bigram_mod

def lemmatize_sentence(sentence, nlp):
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    doc = nlp(" ".join(sentence))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]


def add_text_topic(ds: DataFrame):
    # here we will add the implementation for subject finder
    contents = ds.rdd.map(lambda row: row['content']) \
        .map(lambda content: re.sub('[,\.!?]', '', content)) \
        .map(lambda content: content.lower()) \
        .map(lambda sentence: simple_preprocess(str(sentence), deacc=True)) \
        .cache()
    
    data_words = list(contents.collect())

    bigram_mod = make_grams(data_words)

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'told', 'make', 'edu', 'use',
                       'could', 'many', 'said', 'mr', 'bbc', 'also', 'ms', 'one',
                       'two', 'three', 'year', 'would', 'says', 'government',
                       'people', 'minister', 'may', 'need', 'see'])

    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    texts = contents.map(lambda sentence: [word for word in sentence if word not in stop_words]) \
        .map(lambda sentence: bigram_mod[sentence]) \
        .map(lambda sentence: lemmatize_sentence(sentence, nlp)) \
        .cache()
    
    data_lemmatized = list(texts.collect())
    id2word = corpora.Dictionary(data_lemmatized)

    corpus = list(texts.map(lambda text: id2word.doc2bow(text)).collect())

    get_topic(corpus, id2word)

def add_person_in_row(row: dict):
    persons = person_extractor.get_person_names(row["content"], ner)
    row["persons"] = persons
    return row

def add_persons(ds: DataFrame):
    ds.rdd.map(lambda row: add_person_in_row(row.asDict())).foreach(lambda row: print(row["title"], row["persons"]))

if __name__ == "__main__":
    init()
    sc = init_spark()
    ds = read_dataset(sc).limit(10).cache()
    # add_text_topic(ds)
    add_persons(ds)