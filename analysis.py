import findspark
findspark.init()
import re

from pyspark import SparkConf, SparkContext, RDD
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import countDistinct, count, avg

import topic_extractor
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases, LdaMulticore
from nltk.corpus import stopwords, wordnet
import nltk
import spacy
import gensim.corpora as corpora
import person_extractor

# Uncomment it at first run
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer

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

import gensim.models
def make_bigrams(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return bigram_mod

def right_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return True
    elif treebank_tag.startswith('V'):
        return True
    elif treebank_tag.startswith('N'):
        return True
    elif treebank_tag.startswith('R'):
        return True
    else:
        return False

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatization(doc):
    print("AAA")
    lemmatizer = WordNetLemmatizer()
    a = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(doc) if right_tag(tag)]
    print(a)
    return a

def preprocess_content(row):
    content = row['pc']
    content = re.sub('[,\.!?]', '', content).lower()
    row['pc'] = simple_preprocess(content, deacc=True)
    return row

def remove_stopwords(row, stop_words):
    content = row['pc']
    row['pc'] = [word for word in content if word not in stop_words]
    return row

def add_bag_of_words(row, id2word):
    row['pc'] = id2word.doc2bow(row['pc'])
    return row

def add_preprocesed_content(row: dict):
    row["pc"] = row["content"]
    return row

def add_topic_in_row(row, topic) -> Row:
    row["topic"] = topic
    row.pop("pc", None)
    return Row(**row)

def get_topic(model: LdaMulticore, text):
    topics = model.get_document_topics(text)
    mx = topics[0][1]
    best_topic = topics[0][0]
    for t in topics:
        if t[1] > mx:
            mx = t[1]
            best_topic = t[0]
    return best_topic

def add_text_topic(ds: DataFrame) -> tuple[DataFrame, LdaMulticore]:
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'told', 'make', 'edu', 'use',
                       'could', 'many', 'said', 'mr', 'bbc', 'also', 'ms', 'one',
                       'two', 'three', 'year', 'would', 'says', 'government',
                       'people', 'minister', 'may', 'need', 'see'])

    # here we will add the implementation for subject finder
    texts = ds.rdd.map(lambda row: add_preprocesed_content(row.asDict())) \
        .map(lambda row: preprocess_content(row)) \
        .map(lambda row: remove_stopwords(row, stop_words)) \
        .cache()
        # To do add lemmatization
        # .map(lambda sentence: lemmatization(sentence)) \
    
    data_lemmatized = list(texts.map(lambda row: row['pc']).collect())
    id2word = corpora.Dictionary(data_lemmatized)
    texts = texts.map(lambda row: add_bag_of_words(row, id2word)).cache()
    corpus = list(texts.map(lambda row: row['pc']).collect())
    model = topic_extractor.train_model(corpus, id2word)

    return texts.map(lambda row: add_topic_in_row(row, get_topic(model, row['pc']))).toDF(), model, corpus

def add_person_in_row(row: dict):
    persons = person_extractor.get_person_names(row["content"], ner)
    row["persons"] = persons
    return row

def add_persons(ds: DataFrame) -> DataFrame:
    ds.rdd.map(lambda row: add_person_in_row(row.asDict())).toDF
        

if __name__ == "__main__":
    init()
    sc = init_spark()
    # Here we use only a sample of 10 texts 
    ds = read_dataset(sc).limit(10).cache()
    ds_topic, topic_model, corpus = add_text_topic(ds)
    ds_topic.rdd.foreach(lambda row: print(row["title"], "=>" , row["topic"], topic_model.print_topic(row["topic"], 10)))
    
    
    ds_person = add_persons(ds)
    ds_person.rdd.foreach(lambda row: print(row["title"], row["persons"]))