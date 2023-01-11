import main
import utils
import re

sc = main.init_spark()

df = utils.read_dataset_with_topics(sc).sample(0.15)  # a 15% sample from the dataset
topics = utils.read_topics(sc)

persons_rdd = df.rdd.map(lambda column: (re.findall("'([^']*)'", column[5]), column[9]))
pair_rdd = persons_rdd.flatMap(lambda x: [(i, x[1]) for i in x[0]])
pair_count_rdd = pair_rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

pair_count_df = pair_count_rdd.toDF(["pair", "appearance_number"])
pair_count_df = pair_count_df.selectExpr("pair._1 as name", "pair._2 as topic_id","appearance_number")

result = pair_count_df.join(topics, pair_count_df.topic_id == topics.topic_id, "inner") \
    .drop("topic_id")
result.orderBy(result["appearance_number"].desc()).show(10)
