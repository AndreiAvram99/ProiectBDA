import main
import utils


sc = main.init_spark()

df = utils.read_dataset_with_topics(sc).sample(0.1)  # a 10% sample from the dataset
topics = utils.read_topics(sc)

topic_ids_rdd = df.rdd.map(lambda column: column[9])
topic_ids_df = topic_ids_rdd.map(lambda topic_id: (topic_id, 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda instance: (instance[0], instance[1]))

topic_ids_df_with_stats = topic_ids_df.toDF(["id_of_the_topic", "appearance_number"])

top_topics = topic_ids_df_with_stats \
    .join(topics, topic_ids_df_with_stats.id_of_the_topic == topics.topic_id, "inner") \
    .drop("topic_id")
top_topics.orderBy(top_topics["appearance_number"].desc()).show(10)
