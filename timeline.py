import main
import utils
from pyspark.sql.functions import count, date_format


sc = main.init_spark()

df = utils.read_main_dataset(sc)
df = df.withColumn("published_date", date_format(df['published_date'], 'yyyy-MM'))
topics = utils.read_topics(sc)


month_stats = df.groupBy("published_date", "topic") \
    .agg(count("crt").alias("appearance_number"))

result = month_stats.join(topics, month_stats.topic == topics.topic_id, "inner") \
    .drop("topic")
result.orderBy(result["topic_id"].asc(), result["published_date"].asc()).show(n=500)
