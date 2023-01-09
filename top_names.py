import main
import utils
import re


sc = main.init_spark()

df = utils.read_dataset_with_topics(sc).sample(0.15)  # a 10% sample from the dataset

persons_rdd = df.rdd.map(lambda column: column[5])
persons_df = persons_rdd.flatMap(lambda line: re.findall("'([^']*)'", line)) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda instance: (instance[0], instance[1]))

persons_df_with_statistics = persons_df.toDF(["name", "appearance_number"])
persons_df_with_statistics.orderBy(persons_df_with_statistics["appearance_number"].desc()).show(10)
