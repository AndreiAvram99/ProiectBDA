import pandas as pd
import spacy
# python -m spacy download en_core_web_lg --> to download the large model (560 MB)
# python -m spacy download en_core_web_md --> to download the medium model (40 MB)


ner = spacy.load("en_core_web_lg")
file = open("all_person_names_lg_model.txt", "a", encoding="utf-8")

news_df = pd.read_csv("dataset.csv")
news_content = list(news_df["content"])

for content in news_content:
    labeled_content = ner(content)
    for word in labeled_content.ents:
        # print(word.text, word.label_)
        if word.label_ == "PERSON":
            file.write(word.text + ", ")

file.close()
