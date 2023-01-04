import pandas as pd
import spacy
# python -m spacy download en_core_web_lg --> to download the large model (560 MB)
# python -m spacy download en_core_web_md --> to download the medium model (40 MB)


def get_person_names(content):
    labeled_content = ner(content)
    person_names_by_text = []
    for word in labeled_content.ents:
        if word.label_ == "PERSON":
            person_names_by_text.append(word.text)
    return person_names_by_text


if __name__ == '__main__':
    ner = spacy.load("en_core_web_lg")
    news_df = pd.read_csv("../../../PycharmProjects/person_extractor/dataset.csv")
    news_content = list(news_df["content"])
    for content in news_content:
        print(get_person_names(content))
