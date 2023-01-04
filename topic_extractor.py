import pandas as pd
import spacy
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases
from pprint import pprint


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(local_texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'told', 'make', 'edu', 'use',
                       'could', 'many', 'said', 'mr', 'bbc', 'also', 'ms', 'one',
                       'two', 'three', 'year', 'would', 'says', 'government',
                       'people', 'minister', 'may', 'need', 'see'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in local_texts]


def make_bigrams(local_texts):
    return [bigram_mod[doc] for doc in local_texts]


def make_trigrams(local_texts):
    return [trigram_mod[bigram_mod[doc]] for doc in local_texts]


def lemmatization(local_texts, allowed_postags=None):
    """https://spacy.io/api/annotation"""

    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in local_texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()


def get_topic_modeling_result(data_words, data_words_nostops):
    global bigram_mod, trigram_mod

    # data = pd.read_csv('./dataset.csv')
    # data = data.drop(columns=['crt', 'title', 'published_date', 'authors', 'description', 'section', 'link'], axis=1)
    # Remove punctuation
    # content['data_text_processed'] = content.map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    # content['data_text_processed'] = content['data_text_processed'].map(lambda x: x.lower())

    # data_list_of_sent = content.data_text_processed.values.tolist()
    # data_words = list(sent_to_words(data_list_of_sent))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    # data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Number of topics
    num_topics = 30
    num_words = 30
    # Build LDA model
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
    # Print the Keyword in the "num_topics" topics

    doc_lda = lda_model[corpus]

    # topics_w_words = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    pprint(lda_model.print_topics(num_topics=num_topics, num_words=num_words))
    return lda_model.print_topics(num_topics=num_topics, num_words=num_words)

    # final_sentence_words = []
    # final_sentences = []
    #
    # for topic in topics_w_words:
    #     sentence_words = str(topic[1]).replace("\"", "").split("+")
    #     index = 0
    #     final_sentence_words = []
    #     for sentence_word in sentence_words:
    #         sentence_words[index] = sentence_word.split("*")
    #         index += 1
    #     for nb, word in sentence_words:
    #         final_sentence_words.append(word.strip())
    #
    #     final_sentences.append(final_sentence_words.copy())
    #
    # print(final_sentences)


if __name__ == '__main__':
    data = pd.read_csv('./dataset.csv')
    data = data.drop(columns=['crt', 'title', 'published_date', 'authors', 'description', 'section', 'link'], axis=1)
    # Remove punctuation
    data['data_text_processed'] = data['content'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    data['data_text_processed'] = data['data_text_processed'].map(lambda x: x.lower())



    data_list_of_sent = data.data_text_processed.values.tolist()
    data_words = list(sent_to_words(data_list_of_sent))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]


    # Number of topics
    num_topics = 30
    num_words = 30
    # Build LDA model
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
    # Print the Keyword in the "num_topics" topics

    doc_lda = lda_model[corpus]

    topics_w_words = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    pprint(lda_model.print_topics(num_topics=num_topics, num_words=num_words))

    final_sentence_words = []
    final_sentences = []

    for topic in topics_w_words:
        sentence_words = str(topic[1]).replace("\"", "").split("+")
        index = 0
        final_sentence_words = []
        for sentence_word in sentence_words:
            sentence_words[index] = sentence_word.split("*")
            index += 1
        for nb, word in sentence_words:
            final_sentence_words.append(word.strip())

        final_sentences.append(final_sentence_words.copy())

    print(final_sentences)