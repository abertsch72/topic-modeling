import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.corpus import words

def start():
    newsgroups_train = fetch_20newsgroups(subset="test", shuffle=True, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset="test", shuffle=True, remove=('headers', 'footers', 'quotes'))

    # clean data
    clean_train = clean_data(newsgroups_train.data)
    clean_test = clean_data(newsgroups_test.data)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, smooth_idf=True)
    X = vectorizer.fit_transform(clean_train)
    print(X.shape)

    terms = vectorizer.get_feature_names()
    print(terms)

    from sklearn.decomposition import TruncatedSVD

    # SVD represent documents and terms in vectors
    svd_model = TruncatedSVD(n_components=6, algorithm='randomized', n_iter=100, random_state=122)

    svd_model.fit(X)

    len(svd_model.components_)

    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0], end=' ')
        print("")

    results = svd_model.transform(vectorizer.fit_transform(clean_test))
    print(results)


def clean_data(raw):
    # adapted from https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/

    stop_words = stopwords.words('english')
    dictionary = set(words.words())
    clean_data = []
    for item in raw:
        item = re.sub("[^a-zA-Z#]", " ", item)
        item = item.lower()
        clean_data.append(' '.join([w for w in item.split() if (len(w) > 3 and w not in stop_words and w in dictionary)]))
    return clean_data

start()