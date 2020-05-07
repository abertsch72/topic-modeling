import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.corpus import words
import itertools
import cur
import time
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

NUM_TOPICS = 6


def col_select(num, arr):
    cols_chosen = []
    cols_chosen.append(np.random.randint(0, arr.shape[1]))
    c = arr[:, cols_chosen[0]]
    while(len(cols_chosen) < num):
        next = np.random.randint(0, arr.shape[1])
        if(next not in cols_chosen):
            cols_chosen.append(next)
            col = arr[:, next]
            c = np.column_stack([c, col])
    return c

def CUR_fit_transform(data, k):
    c = col_select(k, data)
    r = np.transpose(col_select(k, np.transpose(data)))
    u = r @ np.linalg.pinv(data) @ c
    a = c @ np.linalg.pinv(u) @ r
    return c, u, r

def sum_cat(data):
    num_each = [0, 0, 0, 0, 0, 0]
    for i in range(len(data)):
        num_each[data[i]] += 1
    return num_each


def find_num_in_each_cat(data):
    computers = [1, 2, 3, 4, 5]
    recreation = [7, 8, 9, 10]
    science = [11, 12, 13, 14]
    forsale = [6]
    politics = [16, 17, 18]
    religion = [0, 15, 19]
    num_each = [0, 0, 0, 0, 0, 0]

    correct = []
    for i in range(len(data)):
        curr = data[i]
        if curr in computers:
            num_each[0] += 1
            correct.append(0)
        elif curr in recreation:
            num_each[1] += 1
            correct.append(1)
        elif curr in science:
            num_each[2] += 1
            correct.append(2)
        elif curr in forsale:
            num_each[3] += 1
            correct.append(3)
        elif curr in politics:
            num_each[4] += 1
            correct.append(4)
        elif curr in religion:
            num_each[5] += 1
            correct.append(5)

    return num_each, correct


def find_top_n(result, n):
    top = []

    for i in range(len(result)):
        curr_max = []
        for j in range(len(result[i])):
            if len(curr_max) < n:
                curr_max.append([j, result[i][j]])
            else:
                for k in range(len(curr_max)):
                    if result[i][j] > curr_max[k][1]:
                        curr_max[k] = [j, result[i][j]]
                        break
        top.append([c[0] for c in curr_max])
    return top


def permute_labels(results, perm=1):
    for i in range(len(results)):
        if (type(results[i]) == int):
            results[i] = (results[i] + perm) % 6
        else:
            for j in range(len(results[i])):
                results[i][j] = (results[i][j] + perm) % 6
    return results


def validate_firm(results, correct, num_each):
    results = find_top_n(results, 1)
    results = [r[0] for r in results]
    accuracy = []
    precision = [[], [], [], [], [], []]
    recall = [[], [], [], [], [], []]
    permutations = itertools.permutations([0, 1, 2, 3, 4, 5])

    for per in permutations:
        true_pos = [0, 0, 0, 0, 0, 0]
        guessed = [0, 0, 0, 0, 0, 0]
        for j in range(len(results)):
            if correct[j] == per[results[j]]:
                true_pos[correct[j]] += 1
            guessed[per[results[j]]] += 1
        accuracy.append(sum(true_pos) / len(results))

        precision.append([])
        recall.append([])

        i = len(precision) - 1
        for k in range(NUM_TOPICS):
            if guessed[k] != 0:
                precision[i].append(true_pos[k] / guessed[k])
            else:
                precision[i].append(0)
            recall[i].append(true_pos[k] / num_each[k])

    print("STRICT VALIDATION: RESULTS")
    print("Accuracy: " + str(max(accuracy)))
    m = accuracy.index(max(accuracy))
    print("Average Precision (detailed on next line): " + str(sum(precision[m])/NUM_TOPICS))
    print(precision[m])
    print("Average Recall (detailed on next line): " + str(sum(recall[m])/NUM_TOPICS))
    print(recall[m])


def validate_lax(results, correct, num_each):
    results = find_top_n(results, 2)

    accuracy = []
    precision = []
    recall = []
    permutations = itertools.permutations([0, 1, 2, 3, 4, 5])

    for per in permutations:
        true_pos = [0, 0, 0, 0, 0, 0]
        guessed = [0, 0, 0, 0, 0, 0]
        for j in range(len(results)):
            if correct[j] == per[results[j][0]] or correct[j] == per[results[j][1]]:
                true_pos[correct[j]] += 1
                guessed[correct[j]] += 1
            else:
                guessed[per[results[j][0]]] += 1
        accuracy.append(sum(true_pos) / len(results))

        precision.append([])
        recall.append([])

        i = len(precision) - 1
        for k in range(NUM_TOPICS):
            if guessed[k] != 0:
                precision[i].append(true_pos[k] / guessed[k])
            else:
                precision[i].append(0)
            recall[i].append(true_pos[k] / num_each[k])
    print("LAX (TOP 2) VALIDATION: RESULTS")
    print("Accuracy: " + str(max(accuracy)))
    m = accuracy.index(max(accuracy))
    print("Average Precision (detailed on next line): " + str(sum(precision[m])/NUM_TOPICS))
    print(precision[m])
    print("Average Recall (detailed on next line): " + str(sum(recall[m])/NUM_TOPICS))
    print(recall[m])

def run_validate_SVD(vectorizer, data, correct_labels, num_each):
    time_start = time.time()
    svd_model = TruncatedSVD(n_components=6, algorithm='randomized', n_iter=100, random_state=122)
    results = svd_model.fit_transform(data)
    time_end = time.time()

    print("SVD RESULTS: \n" + "-" * 70)
    validate_lax(results, correct_labels, num_each)
    validate_firm(results, correct_labels, num_each)
    print("*" * 70)
    print("TIME TAKEN: " + str(time_end - time_start) + " milliseconds")
    print("*" * 70)
    print("TOP TERMS:")
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0], end=' ')
        print("")
    print("\n\n\n")


def run_validate_NMF(vectorizer, data, correct_labels, num_each):
    time_start = time.time()
    nmf_model = NMF(n_components=6, max_iter=100, random_state=122)
    nmf_results = nmf_model.fit_transform(data)
    time_end = time.time()

    print("NMF RESULTS: \n" + "-" * 70)
    validate_lax(nmf_results, correct_labels, num_each)
    validate_firm(nmf_results, correct_labels, num_each)
    print("*" * 70)
    print("TIME TAKEN: " + str(time_end - time_start) + " milliseconds")
    print("*" * 70)
    print("TOP TERMS:")
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(nmf_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0], end=' ')
        print("")
    print("\n\n\n")


def run_validate_slowCUR(vectorizer, data, correct_labels, num_each):
    time_start = time.time()
    c, u, r = CUR_fit_transform(data.toarray(), NUM_TOPICS)
    cur1_results = (data.toarray() @ np.linalg.pinv(r) @ u)
    time_end = time.time()

    print("CUR RESULTS (SLOW VERSION): \n" + "-" * 70)
    validate_lax(cur1_results, correct_labels, num_each)
    validate_firm(cur1_results, correct_labels, num_each)
    print("*" * 70)
    print("TIME TAKEN: " + str(time_end - time_start) + " milliseconds")
    print("*" * 70)
    print("TOP TERMS:")
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(u @ np.linalg.pinv(c)):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0], end=' ')
        print("")
    print("\n\n\n")


def run_validate_CUR(vectorizer, data, correct_labels, num_each):
    time_start = time.time()
    C, U, R = cur.cur_decomposition(data.toarray(), NUM_TOPICS)
    cur2_result = (data.toarray() @ np.linalg.pinv(R) @ U)
    time_end = time.time()

    print("CUR RESULTS (LIBRARY VERSION): \n" + "-" * 70)
    validate_lax(cur2_result, correct_labels, num_each)
    validate_firm(cur2_result, correct_labels, num_each)
    print("*" * 70)
    print("TIME TAKEN: " + str(time_end - time_start) + " milliseconds")
    print("*" * 70)
    print("TOP TERMS:")
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(U @ np.linalg.pinv(C)):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        print("Topic " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0], end=' ')
        print("")
    print("\n\n\n")


def entrypoint():
    newsgroups = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'), random_state=1)
    num_each, correct_labels = find_num_in_each_cat(newsgroups.target)

    # clean data
    clean = clean_data(newsgroups.data)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, smooth_idf=True, max_features=1000)
    X = vectorizer.fit_transform(clean)
    run_validate_SVD(vectorizer, X, correct_labels, num_each)
    run_validate_NMF(vectorizer, X, correct_labels, num_each)
    run_validate_CUR(vectorizer, X, correct_labels, num_each)
    run_validate_slowCUR(vectorizer, X, correct_labels, num_each)
    """
    import umap 
    import matplotlib.pyplot as plt
    
    X_topics = svd_model.fit_transform(X)
    embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

    plt.figure(figsize=(7, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=newsgroups.target,
                s=10,  # size
                edgecolor='none'
                )
    plt.show()u @ np.linalg.pinv(c)
    """


def clean_data(raw):
    # adapted from https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/

    stop_words = stopwords.words('english')
    dictionary = set(words.words())
    clean_data = []
    for item in raw:
        item = re.sub("[^a-zA-Z#]", " ", item)
        item = item.lower()
        clean_data.append(
            ' '.join([w for w in item.split() if (len(w) > 3 and w not in stop_words and w in dictionary)]))

    return clean_data


entrypoint()
