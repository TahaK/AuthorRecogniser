#!/usr/bin/python

import re
import os
import math
import operator
import numpy
from scipy.stats import norm
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

vocabulary = set([])


def occur_dict(items):
    d = {}
    for i in items:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def get_doc_paths_dic_by_writer(base):
    authors_ = os.listdir(base)
    # Build training set dictionary
    dict_by_writer = {}
    for author in authors_:
        # Determining sample files
        author_path = os.path.join(base, author)
        docs = os.listdir(author_path)
        dict_by_writer.setdefault(author, [])
        for doc in docs:
            doc_path = os.path.join(author_path, doc)
            dict_by_writer[author].append(doc_path)

    return dict_by_writer


def tokenize(doc):
    doc = re.sub("[^A-Za-z\xFC-\xFE\xF6\xF0\xE7\xE2\xDC-\xDE\xD6\xD0]", "\n", doc)
    doc = doc.decode('windows-1254').lower()
    words = doc.split()
    map(lambda x: vocabulary.add(x), words)
    return words


# Read Documents and extract document counts

def extract_word_counts(dictionary_of_documents):
    document_count_by_writer = {}
    word_counts_by_writer = {}
    total_word_count_by_writer = {}
    word_length_stats = {}
    for writer in dictionary_of_documents:
        writer_docs = ''
        doc_count = 0
        for docPath in dictionary_of_documents[writer]:
            doc_file_ = open(docPath, 'r')
            doc__ = doc_file_.read()
            writer_docs += doc__
            doc_count += 1

        words = tokenize(writer_docs)
        dictionary = occur_dict(words)

        total_word_count_in_writer = 0
        for counts in dictionary.values():
            total_word_count_in_writer += counts

        word_lengths = map(lambda x: len(x), words)

        mean_word_length = numpy.mean(word_lengths)
        std_word_length = numpy.std(word_lengths)

        word_length_stats[writer] = {'mean': mean_word_length, 'std': std_word_length}
        document_count_by_writer[writer] = doc_count
        word_counts_by_writer[writer] = dictionary
        total_word_count_by_writer[writer] = total_word_count_in_writer

    total_document_count = 0
    for writer in document_count_by_writer:
        total_document_count += document_count_by_writer[writer]

    return document_count_by_writer, word_counts_by_writer, \
           total_document_count, total_word_count_by_writer, word_length_stats


def calculate_likelihood(word, word_count_dict, total_words, author, alfa_):
    word_counts = word_count_dict[author]
    count = alfa_
    if word in word_counts:
        count += word_count_dict[author][word]
    likelihood = float(count) / (total_words[author] + alfa_ * vocabulary.__sizeof__())
    return likelihood


def calculate_prior(document_count_dict, total_docs_count, author):
    prior = float(document_count_dict[author]) / total_docs_count
    return prior


def calculate_word_length_probability(words, word_statistics, author):
    word_lengths = map(lambda x: len(x), words)
    sample_mean = numpy.mean(word_lengths)
    probability = norm.pdf(sample_mean, loc=word_statistics[author]['mean'], scale=word_statistics[author]['std'])
    return probability


# Estimates the author of the document

def estimate_author(document_, document_count_dict, word_count_dict, total_document_count, total_words_dictionary,
                    writer_list, alfa_, extra, word_length_stats_):
    probability_author = {}
    words = tokenize(document_)
    for writer in writer_list:
        probability = 0
        prior = calculate_prior(document_count_dict, total_document_count, writer)
        # print math.log(prior)
        probability += math.log(prior)
        for word in words:
            likelihood_word = calculate_likelihood(word, word_count_dict, total_words_dictionary, writer, alfa_)
            # print math.log(likelihood_word)
            probability += math.log(likelihood_word)

        if extra:
            probability += 100*math.log(calculate_word_length_probability(words, word_length_stats_, writer))

        probability_author[writer] = probability

    most_likely_author = max(probability_author.iteritems(), key=operator.itemgetter(1))[0]
    return most_likely_author


def column(matrix, i):
    return [row[i] for row in matrix]


def calculate_macro_precision(confusion_matrix_):
    precision = []
    predicted_counts = []
    for i in range(0, len(confusion_matrix_)):
        predicted_for_class_i = column(confusion_matrix_, i)
        predicted_counts.append(sum(predicted_for_class_i))

    for j in range(0, len(confusion_matrix_)):
        if predicted_counts[j] != 0:
            precision.append(float(confusion_matrix_[j][j]) / predicted_counts[j])
        else:
            precision.append(0)
    return numpy.mean(precision)


def calculate_macro_recall(confusion_matrix_):
    recall = []
    true_counts = map(lambda row: sum(row), confusion_matrix_)
    for i in range(0, len(confusion_matrix_)):
        if true_counts[i] != 0:
            recall.append(float(confusion_matrix_[i][i])/true_counts[i])
        else:
            recall.append(0)
    return numpy.mean(recall)


def calculate_micro_recall(confusion_matrix_):
    true_counts = sum(map(lambda row: sum(row), confusion_matrix_))
    tp = 0
    for i in range(0, len(confusion_matrix_)):
        tp += confusion_matrix_[i][i]
    return float(tp)/true_counts


def calculate_micro_precision(confusion_matrix_):
    predicted_counts = []
    for i in range(0, len(confusion_matrix_)):
        predicted_for_class_i = column(confusion_matrix_, i)
        predicted_counts.append(sum(predicted_for_class_i))
    predicted_count = sum(predicted_counts)
    tp = 0.0
    for j in range(0, len(confusion_matrix_)):
        tp += confusion_matrix_[j][j]
    return float(tp)/predicted_count


def calculate_f_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main(argv):
    training_dict = get_doc_paths_dic_by_writer(argv[0])
    test_dict_by_writer = get_doc_paths_dic_by_writer(argv[1])
    if argv[2]=="-ext":
        extra_feature = True
    else:
        extra_feature = False

    alfa = 0.0001

    print "Training ."
    document_count_train, word_counts_train, total_doc_count, total_words_dict, word_length_stats = extract_word_counts(
        training_dict)

    authors = os.listdir(argv[1])

    y_true = []
    y_pred = []
    print "Predicting ."
    for author in test_dict_by_writer:
        for docPath in test_dict_by_writer[author]:
            y_true.append(author)
            file_ = open(docPath, 'r')
            document = file_.read()
            predicted_author = estimate_author(document, document_count_train, word_counts_train,
                                               total_doc_count, total_words_dict, authors, alfa, extra_feature,
                                               word_length_stats)
            y_pred.append(predicted_author)
            sys.stdout.write('.')
        print "."

    cm = metrics.confusion_matrix(y_true, y_pred)
    macro_precision_score = calculate_macro_precision(cm)
    macro_recall_score = calculate_macro_recall(cm)

    micro_precision_score = calculate_micro_precision(cm)
    micro_recall_score = calculate_micro_recall(cm)

    print "precision score - macro : " + str(macro_precision_score)
    print "recall score - macro : " + str(macro_recall_score)
    print "f1 score - macro : " + str(calculate_f_score(macro_precision_score, macro_recall_score))

    print "precision score - micro : " + str(micro_precision_score)
    print "recall score - micro : " + str(micro_recall_score)
    print "f1 score - micro : " + str(calculate_f_score(micro_precision_score, micro_recall_score))

    # The rest is just for producing pretty plots :)
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(authors))

        plt.xticks(tick_marks, authors, rotation=90, fontsize='small')
        plt.yticks(tick_marks, authors, fontsize='small')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    print(cm)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
