import sys
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def main(argv):
    output_filename = open(argv[0], 'r')
    test_gold_filename = open(argv[1], 'r')

    gold_words = []
    gold_tags = []

    gold_sentences = test_gold_filename.read().split('\n\n')
    for sentence in gold_sentences:
        word_infos = sentence.split("\n")

        for word_info in word_infos:
            elements = word_info.split("\t")
            if elements.__len__() > 1:
                if elements[1] != '_':
                    gold_tags.append(elements[3])
                    gold_words.append(elements[1])

    output_words = []
    output_tags = []

    output_sentences = output_filename.read().split('\n\n')
    for sentence in output_sentences:
        word_infos = sentence.split("\n")

        for word_info in word_infos:
            elements = word_info.split("|")
            if elements.__len__() > 1:
                if elements[1] != '_':
                    output_tags.append(elements[1])
                    output_words.append(elements[0])

    if output_tags.__len__() != gold_tags.__len__():
        print 'Error output tag number does not equal gold tags number'

    cm = metrics.confusion_matrix(gold_tags, output_tags)
    macro_precision_score = calculate_macro_precision(cm)
    macro_recall_score = calculate_macro_recall(cm)

    micro_precision_score = calculate_micro_precision(cm)
    micro_recall_score = calculate_micro_recall(cm)

    print "precision score - macro : " + str(macro_precision_score)
    print "recall score - macro : " + str(macro_recall_score)
    print "f1 score - macro : " + str(calculate_f_score(macro_precision_score, macro_recall_score))

    print "precision score - micro : " + str(micro_precision_score)
    print "recall score - micro : " + str(micro_recall_score)
    # print "f1 score - micro : " + str(calculate_f_score(micro_precision_score, micro_recall_score))

    # The rest is just for producing pretty plots :)
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        #tick_marks = np.arange(len(authors))

        #plt.xticks(tick_marks, authors, rotation=90, fontsize='small')
        #plt.yticks(tick_marks, authors, fontsize='small')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    print(cm)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    plt.show()

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
    return np.mean(precision)


def calculate_macro_recall(confusion_matrix_):
    recall = []
    true_counts = map(lambda row: sum(row), confusion_matrix_)
    for i in range(0, len(confusion_matrix_)):
        if true_counts[i] != 0:
            recall.append(float(confusion_matrix_[i][i])/true_counts[i])
        else:
            recall.append(0)
    return np.mean(recall)


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


if __name__ == "__main__":
    main(sys.argv[1:])