import shutil
import os
import sys
import re
import pickle


def normalize(twoD_dict):
    for row in twoD_dict:
        total = 0
        for column in twoD_dict[row]:
            total += twoD_dict[row][column]
        for column in twoD_dict[row]:
            twoD_dict[row][column] /= total
    return twoD_dict


def main(argv):

    tags = []
    words = []

    start_element = "<s>"
    cpostag = {'Noun' : 0, 'Adj':0, 'Adv':0, 'Conj':0, 'Det':0,'Dup':0, 'Interj':0, 'Ques':0, 'Verb':0, 'Postp':0, 'Num':0, 'Pron':0, 'Punc':0}
    cpostag_with_start = ["<s>",'Noun' , 'Adj', 'Adv', 'Conj', 'Det','Dup', 'Interj', 'Ques', 'Verb', 'Postp', 'Num', 'Pron', 'Punc']

    transition_counts = {}
    state_counts = {}
    for tag in cpostag_with_start:
        transition_counts[tag] = {'Noun' : 0.0, 'Adj':0.0, 'Adv':0.0, 'Conj':0.0, 'Det':0.0,'Dup':0.0, 'Interj':0.0, 'Ques':0.0, 'Verb':0.0, 'Postp':0.0, 'Num':0.0, 'Pron':0.0, 'Punc':0.0}

    training_file__ = open(argv[0], 'r')
    training_file = training_file__.read()
    sentences = training_file.split('\n\n')
    for sentence in sentences:
        word_infos = sentence.split("\n")

        previous_element = start_element
        for word_info in word_infos:
            elements = word_info.split("\t")
            if elements.__len__() > 1:
                if elements[1] != '_':
                    current_tag = elements[3]
                    if current_tag in cpostag_with_start:
                        transition_counts[previous_element][current_tag] += 1
                        tags.append(current_tag)
                        previous_element = current_tag
                        words.append(elements[1])

    for tag, word in zip(tags, words):
        if word not in state_counts:
            state_counts[word] = {'Noun' : 0.0, 'Adj':0.0, 'Adv':0.0, 'Conj':0.0, 'Det':0.0,'Dup':0.0, 'Interj':0.0, 'Ques':0.0, 'Verb':0.0, 'Postp':0.0, 'Num':0.0, 'Pron':0.0, 'Punc':0.0}
        state_counts[word][tag] = +1

    state_counts = normalize(state_counts)
    transition_counts  =normalize(transition_counts)

    fp_1 = open("state_counts.pkl","w")
    pickle.dump(state_counts, fp_1)
    fp_2 = open("transition_counts.pkl","w")
    pickle.dump(transition_counts, fp_2)


if __name__ == "__main__":
    main(sys.argv[1:])