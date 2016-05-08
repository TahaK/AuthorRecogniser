import sys
import pickle


def get_cpostag():
    return {'Noun': 0, 'Adj': 0, 'Adv': 0, 'Conj':0, 'Det': 0, 'Dup': 0, 'Interj': 0, 'Ques': 0, 'Verb': 0,
            'Postp': 0, 'Num': 0, 'Pron': 0, 'Punc': 0}


def get_cpostag_with_start():
    return ["<s>", 'Noun', 'Adj', 'Adv', 'Conj', 'Det','Dup', 'Interj', 'Ques', 'Verb', 'Postp','Num', 'Pron', 'Punc']


def get_postag():
    return {'Card': 0, 'Ord': 0, 'Percent': 0, 'Range': 0, 'Real': 0, 'Ratio': 0, 'Distrib': 0, 'Time': 0,
                'Inf': 0, 'PastPart': 0, 'FutPart': 0, 'Prop': 0, 'Zero': 0,
                'PastPart': 0, 'FutPart': 0, 'PresPart': 0,
                'DemonsP': 0, 'QuesP': 0, 'ReflexP': 0, 'PersP': 0, 'QuantP': 0,
                'Noun': 0, 'Adj': 0, 'Adv': 0, 'Conj': 0, 'Det': 0, 'Dup': 0, 'Interj': 0, 'Ques': 0, 'Verb': 0, 'Postp': 0,
                'Num': 0, 'Pron': 0, 'Punc': 0}


def get_postag_with_start():
    return ["<s>", 'Card', 'Ord', 'Percent', 'Range', 'Real', 'Ratio', 'Distrib', 'Time',
                            'Inf', 'PastPart', 'FutPart', 'Prop', 'Zero',
                            'PastPart', 'FutPart', 'PresPart',
                            'DemonsP', 'QuesP', 'ReflexP', 'PersP', 'QuantP',
                            'Noun', 'Adj', 'Adv', 'Conj', 'Det', 'Dup', 'Interj', 'Ques', 'Verb', 'Postp',
                            'Num', 'Pron', 'Punc']


def normalize(twoD_dict):
    for row in twoD_dict:
        total = 0
        for column in twoD_dict[row]:
            total += twoD_dict[row][column]
        if total != 0:
            for column in twoD_dict[row]:
                twoD_dict[row][column] /= total
    return twoD_dict


def main(argv):

    tags = []
    words = []

    is_cpostag = True
    pos_name = argv[1]
    if pos_name == '--postag':
        is_cpostag = False

    transition_counts = {}
    state_counts = {}
    if is_cpostag:
        for tag in get_cpostag_with_start():
            transition_counts[tag] = get_cpostag()
    else:
        for tag in get_postag_with_start():
            transition_counts[tag] = get_postag()


    training_file__ = open(argv[0], 'r')
    training_file = training_file__.read()
    sentences = training_file.split('\n\n')
    for sentence in sentences:
        word_infos = sentence.split("\n")

        previous_element = "<s>"
        for word_info in word_infos:
            elements = word_info.split("\t")
            if elements.__len__() > 2:
                if elements[1] != '_':
                    if is_cpostag:
                        current_tag = elements[3]
                        if current_tag in get_cpostag_with_start():
                            transition_counts[previous_element][current_tag] += 1.
                            tags.append(current_tag)
                            previous_element = current_tag
                            words.append(elements[1])
                    else:
                        current_tag = elements[4]
                        if current_tag in get_postag_with_start():
                            transition_counts[previous_element][current_tag] += 1.
                            tags.append(current_tag)
                            previous_element = current_tag
                            words.append(elements[1])

    for tag, word in zip(tags, words):
        if word not in state_counts:
            if is_cpostag:
                state_counts[word] = get_cpostag()
            else:
                state_counts[word] = get_postag()
        state_counts[word][tag] = +1

    state_counts = normalize(state_counts)
    transition_counts = normalize(transition_counts)

    fp_1 = open("state_counts.pkl", "w")
    pickle.dump(state_counts, fp_1)
    fp_2 = open("transition_counts.pkl", "w")
    pickle.dump(transition_counts, fp_2)


if __name__ == "__main__":
    main(sys.argv[1:])
