import sys
import viterbi
import pickle

def main(argv):
    input_file = open(argv[0], 'r')
    output_file = open(argv[1], 'wb')
    sentences = input_file.read().split("\n\n")
    v = viterbi.Viterbi()
    oowv_count = 0.;
    for sentence in sentences:
        word_infos = sentence.split("\n")
        words = []

        for word_info in word_infos:
            elements = word_info.split("\t")
            if elements.__len__() > 1:
                if elements[1] != '_':
                    words.append(elements[1])
        if words.__len__() > 0:
            tags, oowv_counts, tag_names = v.decode(words)

            for word, tag in zip(words, tags):
                output_file.writelines(word+"|"+tag+'\n')

            output_file.writelines('\n')
            oowv_count += oowv_counts

    fp_1 = open("oowv_count.pkl", "w")
    pickle.dump(oowv_count, fp_1)

    fp_2 = open("tag_names.pkl", "w")
    pickle.dump(tag_names, fp_2)

    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])