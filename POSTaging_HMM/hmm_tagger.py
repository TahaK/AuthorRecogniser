import sys
import viterbi


def main(argv):
    input_file = open(argv[0], 'r')
    output_file = open(argv[1], 'wb')
    sentences = input_file.read().split("\n")
    v = viterbi.Viterbi()

    for sentence in sentences:
        words = sentence.split()
        tags = v.decode(words)

        for word, tag in zip(words, tags):
            output_file.write(word+"|"+tag+"\\n")

        output_file.writelines('')

    output_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])