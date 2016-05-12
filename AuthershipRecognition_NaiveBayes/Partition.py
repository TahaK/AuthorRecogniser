#!/usr/bin/python

import shutil
import os
import sys
import random


# Prepare file tree

def remove_directory_recreate(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def main(argv):
    remove_directory_recreate(argv[2])
    remove_directory_recreate(argv[1])

    # Actual partitioning procedure
    base = argv[0]

    writers = os.listdir(base)

    for writer in writers:
        training_count = 0
        test_count = 0

        # Determining sample files
        docs = os.listdir(os.path.join(base, writer))
        trainingSamples = random.sample(docs, int(len(docs) * 0.6))
        for sample in trainingSamples:
            docs.remove(sample)

        # Copying training and test files

        trainingBase = os.path.join(argv[1], writer)
        os.makedirs(trainingBase)

        for sample in trainingSamples:
            src = os.path.join(base, writer, sample)
            dst = os.path.join(trainingBase, sample)
            shutil.copyfile(src, dst)
            training_count += 1

        testBase = os.path.join(argv[2], writer)
        os.makedirs(testBase)

        for sample in docs:
            src = os.path.join(base, writer, sample)
            dst = os.path.join(testBase, sample)
            shutil.copyfile(src, dst)
            test_count += 1

        print writer + " : " + str(training_count) + " /" + str(test_count)

if __name__ == "__main__":
    main(sys.argv[1:])