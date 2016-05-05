import pickle
import numpy as np


def convert_dict_to_array(dict):
    array = []
    for row in dict:
        d = []
        for column in dict[row]:
            d.append(dict[row][column])
        array.append(d)

    return np.array(array)


def dict_to_array(dict):
    for key in dict:
        d = []
        for column in dict[key]:
            d.append(dict[key][column])
        dict[key] = np.array([d])

    return dict


class Viterbi(object):
    def __init__(self):

        fp_1 = open("state_counts.pkl", "r")

        likelihood_dict = pickle.load(fp_1)
        self.likelihood = dict_to_array(likelihood_dict)

        fp_2 = open("transition_counts.pkl", "r")
        transition_dict = pickle.load(fp_2)

        self.transition_table = convert_dict_to_array(transition_dict)
        self.tags = transition_dict["<s>"].keys()
        self.initial_conditions = convert_dict_to_array({'a': transition_dict.pop("<s>")})[0, :]
        self.transition_table = convert_dict_to_array(transition_dict)

        self.state_graph = []
        for tag_name in transition_dict.values()[0]:
            self.state_graph.append(tag_name)

    def decode(self, observations):
        #viterbi = np.zeros((len(self.state_graph) + 2, len(observations)))
        #backpt = np.ones((len(self.state_graph), len(observations)), 'int32') * -1

        #for s in range(0, len(self.state_graph)):
        #    if observations[0] in self.likelihood:
        #        viterbi[s, 0] = self.initial_conditions[self.state_graph[s]] * self.likelihood[observations[0]]
        #    else:
        #        viterbi[s, 0] = self.initial_conditions[self.state_graph[s]]

        viterbi = np.zeros((len(self.state_graph), len(observations)))
        backpt = np.ones((len(self.state_graph), len(observations)), 'int32') * -1

        if observations[0] in self.likelihood:
            bs = self.likelihood[observations[0]]
        else:
            # vector pointing to a noun
            bs = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        viterbi[:, 0] = np.squeeze(self.initial_conditions * bs)

        for t in range(1, len(observations)):
            if observations[t] in self.likelihood:
                bs = self.likelihood[observations[t]]
            else:
                # vector pointing to a noun
                bs = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            # a = (viterbi[s, t - 1, None].dot(bs(s, 1))).max(0)
            viterbi[:, t] = (viterbi[:, t-1, None].dot(bs) * self.transition_table).max(0)
            backpt[:, t] = (np.tile(viterbi[:, t-1, None], [1, len(self.state_graph)]) * self.transition_table).argmax(0)

        tokens = [viterbi[:, -1].argmax()]
        for i in xrange(len(observations)-1, 0, -1):
            tokens.append(backpt[tokens[-1], i])
        tokens = tokens[::-1]
        tags = []
        for token in tokens:
            tags.append(self.tags[token])
        return tags