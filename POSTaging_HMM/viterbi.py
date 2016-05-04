import pickle
import numpy as np


class Viterbi(object):
    def __init__(self):

        fp_1 = open("state_counts.pkl", "r")
        self.likelihood = pickle.load(fp_1)

        fp_2 = open("transition_counts.pkl", "r")
        self.transition_table = pickle.load(fp_2)

        self.initial_conditions = self.transition_table["<s>"]
        self.state_graph = []
        for tag_name in self.initial_conditions:
            self.state_graph.append(tag_name)

    def decode(self, observations):
        viterbi = np.zeros((len(self.state_graph) + 2, len(observations)))
        backpt = np.ones((len(self.state_graph), len(observations)), 'int32') * -1

        for s in range(0, len(self.state_graph)):
            if observations[0] in self.likelihood:
                viterbi[s, 0] = self.initial_conditions[self.state_graph[s]] * self.likelihood[observations[0]]
            else:
                viterbi[s, 0] = self.initial_conditions[self.state_graph[s]]

        # viterbi[:, 0] = np.squeeze(self.initial_conditions * self.likelihood(observations[0]))

        for t in range(1, len(observations)):
            for s in range(0, len(self.state_graph)):
                if observations[t] in self.likelihood:
                    bs = self.likelihood[observations[t]]
                else:
                    # vector pointing to a noun
                    viterbi[s, t] = (viterbi[s, t-1, None].dot(bs) * self.transition_table).max(0)
