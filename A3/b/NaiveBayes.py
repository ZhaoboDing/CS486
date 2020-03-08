import numpy as np
from collections import defaultdict


class NaiveBayes:
    def __init__(self, word_map):
        self.word_set = set(word_map)

    def fit(self, docs, labels):
        num_words = len(self.word_set)
        num_label = len(labels)
        stat = np.zeros((num_words, num_label))

        for document in docs:
            for word in document.word_list:
                stat[word][document.label] += 1

        label_sum = np.sum(stat, axis=1)
        word_sum = np.sum(stat, axis=0)
        # prob_given_label[word][label] stores the value of Pr(word | label)
        self.prob_given_label = (stat + 1) / (label_sum + num_words)
        # prob_given_word[label][word] stores the value of Pr(label | word)
        self.prob_given_word = np.transpose((stat + 1) / (word_sum + num_label))

    def predict(self, word_list, poss_labels):
        p = defaultdict(str)
        for label in poss_labels:
            p[label] = 1
            for word in self.word_set:
                if word in word_list:
                    p[label] *= self.prob_given_label[word][label]
                else:
                    p[label] *= 1 - self.prob_given_label[word][label]

        return max(p, key=p.get)
