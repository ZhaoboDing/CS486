from collections import defaultdict


class NaiveBayes:
    def __init__(self, word_map):
        self.word_set = set(word_map)
        self.prob = defaultdict(lambda: defaultdict(lambda: 0))

    def fit(self, docs):
        stat = defaultdict(lambda: defaultdict(lambda: 0))
        for document in docs:
            for word in document.word_list:
                stat[word][document.label] += 1

        for word in self.word_set:
            total = sum(stat[word].values())
            for label in stat[word].keys():
                self.prob[word][label] = (stat[word][label] + 1) / (total + 2)

    def predict(self, word_list, poss_labels):
        p = defaultdict(str)
        for label in poss_labels:
            p[label] = 1
            for word in self.word_set:
                if word in word_list:
                    p[label] *= self.prob[word][label]
                else:
                    p[label] *= 1 - self.prob[word][label]

        return max(p, key=p.get)
