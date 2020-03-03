from collections import defaultdict


class NaiveBayes:
    def __init__(self, word_map):
        self.word_set = set(word_map)
        self.prob = defaultdict(lambda: defaultdict(lambda: 0))

    def fit(self, docs, labels):
        stat = defaultdict(lambda: defaultdict(lambda: 0))
        for document in docs:
            for word in document.word_list:
                stat[document.label][word] += 1

        for label in labels:
            total = sum(stat[label].values())
            for word in stat[label]:
                self.prob[label][word] = (stat[label][word] + 1) / (total + len(labels))

    def predict(self, word_list, poss_labels):
        p = defaultdict(str)
        for label in poss_labels:
            p[label] = 1
            for word in self.word_set:
                if word in word_list:
                    p[label] *= self.prob[label][word]
                else:
                    p[label] *= 1 - self.prob[label][word]

        return max(p, key=p.get)
