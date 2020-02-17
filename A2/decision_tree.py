import math
from collections import Counter
from .configs import *
from .max_priority_queue import QueueNode
from queue import PriorityQueue


def entropy(p):
    return -p * math.log2(p)


def information_entropy(doc_labels):
    stat = Counter(doc_labels)
    doc_length = len(doc_labels)
    if doc_length:
        return sum([entropy(freq / doc_length) for freq in stat.values()])
    else:
        return 1


def information_gain(docs, word, method=FEATURE_SELECTION_MECHANISM):
    e = [doc.label for doc in docs]
    e1 = [doc.label for doc in docs if word in doc]
    e2 = [doc.label for doc in docs if word not in doc]
    ie = information_entropy(e)
    ie1 = information_entropy(e1)
    ie2 = information_entropy(e2)

    if method is AVERAGE_INFORMATION_GAIN:
        return ie - (ie1 + ie2) / 2
    elif method is WEIGHTED_INFORMATION_GAIN:
        return ie - (len(e1) * ie1 + len(e2) * ie2) / len(docs)


class DecisionTreeNode:
    def __init__(self, docs):
        self.word_id = None
        self.word = None
        self.docs = docs
        self.included = None
        self.excluded = None
        self.order = None
        self.leaf = True
        self.information_gain = 0
        labels = [doc.label for doc in docs]
        self.pred = Counter(labels).most_common(1)[0][0]

    def split(self, word, order):
        in_list = [doc for doc in self.docs if word in doc]
        out_list = [doc for doc in self.docs if word not in doc]
        self.included = DecisionTreeNode(in_list)
        self.excluded = DecisionTreeNode(out_list)

        self.order = order
        self.leaf = False
        del self.docs

        return self.included, self.excluded

    def best_split(self, words, method):
        split_results = {word: information_gain(self.docs, word, method) for word in words}
        chosen_word = max(split_results, key=split_results.get)
        self.word_id = chosen_word
        self.information_gain = split_results[chosen_word]
        return chosen_word, split_results[chosen_word]


class DecisionTree:
    def __init__(self, word_map, algo=WEIGHTED_INFORMATION_GAIN):
        self.root = None
        self.word_map = word_map
        self.word_set = set(word_map)
        self.size = 0
        self.algo = algo

    def train(self, docs, size=AIM_TREE_SIZE):
        self.root = DecisionTreeNode(docs)
        q = PriorityQueue()
        split_word, ig = self.root.best_split(self.word_set, self.algo)
        self.root.word = self.word_map[split_word]
        q.put(QueueNode(ig, self.root, split_word))
        self.size = 0

        while self.size < size:
            node = q.get()
            if node.root.leaf:
                if node.gain:
                    inc, exc = node.root.split(node.split_word, self.size)
                    spi, igi = inc.best_split(self.word_set, self.algo)
                    spe, ige = exc.best_split(self.word_set, self.algo)
                    q.put(QueueNode(igi, inc, spi))
                    q.put(QueueNode(ige, exc, spe))
                    inc.word = self.word_map[spi]
                    exc.word = self.word_map[spe]
                    self.size += 1
                else:
                    break

        while not q.empty():
            node = q.get()
            del node.root.docs

    def predict(self, doc, size=0):
        node = self.root
        while not node.leaf and (size == 0 or node.order <= size):
            if node.word_id in doc:
                node = node.included
            else:
                node = node.excluded

        return node.pred
