import math
from collections import Counter
from .configs import *
from .min_priority_queue import *


def entropy(p):
    return -p * math.log2(p)


def information_entropy(doc_labels):
    stat = Counter(doc_labels)
    doc_length = len(doc_labels)
    return sum([entropy(freq / doc_length) for freq in stat.values()])


def information_gain(docs, word, method=FEATURE_SELECTION_MECHANISM):
    E = [doc.label for doc in docs.values()]
    E1 = [doc.label for doc in docs.values() if word in doc]
    E2 = [doc.label for doc in docs.values() if word not in doc]
    IE = information_entropy(E)
    IE1 = information_entropy(E1)
    IE2 = information_entropy(E2)

    if method is AVERAGE_INFORMATION_GAIN:
        return IE - (IE1 + IE2) / 2
    elif method is WEIGHTED_INFORMATION_GAIN:
        return IE - (len(E1) * IE1 + len(E2) * IE2) / len(docs)


class DecisionTreeNode:
    def __init__(self, docs, used_word):
        self.word = None
        self.docs = docs
        self.usedWord = used_word
        self.included = None
        self.excluded = None
        self.split = False
        self.label = None

    def split(self, word):
        if word in self.usedWord:
            raise LookupError('Undefined word to split: ' + word)

        self.word = word
        self.split = True
        used = self.usedWord | {word}
        in_list = {key: self.docs[key] for key in self.docs if word in self.docs[key]}
        out_list = {key: self.docs[key] for key in self.docs if word not in self.docs[key]}
        self.included = DecisionTreeNode(in_list, used)
        self.excluded = DecisionTreeNode(out_list, used)

        return self.included, self.excluded

    def best_split(self, words):
        valid_words = words - self.usedWord
        split_results = {word: information_gain(self.docs, word) for word in valid_words}
        chosen_word = max(split_results, key=split_results.get)
        return chosen_word, split_results[chosen_word]

    def get_label(self):
        if self.label is None:
            labels = [doc.label for doc in self.docs.values()]
            self.label = max(set(labels), key=labels.count)

        return self.label


class DecisionTree:
    def __init__(self, word_map):
        self.root = None
        self.word_map = word_map
        self.word_set = set(word_map)
        self.size = 0

    def train(self, docs):
        self.root = DecisionTreeNode(docs, set())
        q = MinPQ()
        split_word, ig = self.root.best_split(self.word_set)
        q.push(QueueNode(ig, self.root, split_word))
        self.size = 1

        while self.size <= AIM_TREE_SIZE:
            node = q.pop()
            if node:
                inc, exc = node.root.split(node.split_word)
                spi, igi = inc.best_split(self.word_set)
                spe, ige = exc.best_split(self.word_set)
                q.push(QueueNode(igi, node, spi))
                q.push(QueueNode(ige, node, spe))
                self.size += 1
            else:
                return

    def predict(self, doc):
        node = self.root
        while node.split:
            if node.word in doc:
                node = node.included
            else:
                node = node.excluded

        return node.get_label()
