import math
from collections import Counter
from .configs import *
from .max_priority_queue import MaxPQ, QueueNode


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
    def __init__(self, docs, used_word):
        self.word_id = None
        self.word = None
        self.docs = docs
        self.usedWord = used_word
        self.included = None
        self.excluded = None
        self.terminate = True
        self.label = None
        self.information_gain = 0

    def split(self, word):
        if not self.terminate:
            raise Exception('Unable to split a non-terminal node in decision tree.')
        if word in self.usedWord:
            raise LookupError('Undefined word to split: ' + word)

        used = self.usedWord | {word}
        in_list = [doc for doc in self.docs if word in doc]
        out_list = [doc for doc in self.docs if word not in doc]
        self.included = DecisionTreeNode(in_list, used)
        self.excluded = DecisionTreeNode(out_list, used)

        self.terminate = False
        del self.docs

        return self.included, self.excluded

    def best_split(self, words, method):
        valid_words = words - self.usedWord
        split_results = {word: information_gain(self.docs, word, method) for word in valid_words}
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
        self.root = DecisionTreeNode(docs, set())
        q = MaxPQ()
        split_word, ig = self.root.best_split(self.word_set, self.algo)
        q.push(QueueNode(ig, self.root, split_word))
        self.size = 0

        while self.size < size:
            node = q.pop()
            if node.root.terminate:
                if node.gain:
                    inc, exc = node.root.split(node.split_word)
                    spi, igi = inc.best_split(self.word_set, self.algo)
                    spe, ige = exc.best_split(self.word_set, self.algo)
                    q.push(QueueNode(igi, inc, spi))
                    q.push(QueueNode(ige, exc, spe))
                    self.size += 1
                else:
                    break

        self.clear(self.root)

    def predict(self, doc):
        node = self.root
        while not node.terminate:
            if node.word in doc:
                node = node.included
            else:
                node = node.excluded

        return node.label

    def clear(self, root):
        if root is None:
            return

        root.word = self.word_map[root.word_id]
        if root.terminate:
            if root.label is None:
                labels = [doc.label for doc in root.docs]
                root.label = Counter(labels).most_common(1)[0][0]
                del root.docs
        else:
            self.clear(root.included)
            self.clear(root.excluded)
