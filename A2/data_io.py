from collections import defaultdict
from .configs import *


class Document:
    def __init__(self, word_id=0, label=None):
        self.id = word_id
        self.label = label
        self.wordList = set()

    def __contains__(self, word):
        return word in self.wordList

    def add_word(self, word):
        self.wordList.add(word)


def load_data(filename=train_data_path):
    docs = defaultdict(Document)

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            [doc_id, word_id] = list(map(int, line.strip().split(' ')))
            docs[doc_id].add_word(word_id)

    return docs


def load_labels(docs, filename=train_label_path):
    with open(filename, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            doc_id = index + 1
            label = int(line.strip())
            docs[doc_id].label = label
            docs[doc_id].id = doc_id


def load_words(filename=words_path):
    word_map = dict()

    with open(filename, 'r', encoding='utf-8') as file:
        for index, word in enumerate(file):
            word_map[index] = word.strip()

    return word_map


def load_test_data(filename=test_data_path):
    return load_data(filename)


def load_test_labels(filename=test_label_path):
    labels = dict()
    with open(filename, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            doc_id = index + 1
            label = int(line.strip())
            labels[doc_id] = label

    return labels
