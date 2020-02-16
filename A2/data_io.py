import pickle
from A2.configs import *
from anytree import Node
from anytree.exporter import DotExporter


class Document:
    def __init__(self, word_id=0, label=None):
        self.id = word_id
        self.label = label
        self.wordList = set()

    def __contains__(self, word):
        return word in self.wordList

    def add_word(self, word):
        self.wordList.add(word)


def load_train_data():
    return load_data(train_data_path, train_label_path)


def load_test_data():
    return load_data(test_data_path, test_label_path)


def load_data(data_path, label_path):
    docs = []
    with open(label_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            doc_id = index + 1
            label = int(line.strip())
            docs.append(Document(doc_id, label))

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            [doc_id, word_id] = list(map(int, line.strip().split(' ')))
            docs[doc_id - 1].add_word(word_id)

    return docs


def load_words(filename=words_path):
    word_map = dict()

    with open(filename, 'r', encoding='utf-8') as file:
        for index, word in enumerate(file):
            word_map[index] = word.strip()

    return word_map


def store_tree(tree, tree_path=decision_tree_path):
    with open(tree_path, 'wb') as file:
        pickle.dump(tree, file, pickle.HIGHEST_PROTOCOL)


def load_tree(tree_path=decision_tree_path):
    with open(tree_path, 'rb') as file:
        tree = pickle.load(file)

    return tree


def build_tree(root, edge=None):
    if root is None:
        return

    if root.terminate:
        return Node(id(root), edge=edge, display_name=root.word + " " + str(root.information_gain))
    else:
        children = [build_tree(root.included, "included"),
                    build_tree(root.excluded, "excluded")]
        return Node(id(root), edge=edge, display_name=root.word, children=list(filter(None, children)))


def render(tree, filename=decision_tree_picture_path):
    tree_to_render = build_tree(tree.root)
    DotExporter(tree_to_render).to_picture(filename)
