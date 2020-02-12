from A2.decision_tree import DecisionTree
from A2.data_io import *

if __name__ == '__main__':
    print("Loading training data...")

    docs = load_data()
    load_labels(docs)
    word_map = load_words()

    print("Building decision tree...")
    decision_tree = DecisionTree(word_map)
    decision_tree.train(docs)

    print("Decision tree built.")
    print("Loading test data...")

    test_data = load_test_data()
    test_label = load_test_labels(test_data)

    print("Testing decision tree...")

    N = len(test_label)
    correct = 0
    for index in range(1, N + 1):
        word_list = test_data[index] if index in test_data else Document()
        if test_label[index] == decision_tree.predict(word_list):
            correct += 1

    print("Accuracy: {0} / {1} = {2}".format(correct, N, correct / N))
