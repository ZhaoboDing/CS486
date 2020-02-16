from A2.decision_tree import DecisionTree
from A2.data_io import *


if __name__ == '__main__':
    print("Loading training data...")

    train_data = load_train_data()
    word_map = load_words()

    print("Building decision tree...")
    decision_tree = DecisionTree(word_map)
    decision_tree.train(train_data)
    store_tree(decision_tree)

    print("Decision tree built.")
    print("Loading test data...")

    test_data = load_test_data()

    print("Testing decision tree...")

    N = len(test_data)
    correct = 0
    for doc in test_data:
        if decision_tree.predict(doc) == doc.label:
            correct += 1

    print("Accuracy: {0} / {1} = {2}".format(correct, N, correct / N))
