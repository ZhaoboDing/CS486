import matplotlib.pyplot as plt
from A2.decision_tree import DecisionTree
from A2.configs import AIM_TREE_SIZE, AVERAGE_INFORMATION_GAIN, WEIGHTED_INFORMATION_GAIN
from A2.data_io import load_train_data, load_test_data, load_words, render


def build_decision_tree(size, method):
    train_data = load_train_data()
    word_map = load_words()

    tree = DecisionTree(word_map, method)
    tree.train(train_data, size)

    return tree


def test_decision_tree(tree):
    def test(data):
        correct = incorrect = 0
        for doc in data:
            if tree.predict(doc) == doc.label:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    test_data = load_test_data()
    train_data = load_train_data()

    return test(test_data), test(train_data)


def generate_assignment_files(method):
    suffix = 'average' if method == AVERAGE_INFORMATION_GAIN else 'weighted'
    TREE_SIZE_TO_DRAW = 10
    test_result = []
    train_result = []

    for i in range(AIM_TREE_SIZE):
        """
            The way here we plot accuracy graph is not
            efficient due to abstraction and modularization.
            We re-build the decision tree for every size
            from 1 to 100 since I do not hope to insert code
            for plotting into the training process.
        """

        tree_size = i + 1
        decision_tree = build_decision_tree(tree_size, method)
        test_accuracy, train_accuracy = test_decision_tree(decision_tree)
        test_result.append(test_accuracy)
        train_result.append(train_accuracy)

        if i == TREE_SIZE_TO_DRAW:
            filename = './decision_tree_' + suffix + '.png'
            render(decision_tree, filename)

    size = list(range(1, AIM_TREE_SIZE + 1))
    plt.figure()
    plt.plot(size, train_result, label='training accuracy', color='blue')
    plt.plot(size, test_result, label='testing_accuracy', color='orange')
    plt.legend(loc='lower right')
    plt.xlabel('Number of nodes in decision tree')
    plt.ylabel('Accuracy')
    plt.title(suffix + ' information gain')
    plt.savefig('./accuracy_plot_' + suffix + '.png')


if __name__ == "__main__":
    generate_assignment_files(AVERAGE_INFORMATION_GAIN)
    generate_assignment_files(WEIGHTED_INFORMATION_GAIN)