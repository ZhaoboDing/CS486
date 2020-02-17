import matplotlib.pyplot as plt
from A2.decision_tree import DecisionTree
from A2.configs import AIM_TREE_SIZE, AVERAGE_INFORMATION_GAIN, WEIGHTED_INFORMATION_GAIN
from A2.data_io import load_train_data, load_test_data, load_words, render


def test_decision_tree(tree, size):
    def test(data):
        correct = incorrect = 0
        for doc in data:
            if tree.predict(doc, size) == doc.label:
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

    train_data = load_train_data()
    word_map = load_words()
    decision_tree = DecisionTree(word_map, method)
    decision_tree.train(train_data)

    test_result = []
    train_result = []

    for i in range(AIM_TREE_SIZE):
        tree_size = i + 1
        test_accuracy, train_accuracy = test_decision_tree(decision_tree, tree_size)
        test_result.append(test_accuracy)
        train_result.append(train_accuracy)

    filename = './decision_tree_' + suffix + '.png'
    render(decision_tree, TREE_SIZE_TO_DRAW, filename)

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