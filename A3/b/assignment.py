from A3.b.data_io import load_train_data, load_test_data, load_words
from A3.b.NaiveBayes import NaiveBayes
from A3.b.configs import labels as possible_labels


def test(model, data):
    correct, incorrect = 0, 0
    for doc in data:
        label = doc.label
        pred = model.predict(doc, possible_labels)
        if pred == label:
            correct += 1
        else:
            incorrect += 1

    return correct / (correct + incorrect)

train_data = load_train_data()
test_data = load_test_data()
word_map = load_words()

nb = NaiveBayes(word_map)
nb.fit(train_data)

print(test(nb, train_data))
print(test(nb, test_data))
