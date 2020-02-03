import math
from operator import itemgetter
from collections import Counter
from queue import PriorityQueue

class Document:
    def __init__(self, id, label = None):
        self.id = id
        self.label = label
        self.wordList = set()
    
    def __contains__(self, word):
        return word in self.wordList
    
    def addWord(self, word):
        self.wordList.add(word)
    
    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label
    
class Docs:
    def __init__(self):
        self.fs = dict()

    def addWord(self, doc, word):
        if not doc in self.fs:
            self.fs[doc] = Document(id)
        
        self.fs[doc].addWord(word)
    
    def addLabel(self, doc, label):
        if not doc in self.fs:
            self.fs[doc] = Document(id)
        
        self.fs[doc].setLabel(label)

    def getLoaded(self):
        return self.fs

class DecisionTreeNode:
    def __init__(self, docs, usedWord = set()):
        self.word = None
        self.docs = docs
        self.usedWord = usedWord
        self.contain = None
        self.notCotain = None
        self.leaf = True
    
    def split(self, word):
        if word in self.usedWord:
            return None, None
        
        self.word = word
        self.leaf = False
        used = self.usedWord | {word}
        inList = [doc for doc in self.docs if word in doc]
        outList = [doc for doc in self.docs if not word in doc]
        self.contain = DecisionTreeNode(inList, used)
        self.notCotain = DecisionTreeNode(outList, used)

        return self.contain, self.notCotain

    def bestSplitWord(self, words):
        validWord = words - self.usedWord
        splitResult = {word: informationGain(self.docs, word) for word in validWord}
        print(splitResult)
        chosenWord = max(splitResult, key = itemgetter(1))[0]
        return chosenWord, splitResult[chosenWord]

class HeapNode:
    def __init__(self, gain, root, splitWord):
        self.gain = gain
        self.root = root
        self.splitWord = splitWord

    def __cmp__(self, other):
        if self.gain > other.gain:
            return -1
        elif self.gain == other.gain:
            return 0
        else:
            return 1
    
def loadData(filename):
    docs = Docs()

    with open(filename, 'r') as file:
        for line in file.readlines():
            [docId, wordId] = list(map(int ,line.strip().split(' ')))
            docs.addWord(docId, wordId)
    
    return docs

def loadLabels(docs, filename):
    with open(filename, 'r') as file:
        for index, line in enumerate(file):
            docId = index + 1
            label = int(line.strip())
            docs.addLabel(docId, label)

def loadWords(filename):
    words = set()
    with open(filename, 'rb') as file:
        for word in file.readlines():
            words.add(word.strip())
    
    return words

def informationGain(docs, word, method = 1):
    def informationEntropy(docs):
        def entropy(p):
            return -p * math.log2(p)
    
        stat = Counter([doc.getLabel() for doc in docs.values()])
        return sum([entropy(freq / len(docs)) for freq in stat.values()])
    
    E1 = {key: docs[key] for key in docs if word in docs[key]}
    E2 = {key: docs[key] for key in docs if not word in docs[key]}
    IE = informationEntropy(docs)
    IE1 = informationEntropy(E1)
    IE2 = informationEntropy(E2)

    if method == 1:
        return IE - (IE1 + IE2) / 2
    elif method == 2:
        return IE - (len(IE1) * IE1 + len(IE2) * IE2) / len(docs)

def createTree():
    docs = loadData('trainData.txt')
    loadLabels(docs, 'trainLabel.txt')
    docs = docs.getLoaded()
    words = loadWords('words.txt')

    root = DecisionTreeNode(docs)
    splitWord, IG = root.bestSplitWord(words)
    decisionTreeSize = 0
    q = PriorityQueue()
    q.put(HeapNode(IG, root, splitWord))

    while decisionTreeSize < 100:
        node = q.get()
        if node.gain == 0:
            break

        dtn1, dtn2 = node.root.split(node.splitWord)
        sp1, ig1 = dtn1.bestSplitWord(words)
        sp2, ig2 = dtn2.bestSplitWord(words)
        q.put(HeapNode(ig1, dtn1, sp1))
        q.put(HeapNode(ig2, dtn2, sp2))
    
    return root


if __name__ == '__main__':
    decisionTree = createTree()
    