from queue import PriorityQueue
from functools import total_ordering


class MinPQ:
    def __init__(self):
        self.q = PriorityQueue()

    def pop(self):
        return self.q.get()

    def push(self, node):
        self.q.put(node)


@total_ordering
class QueueNode:
    def __init__(self, gain, root, split_word):
        self.gain = gain
        self.root = root
        self.split_word = split_word

    def __eq__(self, other):
        return self.gain == other.gain

    def __lt__(self, other):
        return self.gain > other.gain
