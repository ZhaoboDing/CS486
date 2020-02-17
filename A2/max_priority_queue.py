from functools import total_ordering


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
