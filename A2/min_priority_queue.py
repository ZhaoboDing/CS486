from queue import PriorityQueue
from functools import total_ordering


class MaxPQ:
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


if __name__ == '__main__':
    # Test comparisons of QueueNode
    qn1 = QueueNode(1, None, None)
    qn2 = QueueNode(1, None, None)
    qn3 = QueueNode(2, None, None)

    assert qn1 == qn2
    assert qn1 > qn3
    assert qn3 <= qn2
    assert qn2 != qn3

    # Test correctness of priority queue
    qn4 = QueueNode(5, None, None)
    qn5 = QueueNode(10, None, None)
    qn6 = QueueNode(23, None, None)
    qn7 = QueueNode(9, None, None)

    queue = MaxPQ()
    queue.push(qn3)
    queue.push(qn6)
    queue.push(qn4)
    assert queue.pop() == qn6
    queue.push(qn7)
    assert queue.pop() is qn7
    queue.push(qn5)
    queue.push(qn1)
    assert queue.pop() is qn5
