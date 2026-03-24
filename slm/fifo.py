import numpy as np


class FIFO:
    size = property(lambda self: self.buffer.shape[1])

    def __init__(self, shape):
        self.buffer = np.zeros(shape)
        self.index = 0

    def reset(self):
        self.buffer.fill(0)
        self.index = 0

    def push(self, value):
        self.buffer[:, self.index] = value
        self.index = (self.index + 1) % self.size

    def get(self):
        return np.concatenate((
            self.buffer[:, self.index:],
            self.buffer[:, :self.index]
        ), axis=1)

    def map(self, func):
        """
        maps a block_fn of the contents of the buffer.
        Warning: the buffer is unordered, so only use functions that are invariant to the ordering (like min, max or mean).
        func must accept an axis keyword argument (np.mean, np.max, np.min all qualify).
        """
        return func(self.buffer, axis=1)
