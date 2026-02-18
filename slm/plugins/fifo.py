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
        ))

    def map(self, func):
        """
        maps a func of the contents of the buffer.
        Warning: the buffer is unordered, so only use functions that are invariant to the ordering (like min, max or mean)
        """
        return np.apply_along_axis(func, 1, self.buffer)
