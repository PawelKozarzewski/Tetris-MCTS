import numpy as np
from numba import jit


class TQueue:
    def __init__(self, size: int = 0):
        self.size = size
        self.queue = TQueue.generate_tetrominos(2 * self.size)

    def get(self, i: int) -> int:
        return self.queue[i]

    def remove_first(self) -> None:
        if self.queue.shape[0] > 0:
            self.queue = np.delete(self.queue, 0)

        if self.queue.shape[0] < self.size:
            new = TQueue.generate_tetrominos(self.size)
            self.queue = np.concatenate((self.queue, new), axis=0)

    def reset(self) -> None:
        self.__init__(self.size)

    @staticmethod
    @jit(nopython=True, cache=True)
    def generate_tetrominos(n: int) -> np.ndarray:
        length = n + (7 - n % 7)
        result = np.zeros(length, dtype='int8')
        for i in range(0, result.shape[0], 7):
            result[i:i + 7] = np.random.permutation(7)
        return result
