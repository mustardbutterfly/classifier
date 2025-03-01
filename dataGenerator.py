import torch


class DataGenerator:
    def generateSample(self, count, length):
        samples = []
        pairs = []
        for i in range(0, count // 2):
            # samples.append(''.join(["1" for _ in range(length)]))
            samples.append([1 for _ in range(length)])
        for i in range(1, count // 2):
            pairs.append((i - 1, i))
        for i in range(count // 2, count):
            # samples.append(''.join(["0" for _ in range(length)]))
            samples.append([0 for _ in range(length)])
        for i in range((count // 2) + 1, count):
            pairs.append((i - 1, i))
        return samples, pairs
