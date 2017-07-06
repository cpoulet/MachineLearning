#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """K-Means clustering"""

    def __init__(self, k):
        self._k     = k     # number of clusters
        self.means  = None  # means location of clusters
    
    def _classify(self, vector):
        """return the closest cluster"""
        return min(range(self._k), key=lambda x:np.sum((vector - self.means[x]) ** 2))
    
    def train(self, vectors):
        self.means = np.array(random.sample(vectors, self._k))
        loc = None
        while True:
            new_loc = list(map(self._classify, np.array(vectors))) # allocation of vector
            if new_loc == loc:  # all vectors were already allocated
                return
            loc = new_loc[:]
            for i in range(self._k): # saving the new cluster means
                p = [vec for vec, l in zip(vectors, loc) if l == i]
                if p:
                    self.means[i] = np.array(p).mean(axis=0)
    
    def show(self, inputs):
        px, py = zip(*inputs)
        kx, ky = zip(*self.means)
        plt.scatter(px, py)
        axes = plt.gca()
        axes.set_xlim([-60,30])
        axes.set_ylim([-30,40])
        plt.plot(kx, ky, 'or')
        plt.show()

def main():
    KM = KMeans(3)
    inputs = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13], [-46, 5], [-34, -1], [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19], [-41, 8], [-11, -6], [-25, -9], [-18, -3]]
    KM.train(inputs)
    KM.show(inputs)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
