#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Kmeans import *

class IMG_Simple():

    def __init__(self, img, k = 5):
        self.k = k
        self.KM = KMeans(k)
        self.old = img
        self.new = None

    def _recolor(self, pixel):
        cluster = self.KM._classify(pixel)
        return self.KM.means[cluster]

    def simplify(self):
        pixels = [pixel for row in self.old for pixel in row]
        self.KM.train(pixels)
        self.new = [[self._recolor(pixel) for pixel in row] for row in self.old]
        return self.new

    def show_img(self):
        f = plt.figure()
        k = f.add_subplot(1,2,1)
        plt.imshow(self.old)
        plt.axis('off')
        k.set_title('Old')
        k = f.add_subplot(1,2,2)
        plt.imshow(self.new)
        plt.axis('off')
        k.set_title('New')
        plt.show()

def main():
    I = IMG_Simple(mpimg.imread('img/SpongeBob.png'), 5)
    I.simplify()
    I.show_img()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
