import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mlp
import fileUtils
import aFunLibrary

#http://rogerdudler.github.io/git-guide/

class DataEntry():
    def __init__(self, inp, exp):
        self.inp = inp
        self.exp = exp

def imageToArray(image):
    return image.reshape(len(image) * len(image[0]))

def digitToArray(digit):
    res = np.zeros(10)
    res[digit] = 1.0
    return res

def showImage(image):
    plt.figure()
    plt.imshow(image, cmap=cm.gray, vmin=0, vmax=255)
    plt.show()

def main():
    """The entry point of the program."""

    network = mlp.MLP(784, (12, 10), 0.01, aFunLibrary.ARCTAN)
    
    #print(network.eval(inp, expected))
    #print(network.getCost())

    trainLabels, trainImages = fileUtils.readMNIST("MNIST//train-labels-idx1-ubyte", "MNIST//train-images-idx3-ubyte")
    testLabels, testImages = fileUtils.readMNIST("MNIST//t10k-labels-idx1-ubyte", "MNIST//t10k-images-idx3-ubyte")

    trainSet = []
    for i in range(len(trainLabels)):
        trainSet.append(DataEntry(imageToArray(trainImages[i]), digitToArray(trainLabels[i])))

    #dataSets.append(DataSet([ digitToArray(lbl) for lbl in trainLabels ], [ imageToArray(img) for img in trainImages ]))

    while True:
        command = input("Enter command: ").split()
        if command[0] == "train":
            sum = 0.0
            print("Training...")
            for i, e in enumerate(trainSet):
                cost = network.train(e.inp, e.exp)
                print("%i of %i: Cost: %.8f" % (i, len(trainSet), cost))
                sum += cost
            print("Average cost: %.8f" % (sum / len(trainSet)))
        elif command[0] == "input":
            i = int(command[1])
            print("Input %s:" % (trainLabels[i]))
            print("Output: %s" % (network.eval(trainSet[i].inp, trainSet[i].exp)))
            # print("Cost: %.8f" % (network.getCost()))

    
if __name__ == '__main__':
    main()