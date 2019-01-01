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

#The MLP works better when the pixels are a value between 0.0f and 1.0f as oposed to 0 to 255?
#Can it be because the expected output is in the range 0.0f to 1.0f?
#Works best when normalized or input and output is in the same order of magnitude?
def imageToArray(image):
    return image.reshape(len(image) * len(image[0])).astype(np.float64) / 255.0

def digitToArray(digit):
    res = np.zeros(10).astype(np.float64)
    res[digit] = 1.0
    return res

def showImage(image):
    plt.figure()
    plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    plt.show()

def maxIndex(x):
    return np.argmax(x)

def main():
    """The entry point of the program."""

    network = mlp.MLP.fromArguments("MyMLP", 784, (12, 10), 0.01, aFunLibrary.ELLIOT_SIG)
    
    #print(network.eval(inp, expected))
    #print(network.getCost())

    trainLabels, trainImages = fileUtils.loadMNIST("mnist//train-labels-idx1-ubyte", "MNIST//train-images-idx3-ubyte")
    testLabels, testImages = fileUtils.loadMNIST("mnist//t10k-labels-idx1-ubyte", "MNIST//t10k-images-idx3-ubyte")

    trainSet = []
    for i in range(len(trainLabels)):
        trainSet.append(DataEntry(imageToArray(trainImages[i]), digitToArray(trainLabels[i])))

    #dataSets.append(DataSet([ digitToArray(lbl) for lbl in trainLabels ], [ imageToArray(img) for img in trainImages ]))

    while True:
        command = input("Enter command: ").split()
        if command[0] == "train":
            n = int(command[1]) if (len(command)) > 1 else 1
            train(network, trainSet, n)
        elif command[0] == "input":
            inputEntry(network, trainSet, int(command[1]))
        elif command[0] == "save":
            fileUtils.saveMLP(network, "networks/mynetwork.txt")
        elif command[0] == "load":
            network = fileUtils.loadMLP("networks/mynetwork.txt")

def inputEntry(network, dataset, i):
    print("Input %s:" % (maxIndex(dataset[i].exp)))
    out = network.eval(dataset[i].inp, dataset[i].exp)
    print("Guess: %i" % maxIndex(out))
    print("Output: %s" % out)
    # print("Cost: %.8f" % (network.getCost()))

def train(network, dataset, n):
    print("Training %i times..." % (n))
    for i in range(n):
        sum = 0.0
        for e in dataset:
            sum += network.train(e.inp, e.exp)
            #print("%i of %i: Cost: %.8f" % (i, len(trainSet), cost))
        print("Set %i of %i: ave. cost: %.16f" % (i + 1, n, sum / len(dataset)))
    print("Done training")
    
if __name__ == '__main__':
    main()