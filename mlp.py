import numpy as np
import aFunLibrary

class Layer:
    """The class representing a single layer in the MLP."""

    def __init__(self, nRow, nCol, aFun):
        self.nRow = nRow
        self.nCol = nCol
        self.w = np.random.rand(nRow, nCol)
        self.b = np.random.rand(nRow)
        self.aFun = aFun
        self.i = np.zeros(nCol)
        self.z = np.zeros(nRow)
        self.o = np.zeros(nRow)
        self.dEdZ = np.zeros(nRow)
        self.wNew = np.zeros((nRow, nCol))
        self.bNew = np.zeros(nRow)

    def eval(self, inp):
        self.i = inp.copy()
        self.z = np.matmul(self.w, inp) + self.b
        self.o = self.aFun.eval(self.z)
        return self.o

class MLP:
    """The class representing the entire MLP."""

    layers = []
    cost = 0

    def __init__(self, inputSize, layerSizes, eta, aFun):
        self.inputSize = inputSize
        current = inputSize
        for ls in layerSizes:
            self.layers.append(Layer(ls, current, aFun))
            current = ls
        self.oL = self.layers[len(self.layers) - 1]
        self.eta = eta
    
    def eval(self, inp, exp):
        #No need to make copy of input as the local
        #reference is reassigned at the eval()

        for l in self.layers:
            inp = l.eval(inp)
        
        d = inp - exp
        self.cost = np.dot(d, d)

        return inp

    def train(self, inp, exp):
        out = self.eval(inp, exp)

        oL = self.oL
        eta = self.eta
        layers = self.layers

        #Output layer
        dEdO = 2 * (out - exp)
        dOdZ = oL.aFun.evalPrim(oL.z)
        oL.dEdZ = dEdO * dOdZ
        oL.bNew = oL.b - eta * oL.dEdZ
        oL.wNew = oL.w - eta * np.outer(oL.dEdZ, oL.i)
        # https://en.wikipedia.org/wiki/Outer_product

        #Hidden layers
        for i in reversed(range(len(layers) - 1)):
            l = layers[i]
            nl = layers[i + 1]

            dOdZ = l.aFun.evalPrim(l.z)
            dEdO = np.matmul(np.transpose(nl.w), nl.dEdZ) 
            l.dEdZ = dOdZ * dEdO
            l.bNew = l.b - eta * l.dEdZ
            l.wNew = l.w - eta * np.outer(l.dEdZ, l.i)

        #Copy over trained values
        for l in layers:
            l.w = l.wNew.copy()
            l.b = l.bNew.copy()

        return self.cost

    def getCost(self):
        return self.cost