import numpy as np
import afun_lib

class Layer(object):
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

    def eval(self, inp):
        self.i = inp.copy()
        self.z = np.matmul(self.w, inp) + self.b
        self.o = self.aFun.function(self.z)
        return self.o

class MLP(object):
    """The class representing the entire MLP."""

    layers = []
    cost = 0

    def __init__(self, name, layers, eta):
        self.name = name
        self.layers = layers
        self.eta = eta
        self.inputSize = layers[0].nCol
        self.oL = layers[len(layers) - 1]

    @classmethod
    def from_arguments(cls, name, inputSize, layerSizes, eta, aFun):
        layers = []
        for ls in layerSizes:
            layers.append(Layer(ls, inputSize, aFun))
            inputSize = ls
        return cls(name, layers, eta)
    
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
        dOdZ = oL.aFun.derivate(oL.z)
        oL.dEdZ = dEdO * dOdZ
        oL.b -= eta * oL.dEdZ
        oL.w -= eta * np.outer(oL.dEdZ, oL.i)
        # https://en.wikipedia.org/wiki/Outer_product

        #Hidden layers
        for i in reversed(range(len(layers) - 1)):
            l = layers[i]
            nl = layers[i + 1]
            dOdZ = l.aFun.derivate(l.z)
            dEdO = np.matmul(np.transpose(nl.w), nl.dEdZ) 
            l.dEdZ = dOdZ * dEdO
            l.b -= eta * l.dEdZ
            l.w -= eta * np.outer(l.dEdZ, l.i)

        return self.cost

    def output(self):
        return self.oL.o