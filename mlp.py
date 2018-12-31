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

    def __init__(self, inputSize, layerSizes, eta):
        self.inputSize = inputSize
        current = inputSize
        for ls in layerSizes:
            self.layers.append(Layer(ls, current, aFunLibrary.ELLIOT_SIG))
            current = ls
        self.outputLayer = self.layers[len(self.layers) - 1]
        self.eta = eta
    
    def eval(self, inp, expected):
        current = inp.copy()
        for l in self.layers:
            current = l.eval(current)
        
        d = current - expected
        self.cost = np.dot(d, d)

        return current

    def train(self, inp, expected):
        output = self.eval(inp, expected)

        #Output layer
        for row in range(self.outputLayer.nRow):
            dEdO = 2 * (output[row] - expected[row])
            dOdZ = self.outputLayer.aFun.evalPrim(self.outputLayer.z[row])
            self.outputLayer.dEdZ[row] = dEdO * dOdZ
            dEdB = dEdO * dOdZ * 1
            self.outputLayer.bNew[row] = self.outputLayer.b[row] - self.eta * dEdB

            for col in range(self.outputLayer.nCol):
                dZdW = self.outputLayer.i[col]
                dEdW = dEdO * dOdZ * dZdW
                self.outputLayer.wNew[row, col] = self.outputLayer.w[row, col] - self.eta * dEdW

        #Hidden layers
        for i in reversed(range(len(self.layers) - 2)):
            l = layers[i]
            nl = layers[i + 1]
            for row in range(l.nRow):
                dOdZ = l.aFun.evalPrim(l.z[row])
                dEdO = 0.0
                for j in range(len(nl.nRow)):
                    dEdO += nl.dEdZ[j] * nl.w[j, row]

                dEdZ = dEdO * dOdZ
                dEdB = dEdZ * 1
                l.bNew[row] = l.b[row] - self.eta * dEdB
                l.dEdZ[row] = dEdZ
                for col in range(l.nCol):
                    dZdW = l.i[col]
                    dEdW = dEdZ * dZdW
                    l.wNew[row, col] = l.w[row, col] - self.eta * dEdW

        #Copy over trained values
        for l in self.layers:
            l.w = l.wNew.copy()
            l.b = l.bNew.copy()

        return self.cost

    def getCost(self):
        return self.cost