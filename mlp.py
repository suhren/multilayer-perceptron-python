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
        self.input = np.zeros(nCol)
        self.z = np.zeros(nRow)
        self.o = np.zeros(nRow)
        self.dEdZ = np.zeros(nRow)
        self.wNew = np.zeros((nRow, nCol))
        self.bNew = np.zeros(nRow)

    def eval(self, input):
        self.input = input.copy
        self.z = np.matmul(self.w, input) + self.b
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
        self.outputLayer = current
        self.eta = eta
    
    def eval(self, input, expected):
        current = input.copy()
        for l in self.layers:
            current = l.eval(current)
        
        d = current - expected
        self.cost = np.dot(d, d)

        return current

    def train(self, input, expected):
        output = self.eval(input, expected)

        #Output layer
        for row in range(0, self.outputLayer.nRow):
            dEdO = 2 * (output[row] - expected[row])
            dOdZ = self.outputLayer.aFun.evalPrim(self.outputLayer.z[row])
            self.outputLayer.dEdZ[row] = dEdO * dOdZ
            dEdB = dEdO * dOdZ * 1
            self.outputLayer.bNew[row] = self.outputLayer.b[row] - self.eta * dEdB

            for col in range(0, self.outputLayer.nCol):
                dZdW = self.outputLayer.input[col]
                dEdW = dEdO * dOdZ * dZdW
                self.outputLayer.wNew[row, col] = self.outputLayer.w[row, col] - self.eta * dEdW

        #Hidden layers
        for i in range(len(self.layer - 2), 0):
            l = layers[i]
            nl = layers[i + 1]
            for row in range(0, l.w.nRow):
                dOdZ = l.aFun.evalPrim(l.z[row])
                dEdO = 0.0
                for j in range(0, len(nl.w.nRow)):
                    dEdO += nl.dEdZ[j] * nl.w[j, row]

                dEdZ = dEdO * dOdZ
                dEdB = dEdZ * 1
                l.bNew[row] = l.b[row] - self.eta * dEdB
                l.dEdZ[row] = dEdZ
                for col in range(0, l.w.nCol):
                    dZdW = l.input[col]
                    dEdW = dEdZ * dZdW
                    l.wNew[row, col] = l.w[row, col] - self.eta * dEdW

        #Copy over trained values
        for l in self.layers:
            l.w = l.wNew.clone()
            l.b = l.bNew.clone()

    def getCost(self):
        return self.cost