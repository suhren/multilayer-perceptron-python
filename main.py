import numpy as np

import aFunLibrary

#http://rogerdudler.github.io/git-guide/

class Layer:
    def __init__(self, nRow, nCol, aFun):
        self.w = np.random.rand(nRow, nCol)
        self.b = np.random.rand(nRow)
        self.aFun = aFun

    def eval(self, input):
        return self.aFun.eval(np.matmul(self.w, input) + self.b)

class MLP:
    layers = []

    def __init__(self, inputSize, layerSizes):
        self.inputSize = inputSize
        current = inputSize
        for ls in layerSizes:
            self.layers.append(Layer(ls, current, aFunLibrary.ELLIOT_SIG))
            current = ls
    
    def eval(self, input):
        current = input.copy()
        for l in self.layers:
            current = l.eval(current)
        return current

def main():
    input = np.zeros(5)
    mlp = MLP(5, (5, 4, 3))
    
    print(mlp.eval(input))

    #a = np.arange(15).reshape(3, 5)
    #print(a)

    #print(layer.w)

if __name__ == '__main__':
    main()