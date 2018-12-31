import numpy as np

class Layer:
    def __init__(self, nRow, nCol):
        self.w = np.random.rand(nRow, nCol)
        self.b = np.random.rand(nRow)

def main():
    layer = Layer(5, 10)

    # wArray = 
    print(input)

    a = np.arange(15).reshape(3, 5)
    print(a)

    print(layer.w)

if __name__ == '__main__':
    main()