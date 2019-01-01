import os
import struct
import numpy as np
from array import array as pyarray

import mlp
import afun_lib

"""
https://gist.github.com/mfathirirhas/f24d61d134b014da029a
Type code	C Type	Python Type	Minimum size in bytes
'b'	signed char	int	1
'B'	unsigned char	int	1
'u'	Py_UNICODE	Unicode character	2
'h'	signed short	int	2
'H'	unsigned short	int	2
'i'	signed int	int	2
'I'	unsigned int	int	2
'l'	signed long	int	4
'L'	unsigned long	int	4
'f'	float	float	4
'd'	double	float	8
"""
def loadMNIST(pathLabels, pathImages):
    fileLabels = open(pathLabels, 'rb')
    magicNumber, size = struct.unpack(">II", fileLabels.read(8))

    if (magicNumber != 2049):
        raise Exception("Magic number in label file should be 2049. The value was: %i" % (magicNumber))

    labels = np.asarray(pyarray("b", fileLabels.read()))
    fileLabels.close()

    fileImage = open(pathImages, 'rb')
    magicNumber, size, nRow, nCol = struct.unpack(">IIII", fileImage.read(16))

    if (magicNumber != 2051):
        raise Exception("Magic number in image file should be 2051. The value was: %i" % (magicNumber))

    imageBytes = pyarray("B", fileImage.read())
    fileImage.close()

    images = np.zeros((size, nRow, nCol), dtype=np.uint8)
    for i in range(size):
        images[i] = np.asarray(imageBytes[nRow * nCol * i : nRow * nCol * (i + 1)]).reshape((nRow, nCol))

    return labels, images

def saveMLP(mlp, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w+") as f:
        f.write("%.16f\n" % (mlp.eta))
        f.write("%i\n" % (mlp.inputSize))
        for l in mlp.layers:
            f.write("%i " % (l.nRow))
        f.write("\n")
        for l in mlp.layers:
            f.write("%s\n" % (l.aFun))
        for l in mlp.layers:
            for row in range(l.nRow):
                for col in range(l.nCol):
                    f.write("%.16f " % (l.w[row][col]))
                f.write("%.16f\n" % (l.b[row]))

def loadMLP(filename):
    with open(filename, "r") as f:
        eta = float(f.readline())
        nInputs = int(f.readline())
        lStrings = f.readline().split()
        dim = []
        for ls in lStrings:
            dim.append(int(ls))
        for i in range(len(lStrings)):
            lStrings[i] = f.readline()
        layers = []
        for i in range(len(dim)):
            nRow = dim[i]
            nCol = nInputs
            l = mlp.Layer(nRow, nCol, afun_lib.from_name(lStrings[i]))
            for row in range(nRow):
                data = [float(i) for i in f.readline().split()]
                l.w[row] = data[0:len(data) - 1]
                l.b[row] = data[len(data) - 1]
            layers.append(l)
            nInputs = nRow

        filename_w_ext = os.path.basename(filename)
        filename = os.path.splitext(filename_w_ext)[0]

        return mlp.MLP(filename, layers, eta)