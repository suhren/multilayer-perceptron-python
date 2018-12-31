import os
import struct
import numpy as np
from array import array as pyarray

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
def readMNIST(pathLabels, pathImages):
    digits=np.arange(10)
    fileLabels = open(pathLabels, 'rb')
    magicNumber, size = struct.unpack(">II", fileLabels.read(8))
    labels = np.asarray(pyarray("b", fileLabels.read()))
    fileLabels.close()

    fileImage = open(pathImages, 'rb')
    magicNumber, size, nRow, nCol = struct.unpack(">IIII", fileImage.read(16))
    imageBytes = pyarray("B", fileImage.read())
    fileImage.close()

    images = np.zeros((size, nRow, nCol), dtype=np.uint8)
    for i in range(size):
        images[i] = np.asarray(imageBytes[nRow * nCol * i : nRow * nCol * (i + 1)]).reshape((nRow, nCol))

    return labels, images