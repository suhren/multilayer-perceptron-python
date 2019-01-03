import utils
import numpy as np

class DataEntry(object):
    def __init__(self, inp, exp):
        self.inp = inp
        self.exp = exp

class DataSet(object):
    entries = []

    def __init__(self, name, nInputs, nOutputs):
        self.name = name
        self.nInputs = nInputs
        self.nOutputs = nOutputs

    def __str__(self):
        return '%s: %i -> %i' % (self.name, self.nInputs, self.nOutputs)

#The MLP works better when the pixels are a value between 0.0f and 1.0f as oposed to 0 to 255?
#Can it be because the expected output is in the range 0.0f to 1.0f?
#Works best when normalized or input and output is in the same order of magnitude?
def _image_to_array(image):
    return image.reshape(len(image) * len(image[0])).astype(np.float64) / 255.0

def _label_to_array(label):
    res = np.zeros(10).astype(np.float64)
    res[label] = 1.0
    return res

_labels_training, _images_training = utils.loadMNIST("mnist//train-labels-idx1-ubyte", "MNIST//train-images-idx3-ubyte")
_labels_test, _images_test = utils.loadMNIST("mnist//t10k-labels-idx1-ubyte", "MNIST//t10k-images-idx3-ubyte")

datasets = []

set_mnist_training = DataSet("MNIST training set", 784, 10)
datasets.append(set_mnist_training)
for i in range(len(_labels_training)):
    set_mnist_training.entries.append(DataEntry(_image_to_array(_images_training[i]), _label_to_array(_labels_training[i])))

set_mnist_test = DataSet("MNIST test set", 784, 10)
datasets.append(set_mnist_test)
for i in range(len(_labels_test)):
    set_mnist_test.entries.append(DataEntry(_image_to_array(_images_test[i]), _label_to_array(_labels_test[i])))