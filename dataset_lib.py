import utils
import numpy as np

class DataEntry():
    def __init__(self, inp, exp):
        self.inp = inp
        self.exp = exp

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

set_mnist_training = []
for i in range(len(_labels_training)):
    set_mnist_training.append(DataEntry(_image_to_array(_images_training[i]), _label_to_array(_labels_training[i])))

set_mnist_test = []
for i in range(len(_labels_test)):
    set_mnist_test.append(DataEntry(_image_to_array(_images_test[i]), _label_to_array(_labels_test[i])))