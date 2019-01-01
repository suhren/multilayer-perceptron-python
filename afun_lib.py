import math
import numpy as np

class ActivationFunction:
    def __init__(self, name, func, derivate):
        self._name = name
        self.function = func
        self.derivate = derivate

    def __str__(self):
        return self._name

IDENTITY = ActivationFunction(
    "Identity",
    lambda x: x,
    lambda x: 1.0)

BINARY_STEP = ActivationFunction(
    "BinaryStep",
    lambda x: 0.0 if (x < 0) else 1.0,
    lambda x: 0.0)

LOGISTIC = ActivationFunction(
    "Logistic",
    lambda x: 1.0 / (1 + math.exp(-x)),
    lambda x: 1.0 / (1 + math.exp(-x)) * (1.0 - 1.0 / (1.0 + math.exp(-x))))

TANH = ActivationFunction(
    "TanH",
    lambda x: math.tanh(x),
    lambda x: 1.0 - pow(math.tanh(x), 2))

ARCTAN = ActivationFunction(
    "ArcTan",
    lambda x: np.arctan(x),
    lambda x: 1.0 / (x**2 + 1))

ELLIOT_SIG = ActivationFunction(
    "ElliotSig",
    lambda x: x / (1.0 + abs(x)),
    lambda x: 1.0 / pow(1.0 + abs(x), 2))

RELU = ActivationFunction(
    "ReLU",
    lambda x: 0.0 if (x < 0) else x,
    lambda x: 0.0 if (x < 0) else 1.0)

RELU_LEAKY = ActivationFunction(
    "ReLULeaky",
    lambda x: 0.01 * x if (x < 0) else x,
    lambda x: 0.01 if (x < 0) else 1.0)
		
SOFT_PLUS = ActivationFunction(
    "SoftPlus",
    lambda x: math.log(1.0 + math.exp(x)),
    lambda x: 1.0 / (1.0 + math.exp(-x)))

SINUSOID = ActivationFunction(
    "Sinusoid",
    lambda x: math.sin(x),
    lambda x: math.cos(x))

SINC = ActivationFunction(
    "Sinc",
    lambda x: 0.0 if (x == 0) else math.sin(x) / x,
    lambda x: 0.0 if (x == 0) else math.cos(x) / x - math.sin(x) / (x**2))

GAUSSIAN = ActivationFunction(
    "Gaussian",
    lambda x: math.exp(-x**2),
    lambda x: -2.0 * x * math.exp(-x**2))
    
def from_name(name):
    return ELLIOT_SIG