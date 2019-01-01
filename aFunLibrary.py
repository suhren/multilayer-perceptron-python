import math
import numpy as np

class ActivationFunction:
    def __init__(self, func, prim):
        self.func = func
        self.prim = prim

    def eval(self, x):
        return self.func(x)

    def evalPrim(self, x):
        return self.prim(x)

IDENTITY = ActivationFunction(
    lambda x: x,
    lambda x: 1.0)

BINARY_STEP = ActivationFunction(
    lambda x: 0.0 if (x < 0) else 1.0,
    lambda x: 0.0)

LOGISTIC = ActivationFunction(
    lambda x: 1.0 / (1 + exp(-x)),
    lambda x: 1.0 / (1 + exp(-x)) * (1.0 - 1.0 / (1.0 + exp(-x))))

TANH = ActivationFunction(
    lambda x: math.tanh(x),
    lambda x: 1.0 - pow(math.tanh(x), 2))

ARCTAN = ActivationFunction(
    lambda x: np.arctan(x),
    lambda x: 1.0 / (x**2 + 1))

ELLIOT_SIG = ActivationFunction(
    lambda x: x / (1.0 + abs(x)),
    lambda x: 1.0 / pow(1.0 + abs(x), 2))

RELU = ActivationFunction(
    lambda x: 0.0 if (x < 0) else x,
    lambda x: 0.0 if (x < 0) else 1.0)

RELU_LEAKY = ActivationFunction(
    lambda x: 0.01 * x if (x < 0) else x,
    lambda x: 0.01 if (x < 0) else 1.0)
		
SOFT_PLUS = ActivationFunction(
    lambda x: math.log(1.0 + exp(x)),
    lambda x: 1.0 / (1.0 + exp(-x)))

SINUSOID = ActivationFunction(
    lambda x: math.sin(x),
    lambda x: math.cos(x))

SINC = ActivationFunction(
    lambda x: 0.0 if (x == 0) else math.sin(x) / x,
    lambda x: 0.0 if (x == 0) else math.cos(x) / x - math.sin(x) / (x**2))

GAUSSIAN = ActivationFunction(
    lambda x: exp(-x**2),
    lambda x: -2.0 * x * exp(-x**2))