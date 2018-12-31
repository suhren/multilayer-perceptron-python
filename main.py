import numpy as np

import mlp

#http://rogerdudler.github.io/git-guide/

def main():
    """The entry point of the program."""

    input = np.zeros(5)
    expected = np.zeros(3)
    network = mlp.MLP(5, (5, 4, 3), 0.01)
    
    print(network.eval(input, expected))
    print(network.getCost())

if __name__ == '__main__':
    main()