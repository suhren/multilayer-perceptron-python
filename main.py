"""
This example uses docopt with the built in cmd module to demonstrate an
interactive command application.
Usage:
    my_program tcp <host> <port> [--timeout=<seconds>]
    my_program serial <port> [--baud=<n>] [--timeout=<seconds>]
    my_program (-i | --interactive)
    my_program (-h | --help | --version)
Options:
    -i, --interactive  Interactive Mode
    -h, --help  Show this screen and exit.
    --baud=<n>  Baudrate [default: 9600]
"""

import sys
import cmd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from docopt import docopt
from docopt import DocoptExit

import mlp
import utils 
import afun_lib
import dataset_lib

#http://rogerdudler.github.io/git-guide/
#https://google.github.io/styleguide/pyguide.html
#https://github.com/docopt/docopt/blob/master/examples/interactive_example.py

def docopt_cmd(func):
    """
    This decorator is used to simplify the try/except block and pass the result
    of the docopt parsing to the called action.
    """
    def fn(self, arg):
        try:
            opt = docopt(fn.__doc__, arg)

        except DocoptExit as e:
            # The DocoptExit is thrown when the args do not match.
            # We print a message to the user and the usage block.

            print('Invalid Command!')
            print(e)
            return

        except SystemExit:
            # The SystemExit exception prints the usage for --help
            # We do not need to do the print here.

            return

        return func(self, opt)

    fn.__name__ = func.__name__
    fn.__doc__ = func.__doc__
    fn.__dict__.update(func.__dict__)
    return fn

class MyInteractive (cmd.Cmd):
    intro = 'Welcome to my interactive program!' \
        + ' (type help for a list of commands.)'
    prompt = '(my_program) '
    file = None

    @docopt_cmd
    def do_tcp(self, arg):
        """Usage: tcp <host> <port> [--timeout=<seconds>]"""
        print(arg)

    @docopt_cmd
    def do_serial(self, arg):
        """Usage: serial <port> [--baud=<n>] [--timeout=<seconds>]
    Options:
        --baud=<n>  Baudrate [default: 9600]
            """
        print(arg)

    def do_quit(self, arg):
        """Quits out of Interactive Mode."""

        print('Good Bye!')
        exit()

def maxIndex(x):
    return np.argmax(x)
    
def showImage(image):
    plt.figure()
    plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    plt.show()

def main():
    """The entry point of the program."""

    opt = docopt(__doc__, sys.argv[1:])

    if opt['--interactive']:
        MyInteractive().cmdloop()

    print(opt)

    """
    network = mlp.MLP.from_arguments('MyMLP', 784, (12, 10), 0.01, afun_lib.ELLIOT_SIG)

    while True:
        command = input('Enter command: ').split()
        if command[0] == 'train':
            n = int(command[1]) if (len(command)) > 1 else 1
            train(network, dataset_lib.set_mnist_training, n)
        elif command[0] == 'input':
            inputEntry(network, dataset_lib.set_mnist_training, int(command[1]))
        elif command[0] == 'save':
            if network is None:
                print('No MLP specified')
                continue
            if len(command) < 2:
                print('Specify a name for the file')
                continue
            utils.saveMLP(network, 'networks/' + command[1] + '.txt')
        elif command[0] == 'load':
            if network is None:
                print('No MLP specified')
                continue
            if len(command) < 2:
                print('Specify a name for the mlp')
                continue
            try:
                print('Loading %s...' % (command[1]))
                network = utils.loadMLP('networks/' + command[1] + '.txt')
                print('Loading done')
            except:
                print('Could not load file')
        else:
            print('Unknown command')
        """

def inputEntry(network, dataset, i):
    print('Input %s:' % (maxIndex(dataset[i].exp)))
    out = network.eval(dataset[i].inp, dataset[i].exp)
    print('Guess: %i' % maxIndex(out))
    print('Output: %s' % out)
    # print('Cost: %.8f' % (network.getCost()))

def train(network, dataset, n):
    print('Training %i times...' % (n))
    for i in range(n):
        sum = 0.0
        for e in dataset:
            sum += network.train(e.inp, e.exp)
            #print('%i of %i: Cost: %.8f' % (i, len(trainSet), cost))
        print('Set %i of %i: ave. cost: %.16f' % (i + 1, n, sum / len(dataset)))
    print('Done training')
    
if __name__ == '__main__':
    main()