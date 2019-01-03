"""
This example uses docopt with the built in cmd module to demonstrate an
interactive command application. Based on https://github.com/docopt/docopt/tree/master/examples.

Usage:
    my_program list (mlp | dat)
    my_program load (mlp | dat) <index>
    my_program input <index>
    my_program train <repeat>
    my_program test
    my_program save <filename>
    my_program (-i | --interactive)
    my_program (-h | --help | --version)

Options:
    -i, --interactive  Interactive Mode
    -h, --help  Show this screen and exit.
"""

import sys
import os
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

def docopt_cmd(func):
    #http://rogerdudler.github.io/git-guide/
    #https://google.github.io/styleguide/pyguide.html
    #https://github.com/docopt/docopt/blob/master/examples/interactive_example.py
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
    
    _mlp = None
    _dat = None

    @docopt_cmd
    def do_list(self, arg):
        """Usage: list (mlp | dat)"""
        if arg['mlp']:
            files = os.listdir('networks/')
            for i, name in enumerate(files):
                print('%i %s' % (i, name))
        elif arg['dat']:
            for i, s in enumerate(dataset_lib.datasets):
                print('%i %s' % (i, s))

    @docopt_cmd
    def do_load(self, arg):
        """Usage: load (mlp | dat) <index>"""
        if arg['mlp']:
            i = int(arg['<index>'])
            files = os.listdir('networks/')
            self._mlp = utils.loadMLP('networks/' + files[i])
            print('Loaded %s' % (files[i]))
        elif arg['dat']:
            i = int(arg['<index>'])
            self._dat = dataset_lib.datasets[i]
            print('Loaded %s' % (self._dat))
            
    @docopt_cmd
    def do_input(self, arg):
        """Usage: input <index>"""
        if self._mlp == None:
            print('No MLP specified')
            return
        if self._dat == None:
            print('No dataset specified')
            return
        inputEntry(self._mlp, self._dat, int(arg['<index>']))
        
    @docopt_cmd
    def do_train(self, arg):
        """Usage: train <repeat>"""
        if self._mlp == None:
            print('No MLP specified')
            return
        if self._dat == None:
            print('No dataset specified')
            return
        train(self._mlp, self._dat, int(arg['<repeat>']))

    @docopt_cmd
    def do_test(self, arg):
        """Usage: test"""
        if self._mlp == None:
            print('No MLP specified')
            return
        if self._dat == None:
            print('No dataset specified')
            return
        test(self._mlp, self._dat)

    @docopt_cmd
    def do_save(self, arg):
        """Usage: save <filename>"""
        if self._mlp == None:
            print('No MLP specified')
            return
        path = 'networks/' + arg['<filename>'] + '.txt'
        utils.saveMLP(self._mlp, path)
        print('Saved MLP as %s' % (path))

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

def inputEntry(network, dataset, i):
    print('Input %s:' % (maxIndex(dataset.entries[i].exp)))
    out = network.eval(dataset.entries[i].inp, dataset.entries[i].exp)
    print('Guess: %i' % maxIndex(out))
    print('Output: %s' % out)
    # print('Cost: %.8f' % (network.getCost()))

def train(network, dataset, n):
    print('Training %i times...' % (n))
    for i in range(n):
        sum = 0.0
        for e in dataset.entries:
            sum += network.train(e.inp, e.exp)
            #print('%i of %i: Cost: %.8f' % (i, len(trainSet), cost))
        print('Set %i of %i: ave. cost: %.16f' % (i + 1, n, sum / len(dataset.entries)))
    print('Done training')

def test(network, dataset):
    total = len(dataset.entries)
    correct = 0
    print('Testing...')
    for e in dataset.entries:
        exp = maxIndex(e.exp)
        out = maxIndex(network.eval(e.inp, e.exp))
        if exp == out:
            correct += 1
    percentage = correct * 100.0 / total
    print('Done testing: Classified %i of %i correct (%.4f%%)' % (correct, total, percentage))

if __name__ == '__main__':
    main()