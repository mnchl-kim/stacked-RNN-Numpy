import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def cross_entropy(x, index):
    """
    Assumption: the ground truth vector contains only one non-zero component with a value of 1
    """

    loss = - np.log(x[index]) if x[index] > 0 else 0
    return loss


def cross_entropy_d(x, index):
    """
    Assumption: the ground truth vector contains only one non-zero component with a value of 1
    """

    x[index] -= 1
    return x


def char_to_ix(chars):
    """
    Make a dictionary that maps a character to an index

    Arguments:
        chars -- list of character set

    Returns:
        dictionary that maps a character to an index
    """

    return {ch: i for i, ch in enumerate(chars)}


def ix_to_char(chars):
    """
    Make a dictionary that maps an index to a character

    Arguments:
        chars -- list of character set

    Returns:
        dictionary that maps an index to a character
    """

    return {i: ch for i, ch in enumerate(chars)}


def one_hot(data, ch2ix):
    """
    Arguments:
        data -- string
        ch2ix -- dictionary that maps a character to an index

    Returns:
        Numpy array, shape = (len(data), len(ch2ix), 1)
    """

    result = []
    for i in range(len(data)):
        x = np.zeros((len(ch2ix), 1))
        if data[i] is not None:
            x[ch2ix[data[i]], 0] = 1
            result.append(x)

    return np.array(result)


def initialize_xavier(first, second):
    """
    Xavier initialization

    Arguments:
        first -- first dimension size
        second -- second dimension size

    Returns:
        W -- Weight matrix initialized by Xavier method
    """

    sd = np.sqrt(2.0 / (first + second))
    W = np.random.randn(first, second) * sd

    return W


class Graph:
    def __init__(self, xlabel, ylabel):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ion()
        plt.show()

    def update(self, x, y, img_name='Figure'):
        plt.plot(x, y, color='xkcd:royal blue')
        plt.show()
        plt.savefig('./figure/' + img_name + '.png')
        plt.pause(0.001)
