from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import jax
from functools import reduce

path = "./"

def save_plot(caption):
    stamp = str(int(datetime.now().timestamp()))

    with open(path + "captions.csv", "a") as file:
        file.write(stamp + "   " + caption + "\n")
    plt.savefig(path + stamp + ".pdf")


def count_params(params):
    if isinstance(params, jax.Array):
            return reduce(lambda a,b : a*b, params.shape, 1)

    acc = 0
    for key in params.keys():
        acc += count_params(params[key])

    return acc