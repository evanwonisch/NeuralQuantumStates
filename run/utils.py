from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime

path = "./"

def save_plot(caption):
    stamp = str(int(datetime.now().timestamp()))

    with open(path + "captions.csv", "a") as file:
        file.write(stamp + "   " + caption + "\n")
    plt.savefig(path + stamp + ".pdf")