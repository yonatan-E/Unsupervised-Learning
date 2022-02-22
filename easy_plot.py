import matplotlib as plt
from enum import Enum
import matplotlib.pyplot as plt



class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

color_to_string = {Color.RED:'red', Color.BLUE:'blue', Color.GREEN:'green'}

def basic_two_axis_plot(data, color_, xlabel_, ylabel_, path=None):
    plt.plot(data, color=color_to_string[color_])
    plt.grid(axis='both', alpha=.3)
    plt.xticks(fontsize=7, alpha=.7)
    plt.yticks(fontsize=7, alpha=.7)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    if(path):
        plt.savefig(path)
    else:
        plt.show()

