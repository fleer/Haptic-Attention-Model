import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np

def load_single(filename):
    """
    loads and returns a single experiment stored in filename
    returns None if file does not exist
    """
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        result = json.load(f)
    return result


style = {
    "linewidth": 2, "alpha": .7, "linestyle": "-", "markersize": 7}

#: default colors used for plotting
default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple']

fig = plt.figure()
min_o = 100.
max_o = 0.
# This is not a nice way to implement the different configuration scripts...
if len(sys.argv) > 1:
    for q in range(1,len(sys.argv)):
        file = load_single(sys.argv[q])

        #x = np.arange(len(file['accuracy']))
        x = np.asarray(file['learning_steps'])
        y_mean = np.asarray(file['accuracy'])
        y_sem = np.asarray(file['accuracy_std'])
        y_mean *= 100.
        y_sem *= 100.
        plt.plot(x, y_mean, color=default_colors[q-1], **style)

        min_ = np.inf
        max_ = - np.inf
        plt.fill_between(x, y_mean - y_sem, y_mean + y_sem, color=default_colors[q-1], alpha=.3)
        max_ = max(np.max(y_mean + y_sem), max_)
        min_ = min(np.min(y_mean - y_sem), min_)

        if min_o > min_:
            min_o = min_
        if max_o < max_:
            max_o = max_

        if file is None:
            print "Wrong file name!"
            sys.exit(0)
else:
    print "Give Results-Files as additional argument! \n " \
          "E.g. python plot.py ./results.json ./results_1.json"
    sys.exit(0)


# adjust visible space
y_lim = [min_o - .1 * abs(max_o - min_o), max_o + .1 * abs(max_o - min_o)]

if min_ != max_:
    plt.ylim(y_lim)

plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Accuracy [%]", fontsize=16)
plt.grid(True)

plt.legend([str(sys.argv[l]) for l in range(1,len(sys.argv))], loc='lower right')

plt.show()

