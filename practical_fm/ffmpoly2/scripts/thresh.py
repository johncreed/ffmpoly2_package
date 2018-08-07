#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

x_cor = []
y_cor = []
with open(sys.argv[1]) as fi:
    for line in fi:
        x, y = line.strip().split(":")
        x_cor.append(int(x))
        y_cor.append(int(y))

print ("max: {} min: {}".format(max(y_cor), min(y_cor)))
x_cor = np.array(np.log(x_cor))
y_cor = np.array(np.log(y_cor/np.sum(y_cor)))


#threshs = [1, 12, 148, 1810, 22000, 268000, 3270000, 39800000] #criteo
#threshs = [8000] #avazu
"""
threshs = [8200] #hw1
for thresh in threshs:
    x_line = np.array(np.log([thresh]))
    plt.vlines(x_line, ymin = -20, ymax = 0, colors = 'R', Linestyles="dashdot")
"""
plt.xlabel(r"$\log(m)$", fontsize = 15)
plt.ylabel(r"$\log(Prob_m)$", fontsize = 15)

plt.plot(x_cor, y_cor, 'o')
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=1000)
