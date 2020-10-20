import numpy as nu
from matplotlib import pyplot as pl
import pandas as pd
import os
import time
import sys
import pickle


minRadius = 1.5e-2


def curveFunction(loms, ws):
    if loms is nu.float:
        return ws[0] + ws[1] * loms**1 + ws[2] * loms**2 + ws[3] * loms**3 + ws[4] * loms**4 + ws[5] * loms**5
    elif len(loms) >= 2:
        zms = nu.zeros(len(loms))
        for i, lo in enumerate(loms):
            zms[i] = ws[0] + ws[1] * lo**1 + ws[2] * lo**2 + ws[3] * lo**3 + ws[4] * lo**4 + ws[5] * lo**5
        return zms
    else:
        raise ValueError


# Main

with open('weights.pickle', 'rb') as file:
    weights = pickle.load(file)
    print(f'weights = {weights}')
with open('averageLosses.pickle', 'rb') as file:
    averageLosses = pickle.load(file)
    print(f'losses = {averageLosses}')
loms = nu.linspace(0, 0.9*minRadius, 100)

for i in range(weights.shape[0]):
    zms = curveFunction(loms, weights[i, :])
    pl.scatter(loms, zms)
    pl.show()
