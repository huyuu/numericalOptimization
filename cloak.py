import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import os
import time
import pickle


# Constant
brDistributionPath = './BrDistribution.csv'
bzDistributionPath = './BzDistribution.csv'
alpha = 1e-4
h = 1e-6
minRadius = 1.5e-2
loms = nu.linspace(0, 0.9*minRadius, 20)
# gloabl variable
ws = nu.zeros(6)

def curveFunction(loms):
    if loms is nu.float:
        return ws[0] +\
        ws[1] * loms**1 +\
        ws[2] * loms**2 +\
        ws[3] * loms**3 +\
        ws[4] * loms**4 +\
        ws[5] * loms**5
    elif len(loms) >= 2:
        zms = nu.zeros(len(loms))
        for i, lo in enumerate(loms):
            zms[i] = ws[0] +\
            ws[1] * lo**1 +\
            ws[2] * lo**2 +\
            ws[3] * lo**3 +\
            ws[4] * lo**4 +\
            ws[5] * lo**5
        return zms
    else:
        raise ValueError


def getVariance(path):
    data = pd.read_csv(path, skiprows=8)
    data.columns = ['r', 'z', 'B']
    data = data.pivot(index='r', columns='z', values='B')
    return nu.var(data.values)


def loss(ws):
    # create index
    rawPath = './curveDistribution.csv'
    # create curve distribution
    zms = curveFunction(loms)
    data = pd.DataFrame({
        'r': loms,
        'z': zms
    })
    data.to_csv(rawPath)
    # get loss
    cookedPath = './BzDistribution.csv'
    while not os.path.exists(cookedPath):
        time.sleep(3)
    loss = getVariance(cookedPath)
    # if we get loss, delete curveDistribution, so make sure comsol wait for enough long time after study is completed.
    os.remove(rawPath)
    return loss


# Main

averageLosses = nu.array([])
while True:
    currentLoss = 0
    # update w
    for i in range(6):
        _wp = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
        _wm = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
        _wp[i] += h
        _wm[i] -= h
        pLoss = loss(_wp)
        mLoss = loss(_wm)
        ws[i] -= alpha * ( pLoss - mLoss )/(2*h)
        currentLoss += (pLoss+mLoss)/2
    averageLosses = nu.append(averageLosses, currentLoss)
    # store losses
    with open('avgerageLosses.pickle', 'wb') as file:
        pickle.dump(averageLosses, file)
