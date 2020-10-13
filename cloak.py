import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import os
import time
import pickle
from scipy.optimize import curve_fit
from numpy import sqrt


# Constant
brDistributionPath = './BrDistribution.csv'
bzDistributionPath = './BzDistribution.csv'
alpha = 1e-4
h = 1e-6
minRadius = 1.5e-2
Z0 = 0.05
loms = nu.linspace(0, 0.9*minRadius, 100)
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
    data.to_csv(rawPath, index=False)
    # get loss
    cookedPath = './BzDistribution.csv'
    while not os.path.exists(cookedPath):
        time.sleep(3)
    loss = getVariance(cookedPath)
    # if we get loss, delete curveDistribution, so make sure comsol wait for enough long time after study is completed.
    os.remove(cookedPath)
    os.remove(rawPath)
    return loss


# Main
# show init ws
# pl.scatter(loms, sqrt(minRadius**2 - loms**2) + Z0-minRadius)
# pl.plot(loms, curveFunction(loms))
# pl.show()

# set avgLosses
if os.path.exists('avgerageLosses.pickle'):
    with open('avgerageLosses.pickle', 'rb') as file:
        averageLosses = pickle.load(file)
else:
    averageLosses = nu.array([])
# set weights and ws
if os.path.exists('weights.pickle'):
    with open('weights.pickle', 'rb') as file:
        weights = pickle.load(file)
    ws = weights[-1, :]
else:
    def wsModel(loms, w0, w1, w2, w3, w4, w5):
        n = len(loms)
        result = nu.concatenate([
            nu.ones(n).reshape(-1, 1),
            loms.reshape(-1, 1),
            (loms**2).reshape(-1, 1),
            (loms**3).reshape(-1, 1),
            (loms**4).reshape(-1, 1),
            (loms**5).reshape(-1, 1)
        ], axis=-1) @ nu.array([w0, w1, w2, w3, w4, w5]).reshape(-1, 1)
        return result.ravel()
    ws, _ = curve_fit(wsModel, xdata=loms, ydata=sqrt(minRadius**2 - loms**2) + Z0-minRadius, p0=ws.tolist())
    ws = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
    weights = nu.array([[ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]]]).reshape(1, -1)
# set step
if os.path.exists('weights.pickle'):
    step = weights.shape[0]
else:
    step = 1
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
    weights = nu.concatenate([weights, ws.reshape(1, -1)])
    print('step: {:>2}, avgLoss: {}'.format(step, currentLoss))
    # store losses
    with open('avgerageLosses.pickle', 'wb') as file:
        pickle.dump(averageLosses, file)
    with open('weights.pickle', 'wb') as file:
        pickle.dump(weights, file)
    # next loop
    step += 1
