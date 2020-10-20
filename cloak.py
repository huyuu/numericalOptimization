import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import os
import time
import pickle
import datetime as dt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, fmin_cg
from numpy import sqrt


# Constant
brDistributionPath = './BrDistribution.csv'
bzDistributionPath = './BzDistribution.csv'
alpha = 1
h = 1e-3
minRadius = 1.5e-2
Z0 = 0.05
loms = nu.linspace(0, 0.9*minRadius, 100)
# gloabl variable
ws = nu.zeros(6)
averageLosses = None


def curveFunction(loms, ws):
    if loms is nu.float:
        return ws[0] + ws[1] * loms**1 + ws[2] * loms**2 + ws[3] * loms**3 + ws[4] * loms**4 + ws[5] * loms**5
    elif len(loms) >= 2:
        zms = nu.zeros(len(loms))
        for i, lo in enumerate(loms):
            zms[i] = ws[0] + ws[1] * lo**1 + ws[2] * lo**2 + ws[3] * lo**3 + ws[4] * lo**4 + ws[5] * lo**5
        return zms
    else:
        print('ValueError')
        raise ValueError


def getVariance(path):
    data = pd.read_csv(path, skiprows=8)
    data.columns = ['r', 'z', 'B']
    data = data.pivot(index='r', columns='z', values='B')
    _var = nu.var(data.iloc[:, 46].values)
    return _var


def loss(ws):
    # create index
    rawPath = './curveDistribution.csv'
    # create curve distribution
    global loms
    zms = curveFunction(loms, ws)
    data = pd.DataFrame({
        'r': loms,
        'z': zms
    })
    # print(data)
    data.to_csv(rawPath, index=False)
    # get loss
    cookedPath = './BzDistribution.csv'
    while True:
        if os.path.exists(cookedPath):
            if os.path.getsize(cookedPath) >= 100:
                break
        time.sleep(3)

    loss = getVariance(cookedPath)
    # if we get loss, delete curveDistribution, so make sure comsol wait for enough long time after study is completed.
    try:
        os.remove(cookedPath)
    except PermissionError:
        time.sleep(2)
        os.remove(cookedPath)

    try:
        os.remove(rawPath)
    except PermissionError:
        time.sleep(2)
        os.remove(rawPath)

    return loss


def callback(ws):
    currentLoss = loss(ws)
    global averageLosses, weights, start, step
    averageLosses = nu.append(averageLosses, currentLoss)
    weights = nu.concatenate([weights, ws.reshape(1, -1)])
    timeDelta = (dt.datetime.now() - start).total_seconds()
    print('step: {:>2}, avgLoss: {}, cost: {:>4.2f}[min]'.format(step, currentLoss, timeDelta/60))
    with open('averageLosses.pickle', 'wb') as file:
        pickle.dump(averageLosses, file)
    with open('weights.pickle', 'wb') as file:
        pickle.dump(weights, file)
    start = dt.datetime.now()
    step += 1


# Main
# show init ws
# pl.scatter(loms, sqrt(minRadius**2 - loms**2) + Z0-minRadius)
# pl.plot(loms, curveFunction(loms))
# pl.show()

# set avgLosses
if os.path.exists('averageLosses.pickle'):
    with open('averageLosses.pickle', 'rb') as file:
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

start = dt.datetime.now()

# while True:
#     # update w
#     for i in range(6):
#         _wp = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
#         _wm = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
#         _wp[i] += h
#         _wm[i] -= h
#         pLoss = loss(_wp)
#         mLoss = loss(_wm)
#         # print('w[{}] -= {} * ({} - {}) / (2*{})'.format(i, alpha, pLoss, mLoss, h))
#         ws[i] -= alpha * ( pLoss - mLoss )/(2*h)
#     currentLoss = loss(ws)
#     averageLosses = nu.append(averageLosses, currentLoss)
#     weights = nu.concatenate([weights, ws.reshape(1, -1)])
#     print('step: {:>2}, avgLoss: {}'.format(step, currentLoss))
#     # store losses
#     with open('averageLosses.pickle', 'wb') as file:
#         pickle.dump(averageLosses, file)
#     with open('weights.pickle', 'wb') as file:
#         pickle.dump(weights, file)
#     # next loop
#     step += 1


result = minimize(fun=loss, x0=ws, method='Nelder-Mead', jac=None, callback=callback, options={'maxiter': 10000, 'disp': True})
# result = fmin_cg(f=loss, x0=ws, maxiter=10000, callback=callback)
