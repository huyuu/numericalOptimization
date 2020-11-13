import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import os
import time
import pickle
import datetime as dt
from scipy.optimize import curve_fit
from scipy.optimize import minimize, fmin_cg, Bounds, LinearConstraint
from numpy import sqrt


# Constant
brDistributionPath = './BrDistribution.csv'
bzDistributionPath = './BzDistribution.csv'
alpha = 1
h = 1e-3
minRadius = 1.5  # 1.5cm
Z0 = 5  # 5cm
loms = nu.linspace(0, 0.9*minRadius, 300)
# gloabl variable
ws = nu.zeros(4)
averageLosses = None
FMThickness = 0.1  # 0.1cm


def curveFunction(loms, ws):
    if loms is nu.float:
        return ws[0] + ws[1] * loms**1 + ws[2] * loms**2 + ws[3] * loms**3# + ws[4] * loms**4 + ws[5] * loms**5
    elif len(loms) >= 2:
        zms = nu.zeros(len(loms))
        for i, lo in enumerate(loms):
            zms[i] = ws[0] + ws[1] * lo**1 + ws[2] * lo**2 + ws[3] * lo**3# + ws[4] * lo**4 + ws[5] * lo**5
        return zms
    else:
        print('ValueError')
        raise ValueError


def isPointOnMagnet(lo, z, ws):
    if z >= 0:
        zm = curveFunction(lo, ws)
        return abs(z-zm) <= FMThickness
    else:
        zm = -curveFunction(lo, ws)
        return abs(z-zm) <= FMThickness


def getVariance(path, ws):
    assert os.path.exists(path)
    global minRadius, Z0
    data = pd.read_csv(path, skiprows=8)
    data.columns = ['r', 'z', 'B']
    data['r'] *= 1e2  # [m] -> [cm]
    data['z'] *= 1e2  # [m] -> [cm]

    # bsOut = nu.array([])
    # bsIn = nu.array([])
    # for i in data.index:
    #     lo = data.iloc[i, 0]
    #     z = data.iloc[i, 1]
    #     z_abs = abs(z)
    #     b = data.iloc[i, 2]
    #     # inside
    #     if lo <= minRadius*0.99 and z_abs <= Z0:
    #         bsIn = nu.append(bsIn, data.iloc[i, 2])
    #     # outside
    #     elif 1.4*minRadius >= lo >= minRadius*1.01 or 1.4*Z0 >= z_abs > Z0:
    #         bsOut = nu.append(bsOut, data.iloc[i, 2])
    #     # mergin
    #     else:
    #         continue
    # assert bsIn.shape[0] >= 1
    # assert bsOut.shape[0] >= 1
    # return bsOut.var() + abs(bsIn).mean()

    data = data.pivot(index='r', columns='z', values='B')
    _var = nu.var(data.iloc[:200*3//4, 46].values)
    _mean = data.iloc[:200*3//4, 46].values.mean()
    return _var + _mean


def loss(ws):
    # create index
    rawPath = './curveDistribution.csv'
    # create curve distribution
    global loms
    zms = curveFunction(loms, ws)
    data = pd.DataFrame({
        'r': loms * 1e-2,
        'z': zms * 1e-2
    })
    # print(data)
    data.to_csv(rawPath, index=False)
    # get loss
    cookedPath = './BzDistribution.csv'
    while True:
        if os.path.exists(cookedPath):
            if os.path.getsize(cookedPath) >= 100:
                break
        time.sleep(1)

    loss = getVariance(cookedPath, ws)
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


def callback(ws, result):
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
    for key, value in result.items():
        print(f'{key}: {value}')
    return False


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
    # def wsModel(loms, w0, w1, w2, w3, w4, w5):
    def wsModel(loms, w0, w1, w2, w3):
        n = len(loms)
        result = nu.concatenate([
            nu.ones(n).reshape(-1, 1),
            loms.reshape(-1, 1),
            (loms**2).reshape(-1, 1),
            (loms**3).reshape(-1, 1)
            # (loms**4).reshape(-1, 1),
            # (loms**5).reshape(-1, 1)
        # ], axis=-1) @ nu.array([w0, w1, w2, w3, w4, w5]).reshape(-1, 1)
        ], axis=-1) @ nu.array([w0, w1, w2, w3]).reshape(-1, 1)
        return result.ravel()
    R = 0.9*minRadius
    ws, _ = curve_fit(wsModel, xdata=loms, ydata=sqrt(R**2 - loms**2) + Z0-R, p0=ws.tolist())
    # ws = nu.array([ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]])
    ws = nu.array([ws[0], ws[1], ws[2], ws[3]])
    # ws = nu.array([0.9*Z0, 0, 0, 0, 0, 0])
    weights = nu.array([[ws[0], ws[1], ws[2], ws[3]]]).reshape(1, -1)
    # weights = nu.array([[ws[0], ws[1], ws[2], ws[3], ws[4], ws[5]]]).reshape(1, -1)
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

ZL = Z0 - 0.9*minRadius
ZU = Z0 + 0.9*minRadius
_loms = nu.linspace(0, 0.9*minRadius, 10)
# _A = nu.array([1, _loms[0], _loms[0]**2, _loms[0]**3, _loms[0]**4, _loms[0]**5]).reshape(1, -1)
_A = nu.array([1, _loms[0], _loms[0]**2, _loms[0]**3]).reshape(1, -1)
for lo in _loms[1:]:
    # _A = nu.concatenate([_A, nu.array([1, lo, lo**2, lo**3, lo**4, lo**5]).reshape(1, -1)])
    _A = nu.concatenate([_A, nu.array([1, lo, lo**2, lo**3]).reshape(1, -1)])
print(_A)
constraint = LinearConstraint(A=_A, lb=ZL, ub=ZU)
result = minimize(fun=loss, x0=ws, method='trust-constr', constraints=constraint, callback=callback, options={'maxiter': 100000, 'disp': True,  'initial_tr_radius': 1, 'verbose': 3, 'barrier_tol': 1e-8})
# result = minimize(fun=loss, x0=ws, method='BFGS', callback=callback)

constraints = []
# for lo in _loms:
#     constraints.append({
#         'type': 'ineq',
#         'fun': lambda w: w[0] + w[1]*lo + w[2]*lo**2 + w[3]*lo**3 + w[4]*lo**4 + w[5]*lo**5 - ZL,
#         # 'jac': lambda xs: nu.array([1, lo, lo**2, lo**3, lo**4, lo**5])
#     })
#     constraints.append({
#         'type': 'ineq',
#         'fun': lambda w: ZU - (w[0] + w[1]*lo + w[2]*lo**2 + w[3]*lo**3 + w[4]*lo**4 + w[5]*lo**5),
#         # 'jac': lambda xs: -1 * nu.array([1, lo, lo**2, lo**3, lo**4, lo**5])
#     })
# result = minimize(fun=loss, x0=ws, method='SLSQP', constraints=constraints, jac=None, callback=callback, options={'maxiter': 10000, 'disp': True})
