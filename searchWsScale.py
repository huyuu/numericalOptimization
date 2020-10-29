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
minRadius = 1.5
Z0 = 5
# gloabl variable
averageLosses = None
FMThickness = 1e-3
points = 5
loms = nu.linspace(0, 0.9*minRadius, points)
ZL = Z0 - 0.9*minRadius
ZU = Z0 + 0.9*minRadius
zms = nu.linspace(ZL, ZU, points)


# Model

def curveFunction(loms, ws):
    if loms is nu.float:
        return sum(( loms**i*w for i, w in enumerate(ws) ))
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
    minRadius = 1.5e-2
    Z0 = 5e-2
    data = pd.read_csv(path, skiprows=8)
    data.columns = ['r', 'z', 'B']
    bsOut = nu.array([])
    bsIn = nu.array([])
    for i in data.index:
        lo = data.iloc[i, 0]
        z = data.iloc[i, 1]
        z_abs = abs(z)
        b = data.iloc[i, 2]
        # inside
        if lo <= minRadius*0.99 and z_abs <= Z0:
            bsIn = nu.append(bsIn, data.iloc[i, 2])
        # outside
        elif 1.4*minRadius >= lo >= minRadius*1.01 or 1.4*Z0 >= z_abs > Z0:
            bsOut = nu.append(bsOut, data.iloc[i, 2])
        # mergin
        else:
            continue
    assert bsIn.shape[0] >= 1
    assert bsOut.shape[0] >= 1
    return bsOut.var() + abs(bsIn).mean()


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


def _model(los, w0, w1, w2, w3):
    return nu.array([ w0 + w1*lo + w2*lo**2 + w3*lo**3 for lo in los ])


def fitWith(samples):
    los = samples[:, 0]
    zs = samples[:, 1]
    _ws, _ = curve_fit(_model, xdata=los, ydata=zs, p0=nu.zeros(4))
    return nu.array([_ws[0], _ws[1], _ws[2], _ws[3]])


# Main

ws = nu.zeros(4).reshape(1, -1)

for z0 in zms:
    for z1 in zms:
        for z2 in zms:
            for z3 in zms:
                for z4 in zms:
                    samples = nu.array([
                        [loms[0], z0],
                        [loms[1], z1],
                        [loms[2], z2],
                        [loms[3], z3],
                        [loms[4], z4],
                    ])
                    ws = nu.concatenate([ws, fitWith(samples).reshape(1, -1)])
ws = ws[1:, :]

for i in range(ws.shape[1]):
    pl.scatter(i*nu.ones(ws.shape[0]), ws[:, i], label=f'w{i}')
pl.yscale('log')
pl.xlabel('w index')
pl.ylabel('value')
pl.legend()
pl.show()
