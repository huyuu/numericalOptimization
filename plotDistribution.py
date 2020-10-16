import numpy as nu
from matplotlib import pyplot as pl
import pandas as pd
import os
import time
import sys


path = sys.argv[1]
data = pd.read_csv(path, skiprows=8)
data.columns = ['r', 'z', 'B']
data = data.pivot(index='r', columns='z', values='B')

pl.contourf(data.index, data.columns, data.values.T, levels=50)
pl.colorbar()
pl.show()
