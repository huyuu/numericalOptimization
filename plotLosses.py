import numpy as nu
import pickle
import pandas as pd
from matplotlib import pyplot as pl


with open('averageLosses.pickle', 'rb') as file:
    losses = pickle.load(file)
pl.plot(losses)
pl.show()
