import numpy as np

# Estimator metrics

def var(x):
    return np.mean(np.square(x)) - np.square(np.mean(x))

def stddev(x):
    return np.sqrt(var(x))

def bias(x, ref):
    return np.mean(x-ref)

def MSE(x, ref):
    return np.mean(np.square(x-ref))

def RMSE(x, ref):
    return np.sqrt(MSE(x, ref))