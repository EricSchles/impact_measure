import numpy as np
from scipy.stats as pareto

def gini_mean_diff(x):
    n = x.shape[0]
    one_size_squared = 1/np.power(n, 2)
    summation = 0
    for i in range(n):
        for j in range(n):
            summation += abs(x[i] - x[j])
    return summation

def pooled_variability_gini_mean_diff(x_1, x_2):
    if (np.var(x_1) > 0) and (np.var(x_2) > 0):
        gmd_x_1 = gini_mean_diff(x_1)
        gmd_x_1_sqrd = np.power(gmd_x_1, 2)
        gmd_x_2_sqrd = np.power(gmd_x_2, 2)
        summation = gmd_x_1_sqrd + gmd_x_2_sqrd
        return np.power(summation/2, 0.5)
    elif np.var(x_1) > 0:
        return gini_mean_diff(x_1)
    elif np.var(x_2) > 0:
        return gini_mean_diff(x_2)
    else:
        return 1e-10
    
def central_tendency_diff(x_1, x_2):
    median_diff = np.median(x_2) - np.median(x_1)
    gmd = pooled_variability_gini_mean_diff(
        x_1, x_2
    )
    return median_diff/gmd

def direction_central_tendency(x_1, x_2):
    if np.median(x_2) < np.median(x_1):
        return -1
    else:
        return 1

def central_tendency_diff_weight(x_1, x_2):
    return min([central_tendency_diff(x_1, x_2), 2])/2

def mle_pareto_param(x):
    x_min = x.min()
    return x.shape[0]/np.sum(np.log(x)/x_min)

def morph_diff(x_1, x_2):
    x1_pareto_param = mle_pareto_param(x_1)
    x2_pareto_param = mle_pareto_param(x_2)
    x1_pdf = pareto.pdf(x_1, x1_pareto_param)
    x2_pdf = pareto.pdf(x_2, x2_pareto_param)
    return np.sum(np.abs(x2_pdf - x1_pdf))

def log_modulus_transform(x):
    return np.sign(x) * np.log(np.abs(x) + 1)
    
def momentum(x):
    abs_L = log_modulus_transform(np.abs(x))
    L = np.sum(log_modulus_transform(x))
    return L/abs_L

def direction_morph(x_1, x_2):
    if momentum(x_2) < momentum(x_1):
        return -1
    else:
        return 1

def impact(x_1, x_2):
    ctdw = central_tendency_diff_weight(x_1, x_2)
    dct = direction_central_tendency(x_1, x_2)
    ctd = central_tendency_diff(x_1, x_2)
    central_tendency = ctdw * dct * ctd
    dm = direction_morph(x_1, x_2)
    md = morph_diff(x_1, x_2)
    morphic = (1 - ctdw) * dm * md
    return central_tendency + morphic
    
