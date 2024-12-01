import numpy as np

np.seterr(divide='ignore', invalid='ignore')
# from sklearn.mixture import GaussianMixture
from math import sqrt, exp
from scipy.special import erf
from scipy.stats import gaussian_kde
import torch



def kde_lossf(gt, pred):
    gt = gt.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    kde_ll = 0
    kde_list = []
    for i in range(pred.shape[0]):#time
        for j in range(pred.shape[1]):#agent
            try:
                kde = gaussian_kde(pred[i, j, :, :].T)
                t = np.clip(kde.logpdf(gt[i, j, :].T), a_min=-20,
                            a_max=None)[0]
                kde_ll += t 
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    return -kde_ll / (pred.shape[0] * pred.shape[1])

