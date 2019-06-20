#!/usr/bin/env python

# call as: python cov2u.py cov.nc

# =======================================
# Version 0.1
# 20 June, 2019
# https://patternizer.github.io
# michael.taylor AT reading DOT ac DOT uk
# =======================================

import os
import os.path
from os import fsync, remove
import glob
import optparse
from  optparse import OptionParser
import sys
import numpy as np
import scipy.linalg as la
import xarray
import seaborn as sns; sns.set(style="darkgrid")
import matplotlib.pyplot as plt; plt.close("all")

def calc_cov2u(N, Xmean, Xcov):
    '''
    Routine to estimate uncertainty from a covariance matrix using Monte Carlo sampling from the underlying distribution. Code adapted from routine coded by Jonathan Mittaz.
    '''

    eigenval, eigenvec = np.linalg.eig(Xcov)
    R = eigenvec
    S = np.diag(np.sqrt(eigenval))
    T = np.matmul(R,S)
    output = np.random.multivariate_normal(Xmean,Xcov,size=N)
    ndims = len(Xcov.shape)
    position = np.zeros((size,ndims))
    final = np.zeros((size,ndims))
    for j in range(ndims):
        position[:,:] = 0.
        position[:,j] = np.random.normal(size=N,loc=0.,scale=1.)
        for i in range(position.shape[0]):
            vector = position[i,:]
            ovector = np.matmul(T,vector)
            final[i,:] = final[i,:]+ovector

    U = np.std(ovector, axis=0)

    return U

if __name__ == "__main__":

    #----------------------------------------------
    # parser = OptionParser("usage: %prog ch cov_file")
    # (options, args) = parser.parse_args()

    # ch = args[0]
    # cov_file = args[1]

    ch = 37
    cov_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'

    ds = xarray.open_dataset(cov_file)

    Xave = ds.parameter
    Xcov = ds.covariance_matrix
    U = calc_cov2u(N,Xave,Xcov)
    
    print('Uncertainty=', U)
    print('** end')


