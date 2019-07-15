#!/usr/bin/env python

# call as: python cov2u.py ch cov_file n_draws
# EXAMPLE:  python cov2u.py 
# EXAMPLE:  python cov2u.py 37 cov.nc 2000

# =======================================
# Version 0.2
# 25 June, 2019
# https://patternizer.github.io
# michael.taylor AT reading DOT ac DOT uk
# =======================================

from  optparse import OptionParser
import numpy as np
import scipy.linalg as la
import xarray
import matplotlib.pyplot as plt

def calc_cov2u(N, Xmean, Xcov):
    '''
    Routine to estimate uncertainty from a covariance matrix using Monte Carlo sampling from the underlying distribution. Code adapted from routine coded by Jonathan Mittaz: get_harm.py
    '''

#    Xmean = np.array([0.,0.])
#    Xcov = np.array([[0.04, 0.112] , [0.112, 0.64]])
    
#    uncert = np.array([[0.2,0.],[0.,0.8]])
    # correl = np.array([[1.,0.7],[0.7,1.]])
    # cov = np.matmul(uncert,correl)
    # orig_cov = np.matmul(cov,uncert)
    # Xcov = np.copy(orig_cov)
#    print('U(target)=', uncert)

    eigenval, eigenvec = np.linalg.eig(Xcov)
    R = eigenvec
    S = np.diag(np.sqrt(eigenval))
    T = np.matmul(R,S)
    output = np.random.multivariate_normal(Xmean, Xcov, size=N)
#    ndims = len(Xcov.shape)
    ndims = Xcov.shape[1]
    position = np.zeros((N, ndims))
    final = np.zeros((N, ndims))
    for j in range(ndims):
        position[:,:] = 0.
        position[:,j] = np.random.normal(size=N, loc=0., scale=1.)
        for i in range(position.shape[0]):
            vector = position[i,:]
            ovector = np.matmul(T,vector)
            final[i,:] = final[i,:]+ovector

    U = np.std(final, axis=0)

    return U

if __name__ == "__main__":

    #----------------------------------------------
    parser = OptionParser("usage: %prog ch cov_file n_draws")
    (options, args) = parser.parse_args()    
    try:
        ch = args[0]
        cov_file = args[1]
        N = args[2]
    except:
        ch = 37
        cov_file = 'FIDUCEO_Harmonisation_Data_' + str(ch) + '.nc'
        N = 2000
    
    ds = xarray.open_dataset(cov_file)
    Xave = ds.parameter
    Xcov = ds.parameter_covariance_matrix
    Xcor = ds.parameter_correlation_matrix
    Xu = ds.parameter_uncertainty
    U = calc_cov2u(N, Xave, Xcov)
    
    print('U(estimate)=', np.diag(U))
    print('** end')


