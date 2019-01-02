#______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLCthe U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
#______________________________________________________________________________

import numpy as np
from random import uniform
from scipy import stats

def lhs(paramDists,nSamples):

    """
    The 'lhs' function accepts a list of parameter distributions from the 
    scipy.stats library for each parameter distribution the user would like to
    sample from as well as the number of samples to generate. The results of
    generating these samples will be a latin hypercube sampling of observations    that can be used to run Monte Carlo like simulations. 

    Arguments:
        paramDists  : list of distributions : from scipy.stats
            examples[stats.norm(0,1), stats.beta(1,5), stats.gamma(2)]
        nSamples    : integer 

    Returns:
        lhs_samples : numpy array           : [nSamples, # paramDists]

    """

    #Verify user inputs
    if type(nSamples) is not int: 
        raise ValueError("nSamples argument is expecting an int")

    for pp in range(len(paramDists)):
        if type(paramDists[pp]) is not stats._distn_infrastructure.rv_frozen:
            raise ValueError("paramDists argument is expecting distributions"
                              " from the scipy.stats library")

    mParams = len(paramDists)
    lhs_0to1 = np.linspace(0,1,nSamples+1)    

    lhs_samples = np.empty((nSamples,mParams))
    for jj in range(mParams):

        for ii in range(nSamples):
            lhs_samples[ii,jj]=uniform(lhs_0to1[ii],lhs_0to1[ii+1])

        #ppf:percent point function (inverse of CDF)
        lhs_samples[:,jj]=paramDists[jj].ppf(lhs_samples[:,jj])
        lhs_samples[:,jj]=np.random.permutation(lhs_samples[:,jj])

    return lhs_samples



if __name__ == "__main__":

    dist1 = stats.beta(0.5,2)
    dist2 = stats.norm(5,0.2)
    dist3 = stats.norm(10,3.4)
    dist4 = stats.beta(1,12)
    dist5 = stats.gamma(2)
    dist6 = stats.norm(-2.7,0.47)
    test = [dist1,dist2,dist3,dist4,dist5,dist6]    
    samples = lhs(test,int(1e5))

