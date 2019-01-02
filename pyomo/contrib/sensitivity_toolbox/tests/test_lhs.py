# ____________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________

"""
Unit Tests for Latin Hypercube Sampling
"""

import pyutilib.th as unittest

import numpy as np
from random import uniform
from scipy import stats

from pyomo.contrib.sensitivity_toolbox.lhs import lhs

class TestSensitivityToolbox(unittest.TestCase):

    def test_bad_arg(self):

        #scipy.stats distribution ValueError
        try:
            Result = lhs(['norm'],10)
            self.fail("Expected ValueError: did not send scipy.stats"
                       "  distribution")
        except ValueError:
            pass

        #int ValueError
        try:
            Result = lhs([stats.norm(0,1)],1e3)
            self.fail("Expected ValueError: did not send int for nSamples")
        except ValueError:
            pass
        
    def test_continuous_dist(self):

        lhs_norm = lhs([stats.norm(5,0.1)],int(1e4))
        
        #upper and lower tail statistics
        lhs_norm_max_std = abs((np.mean(lhs_norm[:,0])-max(lhs_norm[:,0]))
                          /np.std(lhs_norm[:,0]))
        lhs_norm_min_std = ((np.mean(lhs_norm[:,0])-min(lhs_norm[:,0]))
                          /np.std(lhs_norm[:,0]))
        #verify tails are part of the LHS collect
        self.assertTrue(lhs_norm_max_std > 3)
        self.assertTrue(lhs_norm_min_std > 3)
        #verify statistics
        self.assertAlmostEqual(np.mean(lhs_norm[:,0]),5,4)
        self.assertAlmostEqual(np.std(lhs_norm[:,0]),0.1,4)

        lhs_beta = lhs([stats.beta(5,1)],int(1e4))
        mean_beta, var_beta = stats.beta.stats(5,1,moments='mv')

        #lower tail statistics only
        #the beta distribution iwth alpha = 5 and beta = 1 has a long left tail
        lhs_beta_min_std = ((np.mean(lhs_beta[:,0])-min(lhs_beta[:,0]))
                             /np.std(lhs_beta[:,0]))

        #verify tail is part of LHS collect
        self.assertTrue(lhs_beta_min_std > 3)

        #verify statistics
        self.assertAlmostEqual(np.mean(lhs_beta[:,0]),mean_beta,4)
        self.assertAlmostEqual(np.var(lhs_beta[:,0]),var_beta,4)

    def test_discrete_dist(self):

        #Bernoulli distribution will be used to verify statisitics in the
        # discrete case
        lhs_bern = lhs([stats.bernoulli(0.74)],int(1e4))
        mean_bern, var_bern = stats.bernoulli.stats(0.74, moments='mv')

        self.assertAlmostEqual(np.mean(lhs_bern[:,0]),mean_bern,4)
        self.assertAlmostEqual(np.var(lhs_bern[:,0]), var_bern,4)

        #Poisson distribution is used to verify tails are part of the 
        # LHS collect
        lhs_poisson = lhs([stats.poisson(100)],int(1e4))
        lhs_max_std = abs((np.mean(lhs_poisson[:,0])-max(lhs_poisson[:,0]))
                          /np.std(lhs_poisson[:,0]))
        lhs_min_std = ((np.mean(lhs_poisson[:,0])-min(lhs_poisson[:,0]))
                          /np.std(lhs_poisson[:,0]))

        self.assertTrue(lhs_max_std > 3)
        self.assertTrue(lhs_min_std > 3)

        

if __name__=="__main__":
    unittest.main()
