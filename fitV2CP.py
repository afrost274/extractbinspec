#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:50:05 2020

@author: abigail
"""

import time
import numpy as np
import pickle # for loading pickled test data
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import warnings
import readGravity as rg
from scipy.optimize import minimize
import ModelFit as MF
import readGravity as rg



start = time.perf_counter()

#dir = './data/'
dir = './data/'
#file = 'GRAVI.2016-06-18*_singlescivis_singlesciviscalibrated.fits'
file = 'SCI_TYC6265-1977-1_GRAVITY_UT1UT2UT3UT4_MEDIUM-COMBINED_SINGLE_SCI_VIS_CALIBRATED.fits'

method ='SLSQP'#''BFGS' # 'BFGS' # 'Nelson-Mead' 'Powell'

# First guess from ASPRO2
# x, y, fratio
x = -0.49 #-3.3
y = 1.13    #-1.48
fr = 0.7
#nwave = 210 #1700

#load data
data = rg.ReadFilesGRAVITY(dir, file)
nwave = data['nW']

#Initialize first parameters
firstguess = MF.GenParamVector(x, y, fr, nwave)
bounds = MF.BoundsDef(nwave)


#datay, datayerr = MF.GenerateDataVector(data)

#print (firstguess)


# # curve fit the test data
#fitparams, cavmat = curve_fit(MF.ModelFitBin, data, datay, p0=firstguess, sigma=datayerr, bounds=(boundsmin, boundsmax))
res = minimize(MF.ModelChi2Bin, firstguess, args=data, method=method, bounds=bounds, tol=1e-6)
# ydata is vector of v2, vector of diff vis, and final obe closure phase
fitparams = res.x
#print(fitparams)


ymodel = MF.GenerateData(data, fitparams)
chivalue = MF.ModelChi2Bin(fitparams, data)

end = time.perf_counter()
thetime = end - start
print(thetime)

MF.plotFluxRatios(data, fitparams, time=thetime, name=method, dirdat=dir)
MF.giveDataModelChi2(data, ymodel, name=method, dirdat=dir)

MF.PrintFitDetails(fitparams, data, ymodel, method, thetime)

#plot CP
MF.plotCPtriangle(data,ymodel)
MF.plotV2baseline(data,ymodel)
MF.plotCPandV2(data,ymodel)

plt.show()

# plt.figure(0)
# plt.clf()
# plt.plot(datax, datay) # plot the raw data
# # plt.plot(datax, yfit) # plot the equation using the fitted parameters
# plt.show()
# #
# print(fitparams)


#neldermead instead?
# bnds = ((0,1),(0,1),(-2,2),(-2,2),(None,None),(None,None))
# res = minimize(binarymodel, firstguess, method='Nelder-Mead', bounds=bnds, tol=1e-6)
# print res.x

#powell?
# bnds = ((0,1),(0,1),(-2,2),(-2,2),(None,None),(None,None))
# res = minimize(binarymodel, firstguess, method='Powell', bounds=bnds, tol=1e-6)
# print res.x
