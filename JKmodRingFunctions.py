import pylab as pl
import numpy as np
import fileIO
import fitRoutines
from Planck import BB_m
import triangle
import emcee
import oifits
import csv
# from astropy.convolution import Gaussian1DKernel
# from astropy.convolution import Box1DKernel
# from astropy.convolution import convolve
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
# from astropy.io import ascii
from astropy.io import fits
import os
import fnmatch
import math
import gc
import scipy
from astroquery.simbad import Simbad
from shutil import copyfile
from scipy import ndimage
import scipy
import scipy.ndimage as scimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
import matplotlib.cm as cm
import dirtybeam as db

names = {}
names['ACHer'] = 'IRAS18281+2149'
names['AISco'] = 'IRAS17530-3348'
names['ENTrA'] = 'IRAS14524-6838'
names['PSGem'] = 'IRAS07008+1050'
names['HD52961'] = 'IRAS07008+1050'
names['HD93662'] = 'IRAS10456-5712'
names['HD95767'] = 'IRAS11000-6153'
names['HD108015'] = 'IRAS12222-4652'
names['HD213985'] = 'IRAS22327-1731'
names['HR4049'] = 'IRAS10158-2844'
names['IWCar'] = 'IRAS09256-6324'
names['LRSco'] = 'IRAS17243-4348'
names['RSct'] = 'IRAS18448-0545'
names['RUCen'] = 'IRAS12067-4508'
names['SXCen'] = 'IRAS12185-4856'
names['UMon'] = 'IRAS07284-0940'
names['V494Vel'] = ''
names['ARPup'] = 'IRAS08011-3627'
names[''] = ''

modenames = {}
modenames['01'] = 's0'
modenames['02'] = 'sr0'
modenames['03'] = 'sr1'
modenames['04'] = 'sr3'
modenames['05'] = 'br1'
modenames['06'] = 'br2'
modenames['07'] = 'sr4'
modenames['08'] = 'br3'
modenames['09'] = 'sr2'
modenames['10'] = 'br4'
modenames['11'] = 'br5'
modenames['12'] = 's1'
modenames['13'] = 'b1'
modenames['14'] = 'b2'

complexity = ['s0', 's1', 'sr0', 'sr1', 'b1', 'sr2', 'b2', 'sr3', 'sr4', 'br1', 'sr5', 'sr6', 'br2', 'br3', 'br4', 'br5']
complexity2 = ['01', '12', '02', '03',  '13', '09',  '14', '04',  '15',  '05',  '07',  '16',  '06',  '08',  '10',  '11']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


fm = 0.031
allM2s = np.multiply.outer(np.linspace(0.1, 8.0, 3000), np.ones(1000))
allinclinations = np.multiply.outer(np.ones(3000), np.linspace(0.0, 90., 1000))
M1 = 0.6  # 0.75
massfunction = (allM2s*np.sin(allinclinations*np.pi/180.))**3/(M1+allM2s)**2
fittingInd = np.argmin(abs(massfunction-fm), axis=0)

global inclinations
inclinations = allinclinations[0, :]
global fittingM2
fittingM2 = allM2s[fittingInd, 0]

def inform(msg):
    print(bcolors.OKBLUE + msg + bcolors.ENDC)


def inform2(msg):
    print(bcolors.OKGREEN + msg + bcolors.ENDC)


def warn(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def log(msg, dir):
    f = open(dir+"log.txt", "a")
    f.write(msg+"\n")
    f.close()


def WriteREADME(initpars, dir):
    f = open(dir+"log.txt", "a")
    f.write("Models with the following parameters\n")
    for key in initpars:
        f.write(key+"\n")
    f.close()


order = ['got', 'nec', 'nex', 'new', 'oct', 'sep', 'sex', 'qui', 'qua', 'ter', 'bis', '']
order2 = ['mod']

p01 = ['fprim0', 'T']
p02 = ['fprim0', 'T', 'rD', 'rW', 'fbg0']
p03 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA']
p04 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'c1', 's1']
p05 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'fsec0', 'offsetx', 'offsety', 'rM']
p06 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'fsec0', 'offsetx', 'offsety', 'rM', 'c1', 's1']
p07 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'c1', 's1', 'c2', 's2']
p08 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'fsec0', 'offsetx', 'offsety', 'rM', 'c1', 's1', 'c2', 's2']
p09 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'dback']
p10 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'fsec0', 'offsetx', 'offsety', 'rM', 'c1', 's1', 'c2', 's2', 'primD']
p11 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'fsec0', 'offsetx', 'offsety', 'rM', 'c1', 's1', 'c2', 's2', 'primD', 'dsec']
p12 = ['fbg0', 'dback', 'primD']
p13 = ['fbg0', 'dback', 'primD', 'offsetx', 'offsety', 'dsec', 'fsec0']
p14 = ['fbg0', 'dback', 'primD', 'offsetx', 'offsety', 'dsec', 'secD', 'fsec0']
p15 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'offsetx', 'offsety', 'c1', 's1']
p16 = ['fprim0', 'T', 'rD', 'rW', 'fbg0', 'inc', 'PA', 'offsetx', 'offsety', 'c1', 's1', 'c2', 's2']

models = [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16]


def giveDataModelChi2(dirdat, filedat, filemod, name='OUTPUT'):
    data = ReadFilesPionier(dirdat, filedat)
    model = ReadFilesPionier(dirdat, filemod)

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    V2mod, tmp1 = model['v2']
    CPmod, tmp2 = model['cp']
    #for i in np.arange(len(V2data)):
    #    print(V2err[i] - tmp1[i])

    #for i in np.arange(len(CPdata)):
    #    print(CPerr[i] - tmp2[i])


    base, Bmax = Bases(data)

    # compute chi2
    chi2V2 = (V2data - V2mod)**2 / V2err**2
    chi2CP = (CPdata - CPmod)**2 / CPerr**2

    chi2 = (np.sum(chi2V2) + np.sum(chi2CP)) / (len(V2err) + len(CPerr))

    chi2V2 = np.sum(chi2V2) / len(V2err)
    chi2CP = np.sum(chi2CP) / len(CPerr)

    print (chi2, chi2V2, chi2CP)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0


    fig, ((ax11, ax21), (ax12, ax22)) = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=5, label='model', zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=3, label='data')
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0, 100)
    ax11.text(2, 0.05, name)

    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=3, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='white')

    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=5, label='model',zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=3, label='data')
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0, 100)


    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=3, label='model')
    plt.setp(ll, markerfacecolor='white')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    plt.savefig(dirdat + name+ '_ImageVsData.pdf')

    plt.show()
    plt.close()



def Fetchingchi22(star):
    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/'

    fw = open(dir+star+'_BICmmod.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dir+'Data/', star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    for i in [0]:
        BIC0 = 1e10
        prnt = 0
        if i < 10:
            num = 'mfit0'+str(i)
        else:
            num = 'mfit'+str(i)
        for step in order2:
            try:
                f = open(dir+num+step+'/'+star+'_bestfit.txt')
                lines = f.readlines()
                params = lines[0]
                values = lines[1]
                params = params.strip()
                values = values.strip()
                params = params.split()
                values = values.split()
                chi2 = values[0]
                if chi2 == 'nan':
                    chi2 = float(values[1])
                else:
                    chi2 = float(chi2)
                npar = len(models[i-1])
                chi2 *= nTot
                BIC = chi2 + npar * np.log(nTot)
                if BIC < BIC0:
                    BIC0 = BIC
                    chi2_0 = chi2
                    step0 = step
                f.close()
                prnt = 1
            except:
                _a_=1
        if prnt == 1:
            fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
    fw.close()


def Fetchingchi22mod(star):
    dir1 = '/Users/jacques/Work/pAGBPIONIERsurvey/'
    dir2 = '/Users/jacques/Work/pAGBPIONIERsurvey/mod/'

    fw = open(dir2+star+'_BICmod.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dir1+'Data/', star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    for i in [1, 12, 2, 3, 13, 9, 14, 4, 15, 5, 7, 16, 6, 8, 10, 11]:
        if i==2 or i==3 or i==9 or i==4 or i==15 or i==5 or i==7 or i==16 or i==6 or i==8 or i==10 or i==11:
            warn('Ringmodels')
            dir = dir2
        else:
            warn('No Ring')
            dir = dir1
        BIC0 = 1e10
        prnt = 0
        if i < 10:
            num = 'fit0'+str(i)
        else:
            num = 'fit'+str(i)
        for step in order:
            try:
                f = open(dir+num+step+'/'+star+'_bestfit.txt')
                lines = f.readlines()
                params = lines[0]
                values = lines[1]
                params = params.strip()
                values = values.strip()
                params = params.split()
                values = values.split()
                chi2 = values[0]
                if chi2 == 'nan':
                    chi2 = float(values[1])
                else:
                    chi2 = float(chi2)
                npar = len(models[i-1])
                chi2 *= nTot
                BIC = chi2 + npar * np.log(nTot)
                if BIC < BIC0:
                    BIC0 = BIC
                    chi2_0 = chi2
                    step0 = step
                    if not os.path.exists(dir2+'m'+num):
                        os.makedirs(dir2+'m'+num)
                    copyfile(dir+num+step+'/'+star+'_bestfit.txt', dir2+'m'+num+'/'+star+'_bestfit.txt')
                f.close()
                prnt = 1
            except:
                _a_=1
        if prnt == 1:
            fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
    fw.close()


def Fetchingchi2(star):
    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/'

    fw = open(dir+star+'_BICm.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dir+'Data/', star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    for i in range(1, 15):
        BIC0 = 1e10
        prnt = 0
        if i < 10:
            num = 'mfit0'+str(i)
        else:
            num = 'mfit'+str(i)
        for step in order:
            try:
                f = open(dir+num+step+'/'+star+'_bestfit.txt')
                lines = f.readlines()
                params = lines[0]
                values = lines[1]
                params = params.strip()
                values = values.strip()
                params = params.split()
                values = values.split()
                chi2 = values[0]
                if chi2 == 'nan':
                    chi2 = float(values[1])
                else:
                    chi2 = float(chi2)
                npar = len(models[i-1])
                chi2 *= nTot
                BIC = chi2 + npar * np.log(nTot)
                if BIC < BIC0:
                    BIC0 = BIC
                    chi2_0 = chi2
                    step0 = step
                f.close()
                prnt = 1
            except:
                _a_=1
        if prnt == 1:
            try:
                f = open(dir+num+'mod'+'/'+star+'_bestfit.txt')
                lines = f.readlines()
                params = lines[0]
                values = lines[1]
                params = params.strip()
                values = values.strip()
                params = params.split()
                values = values.split()
                chi2 = values[0]
                if chi2 == 'nan':
                    chi2 = float(values[1])
                else:
                    chi2 = float(chi2)
                npar = len(models[i-1])
                chi2 *= nTot
                BIC = chi2 + npar * np.log(nTot)
                BIC0 = BIC
                chi2_0 = chi2
                step0 = step
                f.close()
                fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
            except:
                fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
    fw.close()


def Fetchingchi2finalmod(star):
    dirm1 = '/Users/jacques/Work/pAGBPIONIERsurvey/mod/'
    dirm2 = '/Users/jacques/Work/pAGBPIONIERsurvey/'
    dirg1 = '/Users/jacques/Work/GeneticMod/'
    dirg2 = '/Users/jacques/Work/GeneticAlgoTest/'

    fw = open(dirm1+star+'_BICmod.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dirm1+'../Data/', star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    for i in range(1, 17):
        if i==2 or i==3 or i==9 or i==4 or i==15 or i==5 or i==7 or i==16 or i==6 or i==8 or i==10 or i==11:
            warn('Ringmodels')
            dirm = dirm1
            dirg = dirg1
        else:
            warn('No Ring')
            dirm = dirm2
            dirg = dirg2
        BIC0 = 1e10
        prnt = 0
        if i < 10:
            numm = 'fit0'+str(i)
            numg = 'fit0'+str(i)
        else:
            numm = 'fit'+str(i)
            numg = 'fit'+str(i)
        for step in order:
            try:
                #print(dirm+numm+step+'/'+star+'_bestfit.txt')
                fm = open(dirm+numm+step+'/'+star+'_bestfit.txt')
                #fg = open(dirg+numg+step+'/'+star+'_bestfit.txt')
                mlines = fm.readlines()
                #glines = fg.readlines()
                mparams = mlines[0]
                mvalues = mlines[1]
                #gparams = glines[0]
                #gvalues = glines[1]
                mparams = mparams.strip()
                mvalues = mvalues.strip()
                #gparams = gparams.strip()
                #gvalues = gvalues.strip()
                mparams = mparams.split()
                mvalues = mvalues.split()
                #gparams = gparams.split()
                #gvalues = gvalues.split()
                chi2m = mvalues[0]
                if chi2m == 'nan':
                    chi2m = float(mvalues[1])
                else:
                    chi2m = float(chi2m)
                #chi2g = gvalues[0]
                #if chi2g == 'nan':
                #    chi2g = float(gvalues[1])
                #else:
                #    chi2g = float(chi2g)
                npar = len(models[i-1])
                #chi2g *= nTot
                chi2m *= nTot
                BICm = chi2m + npar * np.log(nTot)
                #BICg = chi2g + npar * np.log(nTot)
                if BICm < BIC0:
                    BIC0 = BICm
                    chi2_0 = chi2m
                    step0 = step
                    num = numm
                #if BICg < BIC0:
                #    BIC0 = BICg
                #    chi2_0 = chi2g
                #    step0 = step
                #    num = numg
                fm.close()
                #fg.close()
                prnt = 1
            except IOError:
                _a_=1
                warn('Error for '+step)
            #except:
            #    warn('Other error...')
        if prnt == 1:
            copyfile(dirm+num+step0+'/'+star+'_bestfit.txt', dirm1+'m'+numm+'/'+star+'_bestfit.txt')
            copyfile(dirm+num+step0+'/'+star+'_quantiles.txt', dirm1+'m'+numm+'/'+star+'_quantiles.txt')
                #f = open(dirm1+'m'+numm+'/'+star+'_bestfit.txt')
                #lines = f.readlines()
                #params = lines[0]
                #values = lines[1]
                #params = params.strip()
                #values = values.strip()
                #params = params.split()
                #values = values.split()
                #chi2 = values[0]
                #if chi2 == 'nan':
                #    chi2 = float(values[1])
                #else:
                #    chi2 = float(chi2)
            npar = len(models[i-1])
                #chi2 *= nTot
                #BIC = chi2 + npar * np.log(nTot)
                #BIC0 = BIC
                #chi2_0 = chi2
                #step0 = step
                #f.close()
            fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
            inform('YEEEEES')
    fw.close()



def Fetchingchi2indiv(dir, star, npar):

    fw = open(dir+'/'+star+'_BICmod.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dir+'/../../Data/', '*'+star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP


    fm = open(dir+'/'+star+'_bestfit.txt')
    try:
        mlines = fm.readlines()
        mparams = mlines[0]
        mvalues = mlines[1]
        mparams = mparams.strip()
        mvalues = mvalues.strip()
        mparams = mparams.split()
        mvalues = mvalues.split()
        chi2m = mvalues[0]
        if chi2m == 'nan':
            chi2m = float(mvalues[1])
        else:
            chi2m = float(chi2m)
        chi2m *= nTot
        BICm = chi2m + npar * np.log(nTot)

        fm.close()
        prnt = 1
    except IOError:
        _a_=1
        warn('Error...Wrong directory?')
    fw.write('{}   {}   {}   {}   {}   {}\n'.format(dir[-14:], BICm, chi2m, npar, nTot, npar+np.log(nTot)))
    inform('YEEEEES')
    fw.close()
    inform('Wrote the following file {}'.format(dir+star+'_BICmod.txt'))




def Fetchingchi2final(star):
    dirm = '/Users/jacques/Work/pAGBPIONIERsurvey/'
    dirg = '/Users/jacques/Work/GeneticAlgoTest/'

    fw = open(dirm+star+'_BICm.txt', 'w')
    fw.write('model    BIC    chi2   npar   ndata   npar*log(ndata)\n')

    data = ReadFilesPionier(dirm+'Data/', star+'*.fits')
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    for i in range(1, 17):
        BIC0 = 1e10
        prnt = 0
        if i < 10:
            numm = 'mfit0'+str(i)
            numg = 'gfit0'+str(i)
        else:
            numm = 'mfit'+str(i)
            numg = 'gfit'+str(i)
        for step in order:
            try:
                fm = open(dirm+numm+step+'/'+star+'_bestfit.txt')
                fg = open(dirg+numg+step+'/'+star+'_bestfit.txt')
                mlines = fm.readlines()
                glines = fg.readlines()
                mparams = mlines[0]
                mvalues = mlines[1]
                gparams = glines[0]
                gvalues = glines[1]
                mparams = mparams.strip()
                mvalues = mvalues.strip()
                gparams = gparams.strip()
                gvalues = gvalues.strip()
                mparams = mparams.split()
                mvalues = mvalues.split()
                gparams = gparams.split()
                gvalues = gvalues.split()
                chi2m = mvalues[0]
                if chi2m == 'nan':
                    chi2m = float(mvalues[1])
                else:
                    chi2m = float(chi2m)
                chi2g = gvalues[0]
                if chi2g == 'nan':
                    chi2g = float(gvalues[1])
                else:
                    chi2g = float(chi2g)
                npar = len(models[i-1])
                chi2g *= nTot
                chi2m *= nTot
                BICm = chi2m + npar * np.log(nTot)
                BICg = chi2g + npar * np.log(nTot)
                if BICm < BIC0:
                    BIC0 = BICm
                    chi2_0 = chi2m
                    step0 = step
                    num = numm
                if BICg < BIC0:
                    BIC0 = BICg
                    chi2_0 = chi2g
                    step0 = step
                    num = numg
                fm.close()
                fg.close()
                prnt = 1
            except IOError:
                _a_=1
            #except:
            #    warn('Other error...')
        if prnt == 1:
            try:
                f = open(dirm+num+'mod'+'/'+star+'_bestfit.txt')
                lines = f.readlines()
                params = lines[0]
                values = lines[1]
                params = params.strip()
                values = values.strip()
                params = params.split()
                values = values.split()
                chi2 = values[0]
                if chi2 == 'nan':
                    chi2 = float(values[1])
                else:
                    chi2 = float(chi2)
                npar = len(models[i-1])
                chi2 *= nTot
                BIC = chi2 + npar * np.log(nTot)
                BIC0 = BIC
                chi2_0 = chi2
                step0 = step
                f.close()
                fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
                inform('YEEEEES')
            except:
                fw.write('{}   {}   {}   {}   {}   {}\n'.format(num+step0, BIC0, chi2_0, npar, nTot, npar+np.log(nTot)))
    fw.close()


def PlotBICfinal(star):
    dirm = '/Users/jacques/Work/pAGBPIONIERsurvey/'
    dirg = '/Users/jacques/Work/GeneticAlgoTest/'
    inform('PlottingBIC for '+star)
    f = open(dirm+star+'_BICm.txt')
    lines = f.readlines()
    nl = len(lines)
    BIC = []
    chi2 = []
    params = lines[0]
    params = params.strip()
    params = params.split()
    models = []
    for k in range(1,nl):
        values = lines[k]
        values = values.strip()
        values = values.split()
        model = values[0]
        zeBIC = float(values[1])
        zechi2 = float(values[2])
        BIC = np.append(BIC, zeBIC)
        models = np.append(models, model)
        chi2 = np.append(chi2, zechi2)

    models2 = []
    chi22 = []
    BIC2 = []
    modelsn = np.array(models.copy())
    for i in range(len(models)):
        modelsn[i] = models[i][4:6]

    for i in range(len(complexity2)):
        id = np.where(modelsn==complexity2[i])
        if len(id[0]) != 0:
            models2 = np.append(models2, complexity[i])
            BIC2 = np.append(BIC2, BIC[id])
            chi22 = np.append(chi22, chi2[id])
        else:
            warn('No {} model found for {}'.format(complexity[i], star))

    #models2 = np.append(models2, modnames[models[i]])

    minBIC = BIC2.min()
    idBIC = BIC2.argmin()

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(models2, BIC2, label='BIC')
    mod = np.arange(len(models2))
    ax.set_xticks(mod)
    ax.set_xticklabels(models2)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('BIC')
    ax.set_xlabel('models')
    # ax.tick_params(axis='y', labelcolor='red')
    plt.yscale('log')
    # ax.set_xlim((300, 350))
    # ax.set_title('How fast do you want to go today?')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('$\chi^2$', color='red')
    ax.plot(models2, chi22, color='red', label='$\chi^2$')
    plt.axvline(x=idBIC, ls='--', color='gray')
    plt.axhline(y=minBIC, ls='--', color='gray')
    # ax2.tick_params(axis='y', labelcolor='red')
    legend = ax.legend(loc='upper right', shadow=True)

    fig.tight_layout()
    # ax2.set_xlabel('r$\chi^2$')
    plt.yscale('log')
    plt.title(star)
    plt.savefig(dirm+star+'_BICm.pdf')
    # plt.show()
    plt.close()


def PlotBICfinalmod(star, idx):
    dirm = '/Users/jacques/Work/pAGBPIONIERsurvey/mod/'
    dirg = '/Users/jacques/Work/GeneticMod/'
    inform('PlottingBIC for '+star)
    f = open(dirm+star+'_BICmod.txt')
    lines = f.readlines()
    nl = len(lines)
    BIC = []
    chi2 = []
    params = lines[0]
    params = params.strip()
    params = params.split()
    models = []
    for k in range(1,nl):
        values = lines[k]
        values = values.strip()
        values = values.split()
        model = values[0]
        zeBIC = float(values[1])
        zechi2 = float(values[2])
        BIC = np.append(BIC, zeBIC)
        models = np.append(models, model)
        chi2 = np.append(chi2, zechi2)

    models2 = []
    chi22 = []
    BIC2 = []
    modelsn = np.array(models.copy())

    for i in range(len(models)):
        modelsn[i] = models[i][3:5]
    for i in range(len(complexity2)):
        id = np.where(modelsn==complexity2[i])
        if len(id[0]) != 0:
            models2 = np.append(models2, complexity[i])
            BIC2 = np.append(BIC2, BIC[id])
            chi22 = np.append(chi22, chi2[id])
        else:
            warn('No {} model found for {}'.format(complexity[i], star))

    #models2 = np.append(models2, modnames[models[i]])

    minBIC = BIC2.min()
    idBIC = BIC2.argmin()

    selBIC = []
    selmodel = []
    selid = []
    selchi22 = []
    for i in range(len(complexity2)):
        if BIC2[i] <= minBIC +10:
            selBIC = np.append(selBIC, BIC2[i])
            selmodel = np.append(selmodel, models2[i])
            selid = np.append(selid, i)

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(models2, BIC2, label='BIC')
    mod = np.arange(len(models2))
    ax.set_xticks(mod)
    ax.set_xticklabels(models2)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('BIC', fontsize=15)
    ax.set_xlabel('models', fontsize=15)
    # ax.tick_params(axis='y', labelcolor='red')
    plt.yscale('log')
    # ax.set_xlim((300, 350))
    # ax.set_title('How fast do you want to go today?')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('$\chi^2$', color='red')
    ax.plot(models2, chi22, color='red', label='$\chi^2$')
    plt.axvline(x=idBIC, ls='--', color='gray')
    plt.axvline(x=selid, ls='--', color='gray')
    plt.axhline(y=minBIC, ls='--', color='gray')
    minBIC2 = np.zeros(len(complexity2)) + minBIC
    ax.fill_between(models, minBIC2, minBIC2+10)
    # ax2.tick_params(axis='y', labelcolor='red')
    legend = ax.legend(loc='upper right', shadow=True)


    # ax2.set_xlabel('r$\chi^2$')
    plt.yscale('log')
    plt.title(star + ' #'+str(idx), fontsize=18)
    fig.tight_layout()
    plt.savefig(dirm+star+'_BICm.pdf')
    # plt.show()
    plt.close()


def PlotBIC(star):
    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/'
    inform('PlottingBIC for '+star)
    f = open(dir+star+'_BICm.txt')
    lines = f.readlines()
    nl = len(lines)
    BIC = []
    chi2 = []
    params = lines[0]
    params = params.strip()
    params = params.split()
    models = []
    for k in range(1,nl):
        values = lines[k]
        values = values.strip()
        values = values.split()
        model = values[0]
        zeBIC = float(values[1])
        zechi2 = float(values[2])
        BIC = np.append(BIC, zeBIC)
        models = np.append(models, model)
        chi2 = np.append(chi2, zechi2)

    minBIC = BIC.min()
    idBIC = BIC.argmin()

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(models, BIC, label='BIC')
    mod = np.arange(len(models))
    ax.set_xticks(mod)
    ax.set_xticklabels(models)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('BIC')
    ax.set_xlabel('models')
    # ax.tick_params(axis='y', labelcolor='red')
    plt.yscale('log')
    # ax.set_xlim((300, 350))
    # ax.set_title('How fast do you want to go today?')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('$\chi^2$', color='red')
    ax.plot(models, chi2, color='red', label='$\chi^2$')
    plt.axvline(x=idBIC, ls='--', color='gray')
    plt.axhline(y=minBIC, ls='--', color='gray')
    # ax2.tick_params(axis='y', labelcolor='red')
    legend = ax.legend(loc='upper right', shadow=True)

    fig.tight_layout()
    # ax2.set_xlabel('r$\chi^2$')
    plt.yscale('log')
    plt.title(star)
    plt.savefig(dir+star+'_BICm.pdf')
    plt.show()
    plt.close()


def PlotBICmod(star):
    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/mod/'
    inform('PlottingBIC for '+star)
    f = open(dir+star+'_BICmod.txt')
    lines = f.readlines()
    nl = len(lines)
    BIC = []
    chi2 = []
    params = lines[0]
    params = params.strip()
    params = params.split()
    models = []
    for k in range(1,nl):
        values = lines[k]
        values = values.strip()
        values = values.split()
        model = values[0]
        zeBIC = float(values[1])
        zechi2 = float(values[2])
        BIC = np.append(BIC, zeBIC)
        models = np.append(models, model)
        chi2 = np.append(chi2, zechi2)

    minBIC = BIC.min()
    idBIC = BIC.argmin()

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(models, BIC, label='BIC')
    mod = np.arange(len(models))
    ax.set_xticks(mod)
    ax.set_xticklabels(models)
    # ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylabel('BIC')
    ax.set_xlabel('models')
    # ax.tick_params(axis='y', labelcolor='red')
    plt.yscale('log')
    # ax.set_xlim((300, 350))
    # ax.set_title('How fast do you want to go today?')
    # ax2 = ax.twinx()
    # ax2.set_ylabel('$\chi^2$', color='red')
    ax.plot(models, chi2, color='red', label='$\chi^2$')
    plt.axvline(x=idBIC, ls='--', color='gray')
    plt.axhline(y=minBIC, ls='--', color='gray')
    # ax2.tick_params(axis='y', labelcolor='red')
    legend = ax.legend(loc='upper right', shadow=True)

    fig.tight_layout()
    # ax2.set_xlabel('r$\chi^2$')
    plt.yscale('log')
    plt.title(star)
    plt.savefig(dir+star+'_BICm.pdf')
    #plt.show()
    plt.close()


def StarSpectrumSED(star):

    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/GalacticDiscs/disks/'
    try:
        hdu = fits.open(dir+star+'.fits')
        table = hdu[2].data
        flux = table.field('flux')
        wave = table.field('wave')*1e-10

#    flux = convolve(flux, Gaussian1DKernel(10, len(wave)) )
        flux = convolve(flux, Box1DKernel(10))

        wavemin = 0 # 1.4e-6
        wavemax = 1e1 # 1.9e-6

        wave0 = []
        flux0 = []

        for i in np.arange(0, len(wave)):
            if (wave[i] > wavemin):
                if (wave[i] < wavemax):
                    wave0 = np.append(wave0, wave[i])
                    flux0 = np.append(flux0, flux[i])
        flux00 = np.interp(1.65e-6, wave, flux)

        flux0 /= flux00

    except:
        warn('No SED fits file found...')
        warn('Assuming a Rayleigh-Jeans tail for the stellar spectrum')
        wave0 = []
        flux0 = []

    return wave0, flux0


def GiveLum(star):
    zestar = names[star]
    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/GalacticDiscs/disks/'
    file = 'OverviewDisks.csv'
    with open(dir+file, 'rb') as csvfile:
        doc = csv.reader(csvfile, delimiter='&')
        i = 0
        for row in doc:
            if i != 0:
                name = row[0]
                name = name[0:-5]
                if name == zestar:
                    R = row[2]
                    Teff = row[1]
                i += 1
            else:
                i += 1

def LDSED(star):

    dir = '/Users/jacques/Work/pAGBPIONIERsurvey/GalacticDiscs/disks/'

    #GiveLum(star)
    try:
        hdu = fits.open(dir+star+'.fits')
    except:
        hdu = fits.open(dir+names[star]+'.fits')
    table = hdu[2].data
    flux = table.field('dered_flux')
    wave = table.field('wave')
    F = scipy.integrate.simps(flux, x=wave)
    return F

def LookIndivEvolution(star):

    inform2('Looking for the evolution of '+star)

    inform('Fetching chi2...')
    Fetchingchi2(star)


def FetchParams(star, dir):
    try:
        f = open(dir+ star + '_bestfit.txt', 'r')
        lines = f.readlines()
        params = lines[0]
        values = lines[1]
        params = params.strip()
        values = values.strip()
        params = params.split()
        values = values.split()
        parameters = {}
        for i in range(0, len(params)):
            parameters[params[i]] = float(values[i])
        return parameters
    except IOError:
        print('cannot open', dir+star+"_bestfit.txt")
    except UnboundLocalError:
        print('No file named', dir+star+"_bestfit.txt")


def CheckParams(inits, pars, scalepars, defaultpars):

    for key in inits:
        if key not in pars.keys():
            pars[key] = defaultpars[key]

    for key in pars.keys():
        if key not in inits:
            del pars[key]
            if key in scalepars.keys():
                del scalepars[key]

    for key in defaultpars.keys():
        if key in inits:
            del defaultpars[key]

    return pars, scalepars


def CheckParams2(inits, pars, scalepars, defaultpars, defaultscale, previouspars):

    for key in inits:
        if key not in pars.keys():
            pars[key] = defaultpars[key]
        if key not in scalepars.keys():
            scalepars[key] = defaultscale[key]

    for key in pars.keys():
        if key not in inits:
            del pars[key]
            if key in scalepars.keys():
                del scalepars[key]
        else:
            if key not in previouspars:
                scalepars[key] = defaultscale[key]

    for key in defaultpars.keys():
        if key in inits:
            del defaultpars[key]

    return pars, scalepars, defaultpars


def ScaleParams(pars, scale):
    pars2 = pars.copy()
    for key, val in pars.iteritems():
        pars2[key] = val*scale

    return pars2


def ScaleParams2(pars, ascale, gscale):
    pars2 = pars.copy()
    for key, val in pars.iteritems():
        pars2[key] = np.max([val*gscale, ascale])

    return pars2


def ReadTargets(file):

    list = ()
    f = open(file, 'r')
    for line in f:
        list = np.append(list, line.strip())

    return list



def ReadFilesPionier(dir, files):

    listOfFiles = os.listdir(dir)
    #print listOfFiles
    #print files
    pattern = files
    i = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            i += 1
            print ('Reading '+entry+'...')
            if i == 1:
                data = read_PIONIER(dir+entry)
            else:
                datatmp = read_PIONIER(dir+entry)
                # Appending all the stuff together
                # Starting with u coordinates
                ut, u1t, u2t, u3t = datatmp['u']
                u, u1, u2, u3 = data['u']
                u = np.append(u, ut)
                u1 = np.append(u1, u1t)
                u2 = np.append(u2, u2t)
                u3 = np.append(u3, u3t)
                data['u'] = (u, u1, u2, u3)
                # v coordinates
                vt, v1t, v2t, v3t = datatmp['v']
                v, v1, v2, v3 = data['v']
                v = np.append(v, vt)
                v1 = np.append(v1, v1t)
                v2 = np.append(v2, v2t)
                v3 = np.append(v3, v3t)
                data['v'] = (v, v1, v2, v3)
                # wavelength tables
                wavet, wavecpt = datatmp['wave']
                wave, wavecp = data['wave']
                wave = np.append(wave, wavet)
                wavecp = np.append(wavecp, wavecpt)
                data['wave'] = (wave, wavecp)
                # Visibility squared
                v2t, v2et = datatmp['v2']
                v2, v2e = data['v2']
                v2 = np.append(v2, v2t)
                v2e = np.append(v2e, v2et)
                data['v2'] = (v2, v2e)
                # closure phases
                cpt, cpet = datatmp['cp']
                cp, cpe = data['cp']
                cp = np.append(cp, cpt)
                cpe = np.append(cpe, cpet)
                data['cp'] = (cp, cpe)

    return data


def read_GRAVITY(file):
    GRAVITY = oifits.open(file)
    GRAVITYv2 = GRAVITY.allvis2
    GRAVITYcp = GRAVITY.allt3

    wave = np.array(GRAVITYv2['eff_wave'])
    u = np.array(GRAVITYv2['ucoord'])/wave
    v = np.array(GRAVITYv2['vcoord'])/wave
    V2 = np.array(GRAVITYv2['vis2data'])
    V2err = np.array(GRAVITYv2['vis2err'])
    wavecp = np.array(GRAVITYcp['eff_wave'])
    u1 = np.array(GRAVITYcp['u1coord'])/wavecp
    v1 = np.array(GRAVITYcp['v1coord'])/wavecp
    u2 = np.array(GRAVITYcp['u2coord'])/wavecp
    v2 = np.array(GRAVITYcp['v2coord'])/wavecp
    u3 = u1 + u2
    v3 = v1 + v2
    CP = np.array(GRAVITYcp['t3phi'])
    CPerr = np.array(GRAVITYcp['t3phierr'])

    # TODO: Temporary fix for 0 errorbars
    CPerr = CPerr + (CPerr == 0)*1e12
    V2err = V2err + (V2err == 0)*1e12

    data = {}
    data['u'] = (u, u1, u2, u3)
    data['v'] = (v, v1, v2, v3)
    data['wave'] = (wave, wavecp)
    data['v2'] = (V2, V2err)
    data['cp'] = (CP, CPerr)

    return data


def readtimebase(file):
    inform2('Opening the following file: '+file)
    dataSC = {}  # dicoinit()
    hdul = fits.open(file)
    err = False
    i = 0
    while err == False:
        i += 1
        try:
            extname = hdul[i].header['EXTNAME']
            print ('Reading '+extname)
            if extname == 'OI_VIS2':
                insname = hdul[i].header['INSNAME']
                uvis, vvis, mjd = [], [], []
                #for j in range(len(hdul[i].data['MJD'])):
                mjd = hdul[i].data['MJD']
                uvis = hdul[i].data['UCOORD']
                vvis = hdul[i].data['VCOORD']
        except IndexError:
            err = True
    time = np.average(mjd)
    bases = np.sqrt(uvis**2 + vvis**2)
    bmax = np.max(bases)
    bmin = np.min(bases)
    nobs = int(len(bases)/6.)
    return time, bmax, bmin, nobs



def plotBaseTime(time, bmax, bmin, nobs, P, target, dir):
    time -= np.min(time)
    fig, ax= plt.subplots()
    plt.scatter(time, bmin)
    plt.scatter(time, bmax)
    for i in np.arange(len(time)):
        plt.plot([time[i],time[i]], [bmin[i], bmax[i]])
        plt.text(time[i], bmin[i]-3, nobs[i], fontsize='x-small')
    #plt.text(0, 10, 'P={}days'.format(P)  )
    plt.plot([0, 0.1*P], [140,140], '--', color='black')
    plt.text(0, 148, '10% of the period'  )
    ax.set_ylim((150, 0))
    plt.title('{} (P={}days)'.format(target, P) )
    ax.set_ylabel('B (m)', fontsize=15)
    ax.set_xlabel('days since first obs.', fontsize=15)
    plt.savefig(dir+target+'_basetime.pdf')
    plt.show()
    plt.close()



def readPIONIERtimebase(dir, files):
    listOfFiles = os.listdir(dir)
    pattern = files
    i = 0
    TIME = []
    BMIN = []
    BMAX = []
    NOBS = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            i += 1
            print ('Reading '+entry+'...')
            if i == 1:
                time, basemax, basemin, nobs = readtimebase(dir+entry)
            else:
                time, basemax, basemin, nobs = readtimebase(dir+entry)
            TIME.append(time)
            BMIN.append(basemin)
            BMAX.append(basemax)
            NOBS.append(nobs)
    return TIME, BMIN, BMAX, NOBS


def read_PIONIER(file):
    PIONIER = oifits.open(file)
    PIONIERv2 = PIONIER.allvis2
    PIONIERcp = PIONIER.allt3

    wave = np.array(PIONIERv2['eff_wave'])
    u = np.array(PIONIERv2['ucoord'])/wave
    v = np.array(PIONIERv2['vcoord'])/wave
    V2 = np.array(PIONIERv2['vis2data'])
    V2err = np.array(PIONIERv2['vis2err'])
    wavecp = np.array(PIONIERcp['eff_wave'])
    u1 = np.array(PIONIERcp['u1coord'])/wavecp
    v1 = np.array(PIONIERcp['v1coord'])/wavecp
    u2 = np.array(PIONIERcp['u2coord'])/wavecp
    v2 = np.array(PIONIERcp['v2coord'])/wavecp
    u3 = u1 + u2
    v3 = v1 + v2
    CP = np.array(PIONIERcp['t3phi'])
    CPerr = np.array(PIONIERcp['t3phierr'])

    # print('the keys:', PIONIERv2.mask)
    # TODO: Temporary fix for 0 errorbar
    CPerr = CPerr + (CPerr == 0)*1e12
    V2err = V2err + (V2err == 0)*1e12

    data = {}
    data['u'] = (u, u1, u2, u3)
    data['v'] = (v, v1, v2, v3)
    data['wave'] = (wave, wavecp)
    data['v2'] = (V2, V2err)
    data['cp'] = (CP, CPerr)

    return data


def Bases(data):

    # gives you back the bases lengths for V2 and cpres
    u = data['u']
    v = data['v']
    wave = data['wave']
    base = np.sqrt(u[0]**2 + v[0]**2)
    B1 = np.sqrt(u[1]**2 + v[1]**2)
    B2 = np.sqrt(u[2]**2 + v[2]**2)
    B3 = np.sqrt(u[3]**2 + v[3]**2)
    Bmax = np.maximum(B1, B2, B3)

    return base, Bmax


def modring_fit_plot(data, pars, dir='./', save=False, name='anonymous', idx=''):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    u = data['u']
    v = data['v']
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Loading the model
    wave0 = np.array([1.65e-6])
    if name != 'anonymous':
        wavespec, starspec = StarSpectrumSED(name)
    else:
        wavespec, starspec = [], []

    V2mod, CPmod = modring_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    fig, ((ax11, ax21), (ax12, ax22)) = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=5, label='model', zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=3, label='data')
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0, 100)
    ax11.text(2, 0.05, name+' #'+str(idx))

    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=3, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='white')

    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=5, label='model',zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=3, label='data')
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0, 100)


    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=3, label='model')
    plt.setp(ll, markerfacecolor='white')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsData.pdf')

    # plt.show()



def modringshift_fit_plot(data, pars, dir='./', save=False, name='anonymous', idx=''):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    u = data['u']
    v = data['v']
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Loading the model
    wave0 = np.array([1.65e-6])
    if name != 'anonymous':
        wavespec, starspec = StarSpectrumSED(name)
    else:
        wavespec, starspec = [], []

    V2mod, CPmod = modringshift_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    fig, ((ax11, ax21), (ax12, ax22)) = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=1, label='model', zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=1, label='data',elinewidth=1)
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0, 100)
    ax11.text(2, 0.05, name+' #'+str(idx))

    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=1, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='white')

    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=1, label='model',zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=1, label='data', elinewidth=1)
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0, 100)

    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=1, label='model')
    plt.setp(ll, markerfacecolor='white')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsData.pdf')

    # plt.show()

def modringshift2_fit_plotsep(data, pars, dir='./', save=False, name='anonymous', idx='', xlog=False):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    u = data['u']
    v = data['v']
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Loading the model
    wave0 = np.array([1.65e-6])
    if name != 'anonymous':
        wavespec, starspec = StarSpectrumSED(name)
    else:
        wavespec, starspec = [], []

    V2mod, CPmod = modringshift2_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    fig, (ax11, ax12) = plt.subplots(nrows=2, ncols = 1, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    #fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=.02, label='data',elinewidth=.02)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=.1, label='model', zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0, 100)
    if xlog==True:
        ax11.set_xlim(min(base)-1, 100)
        ax11.set_xscale('log')
    # ax11.text(60, 0.75, name)
    #    ax11.text(60, 0.75, name+' #'+str(idx))


    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=0.1, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='blue')

    fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsDataV2.pdf')
    plt.close()


    fig, (ax21, ax22) = plt.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    #fig.subplots_adjust(hspace=0., wspace=0.3)

    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=.02, label='data', elinewidth=.02)
    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=.1, label='model',zorder=-32)
    plt.setp(ll, markerfacecolor='blue')
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0, 100)
    if xlog==True:
        ax21.set_xlim(min(Bmax)-1, 100)
        ax21.set_xscale('log')

    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=0.1, label='model')
    plt.setp(ll, markerfacecolor='blue')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsDataCP.pdf')

    # plt.show()



def modringshift2_fit_plot(data, pars, dir='./', save=False, name='anonymous', idx='', xlog=False):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    u = data['u']
    v = data['v']
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Loading the model
    wave0 = np.array([1.65e-6])
    if name != 'anonymous':
        wavespec, starspec = StarSpectrumSED(name)
    else:
        wavespec, starspec = [], []

    V2mod, CPmod = modringshift2_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    fig, ((ax11, ax21), (ax12, ax22)) = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=.02, label='data',elinewidth=.02)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=.1, label='model', zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0, 100)
    if xlog==True:
        ax11.set_xlim(min(base)-1, 100)
        ax11.set_xscale('log')
    # ax11.text(60, 0.75, name)
    #    ax11.text(60, 0.75, name+' #'+str(idx))


    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=0.1, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='white')


    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=.02, label='data', elinewidth=.02)
    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=.1, label='model',zorder=-32)
    plt.setp(ll, markerfacecolor='white')
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0, 100)
    if xlog==True:
        ax21.set_xlim(min(Bmax)-1, 100)
        ax21.set_xscale('log')

    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=0.1, label='model')
    plt.setp(ll, markerfacecolor='white')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsData.pdf')

    # plt.show()


def plot_image_bootstrap(dir, file1, file2, name='TEST', dext=10):
    hdu = fits.open(dir+file1)
    boot = hdu[0].data
    sig = boot.std(0)
    hdu = fits.open(dir+file2)
    imgf = hdu[0].data

    imgf = imgf[:,::-1]
    sig = sig[:,::-1]

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(imgf/np.max(imgf), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[5], linewidths=0.5, colors='white')
    ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootonimage.pdf')
    # plt.show()
    plt.close()


def plot_image_bootstrap_beam(dir, file1, file2, data, name='TEST', dext=10):
    hdu = fits.open(dir+file1)
    boot = hdu[0].data
    sig = boot.std(0)
    hdu = fits.open(dir+file2)
    imgf = hdu[0].data

    imgf = imgf[:,::-1]
    sig = sig[:,::-1]

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(imgf/np.max(imgf), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[5], linewidths=0.5, colors='white')
    ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')

    params = db.giveDirtyBeam(data)
    db.plotBeam(ax, n, psy, params, loc=7.5)

    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootonimage.pdf')
    # plt.show()
    plt.close()


def plot_bootstrap(dir, file, name='TEST', dext=0, method='median'):
    hdu = fits.open(dir+file)
    boot = hdu[0].data
    if method == 'median':
        imgf = np.median(boot,0)
    elif method == 'mean':
        imgf = boot.mean(0)
    else:
        raise Exception('method should be either median or mean')
    sig = boot.std(0)

    imgf = imgf[:,::-1]
    sig = sig[:,::-1]

    shift = [0.5, -0.5]

    imgf = scimage.shift(imgf, shift)
    sig = scimage.shift(sig, shift)

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(imgf/np.max(imgf), extent=[-d, d, d, -d], cmap='gist_heat_r', vmin=0, vmax=1)
    plt.plot(psx/2, psy/2, 'b*')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[2,3], linewidths=0.5, colors='black')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[1], linewidths=0.5, colors='black', linestyles='dotted')
    #ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootimage.pdf')
    # plt.show()
    plt.close()


    asym = imgf-imgf[::-1,::-1]
    asym = asym * (asym>0)
    fig, ax = plt.subplots()
    cs = ax.imshow(asym/np.max(imgf), extent=[-d, d, d, -d], cmap='gist_heat_r', vmin=0, vmax=1)
    plt.plot(psx/2, psy/2, 'b*')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[2], linewidths=0.5, colors='black')
    #CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[1], linewidths=0.5, colors='black', linestyles='dotted')
    #ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootasymimage.pdf')
    # plt.show()
    plt.close()


def plot_bootstrap_beam(dir, file, data, name='TEST', dext=0, method='median'):
    hdu = fits.open(dir+file)
    boot = hdu[0].data
    if method == 'median':
        imgf = np.median(boot,0)
    elif method == 'mean':
        imgf = boot.mean(0)
    else:
        raise Exception('method should be either median or mean')
    sig = boot.std(0)

    imgf = imgf[:,::-1]
    sig = sig[:,::-1]

    shift = [0.5, -0.5]

    imgf = scimage.shift(imgf, shift)
    sig = scimage.shift(sig, shift)

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(imgf/np.max(imgf), extent=[-d, d, d, -d], cmap='gist_heat_r', vmin=0, vmax=1)
    plt.plot(psx/2, psy/2, 'b*')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[2,3], linewidths=0.5, colors='black')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[1], linewidths=0.5, colors='black', linestyles='dotted')
    params = db.giveDirtyBeam(data)
    db.plotBeam(ax, n, psy, params, loc=7.5)
    #ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootimage.pdf')
    # plt.show()
    plt.close()


    asym = imgf-imgf[::-1,::-1]
    asym = asym * (asym>0)
    fig, ax = plt.subplots()
    cs = ax.imshow(asym/np.max(imgf), extent=[-d, d, d, -d], cmap='gist_heat_r', vmin=0, vmax=1)
    plt.plot(psx/2, psy/2, 'b*')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    params = db.giveDirtyBeam(data)
    db.plotBeam(ax, n, psy, params, loc=7.5)
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[2], linewidths=0.5, colors='black')
    #CS = ax.contour(X, -Y, imgf/(sig+1e-20), levels=[1], linewidths=0.5, colors='black', linestyles='dotted')
    #ax.clabel(CS, inline=1, fontsize=7, fmt='%.0f')
    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_bootasymimage.pdf')
    # plt.show()
    plt.close()






def plot_image(dir, file, name='TEST', dext=0):
    hdu = fits.open(dir+file)
    img = hdu[0].data
    img = img[:,::-1]

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(img/np.max(img), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')

    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_image.pdf')
    plt.close()


def plot_image_beam(dir, file, data, name='TEST', dext=0):
    hdu = fits.open(dir+file)
    img = hdu[0].data
    img = img[:,::-1]

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(img/np.max(img), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')

    params = db.giveDirtyBeam(data)
    db.plotBeam(ax, n, psy, params, loc=7.5)

    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_image.pdf')
    plt.close()


def plot_asym(dir, file, name='TEST', dext=0):
    hdu = fits.open(dir+file)
    img = hdu[0].data
    img = img[:,::-1]
    asym = img - img[::-1,::-1]
    asym = asym * (asym>0)

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(asym/np.max(img), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')

    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_asym.pdf')
    plt.close()


def plot_asymboot(dir, file, name='TEST', dext=0, method='median', shift=[0,0]):
    hdu = fits.open(dir+file)
    boot = hdu[0].data
    if method == 'median':
        imgf = np.median(boot,0)
    elif method == 'mean':
        imgf = boot.mean(0)
    else:
        raise Exception('method should be either median or mean')
    sig = boot.std(0)

    imgf = imgf[:,::-1]
    imgf = scimage.shift(imgf, shift)
    asym = imgf - imgf[::-1,::-1]
    asym = asym * (asym>0)

    unit1 = hdu[0].header['CTYPE1']
    if unit1=='milliarcescond':
        psx = hdu[0].header['CDELT1']
        psy = hdu[0].header['CDELT2']
        valx0 = hdu[0].header['CRVAL1']
        valy0 = hdu[0].header['CRVAL2']

    x0 = hdu[0].header['CRPIX1']
    y0 = hdu[0].header['CRPIX2']
    n = hdu[0].header['NAXIS1']
    xr = np.arange(n)+1
    x = valx0 - (xr-x0)*psx
    y = valy0 - (xr-y0)*psy
    d = n*psy/2.

    fs = hdu[0].header['FSVAL']
    de = hdu[0].header['DEVAL']

    X, Y = np.meshgrid(x, y)

## Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow(asym/np.max(imgf), extent=[-d, d, d, -d], cmap='inferno')
    plt.plot(psx/2, psy/2, 'c+')
    plt.axis([d, -d, -d, d])
    if dext!=0:
        plt.axis([dext, -dext, -dext, dext])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')

    cbar = fig.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+name+'_asymboot.pdf')
    plt.close()


def plot_fmap(dir, file, name='TEST', dstar=-4):

    # Loading data
    hdu = fits.open(dir+file)
    data = hdu[0].data
    # fdata = data[0,::-1,:]
    # frgl = data[1,::-1,:]
    f = data[2,::-1,:]

    # Loading coordinates
    dedelta = hdu[0].header['CDELT1']
    fsdelta = hdu[0].header['CDELT2']
    valde0 = hdu[0].header['CRVAL1']+dstar+4
    valfs0 = hdu[0].header['CRVAL2']
    de0 = hdu[0].header['CRPIX1']
    fs0 = hdu[0].header['CRPIX2']
    nd = hdu[0].header['NAXIS1']
    nf = hdu[0].header['NAXIS2']
    der = np.arange(nd)+1
    fsr = np.arange(nf)+1
    de = valde0 + (der-de0) *dedelta
    fs = valfs0 + (fsr-fs0) *fsdelta
    fs = fs[::-1]
    # corners for image plot_chrom
    de0 = de[0] -dedelta/2.
    de1 = de[-1]+dedelta/2.
    fs0 = fs[0] -fsdelta/2.
    fs1 = fs[-1]+fsdelta/2.
    DE, FS = np.meshgrid(de, fs)

    idx = np.argmin(f)
    idx = np.unravel_index(idx, f.shape)

    zede = DE[idx[0],idx[1]]
    zefs = FS[idx[0],idx[1]]

    aspect_ratio = (de0 -de1) / (-fs0 + fs1)
    # P = np.exp(-( f-1.*np.min(f) )/2.)
    # P = P / np.sum(P)

    # plot
    fig, ax = plt.subplots()
    cs = ax.imshow(np.log(f), extent=[de0, de1, fs1, fs0], cmap='inferno', vmax=8.5)
    # cs = ax.imshow(P, extent=[de0, de1, fs0, fs1], cmap='inferno')
    plt.plot(zede, zefs, 'go')
    # plt.axis([d/3, -d/3, -d/3, d/3])
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel('$d_\mathrm{env}$', fontsize=14)
    ax.set_ylabel('$f_\mathrm{prim}$', fontsize=14)
    # CS = ax.contour(-X, -Y, img/(sig+1e-20), levels=[0,3], linewidths=0.5, cmap='gray_r')
    # ax.clabel(CS, inline=1, fontsize=8, fmt='%.0f')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(cs, cax=cax)

    cbar = fig.colorbar(cs, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'$\log f$', fontsize=14)
    # plt.title('fs={0};de={1}'.format(fs, de))
    plt.tight_layout()
    plt.savefig(dir+name+'_chrom.pdf')
    # plt.show()
    plt.close()


def plot_fmaplog(dir, file, name='TEST', dstar=-4):

    # Loading data
    hdu = fits.open(dir+file)
    data = hdu[0].data
    # fdata = data[0,::-1,:]
    # frgl = data[1,::-1,:]
    f = data[2,::-1,:]

    # Loading coordinates
    dedelta = hdu[0].header['CDELT1']
    fsdelta = hdu[0].header['CDELT2']
    valde0 = hdu[0].header['CRVAL1']+dstar+4
    valfs0 = hdu[0].header['CRVAL2']
    de0 = hdu[0].header['CRPIX1']
    fs0 = hdu[0].header['CRPIX2']
    nd = hdu[0].header['NAXIS1']
    nf = hdu[0].header['NAXIS2']
    der = np.arange(nd)+1
    fsr = np.arange(nf)+1
    de = valde0 + (der-de0) *dedelta
    fs = valfs0 + (fsr-fs0) *fsdelta
    fs = fs[::-1]
    # corners for image plot_chrom
    de0 = de[0] -dedelta/2.
    de1 = de[-1]+dedelta/2.
    fs0 = fs[0] -fsdelta/2.
    fs1 = fs[-1]+fsdelta/2.
    DE, FS = np.meshgrid(de, fs)

    idx = np.argmin(f)
    idx = np.unravel_index(idx, f.shape)

    zede = DE[idx[0],idx[1]]
    zefs = FS[idx[0],idx[1]]

    aspect_ratio = (de0 -de1) / (-fs0 + fs1)
    # P = np.exp(-( f-1.*np.min(f) )/2.)
    P = np.exp(-1*(f-1.0420*np.min(f))/2)
    P = P / np.sum(P)


    # plot
    fig, ax = plt.subplots()
    cs = ax.imshow(P, extent=[de0, de1, fs1, fs0], cmap='inferno')
    # cs = ax.imshow(P, extent=[de0, de1, fs0, fs1], cmap='inferno')
    plt.plot(zede, zefs, 'g+')
    # plt.axis([d/3, -d/3, -d/3, d/3])
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel('$d_\mathrm{env}$', fontsize=14)
    ax.set_ylabel('$f_\mathrm{prim}$', fontsize=14)
    # CS = ax.contour(-X, -Y, img/(sig+1e-20), levels=[0,3], linewidths=0.5, cmap='gray_r')
    # ax.clabel(CS, inline=1, fontsize=8, fmt='%.0f')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(cs, cax=cax)

    cbar = fig.colorbar(cs, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'$P$', fontsize=14)
    # plt.title('fs={0};de={1}'.format(fs, de))
    plt.tight_layout()
    plt.savefig(dir+name+'_chrom3.pdf')
    # plt.show()
    plt.close()





from matplotlib.ticker import FuncFormatter

def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.2g' % x

scientific_formatter = FuncFormatter(scientific)


def plot_Lcurve(dir, file, name='TEST'):

    mu = np.logspace(6, 12, num = 61)
    print(mu[::10])
    print(np.log10(mu[::10]))
    fig, ax = plt.subplots()


    hdu = fits.open(dir+file)
    chi2, frgl = hdu[0].data
    #  ax.ticklabel_format(style='sci')
    ax.yaxis.set_major_formatter(scientific_formatter)
    ax.xaxis.set_major_formatter(scientific_formatter)

    ax.plot(chi2, frgl, '-', linewidth=0.2, markersize=0.2)
    colors = cm.jet(np.linspace(0, 1, len(chi2)/10+1))
    for i in np.arange(len(colors)):
        ax.scatter( chi2[i*10], frgl[i*10], c=colors[i], cmap='jet', label='log $\mu$={}'.format(np.log10(mu[10*i]) ) )


    ax.legend(loc='upper right', fontsize=10)
    # plt.colorbar(cs, fraction=0.046, pad=0.04)
    plt.yscale('log')
    plt.xscale('log')

    ax.set_ylabel('$f_\mathrm{rgl}$', fontsize=15)
    ax.set_xlabel('$f_\mathrm{data}$', fontsize=15)

    ax.set_ylim(3e-9, 5e-6)
    ax.set_xlim(4e3, 4.6e3)

    plt.xticks(list(plt.xticks()[0]) + [4.2e3, 4.5e3])
    ax.set_xlim(4e3, 4.6e3)

    plt.tight_layout()
    plt.savefig(dir+name+'_Lcurve.pdf')
    plt.show()
    plt.close()


def plot_fmap2(dir, file, name='TEST', dstar=-4):

    # Loading data
    hdu = fits.open(dir+file)
    data = hdu[0].data
    # fdata = data[0,::-1,:]
    # frgl = data[1,::-1,:]
    f = data[2,::-1,:]

    # Loading coordinates
    dedelta = hdu[0].header['CDELT1']
    fsdelta = hdu[0].header['CDELT2']
    valde0 = hdu[0].header['CRVAL1']+dstar+4
    valfs0 = hdu[0].header['CRVAL2']
    de0 = hdu[0].header['CRPIX1']
    fs0 = hdu[0].header['CRPIX2']
    nd = hdu[0].header['NAXIS1']
    nf = hdu[0].header['NAXIS2']
    der = np.arange(nd)+1
    fsr = np.arange(nf)+1
    de = valde0 + (der-de0) *dedelta
    fs = valfs0 + (fsr-fs0) *fsdelta
    fs = fs[::-1]
    # corners for image plot_chrom
    de0 = de[0] -dedelta/2.
    de1 = de[-1]+dedelta/2.
    fs0 = fs[0] -fsdelta/2.
    fs1 = fs[-1]+fsdelta/2.
    DE, FS = np.meshgrid(de, fs)

    idx = np.argmin(f)
    idx = np.unravel_index(idx, f.shape)

    zede = DE[idx[0],idx[1]]
    zefs = FS[idx[0],idx[1]]

    aspect_ratio = (de0 -de1) / (-fs0 + fs1)
    # P = np.exp(-( f-1.*np.min(f) )/2.)
    # P = P / np.sum(P)

    # plot
    fig, ax = plt.subplots()
    cs = ax.imshow(np.log(f), extent=[de0, de1, fs1, fs0], cmap='inferno', vmax=8.5)
    # cs = ax.imshow(P, extent=[de0, de1, fs0, fs1], cmap='inferno')
    plt.plot(zede, zefs, 'go')
    ax.axhline(y=0.28, color='cyan')
    plt.plot(0.85+dstar+4, 0.28, 'co')
    # plt.axis([d/3, -d/3, -d/3, d/3])
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel('$d_\mathrm{env}$', fontsize=15)
    ax.set_ylabel('$f^*_\mathrm{0}$', fontsize=15)
    # CS = ax.contour(-X, -Y, img/(sig+1e-20), levels=[0,3], linewidths=0.5, cmap='gray_r')
    # ax.clabel(CS, inline=1, fontsize=8, fmt='%.0f')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(cs, cax=cax)

    cbar = fig.colorbar(cs, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'$\log f$', fontsize=15)
    # plt.title('fs={0};de={1}'.format(fs, de))
    plt.tight_layout()
    plt.savefig(dir+name+'_chrom2.pdf')
    # plt.show()
    plt.close()



def data_plot_PIONIER(data, dir='./', save=False, name='Data.pdf', Blim=100, CPext=200, V2min=0.0, V2max=1.0, lines=False, xlog=False, ylog=False):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6
    waveCP = wave[1]
    waveV2 = wave[0]
    waveV2 *= 1e6
    waveCP *= 1e6

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Actual plot
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    sc = ax1.scatter(base[maskv2], V2data[maskv2], s=0.1, c=waveV2[maskv2], cmap='gist_rainbow_r')
    clb = fig.colorbar(sc, cax=cax)
    clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad=15)
    a, b, c = ax1.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2],marker='',elinewidth=0.05, ls='', zorder=0)
    color = clb.to_rgba(waveV2[maskv2])
    c[0].set_color(color)
    ax1.axhline(0)
    ax1.set_ylim(V2min, V2max)
    ax1.set_xlim(0, Blim)
    if xlog==True:
        ax1.set_xlim(min(base)-1, Blim)
        ax1.set_xscale('log')
    if ylog==True:
        ax1.set_ylim(0.0001, 1)
        ax1.set_yscale('log')

    ax1.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    ax1.set_title('Squared visibilities')
    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    ax2.axhline(y=0, ls='--', c='grey', lw=0.3)
    sc = ax2.scatter(Bmax[maskcp], CPdata[maskcp], s=0.1, c=waveCP[maskcp], cmap='gist_rainbow_r')
    ax2.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    ax2.set_title('Closure phases')
    ax2.set_xlim(0,Blim)
    ax2.set_ylim(-CPext, CPext)
    if xlog==True:
        ax2.set_xlim(min(Bmax)-1, Blim)
        ax2.set_xscale('log')

    #ax2.text(5, 22, name, fontsize=12)
    a2, b2, c2 = ax2.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp],elinewidth=0.05, marker='', ls='', zorder=0)
    colorCP = clb.to_rgba(waveCP[maskcp])
    c2[0].set_color(colorCP)

    if save:
        plt.savefig(dir + name + '_Data.pdf')
        #plt.savefig(dir + name[:-5] + '_Data.pdf')

    # plt.show()

    u, u1, u2, u3 = data['u']
    v, v1, v2, v3 = data['v']
    u*=1e-6
    v*=1e-6

    # Actual plot
    fig, ax = plt.subplots(1, 1)
    cax = fig.add_axes(ax)
    sc = ax.scatter([u, -u], [v,-v], c=[waveV2,waveV2], s=0.1, cmap='gist_rainbow_r')
    #sc = ax.scatter(-u, -v, c=waveV2, s=0.5, cmap='gist_rainbow_r')
    clb = fig.colorbar(sc)
    clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad=15)
    ax.set_xlabel('u (M$\lambda$)', fontsize=8)
    ax.set_ylabel('v (M$\lambda$)', fontsize=8)
    ax.set_title('uv plane')
    ax.set_ylim(-Blim, Blim)
    ax.set_xlim(Blim, -Blim)
    fig.tight_layout()
    # ax.text(0, 80, name, fontsize=12)
    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    if save:
        #plt.savefig(dir + name[:-8] +'_Datauv.pdf')
        plt.savefig(dir + name +'_Datauv.pdf')

    #plt.show()
    plt.close()


def data_plot_PIONIERsep(data, dir='./', save=False, name='Data.pdf', Blim=100, CPext=200, V2min=0.0, V2max=1.0, lines=False, xlog=False):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6
    waveCP = wave[1]
    waveV2 = wave[0]
    waveV2 *= 1e6
    waveCP *= 1e6

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Actual plot
    fig, ax = plt.subplots()

    #fig.subplots_adjust(right=0.8)
    #cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    sc = ax.scatter(base[maskv2], V2data[maskv2], s=0.1, c=waveV2[maskv2], cmap='gist_rainbow_r')
    clb = fig.colorbar(sc)
    clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad = 20, fontsize=14)
    a, b, c = ax.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2],marker='',elinewidth=0.05, ls='', zorder=0)
    color = clb.to_rgba(waveV2[maskv2])
    c[0].set_color(color)
    ax.axhline(0)
    ax.set_ylim(V2min, V2max)
    ax.set_xlim(0, Blim)
    if xlog==True:
        ax.set_xlim(min(base)-1, Blim)
        ax.set_xscale('log')

    ax.set_xlabel(r'B (M$\lambda$)', fontsize=14)
    ax.set_title('Squared visibilities',fontsize=16)
    fig.tight_layout()

    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')
    if save:
        plt.savefig(dir + name + '_DataV2.pdf')

    plt.close()


    fig, ax = plt.subplots()

    ax.axhline(y=0, ls='--', c='grey', lw=0.3)
    sc = ax.scatter(Bmax[maskcp], CPdata[maskcp], s=0.1, c=waveCP[maskcp], cmap='gist_rainbow_r')
    ax.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=14)
    ax.set_title('Closure phases', fontsize=16)
    ax.set_xlim(0,Blim)
    ax.set_ylim(-CPext, CPext)
    if xlog==True:
        ax.set_xlim(min(Bmax)-1, Blim)
        ax.set_xscale('log')

    #ax2.text(5, 22, name, fontsize=12)
    a2, b2, c2 = ax.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp],elinewidth=0.05, marker='', ls='', zorder=0)
    clb = fig.colorbar(sc)
    colorCP = clb.to_rgba(waveCP[maskcp])
    clb.set_label(r'Wavelength ($\mu$m)', rotation=270, fontsize=14, labelpad = 20)
    c2[0].set_color(colorCP)
    fig.tight_layout()

    if save:
        plt.savefig(dir + name + '_DataCP.pdf')
        #plt.savefig(dir + name[:-5] + '_Data.pdf')

    plt.close()

    u, u1, u2, u3 = data['u']
    v, v1, v2, v3 = data['v']
    u*=1e-6
    v*=1e-6

    # Actual plot
    fig, ax = plt.subplots(1, 1)
    cax = fig.add_axes(ax)
    sc = ax.scatter([u, -u], [v,-v], c=[waveV2,waveV2], s=0.1, cmap='gist_rainbow_r')
    clb = fig.colorbar(sc)
    color = clb.to_rgba(waveV2[maskv2])
    clb.set_label(r'Wavelength ($\mu$m)', rotation=270, labelpad = 20, fontsize=14)
    #sc = ax.scatter(-u, -v, c=waveV2, s=0.5, cmap='gist_rainbow_r')
    ax.set_xlabel('u (M$\lambda$)', fontsize=14)
    ax.set_ylabel('v (M$\lambda$)', fontsize=14)
    ax.set_title('(u,v)-plane', fontsize=16)
    ax.set_ylim(-Blim, Blim)
    ax.set_xlim(Blim, -Blim)
    fig.tight_layout()
    # ax.text(0, 80, name, fontsize=12)
    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    if save:
        #plt.savefig(dir + name[:-8] +'_Datauv.pdf')
        plt.savefig(dir + name +'_Datauv.pdf')

    #plt.show()
    plt.close()


def data_plot_SAMPIO(dataPIO, dataSAM, dir='./', save=False, name='DataSAMPIO.pdf', Blim=100, CPext=200, V2min=0.0, V2max=1.0, lines=False, xlog=True):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset PIO
    waveP = dataPIO['wave']
    V2dataP, V2errP = dataPIO['v2']
    CPdataP, CPerrP = dataPIO['cp']
    baseP, BmaxP = Bases(dataPIO)

    # Loading the dataset SAM
    waveS = dataSAM['wave']
    V2dataS, V2errS = dataSAM['v2']
    CPdataS, CPerrS = dataSAM['cp']
    baseS, BmaxS = Bases(dataSAM)

    # Setting things for the plot
    baseP *= 1e-6
    BmaxP *= 1e-6
    waveCPP = waveP[1]
    waveV2P = waveP[0]
    waveV2P *= 1e6
    waveCPP *= 1e6

    maskv2P = V2errP < 0.5
    maskcpP = CPerrP < 2.0

    # Setting things for the plot
    baseS *= 1e-6
    BmaxS *= 1e-6
    waveCPS = waveS[1]
    waveV2S = waveS[0]
    waveV2S *= 1e6
    waveCPS *= 1e6

    maskv2S = V2errS < 0.5
    maskcpS = CPerrS < 2.0

    # Actual plot
    fig, (ax1, ax2) = plt.subplots(1, 2)

    #fig.subplots_adjust(right=0.8)
    #cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    ax1.scatter(baseP[maskv2P], V2dataP[maskv2P], s=0.1, c='blue', label='VLTI/PIONIER (actual)')
    ax1.errorbar(baseP[maskv2P], V2dataP[maskv2P], yerr=V2errP[maskv2P], marker='',elinewidth=0.2, ls='', zorder=0, ecolor='blue')

    ax1.scatter(baseS[maskv2S], V2dataS[maskv2S], s=0.1, c='red', label='SPHERE/SAM (simulated)')
    ax1.errorbar(baseS[maskv2S], V2dataS[maskv2S], yerr=V2errS[maskv2S],marker='',elinewidth=0.2, ls='', zorder=0, ecolor='red')

    ax1.axhline(0)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(0.5, Blim)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    ax1.set_title('Squared visibilities')
    legend1 = ax1.legend(loc='lower left', fontsize='small')

    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    ax2.axhline(y=0, ls='--', c='grey', lw=0.3)
    ax2.scatter(BmaxP[maskcpP], CPdataP[maskcpP], s=0.2,  c='blue')
    ax2.scatter(BmaxS[maskcpS], CPdataS[maskcpS], s=0.2,  c='red')
    ax2.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    ax2.set_title('Closure phases')
    ax2.set_xlim(0.5,Blim)
    ax2.set_xscale('log')

    ax2.set_ylim(-CPext, CPext)
    #ax2.text(5, 22, name, fontsize=12)
    ax2.errorbar(BmaxP[maskcpP], CPdataP[maskcpP], yerr=CPerrP[maskcpP], elinewidth=0.2, marker='', ls='', zorder=0, ecolor='blue')
    ax2.errorbar(BmaxS[maskcpS], CPdataS[maskcpS], yerr=CPerrS[maskcpS], elinewidth=0.2, marker='', ls='', zorder=0, ecolor='red')

    if save:
        plt.savefig(dir + name + '_DataSAMPIO.pdf')
        #plt.savefig(dir + name[:-5] + '_Data.pdf')

    # plt.show()

    uP, u1P, u2P, u3P = dataPIO['u']
    vP, v1P, v2P, v3P = dataPIO['v']
    uP*=1e-6
    vP*=1e-6

    uS, u1S, u2S, u3S = dataSAM['u']
    vS, v1S, v2S, v3S = dataSAM['v']
    uS*=1e-6
    vS*=1e-6

    # Actual plot
    fig, ax = plt.subplots(1, 1)
    cax = fig.add_axes(ax)
    ax.scatter([uP, -uP], [vP,-vP], s=0.5, color='blue')
    ax.scatter([uS, -uS], [vS,-vS], s=0.5, color='red')

    #sc = ax.scatter(-u, -v, c=waveV2, s=0.5, cmap='gist_rainbow_r')
    ax.set_xlabel('u (M$\lambda$)', fontsize=8)
    ax.set_ylabel('v (M$\lambda$)', fontsize=8)
    ax.set_title('uv plane')
    ax.set_ylim(-Blim, Blim)
    ax.set_xlim(Blim, -Blim)
    # ax.text(0, 80, name, fontsize=12)
    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    if save:
        #plt.savefig(dir + name[:-8] +'_Datauv.pdf')
        plt.savefig(dir + name +'_DatauvSAMPIO.pdf')

    #plt.show()
    plt.close()




def data_plot_PIONIER_vis2d(data, dir='./', save=False, name='DataVis2D.pdf', Blim=100, V2min=0.0, V2max=1.0, lines=False):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6
    waveV2 = wave[0]
    waveV2 *= 1e6

    u, u1, u2, u3 = data['u']
    v, v1, v2, v3 = data['v']
    u*=1e-6
    v*=1e-6

    maskv2 = V2err < 0.5

    V2data2 = V2data[maskv2]
    V2err2 = V2err[maskv2]
    u2 = u[maskv2]
    v2 = v[maskv2]
    waveV22 = waveV2[maskv2]

    maskv2 = waveV22 > 1.75

    V2data3 = V2data2[maskv2]
    V2err3 = V2err2[maskv2]
    u3 = u2[maskv2]
    v3 = v2[maskv2]

    weight = 1/V2err3**2

    weight /= max(weight)
    weight *= 100


    # Actual plot
    fig, ax = plt.subplots(1, 1)
    cax = fig.add_axes(ax)
    sc = ax.scatter(u3, v3, c=V2data3, s=weight, cmap='gist_rainbow_r')
    sc = ax.scatter(-u3, -v3, c=V2data3, s=weight, cmap='gist_rainbow_r')
    clb = fig.colorbar(sc)
    clb.set_label(r'V2', rotation=270, labelpad=15)
    ax.set_xlabel('u', fontsize=8)
    ax.set_ylabel('v', fontsize=8)
    ax.set_title('uv plane')
    ax.set_ylim(-Blim, Blim)
    ax.set_xlim(Blim, -Blim)
    # ax.text(0, 80, name, fontsize=12)
    # ax2.scatter(Bmax, CPdata, c=color, cmap='gist_rainbow_r')

    if save:
        #plt.savefig(dir + name[:-8] +'_Datauv.pdf')
        plt.savefig(dir + name +'_Data_vis2D.pdf')

    plt.show()




def data_plot_PIONIER3(data, dir='./', save=False, name='test'):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6
    waveCP = wave[1]
    waveV2 = wave[0]
    waveV2 *= 1e6
    waveCP *= 1e6

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    V2data = V2data[maskv2]
    base = base[maskv2]
    waveV2 = waveV2[maskv2]

    CPdata = CPdata[maskcp]
    Bmax = Bmax[maskcp]
    waveCP = waveCP[maskcp]

    nb = 0
    for k in range(len(base)):
        if k == 0:
            base0 = base[k]*waveV2[k]
        else:
            if abs(base[k]*waveV2[k] - base0) > 1e-20:
                nb += 1
                base0 = base[k]*waveV2[k]

    noft = nb/2
    nb /= 2
    if nb / 3. == int(nb/3.):
        ncols = 3
        nrows = nb / 3
    else:
        ncols = 3
        nrows = nb / 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    nb = 0
    for k in range(len(base)):
        if k == 0:
            base0 = base[k]*waveV2[k]
            zebase = base[k]
            zeV2 = V2data[k]
            zewav = waveV2[k]
        else:
            if abs(base[k]*waveV2[k] == base0) < 1e-20:
                zebase = np.append(zebase, base[k])
                zeV2 = np.append(zeV2, V2data[k])
                zewav = np.append(zewav, waveV2[k])
            else:
                if nb > noft:
                    ax = axes.flat[nb-noft-1]
                    ax.plot( zewav, zeV2)
                    ax.set_ylim(0, 0.5)
                    #ax.set_xlim(0, 100)
                nb += 1
                base0 = base[k]*waveV2[k]
                zebase = base[k]
                zeV2 = V2data[k]
                zewav = waveV2[k]

    plt.tight_layout()
    if save:
        fig.savefig(dir + name +'_V2')
    # plt.show()

    plt.show()

    plt.close()

    nb = 0
    for k in range(len(Bmax)):
        if k == 0:
            base0 = Bmax[k]*waveCP[k]
        else:
            if abs(Bmax[k]*waveCP[k] - base0) > 1e-20 :
                nb += 1
                base0 = Bmax[k]*waveCP[k]

    noft = nb/2
    nb /= 2
    if nb / 2. == int(nb/2.):
        ncols = 2
        nrows = nb / 2
    else:
        ncols = 2
        nrows = nb / 2 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    nb = 0
    for k in range(len(Bmax)):
        if k == 0:
            base0 = Bmax[k]*waveCP[k]
            zebase = Bmax[k]
            zeCP = CPdata[k]
            zewav = waveCP[k]
        else:
            if abs(Bmax[k]*waveCP[k] - base0) < 1e-20 :
                zebase = np.append(zebase, Bmax[k])
                zeCP = np.append(zeCP, CPdata[k])
                zewav = np.append(zewav, waveCP[k])
            else:
                if nb > noft:
                    ax = axes.flat[nb-noft-1]
                    ax.plot( zewav, zeCP)
                    #ax.set_ylim(0, 0.5)
                nb += 1
                base0 = Bmax[k]*waveCP[k]
                zebase = Bmax[k]
                zeCP = CPdata[k]
                zewav = waveCP[k]
    plt.tight_layout()
    if save:
        plt.savefig(dir + name +'_CP')

    plt.show()

    plt.close()








def modbinbg_fit_plot(data, pars, dir='./', save=False, name='anonymous', idx=''):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    u = data['u']
    v = data['v']
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    # Loading the model
    wave0 = np.array([1.65e-6])
    if name != 'anonymous':
        wavespec, starspec = StarSpectrumSED(name)
    else:
        wavespec, starspec = [], []

    V2mod, CPmod = modbinbg_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # Computing the Residuals
    resV2 = V2mod - V2data
    resV2 /= V2err
    resCP = CPmod - CPdata
    resCP /= CPerr

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6

    fig, ((ax11, ax21), (ax12, ax22)) = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw = {'height_ratios':[6, 1]})
    fig.subplots_adjust(hspace=0., wspace=0.3)

    # fig, (vis, cp) = plt.subplot(121)
    ll = ax11.plot(base[maskv2], V2mod[maskv2], 'o', c='blue', markersize=5, label='model')
    plt.setp(ll, markerfacecolor='white')
    ax11.errorbar(base[maskv2], V2data[maskv2], yerr=V2err[maskv2], fmt='o', color='black', markersize=3, label='data')
    ax11.set_title('Squared visibilities')
    ax11.set_ylim((0, 1))
    legend1 = ax11.legend(loc='upper right', fontsize='small')
    ax11.set_yticks(np.arange(0, 1.1, 0.2))
    ax11.set_xlim(0)
    ax11.text(2,0.05,name+' #'+str(idx))

    ax12.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax12.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax12.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax12.plot(base[maskv2], resV2[maskv2], 'o', c='blue', markersize=3, label='model')
    # ax12.set_yticks(np.arange(-9, 9.1, 3))
    ax12.set_ylim(-9, 9)
    ax12.set_ylabel(r'Error ($\sigma_{V^2}$)', fontsize=8)
    ax12.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    plt.setp(ll, markerfacecolor='white')

    ll = ax21.plot(Bmax[maskcp], CPmod[maskcp], 'o', c='blue', markersize=5, label='model')
    plt.setp(ll, markerfacecolor='white')
    ax21.errorbar(Bmax[maskcp], CPdata[maskcp], yerr=CPerr[maskcp], fmt='o', color='black', markersize=3, label='data')
    ax21.set_title('Closure phases')
    legend2 = ax21.legend(loc='upper left', fontsize='small')
    # ax21.set_yticks(np.arange(-25, 25, 5))
    ax21.set_ylim(-30, 30)
    ax21.set_xlim(0)

    ax22.axhline(0,  linestyle='-', linewidth=0.8, color='black')
    ax22.axhline(-5,  linestyle='--', linewidth=0.8, color='black')
    ax22.axhline(5,  linestyle='--', linewidth=0.8, color='black')
    ll = ax22.plot(Bmax[maskcp], resCP[maskcp], 'o', c='blue', markersize=3, label='model')
    plt.setp(ll, markerfacecolor='white')
    # ax22.set_yticks(np.arange(-10, 10, 2))
    ax22.set_ylim(-9, 9)
    ax22.set_ylabel(r'Error ($\sigma_\mathrm{CP}$)', fontsize=8)
    ax22.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    # ax22.set_yticks(np.arange(-9, 9.1, 3))
    # legend.get_frame().set_facecolor('#00FFCC')

    # fig.tight_layout()
    if save:
        plt.savefig(dir + name+ '_ModelVsData.pdf')


def data_plot_PIONIER2(data, dir='./', save=False, name='test'):

    # Plot the vis2 and cp from the data and the model

    # Loading the dataset
    wave = data['wave']
    V2data, V2err = data['v2']
    CPdata, CPerr = data['cp']
    base, Bmax = Bases(data)

    # Setting things for the plot
    base *= 1e-6
    Bmax *= 1e-6
    waveCP = wave[1]
    waveV2 = wave[0]
    waveV2 *= 1e6
    waveCP *= 1e6

    maskv2 = V2err < 0.5
    maskcp = CPerr < 10.0

    V2data = V2data[maskv2]
    base = base[maskv2]
    waveV2 = waveV2[maskv2]

    CPdata = CPdata[maskcp]
    Bmax = Bmax[maskcp]
    waveCP = waveCP[maskcp]

    # Actual plot
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    nb = 0
    for k in range(len(base)):
        if k == 0:
            base0 = base[k]*waveV2[k]
            zebase = base[k]
            zeV2 = V2data[k]
        else:
            if abs(base[k]*waveV2[k] - base0) < 1e-10:
                zebase = np.append(zebase, base[k])
                zeV2 = np.append(zeV2, V2data[k])
            else:
                nb += 1
                ax1.plot(zebase, zeV2)
                base0 = base[k]*waveV2[k]
                zebase = base[k]
                zeV2 = V2data[k]

    noft = nb/2
    nb /= 2
    if nb / 3. == int(nb/3.):
        ncols = 3
        nrows = nb / 3
    else:
        ncols = 3
        nrows = nb / 3 + 1

    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 4.5))

    nb = 0
    for k in range(len(base)):
        if k == 0:
            base0 = base[k]*waveV2[k]
            zebase = base[k]
            zeV2 = V2data[k]
            zewav = waveV2[k]
        else:
            if abs(base[k]*waveV2[k] - base0) < 1e-10:
                zebase = np.append(zebase, base[k])
                zeV2 = np.append(zeV2, V2data[k])
                zewav = np.append(zewav, waveV2[k])
            else:
                if nb > noft:
                    ax = axes.flat[nb-noft-1]
                    ax.plot( zewav, zeV2)
                    ax.set_ylim(0, 0.5)
                    #ax.set_xlim(0, 100)
                nb += 1
                base0 = base[k]*waveV2[k]
                zebase = base[k]
                zeV2 = V2data[k]
                zewav = waveV2[k]

    plt.tight_layout()
    if save:
        fig2.savefig(dir + name +'_V2')
    # plt.show()

    ax1.set_ylim(0, 1)
    ax1.set_xlim(0)
    ax1.set_xlabel(r'B (M$\lambda$)', fontsize=8)
    ax1.set_title('Squared visibilities')

    nb = 0
    for k in range(len(Bmax)):
        if k == 0:
            base0 = Bmax[k]*waveCP[k]
            zebase = Bmax[k]
            zeCP = CPdata[k]
        else:
            if abs(Bmax[k]*waveCP[k] - base0) < 1e-10 :
                zebase = np.append(zebase, Bmax[k])
                zeCP = np.append(zeCP, CPdata[k])
            else:
                nb += 1
                ax2.plot(zebase, zeCP)
                base0 = Bmax[k]*waveCP[k]
                zebase = Bmax[k]
                zeCP = CPdata[k]


    noft = nb/2
    nb /= 2
    if nb / 3. == int(nb/3.):
        ncols = 3
        nrows = nb / 3
    else:
        ncols = 3
        nrows = nb / 3 + 1

    fig3, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 4.5))

    nb = 0
    for k in range(len(Bmax)):
        if k == 0:
            base0 = Bmax[k]*waveCP[k]
            zebase = Bmax[k]
            zeCP = CPdata[k]
            zewav = waveCP[k]
        else:
            if abs(Bmax[k]*waveCP[k] - base0) < 1e-10 :
                zebase = np.append(zebase, Bmax[k])
                zeCP = np.append(zeCP, CPdata[k])
                zewav = np.append(zewav, waveCP[k])
            else:
                if nb > noft-1:
                    ax = axes2.flat[nb-noft-1]
                    ax.plot( zewav, zeCP)
                    #ax.set_ylim(0, 0.5)
                nb += 1
                ax2.plot(zewav, zeCP)
                base0 = Bmax[k]*waveCP[k]
                zebase = Bmax[k]
                zeCP = CPdata[k]
                zewav = waveCP[k]

    ax2.set_xlabel(r'B$_\mathrm{max}$ (M$\lambda$)', fontsize=8)
    ax2.set_title('Closure phases')
    ax2.set_xlim(0)
    ax2.set_ylim(-100, 100)

    if save:
        fig3.savefig(dir + name +'_CP')

    if save:
        fig.savefig(dir + name +'_data')

    plt.show()


def xycoord(n=256, ps=0.1):
    # Creates arrays of coordinates for an image
    fov = n * ps / 2.
    x = y = np.arange(-fov, fov, ps)
    X, Y = np.meshgrid(x, y)

    return (X, Y)


def modringImage(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring0 = 1.-fprim0-fsec0-fbg0
    rM = pars['rM']
    xprim = pars['offsetx']
    yprim = pars['offsety']
    rD = pars['rD']
    rW = pars['rW']
    c1 = pars['c1']
    s1 = pars['s1']
    c2 = pars['c2']
    s2 = pars['s2']
    c3 = pars['c3']
    s3 = pars['s3']
    PA = pars['PA']
    inc = pars['inc']
    primD = pars['primD']
    secD = pars['secD']

    xsec = -1.0 * rM * xprim
    ysec = -1.0 * rM * yprim

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = xprim * np.cos(PA*np.pi/180.) - yprim * np.sin(PA*np.pi/180.)
    yprim2 = xprim * np.sin(PA*np.pi/180.) + yprim * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    theta = np.pi / 2. - PA * np.pi / 180.
    inc = np.pi * inc / 180.

    x2 = x * np.cos(theta) + y * np.sin(theta)
    y2 = y * np.cos(theta) - x * np.sin(theta)
    y2 /= np.cos(inc)
    alpha = np.arctan2(y2, x2)
    width = rD * rW /2.
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    ring = 1. / (sigma * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD/2.)**2 / (2 * sigma**2) )
    ring *= 1 - c1 * np.cos(alpha) - s1 * np.sin(alpha) + c2 * np.cos(2*alpha) + s2 * np.sin(2*alpha) - c3 * np.cos(3*alpha) + s3 * np.sin(3*alpha)

    img = fring0 * ring + fbg0 * np.ones((n, n)) / (n*n)
    img /= np.max(img)

    # Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='gist_heat', vmin=0)

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        #try:
        if ( sepx**2 / (rD/3.*np.cos(inc))**2 + sepy**2 / (rD/3.)**2 < 1 ):
            circle = plt.Circle((xprim, yprim), primD/2., color='blue', fill=True)
        else:
            circle = plt.Circle((xprim, yprim), primD/2., color='cyan', fill=True)
        ax.add_artist(circle)
        #except:
        #    plt.plot(xprim, yprim, 'r+')
    if ( (fsec0 > 0) & (fsec0 <= 1)  ):
        #try:
        if ( sepx**2 / (rD/3.*np.cos(inc))**2 + sepy**2 / (rD/3.)**2 < 1 ):
            circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
        else:
            circle2 = plt.Circle((xsec, ysec), secD/2., color='cyan', fill=True)
        ax.add_artist(circle2)
        #except:
        #    plt.plot(xsec, ysec, 'g+')

    plt.axis([d, -d, -d, d])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    cbar = fig.colorbar(cs)
    plt.title(name)
    fig.tight_layout()

    if save:
        plt.savefig(dir + name + '_imageFINAL.pdf')

    #plt.show()

    plt.close()


def modringImage(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring0 = 1.-fprim0-fsec0-fbg0
    rM = pars['rM']
    xprim = pars['offsetx']
    yprim = pars['offsety']
    rD = pars['rD']
    rW = pars['rW']
    c1 = pars['c1']
    s1 = pars['s1']
    c2 = pars['c2']
    s2 = pars['s2']
    c3 = pars['c3']
    s3 = pars['s3']
    PA = pars['PA']
    inc = pars['inc']
    primD = pars['primD']
    secD = pars['secD']

    xsec = -1.0 * rM * xprim
    ysec = -1.0 * rM * yprim

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = xprim * np.cos(PA*np.pi/180.) - yprim * np.sin(PA*np.pi/180.)
    yprim2 = xprim * np.sin(PA*np.pi/180.) + yprim * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    theta = np.pi / 2. - PA * np.pi / 180.
    inc = np.pi * inc / 180.

    x2 = x * np.cos(theta) + y * np.sin(theta)
    y2 = y * np.cos(theta) - x * np.sin(theta)
    y2 /= np.cos(inc)
    alpha = np.arctan2(y2, x2)
    width = rD * rW /2.
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    ring = 1. / (sigma * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD/2.)**2 / (2 * sigma**2) )
    ring *= 1 - c1 * np.cos(alpha) - s1 * np.sin(alpha) + c2 * np.cos(2*alpha) + s2 * np.sin(2*alpha) - c3 * np.cos(3*alpha) + s3 * np.sin(3*alpha)

    img = fring0 * ring + fbg0 * np.ones((n, n)) / (n*n)
    img /= np.max(img)

    # Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='gist_heat', vmin=0)

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        #try:
        if ( sepx**2 / (rD/3.*np.cos(inc))**2 + sepy**2 / (rD/3.)**2 < 1 ):
            circle = plt.Circle((xprim, yprim), primD/2., color='blue', fill=True)
        else:
            circle = plt.Circle((xprim, yprim), primD/2., color='cyan', fill=True)
        ax.add_artist(circle)
        #except:
        #    plt.plot(xprim, yprim, 'r+')
    if ( (fsec0 > 0) & (fsec0 <= 1)  ):
        #try:
        if ( sepx**2 / (rD/3.*np.cos(inc))**2 + sepy**2 / (rD/3.)**2 < 1 ):
            circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
        else:
            circle2 = plt.Circle((xsec, ysec), secD/2., color='cyan', fill=True)
        ax.add_artist(circle2)
        #except:
        #    plt.plot(xsec, ysec, 'g+')

    plt.axis([d, -d, -d, d])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    cbar = fig.colorbar(cs)
    plt.title(name)
    fig.tight_layout()

    if save:
        plt.savefig(dir + name + '_imageFINAL.pdf')

    #plt.show()

    plt.close()


def modringshiftImage(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring20 = pars['fring20']
    fring10 = 1.-fprim0-fsec0-fbg0-fring20
    rM = pars['rM']
    xprim = pars['offsetx']
    yprim = pars['offsety']
    xring2 = pars['xring2']
    yring2 = pars['yring2']
    rD1 = pars['rD1']
    rW1 = pars['rW1']
    rD2 = pars['rD2']
    rW2 = pars['rW2']
    c11 = pars['c11']
    s11 = pars['s11']
    c12 = pars['c12']
    s12 = pars['s12']
    c13 = pars['c13']
    s13 = pars['s13']
    rD2 = pars['rD2']
    rW2 = pars['rW2']
    c21 = pars['c21']
    s21 = pars['s21']
    c22 = pars['c22']
    s22 = pars['s22']
    c23 = pars['c23']
    s23 = pars['s23']
    PA = pars['PA']
    inc = pars['inc']
    primD = pars['primD']
    secD = pars['secD']

    xsec = -1.0 * rM * xprim
    ysec = -1.0 * rM * yprim

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = xprim * np.cos(PA*np.pi/180.) - yprim * np.sin(PA*np.pi/180.)
    yprim2 = xprim * np.sin(PA*np.pi/180.) + yprim * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    theta = np.pi / 2. - PA * np.pi / 180.
    inc = np.pi * inc / 180.

    x2 = x * np.cos(theta) + y * np.sin(theta)
    y2 = y * np.cos(theta) - x * np.sin(theta)
    y2 /= np.cos(inc)
    alpha = np.arctan2(y2, x2)

    width1 = rD1 * rW1 /2.
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    ring1 = 1. / (sigma1 * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD1/2.)**2 / (2 * sigma1**2) )
    ring1 *= 1 - c11 * np.cos(alpha) + s11 * np.sin(alpha) + c12 * np.cos(2*alpha) + s12 * np.sin(2*alpha) - c13 * np.cos(3*alpha) + s13 * np.sin(3*alpha)

    width2 = rD2 * rW2 /2.
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))
    ring2 = 1. / (sigma2 * 2 * np.pi) * np.exp(- (np.sqrt((x2-xring2)**2 + (y2-yring2)**2) - rD2/2.)**2 / (2 * sigma2**2) )
    ring2 *= 1 - c21 * np.cos(alpha) + s21 * np.sin(alpha) + c22 * np.cos(2*alpha) + s22 * np.sin(2*alpha) - c23 * np.cos(3*alpha) + s23 * np.sin(3*alpha)

    img = fring10 * ring1 + fring20 * ring2 + fbg0 * np.ones((n, n)) / (n*n)
    img /= np.max(img)

    # Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='gist_heat', vmin=0)

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        #try:
        if ( sepx**2 / (rD1/3.*np.cos(inc))**2 + sepy**2 / (rD1/3.)**2 < 1 ):
            circle = plt.Circle((xprim, yprim), primD/2., color='blue', fill=True)
        else:
            circle = plt.Circle((xprim, yprim), primD/2., color='cyan', fill=True)
        ax.add_artist(circle)
        #except:
        #    plt.plot(xprim, yprim, 'r+')
    if ( (fsec0 > 0) & (fsec0 <= 1)  ):
        #try:
        if ( sepx**2 / (rD/3.*np.cos(inc))**2 + sepy**2 / (rD1/3.)**2 < 1 ):
            circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
        else:
            circle2 = plt.Circle((xsec, ysec), secD/2., color='cyan', fill=True)
        ax.add_artist(circle2)
        #except:
        #    plt.plot(xsec, ysec, 'g+')

    plt.axis([d, -d, -d, d])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    cbar = fig.colorbar(cs)
    plt.title(name)
    fig.tight_layout()

    if save:
        plt.savefig(dir + name + '_imageFINAL.pdf')

    #plt.show()

    plt.close()


def modringshift2Image(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring20 = pars['fring20']
    fring10 = 1.-fprim0-fsec0-fbg0-fring20
    rM = pars['rM']
    xprim = pars['offsetx']
    yprim = pars['offsety']
    Rring = pars['Rring']
    PA = pars['PA']
    xring2 = Rring * np.sin((PA-90)*np.pi/180. + np.pi)
    yring2 = Rring * np.cos((PA-90)*np.pi/180. + np.pi)
    rD1 = pars['rD1']
    rW1 = pars['rW1']
    c11 = pars['c11']
    s11 = pars['s11']
    c12 = pars['c12']
    s12 = pars['s12']
    c13 = pars['c13']
    s13 = pars['s13']
    rD2 = pars['rD2']
    rW2 = pars['rW2']
    c21 = pars['c21']
    s21 = pars['s21']
    c22 = pars['c22']
    s22 = pars['s22']
    c23 = pars['c23']
    s23 = pars['s23']
    inc = pars['inc']
    primD = pars['primD']
    secD = pars['secD']

    xsec = -1.0 * rM * xprim
    ysec = -1.0 * rM * yprim

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = xprim * np.cos(PA*np.pi/180.) - yprim * np.sin(PA*np.pi/180.)
    yprim2 = xprim * np.sin(PA*np.pi/180.) + yprim * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    theta = np.pi / 2. - PA * np.pi / 180.
    inc = np.pi * inc / 180.

    x2 = x * np.cos(theta) + y * np.sin(theta)
    y2 = y * np.cos(theta) - x * np.sin(theta)
    y2 /= np.cos(inc)
    alpha = np.arctan2(y2, x2)

    width1 = rD1 * rW1 /2.
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    ring1 = 1. / (sigma1 * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD1/2.)**2 / (2 * sigma1**2) )
    ring1 *= 1 - c11 * np.cos(alpha) - s11 * np.sin(alpha) + c12 * np.cos(2*alpha) + s12 * np.sin(2*alpha) - c13 * np.cos(3*alpha) - s13 * np.sin(3*alpha)

    width2 = rD2 * rW2 /2.
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))
    ring2 = 1. / (sigma2 * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD2/2.)**2 / (2 * sigma2**2) )
    ring2 *= 1 - c21 * np.cos(alpha) - s21 * np.sin(alpha) + c22 * np.cos(2*alpha) + s22 * np.sin(2*alpha) - c23 * np.cos(3*alpha) - s23 * np.sin(3*alpha)
    ring2 = scimage.shift(ring2, (yring2/ps, xring2/ps))

    img = fring10 * ring1 + fring20 * ring2 + fbg0 * np.ones((n, n)) / (n*n)
    img /= np.max(img)

    # Plot the image
    fig, ax = plt.subplots()
    cs = ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='inferno', vmin=0)

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        #try:
        if ( sepx**2 / (rD1/3.*np.cos(inc))**2 + sepy**2 / (rD1/3.)**2 < 1 ):
            circle = plt.Circle((xprim, yprim), primD/2., color='green', fill=True)
        else:
            circle = plt.Circle((xprim, yprim), primD/2., color='green', fill=True)
        ax.add_artist(circle)
        plt.plot(xprim, yprim, 'c+', markersize=8)
        #except:
        #    plt.plot(xprim, yprim, 'r+')
    if ( (fsec0 > 0) & (fsec0 <= 1)  ):
        #try:
        if ( sepx**2 / (rD1/3.*np.cos(inc))**2 + sepy**2 / (rD1/3.)**2 < 1 ):
            circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
        else:
            circle2 = plt.Circle((xsec, ysec), secD/2., color='cyan', fill=True)
        ax.add_artist(circle2)
        #except:
        #    plt.plot(xsec, ysec, 'g+')

    plt.axis([d, -d, -d, d])
    ax.set_xlabel('$\Delta$ra (mas)')
    ax.set_ylabel('$\Delta$dec (mas)')
    cbar = fig.colorbar(cs)
    plt.title(name)
    fig.tight_layout()

    if save:
        plt.savefig(dir + name + '_imageFINAL.pdf')

    #plt.show()

    plt.close()


def modringImage22(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring0 = 1.-fprim0-fsec0-fbg0
    rM = pars['rM']
    xsec = pars['offsetx']
    ysec = pars['offsety']
    rD = pars['rD']
    rW = pars['rW']
    c1 = pars['c1']
    s1 = pars['s1']
    c2 = pars['c2']
    s2 = pars['s2']
    c3 = pars['c3']
    s3 = pars['s3']
    PA = pars['PA']
    inc = pars['inc']

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = -1.0 * rM * xsec2
    yprim2 = -1.0 * rM * ysec2

    theta = np.pi / 2. - PA * np.pi / 180.
    inc = np.pi * inc / 180.

    x2 = x * np.cos(theta) + y * np.sin(theta)
    y2 = y * np.cos(theta) - x * np.sin(theta)
    y2 /= np.cos(inc)
    alpha = np.arctan2(y2, x2)
    width = rD * rW
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    ring = 1. / (sigma * 2 * np.pi) * np.exp(- (np.sqrt(x2**2 + y2**2) - rD/2.)**2 / (2 * sigma**2) )
    ring *= 1 - c1 * np.cos(alpha) - s1 * np.sin(alpha) + c2 * np.cos(2*alpha) + s2 * np.sin(2*alpha) - c3 * np.cos(3*alpha) + s3 * np.sin(3*alpha)

    img = fring0 * ring + fbg0 * np.ones((n, n)) / (n*n)

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='hot')

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        try:
            if ( ( np.abs(xprim2) < rD/3 *np.cos(inc*np.pi/180.) ) & ( np.abs(yprim2) < rD/3 ) ):
                circle = plt.Circle((xprim, yprim), primD/2., color='red', fill=True)
            else:
                circle = plt.Circle((xprim, yprim), primD/2., color='yellow', fill=True)
            ax.add_artist(circle)
        except:
            plt.plot(xprim, yprim, 'r+')
    if ( (fsec0 > 0) & (fsec0 <= 1)  ):
        try:
            if ( np.abs(xsec2) < rD/3 *np.cos(inc*np.pi/180.) ) & ( np.abs(ysec2) < rD/3 ):
                circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
            else:
                circle2 = plt.Circle((xsec, ysec), secD/2., color='yellow', fill=True)
            ax.add_artist(circle2)
        except:
            plt.plot(xsec, ysec, 'g+')

    plt.axis([d, -d, -d, d])

    if save:
        plt.savefig(dir + name + '_image.pdf')

    # plt.show()

    plt.close()


def offsetstar_model(allu, allv, X, Y, primobj):

    #
    # single offset-star model
    #

    u, u1, u2, u3 = allu
    v, v1, v2, v3 = allv
    primobj.x = X
    primobj.y = Y

    return (primobj.evaluateComplex((u, v)), primobj.evaluateComplex((u1, v1)), primobj.evaluateComplex((u2, v2)), primobj.evaluateComplex((-u3, -v3)))

def offsetstar_modelVis(u, v, X, Y, primobj):

    #
    # single offset-star model
    #

    primobj.x = X
    primobj.y = Y

    return primobj.evaluateComplex((u, v))



def modring_model(allu, allv, rD, rW, c1, s1, c2, s2, c3, s3, PA, inc, ringobj):

    #
    # modulated ring model
    #

    u,u1,u2,u3 = allu
    v,v1,v2,v3 = allv
    ringobj.diameter = rD
    ringobj.width = rW
    ringobj.c1 = c1
    ringobj.s1 = s1
    ringobj.c2 = c2
    ringobj.s2 = s2
    ringobj.c3 = c3
    ringobj.s3 = s3
    ringobj.PA = PA
    ringobj.inclination = inc

    return (ringobj.evaluateComplex((u,v)),ringobj.evaluateComplex((u1,v1)),ringobj.evaluateComplex((u2,v2)),ringobj.evaluateComplex((-u3,-v3)))


def modringshift_model(allu, allv, rD, rW, c1, s1, c2, s2, c3, s3, PA, inc, xring, yring, ringobj):

    #
    # modulated ring model
    #

    u,u1,u2,u3 = allu
    v,v1,v2,v3 = allv
    ringobj.diameter = rD
    ringobj.width = rW
    ringobj.c1 = c1
    ringobj.s1 = s1
    ringobj.c2 = c2
    ringobj.s2 = s2
    ringobj.c3 = c3
    ringobj.s3 = s3
    ringobj.PA = PA
    ringobj.inclination = inc
    ringobj.x0 = xring
    ringobj.y0 = yring

    return (ringobj.evaluateComplex((u,v)),ringobj.evaluateComplex((u1,v1)),ringobj.evaluateComplex((u2,v2)),ringobj.evaluateComplex((-u3,-v3)))


def modringshift2_model(allu, allv, rD, rW, c1, s1, c2, s2, c3, s3, PA, inc, xring, yring, ringobj):

    #
    # modulated ring model
    #

    u,u1,u2,u3 = allu
    v,v1,v2,v3 = allv
    ringobj.diameter = rD
    ringobj.width = rW
    ringobj.c1 = c1
    ringobj.s1 = s1
    ringobj.c2 = c2
    ringobj.s2 = s2
    ringobj.c3 = c3
    ringobj.s3 = s3
    ringobj.PA = PA
    ringobj.inclination = inc
    ringobj.x0 = xring
    ringobj.y0 = yring

    return (ringobj.evaluateComplex((u,v)),ringobj.evaluateComplex((u1,v1)),ringobj.evaluateComplex((u2,v2)),ringobj.evaluateComplex((-u3,-v3)))



def lnprior_binarymodring(pars):

    #
    # log-prior function
    #

    rD,rW,c1,s1,c2,s2,c3,s3,PA,inc,T,offsetx,offsety,deltainc,fprim0,fsec0,fbg0,dprim,dsec,Tsec,dback,logfactor,rM,primD = pars['rD'], \
                                   pars['rW'],pars['c1'],pars['s1'],pars['c2'],pars['s2'],pars['c3'],pars['s3'],pars['PA'],pars['inc'], \
                                   pars['T'],pars['offsetx'],pars['offsety'],pars['deltainc'],pars['fprim0'], \
                                   pars['fsec0'],pars['fbg0'],pars['dprim'],pars['dsec'],pars['Tsec'],pars['dback'],pars['logfactor'], \
                                   pars['rM'], pars['primD']

    xsec = -1.0 * rM * offsetx * (fsec0 != 0)
    ysec = -1.0 * rM * offsety * (fsec0 != 0)

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = offsetx * np.cos(PA*np.pi/180.) - offsety * np.sin(PA*np.pi/180.)
    yprim2 = offsetx * np.sin(PA*np.pi/180.) + offsety * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    if (np.sqrt(c1**2+s1**2) <= 1.) & (np.sqrt(c2**2+s2**2) <= 1.) & (np.sqrt(c3**2+s3**2) <= 1.) & (inc<=90.) & (inc>=0.) & (PA<=360.) \
    & (PA>=0.) & (rW>=0.) & (rW<=5.) & (rD>=0.01) & (rD<=501.) & (fprim0<=1.0) & (fprim0>=0.0) & (fsec0<=1.0) \
    & (fsec0>=0.0) & (fbg0<=1.0) & (fbg0>=0.0) & (fprim0+fsec0+fbg0<=1.0) & (T <= 10000.) & (T >= 500.) \
    & (offsetx<=30.0) & (offsetx>=-30.0)& (rM<=20.0) & (rM>=0.0) & ( sepx**2 / (rD/3.*np.cos(inc*np.pi/180.))**2 + sepy**2 / (rD/3.)**2 < 1 )\
    & (offsety<=30.0) & (offsety>=-30.0) & (deltainc>=0.0) & (deltainc<=90.-inc) & (dprim>=-4.) & (dprim<=4.) \
    & (primD>=0.01) & (dsec>=-4.) & (dsec<=4.) & (Tsec>=1500.) & (Tsec<=8000.) & (dback>=-4.) & (dback<=4.) & (logfactor>=-15.) & (logfactor<=1.):
        return 0.

    return -np.inf


def lnprior_binarymodringshift(pars):

    #
    # log-prior function
    #

    rD1,rW1,rD2,rW2,c11,s11,c12,s12,c13,s13,c21,s21,c22,s22,c23,s23,PA,inc,T1,T2,offsetx,offsety,deltainc,\
                                   fprim0,fsec0,fbg0,fring20,dprim,dsec,Tsec,dback,\
                                   logfactor,rM,primD,xring,yring2 = pars['rD1'],pars['rW1'],pars['rD2'], pars['rW2'], \
                                   pars['c11'],pars['s11'],pars['c12'],pars['s12'],pars['c13'],pars['s13'], \
                                   pars['c21'],pars['s21'],pars['c22'],pars['s22'],pars['c23'],pars['s23'], \
                                   pars['PA'],pars['inc'], \
                                   pars['T1'],pars['T2'],pars['offsetx'],pars['offsety'],pars['deltainc'],pars['fprim0'], \
                                   pars['fsec0'],pars['fbg0'],pars['fring20'],pars['dprim'],pars['dsec'],pars['Tsec'],\
                                   pars['dback'],pars['logfactor'], \
                                   pars['rM'], pars['primD'],pars['xring2'],pars['yring2']

    xsec = -1.0 * rM * offsetx * (fsec0 != 0)
    ysec = -1.0 * rM * offsety * (fsec0 != 0)

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = offsetx * np.cos(PA*np.pi/180.) - offsety * np.sin(PA*np.pi/180.)
    yprim2 = offsetx * np.sin(PA*np.pi/180.) + offsety * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

    if (np.sqrt(c11**2+s11**2) <= 1.) & (np.sqrt(c12**2+s12**2) <= 1.) & (np.sqrt(c13**2+s13**2) <= 1.) & \
        (np.sqrt(c21**2+s21**2) <= 1.) & (np.sqrt(c22**2+s22**2) <= 1.) & (np.sqrt(c23**2+s23**2) <= 1.) & \
        (inc<=90.) & (inc>=0.) & (PA<=360.) \
        & (PA>=0.) & (rW1>=0.) & (rW1<=5.) & (rW2>=0.) & (rW2<=5.) & \
        (rD1>=0.01) & (rD1<=501.) & (rD2>=0.01) & (rD2<=501.) & (fprim0<=1.0) & (fprim0>=0.0) & (fsec0<=1.0) \
        & (fsec0>=0.0) & (fbg0<=1.0) & (fbg0>=0.0) & (fprim0+fsec0+fbg0+fring20<=1.0) & (fring20<=1.0) & (fring20>=0.0) &\
         (T1 <= 10000.) & (T1 >= 500.) & (T2 <= 10000.) & (T2 >= 500.) \
        & (offsetx<=30.0) & (offsetx>=-30.0)& (rM<=20.0) & (rM>=0.0) & ( sepx**2 / (rD1/3.*np.cos(inc*np.pi/180.))**2 + sepy**2 / (rD1/3.)**2 < 1 )\
        & (offsety<=30.0) & (offsety>=-30.0) & (deltainc>=0.0) & (deltainc<=90.-inc) & (dprim>=-4.) & (dprim<=4.) \
        & (primD>=0.01) & (dsec>=-4.) & (dsec<=4.) & (Tsec>=1500.) & (Tsec<=8000.) & (dback>=-4.) & (dback<=4.) &\
         (logfactor>=-15.) & (logfactor<=1.) &  (rD1 < rD2):
        return 0.

    return -np.inf


def lnprior_binarymodringshift2(pars):

    #
    # log-prior function
    #

    rD1,rW1,rD2,rW2,c11,s11,c12,s12,c13,s13,c21,s21,c22,s22,c23,s23,PA,inc,T1,T2,offsetx,offsety,deltainc,\
                                   fprim0,fsec0,fbg0,fring20,dprim,dsec,Tsec,dback,\
                                   logfactor,rM,primD,Rring = pars['rD1'],pars['rW1'],pars['rD2'], pars['rW2'], \
                                   pars['c11'],pars['s11'],pars['c12'],pars['s12'],pars['c13'],pars['s13'], \
                                   pars['c21'],pars['s21'],pars['c22'],pars['s22'],pars['c23'],pars['s23'], \
                                   pars['PA'],pars['inc'], \
                                   pars['T1'],pars['T2'],pars['offsetx'],pars['offsety'],pars['deltainc'],pars['fprim0'], \
                                   pars['fsec0'],pars['fbg0'],pars['fring20'],pars['dprim'],pars['dsec'],pars['Tsec'],\
                                   pars['dback'],pars['logfactor'], \
                                   pars['rM'], pars['primD'], pars['Rring']

    xsec = -1.0 * rM * offsetx * (fsec0 != 0)
    ysec = -1.0 * rM * offsety * (fsec0 != 0)

    xsec2 = xsec * np.cos(PA*np.pi/180.) - ysec * np.sin(PA*np.pi/180.)
    ysec2 = xsec * np.sin(PA*np.pi/180.) + ysec * np.cos(PA*np.pi/180.)

    xprim2 = offsetx * np.cos(PA*np.pi/180.) - offsety * np.sin(PA*np.pi/180.)
    yprim2 = offsetx * np.sin(PA*np.pi/180.) + offsety * np.cos(PA*np.pi/180.)

    sepx = np.abs(xprim2 - xsec2)
    sepy = np.abs(yprim2 - ysec2)

# & ( sepx**2 / (rD1/3.*np.cos(inc*np.pi/180.))**2 + sepy**2 / (rD1/3.)**2 < 1 )

    if (np.sqrt(c11**2+s11**2) <= 1.) & (np.sqrt(c12**2+s12**2) <= 1.) & (np.sqrt(c13**2+s13**2) <= 1.) & \
        (np.sqrt(c21**2+s21**2) <= 1.) & (np.sqrt(c22**2+s22**2) <= 1.) & (np.sqrt(c23**2+s23**2) <= 1.) & \
        (inc<=90.) & (inc>=0.) & (PA<=360.) \
        & (PA>=0.) & (rD1*rW1>=0.8) & (rW1<=5.) & (rW2*rD2>=0.8) & (rW2<=5.) & \
        (rD1>=0.01) & (rD1<=501.) & (rD2>=0.01) & (rD2<=501.) & (fprim0<=1.0) & (fprim0>=0.0) & (fsec0<=1.0) \
        & (fsec0>=0.0) & (fbg0<=1.0) & (fbg0>=0.0) & (fprim0+fsec0+fbg0+fring20<=1.0) & (fring20<=1.0) & (fring20>=0.0) &\
         (T1 <= 10000.) & (T1 >= 500.) & (T2 <= 10000.) & (T2 >= 500.) \
        & (offsetx<=30.0) & (offsetx>=-30.0)& (rM<=20.0) & (rM>=0.0) \
        & (offsety<=30.0) & (offsety>=-30.0) & (deltainc>=0.0) & (deltainc<=90.-inc) & (dprim>=-4.) & (dprim<=4.) \
        & (primD>=0.01) & (dsec>=-4.) & (dsec<=4.) & (Tsec>=1500.) & (Tsec<=8000.) & (dback>=-4.) & (dback<=4.) &\
         (logfactor>=-15.) & (logfactor<=1.) &  (rD1 < rD2) & (rD1*np.cos(inc*np.pi/180.)/2. + rW1*rD1/6. < rD2*np.cos(inc*np.pi/180.)/2. - rW2*rD2/6. - np.abs(Rring) ):
        return 0.

    return -np.inf


def lnprior_binarymodbg(pars):

    #
    # log-prior function
    #

    offsetx,offsety,fsec0,fbg0,dprim,dsec,dback,rM,primD,secD = pars['offsetx'],pars['offsety'],pars['fsec0'], \
                                   pars['fbg0'],pars['dprim'],pars['dsec'],pars['dback'], pars['rM'], pars['primD']\
                                   , pars['secD']

    if (fsec0 <= 1.0) & (fsec0 >= 0.0) & (fbg0 <= 1.0) & (fbg0 >= 0.0) & (fsec0+fbg0 <= 1.0) \
    & (offsetx<=50.0) & (offsetx>=-50.0)& (rM<=100.0) & (rM>=0.0)\
    & (offsety<=50.0) & (offsety>=-50.0) & (dprim>=-4.) & (dprim<=4.) & (secD>=0.001)  \
    & (primD>=0.01) & (dsec>=-4.) & (dsec<=4.) & (dback>=-4.) & (dback<=4.):
            return 0.

    return -np.inf


def modring_model_chrom(pars, u, v, wave, wave0, wavespec=[], starspec=[]):

    # unpacking parameters
    # print(pars['T'])
    T = np.array([pars['T']])
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring0 = 1.-fprim0-fsec0-fbg0
    dprim = pars['dprim']
    dsec = pars['dsec']
    # Tsec = np.array([pars['Tsec']])
    dback = pars['dback']
    rM = pars['rM']
    #d = pars['d']
    x1 = pars['offsetx']
    y1 = pars['offsety']
    primD = pars['primD']
    secD = pars['secD']

    x2 = -1.0 * x1 * rM
    y2 = -1.0 * y1 * rM

    ringobj = fitRoutines.ModulatedGaussianRing(pars['rD'], pars['rW'], pars['c1'], pars['c2'], pars['c3'], pars['s1'], pars['s2'], pars['s3'], pars['PA'], pars['inc'])
    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    modringVis = modring_model(u, v, pars['rD'], pars['rW'], pars['c1'], pars['s1'], pars['c2'], pars['s2'], pars['c3'], pars['s3'], pars['PA'], pars['inc'], ringobj)
    primVis = offsetstar_model(u, v, x1, y1, primobj)
    secVis = offsetstar_model(u, v, x2, y2, secobj)

    # the blackbody flux scaling
    if starspec == []:
        fprim = fprim0 * (wave[0]/wave0)**dprim
        fprimCP = fprim0 * (wave[1]/wave0)**dprim
        fbg = fbg0 * (wave[0]/wave0)**dback
        fbgCP = fbg0 * (wave[1]/wave0)**dback
        fsec = fsec0 * (wave[0]/wave0)**dsec
        fsecCP = fsec0 * (wave[1]/wave0)**dsec
    else:
        fprim = fprim0 * np.interp( wave[0], wavespec, starspec)
        fprimCP = fprim0 * np.interp( wave[1], wavespec, starspec)
        if dback == -4:
            fbg = fbg0 * np.interp( wave[0], wavespec, starspec)
            fbgCP = fbg0 * np.interp( wave[1], wavespec, starspec)
        else:
            fbg = fbg0 * (wave[0]/wave0)**dback
            fbgCP = fbg0 * (wave[1]/wave0)**dback
        if dsec == -4:
            fsec = fsec0 * np.interp( wave[0], wavespec, starspec)
            fsecCP = fsec0 * np.interp( wave[1], wavespec, starspec)
        else:
            fsec = fsec0 * (wave[0]/wave0)**dsec
            fsecCP = fsec0 * (wave[1]/wave0)**dsec

    BBnorm = BB_m(T, wave0*1e6)[0, :]
    BB0 = BB_m(T, wave[0]*1e6)[0, :]
    BB1 = BB_m(T, wave[1]*1e6)[0, :]

    fring = fring0 * BB0 / BBnorm
    fringCP = fring0 * BB1 / BBnorm

    ### Test of the flux chromatism
    #fig, ax = plt.subplots()
    #ax.plot(wave[0], fprim, '.', label='primary from SED')
    #ax.plot(wave[0], fsec, '.', label='secondary')
    #ax.plot(wave[0], fbg, '.', label='background')
    #ax.plot(wave[0], fring, '.', label='ring')
    #ax.legend(loc='upper right')
    #plt.show()
    ###

    # the total visibilities and bispectrum -> closure phase
    totalVis = fprim * primVis[0] + fsec * secVis[0] + fring * modringVis[0]
    totalVis /= fprim + fsec + fring + fbg

    totalVis1 = fprimCP * primVis[1] + fsecCP * secVis[1] + fringCP * modringVis[1]
    totalVis1 /= fprimCP + fsecCP + fringCP + fbgCP

    totalVis2 = fprimCP * primVis[2] + fsecCP * secVis[2] + fringCP * modringVis[2]
    totalVis2 /= fprimCP + fsecCP + fringCP + fbgCP

    totalVis3 = fprimCP * primVis[3] + fsecCP * secVis[3] + fringCP * modringVis[3]
    totalVis3 /= fprimCP + fsecCP + fringCP + fbgCP

    V2 = abs(totalVis)**2

    bispectrum = totalVis1*totalVis2*totalVis3
    CP = np.angle(bispectrum, deg=True)  # ,deg=True)

    return V2, CP


def modringshift_model_chrom(pars, u, v, wave, wave0, wavespec=[], starspec=[]):

    # unpacking parameters
    # print(pars['T'])
    T1 = np.array([pars['T1']])
    T2 = np.array([pars['T2']])
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring20 = pars['fring20']
    fring10 = 1.-fprim0-fsec0-fbg0-fring20
    dprim = pars['dprim']
    dsec = pars['dsec']
    # Tsec = np.array([pars['Tsec']])
    dback = pars['dback']
    rM = pars['rM']
    #d = pars['d']
    x1 = pars['offsetx']
    y1 = pars['offsety']
    xring2 = pars['xring2']
    yring2 = pars['yring2']
    primD = pars['primD']
    secD = pars['secD']

    x2 = -1.0 * x1 * rM
    y2 = -1.0 * y1 * rM

    ringobj1 = fitRoutines.ModulatedGaussianRingShift(pars['rD1'], pars['rW1'], pars['c11'], pars['c12'], pars['c13'], pars['s11'], pars['s12'], pars['s13'], pars['PA'], pars['inc'],0,0)
    ringobj2 = fitRoutines.ModulatedGaussianRingShift(pars['rD2'], pars['rW2'], pars['c21'], pars['c22'], pars['c23'], pars['s21'], pars['s22'], pars['s23'], pars['PA'], pars['inc'],xring2,yring2)
    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    modring1Vis = modringshift_model(u, v, pars['rD1'], pars['rW1'], pars['c11'], pars['s11'], pars['c12'], pars['s12'], pars['c13'], pars['s13'], pars['PA'], pars['inc'],0,0, ringobj1)
    modring2Vis = modringshift_model(u, v, pars['rD2'], pars['rW2'], pars['c21'], pars['s21'], pars['c22'], pars['s22'], pars['c23'], pars['s23'], pars['PA'], pars['inc'],xring2,yring2, ringobj2)
    primVis = offsetstar_model(u, v, x1, y1, primobj)
    secVis = offsetstar_model(u, v, x2, y2, secobj)

    # the blackbody flux scaling
    if starspec == []:
        fprim = fprim0 * (wave[0]/wave0)**dprim
        fprimCP = fprim0 * (wave[1]/wave0)**dprim
        fbg = fbg0 * (wave[0]/wave0)**dback
        fbgCP = fbg0 * (wave[1]/wave0)**dback
        fsec = fsec0 * (wave[0]/wave0)**dsec
        fsecCP = fsec0 * (wave[1]/wave0)**dsec
    else:
        fprim = fprim0 * np.interp( wave[0], wavespec, starspec)
        fprimCP = fprim0 * np.interp( wave[1], wavespec, starspec)
        if dback == -4:
            fbg = fbg0 * np.interp( wave[0], wavespec, starspec)
            fbgCP = fbg0 * np.interp( wave[1], wavespec, starspec)
        else:
            fbg = fbg0 * (wave[0]/wave0)**dback
            fbgCP = fbg0 * (wave[1]/wave0)**dback
        if dsec == -4:
            fsec = fsec0 * np.interp( wave[0], wavespec, starspec)
            fsecCP = fsec0 * np.interp( wave[1], wavespec, starspec)
        else:
            fsec = fsec0 * (wave[0]/wave0)**dsec
            fsecCP = fsec0 * (wave[1]/wave0)**dsec

    BBnorm = BB_m(T1, wave0*1e6)[0, :]
    BB0 = BB_m(T1, wave[0]*1e6)[0, :]
    BB1 = BB_m(T1, wave[1]*1e6)[0, :]

    fring1 = fring10 * BB0 / BBnorm
    fring1CP = fring10 * BB1 / BBnorm

    BBnorm = BB_m(T2, wave0*1e6)[0, :]
    BB0 = BB_m(T2, wave[0]*1e6)[0, :]
    BB1 = BB_m(T2, wave[1]*1e6)[0, :]

    fring2 = fring20 * BB0 / BBnorm
    fring2CP = fring20 * BB1 / BBnorm

    ### Test of the flux chromatism
    #fig, ax = plt.subplots()
    #ax.plot(wave[0], fprim, '.', label='primary from SED')
    #ax.plot(wave[0], fsec, '.', label='secondary')
    #ax.plot(wave[0], fbg, '.', label='background')
    #ax.plot(wave[0], fring, '.', label='ring')
    #ax.legend(loc='upper right')
    #plt.show()
    ###

    # the total visibilities and bispectrum -> closure phase
    totalVis = fprim * primVis[0] + fsec * secVis[0] + fring1 * modring1Vis[0] + fring2 * modring2Vis[0]
    totalVis /= fprim + fsec + fring1 + fring2 + fbg

    totalVis1 = fprimCP * primVis[1] + fsecCP * secVis[1] + fring1CP * modring1Vis[1] + fring2CP * modring2Vis[1]
    totalVis1 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    totalVis2 = fprimCP * primVis[2] + fsecCP * secVis[2] + fring1CP * modring1Vis[2] + fring2CP * modring2Vis[2]
    totalVis2 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    totalVis3 = fprimCP * primVis[3] + fsecCP * secVis[3] + fring1CP * modring1Vis[3] + fring2CP * modring2Vis[3]
    totalVis3 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    V2 = abs(totalVis)**2

    bispectrum = totalVis1*totalVis2*totalVis3
    CP = np.angle(bispectrum, deg=True)  # ,deg=True)

    return V2, CP


def modringshift2_model_chrom(pars, u, v, wave, wave0, wavespec=[], starspec=[]):

    # unpacking parameters
    # print(pars['T'])
    T1 = np.array([pars['T1']])
    T2 = np.array([pars['T2']])
    fprim0 = pars['fprim0']
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fring20 = pars['fring20']
    fring10 = 1.-fprim0-fsec0-fbg0-fring20
    dprim = pars['dprim']
    dsec = pars['dsec']
    # Tsec = np.array([pars['Tsec']])
    dback = pars['dback']
    rM = pars['rM']
    #d = pars['d']
    x1 = pars['offsetx']
    y1 = pars['offsety']
    Rring = pars['Rring']
    PA = pars['PA']
    xring2 = Rring * np.sin((PA-90)*np.pi/180)
    yring2 = Rring * np.cos((PA-90)*np.pi/180)
    primD = pars['primD']
    secD = pars['secD']

    x2 = -1.0 * x1 * rM
    y2 = -1.0 * y1 * rM

    ringobj1 = fitRoutines.ModulatedGaussianRingShift(pars['rD1'], pars['rW1'], pars['c11'], pars['c12'], pars['c13'], pars['s11'], pars['s12'], pars['s13'], pars['PA'], pars['inc'],0,0)
    ringobj2 = fitRoutines.ModulatedGaussianRingShift(pars['rD2'], pars['rW2'], pars['c21'], pars['c22'], pars['c23'], pars['s21'], pars['s22'], pars['s23'], pars['PA'], pars['inc'],xring2,yring2)
    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    modring1Vis = modringshift_model(u, v, pars['rD1'], pars['rW1'], pars['c11'], pars['s11'], pars['c12'], pars['s12'], pars['c13'], pars['s13'], pars['PA'], pars['inc'],0,0, ringobj1)
    modring2Vis = modringshift_model(u, v, pars['rD2'], pars['rW2'], pars['c21'], pars['s21'], pars['c22'], pars['s22'], pars['c23'], pars['s23'], pars['PA'], pars['inc'],xring2,yring2, ringobj2)
    primVis = offsetstar_model(u, v, x1, y1, primobj)
    secVis = offsetstar_model(u, v, x2, y2, secobj)

    # the blackbody flux scaling
    if starspec == []:
        fprim = fprim0 * (wave[0]/wave0)**dprim
        fprimCP = fprim0 * (wave[1]/wave0)**dprim
        fbg = fbg0 * (wave[0]/wave0)**dback
        fbgCP = fbg0 * (wave[1]/wave0)**dback
        fsec = fsec0 * (wave[0]/wave0)**dsec
        fsecCP = fsec0 * (wave[1]/wave0)**dsec
    else:
        fprim = fprim0 * np.interp( wave[0], wavespec, starspec)
        fprimCP = fprim0 * np.interp( wave[1], wavespec, starspec)
        if dback == -4:
            fbg = fbg0 * np.interp( wave[0], wavespec, starspec)
            fbgCP = fbg0 * np.interp( wave[1], wavespec, starspec)
        else:
            fbg = fbg0 * (wave[0]/wave0)**dback
            fbgCP = fbg0 * (wave[1]/wave0)**dback
        if dsec == -4:
            fsec = fsec0 * np.interp( wave[0], wavespec, starspec)
            fsecCP = fsec0 * np.interp( wave[1], wavespec, starspec)
        else:
            fsec = fsec0 * (wave[0]/wave0)**dsec
            fsecCP = fsec0 * (wave[1]/wave0)**dsec

    BBnorm = BB_m(T1, wave0*1e6)[0, :]
    BB0 = BB_m(T1, wave[0]*1e6)[0, :]
    BB1 = BB_m(T1, wave[1]*1e6)[0, :]

    fring1 = fring10 * BB0 / BBnorm
    fring1CP = fring10 * BB1 / BBnorm

    BBnorm = BB_m(T2, wave0*1e6)[0, :]
    BB0 = BB_m(T2, wave[0]*1e6)[0, :]
    BB1 = BB_m(T2, wave[1]*1e6)[0, :]

    fring2 = fring20 * BB0 / BBnorm
    fring2CP = fring20 * BB1 / BBnorm

    ### Test of the flux chromatism
    #fig, ax = plt.subplots()
    #ax.plot(wave[0], fprim, '.', label='primary from SED')
    #ax.plot(wave[0], fsec, '.', label='secondary')
    #ax.plot(wave[0], fbg, '.', label='background')
    #ax.plot(wave[0], fring, '.', label='ring')
    #ax.legend(loc='upper right')
    #plt.show()
    ###

    # the total visibilities and bispectrum -> closure phase
    totalVis = fprim * primVis[0] + fsec * secVis[0] + fring1 * modring1Vis[0] + fring2 * modring2Vis[0]
    totalVis /= fprim + fsec + fring1 + fring2 + fbg

    totalVis1 = fprimCP * primVis[1] + fsecCP * secVis[1] + fring1CP * modring1Vis[1] + fring2CP * modring2Vis[1]
    totalVis1 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    totalVis2 = fprimCP * primVis[2] + fsecCP * secVis[2] + fring1CP * modring1Vis[2] + fring2CP * modring2Vis[2]
    totalVis2 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    totalVis3 = fprimCP * primVis[3] + fsecCP * secVis[3] + fring1CP * modring1Vis[3] + fring2CP * modring2Vis[3]
    totalVis3 /= fprimCP + fsecCP + fring1CP + fring2CP + fbgCP

    V2 = abs(totalVis)**2

    bispectrum = totalVis1*totalVis2*totalVis3
    CP = np.angle(bispectrum, deg=True)  # ,deg=True)

    return V2, CP


def chi2_binarymodringshift(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=[], starspec=[]):

    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    V2mod, CPmod = modringshift_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    c2v2 = (vis2 - V2mod)**2
    c2v2 /= vis2err**2
    c2v2 = np.sum(c2v2) / nV2

    c2cp = (cp - CPmod)**2
    c2cp /= cperr**2
    c2cp = np.sum(c2cp) / nCP

    c2tot = nV2 * c2v2 + nCP * c2cp
    c2tot /= nTot

    return c2v2, c2cp, c2tot


def chi2_binarymodringshift2(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=[], starspec=[]):

    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    V2mod, CPmod = modringshift2_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    c2v2 = (vis2 - V2mod)**2
    c2v2 /= vis2err**2
    c2v2 = np.sum(c2v2) / nV2

    c2cp = (cp - CPmod)**2
    c2cp /= cperr**2
    c2cp = np.sum(c2cp) / nCP

    c2tot = nV2 * c2v2 + nCP * c2cp
    c2tot /= nTot

    return c2v2, c2cp, c2tot


def chi2_binarymodring(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=[], starspec=[]):

    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    V2mod, CPmod = modring_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    c2v2 = (vis2 - V2mod)**2
    c2v2 /= vis2err**2
    c2v2 = np.sum(c2v2) / nV2

    c2cp = (cp - CPmod)**2
    c2cp /= cperr**2
    c2cp = np.sum(c2cp) / nCP

    c2tot = nV2 * c2v2 + nCP * c2cp
    c2tot /= nTot

    return c2v2, c2cp, c2tot


def lnlike_binarymodring(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ringobj, primobj, secobj, wavespec=[], starspec=[]):

    V2mod, CPmod = modring_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # the log-probability terms
    visres = vis2 - V2mod
    invvissigma = 1./(vis2err**2 + np.exp(2*pars['logfactor'])*V2mod**2)
    visterm = -.5*np.sum(visres**2*invvissigma + np.log(2*np.pi/invvissigma))
    cpres = np.angle(np.exp(1j*(cp - CPmod)*np.pi/180), deg=True)
    invcpsigma = 1./(cperr**2 + np.exp(2*pars['logfactor'])*CPmod**2)
    cpterm = -.5*np.sum(cpres**2*invcpsigma + np.log(2*np.pi/invcpsigma))

    return visterm+cpterm


def lnlike_binarymodringshift(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, wavespec=[], starspec=[]):

    V2mod, CPmod = modringshift_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # the log-probability terms
    visres = vis2 - V2mod
    invvissigma = 1./(vis2err**2 + np.exp(2*pars['logfactor'])*V2mod**2)
    visterm = -.5*np.sum(visres**2*invvissigma + np.log(2*np.pi/invvissigma))
    cpres = np.angle(np.exp(1j*(cp - CPmod)*np.pi/180), deg=True)
    invcpsigma = 1./(cperr**2 + np.exp(2*pars['logfactor'])*CPmod**2)
    cpterm = -.5*np.sum(cpres**2*invcpsigma + np.log(2*np.pi/invcpsigma))

    return visterm+cpterm


def lnlike_binarymodringshift2(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, wavespec=[], starspec=[]):

    V2mod, CPmod = modringshift2_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # the log-probability terms
    visres = vis2 - V2mod
    invvissigma = 1./(vis2err**2 + np.exp(2*pars['logfactor'])*V2mod**2)
    visterm = -.5*np.sum(visres**2*invvissigma + np.log(2*np.pi/invvissigma))
    cpres = np.angle(np.exp(1j*(cp - CPmod)*np.pi/180), deg=True)
    invcpsigma = 1./(cperr**2 + np.exp(2*pars['logfactor'])*CPmod**2)
    cpterm = -.5*np.sum(cpres**2*invcpsigma + np.log(2*np.pi/invcpsigma))

    return visterm+cpterm


def lnlike_binarymodbg(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, primobj, secobj, wavespec=[], starspec=[]):

    V2mod, CPmod = modbinbg_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    # the log-probability terms
    visres = vis2 - V2mod
    invvissigma = 1./(vis2err**2 + np.exp(2*pars['logfactor'])*V2mod**2)
    visterm = -.5*np.sum(visres**2*invvissigma + np.log(2*np.pi/invvissigma))
    cpres = np.angle(np.exp(1j*(cp - CPmod)*np.pi/180), deg=True)
    invcpsigma = 1./(cperr**2 + np.exp(2*pars['logfactor'])*CPmod**2)
    cpterm = -.5*np.sum(cpres**2*invcpsigma + np.log(2*np.pi/invcpsigma))

    return visterm+cpterm


def lnprob_binarymodring(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ringobj, primobj, secobj, fitpars, fixedpars, wavespec, starspec):

    #
    # log-probability function
    #

    newpars = fixedpars.copy()
    for x in fitpars.keys():
        newpars[x] = pars[fitpars[x]]

    lp = lnprior_binarymodring(newpars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_binarymodring(newpars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ringobj, primobj, secobj, wavespec=wavespec, starspec=starspec)


def lnprob_binarymodringshift(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, fitpars, fixedpars, wavespec, starspec):

    #
    # log-probability function
    #

    newpars = fixedpars.copy()
    for x in fitpars.keys():
        newpars[x] = pars[fitpars[x]]

    lp = lnprior_binarymodringshift(newpars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_binarymodringshift(newpars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, wavespec=wavespec, starspec=starspec)


def lnprob_binarymodringshift2(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, fitpars, fixedpars, wavespec, starspec):

    #
    # log-probability function
    #

    newpars = fixedpars.copy()
    for x in fitpars.keys():
        newpars[x] = pars[fitpars[x]]

    lp = lnprior_binarymodringshift2(newpars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_binarymodringshift2(newpars, u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, wavespec=wavespec, starspec=starspec)



def lnprob_binarymodbg(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, primobj, secobj, fitpars, fixedpars, wavespec, starspec):

    #
    # log-probability function
    #

    newpars = fixedpars.copy()
    for x in fitpars.keys():
        newpars[x] = pars[fitpars[x]]

    lp = lnprior_binarymodbg(newpars)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_binarymodbg(newpars, u, v, wave, vis2, vis2err, cp, cperr, wave0, primobj, secobj, wavespec=wavespec, starspec=starspec)



def PlotBestModelMCMC (data, file, dir='./', save=False, name='anonymous'):

    f = open(file, 'r')
    header = f.readline()
    params = f.readline()
    f.close()

    # print(header)
    header = header.replace(' \n','')
    headers = header.split('        ')

    pars = params.split('        ')
    pars = np.array(pars)

    params = {}
    for i, obj in enumerate(headers):
        obj = obj.strip(' ')
        if obj != '':
            # print(obj, pars[i])
            params[obj] = float(pars[i])

    modring_fit_plot(data, params, save=save, name= name, dir=dir)
    modringImage(params, n=256, ps=0.15, save=save, name=name, dir=dir)


def PlotBestModelMCMCshift (data, file, dir='./', save=False, name='anonymous'):

    f = open(file, 'r')
    header = f.readline()
    params = f.readline()
    f.close()

    # print(header)
    header = header.replace(' \n','')
    headers = header.split('        ')

    pars = params.split('        ')
    pars = np.array(pars)

    params = {}
    for i, obj in enumerate(headers):
        obj = obj.strip(' ')
        if obj != '':
            # print(obj, pars[i])
            params[obj] = float(pars[i])

    modringshift_fit_plot(data, params, save=save, name= name, dir=dir)
    modringshiftImage(params, n=256, ps=0.1, save=save, name=name, dir=dir)


def PlotBestModelMCMCshift2 (data, file, dir='./', save=False, name='anonymous', xlog=False):

    f = open(file, 'r')
    header = f.readline()
    params = f.readline()
    f.close()

    # print(header)
    header = header.replace(' \n','')
    headers = header.split('        ')

    pars = params.split('        ')
    pars = np.array(pars)

    params = {}
    for i, obj in enumerate(headers):
        obj = obj.strip(' ')
        if obj != '':
            # print(obj, pars[i])
            params[obj] = float(pars[i])

    modringshift2_fit_plotsep(data, params, save=save, name= name, dir=dir, xlog=xlog)
    modringshift2Image(params, n=200, ps=0.1, save=save, name=name, dir=dir)


def PlotBestModelMCMCbinbg (data, file, dir='./', save=False, name='anonymous'):

    f = open(file, 'r')
    header = f.readline()
    params = f.readline()
    f.close()

    # print(header)
    header = header.replace(' \n','')
    headers = header.split('        ')

    pars = params.split('        ')
    pars = np.array(pars)

    params = {}
    for i, obj in enumerate(headers):
        obj = obj.strip(' ')
        if obj != '':
            # print(obj, pars[i])
            params[obj] = float(pars[i])

    modbinbg_fit_plot(data, params, save=save, name= name, dir=dir)
    modbinbgImage(params, n=256, ps=0.15, save=save, name=name, dir=dir)


def modbinbgImage(pars, n=256, ps=0.1, lim=0, dir='./', save=False, name='anonymous'):
    # It will plot an image of the model
    # img = np.zeros(n, n)
    fov = ps * n / 2.
    if lim == 0:
        d = fov
    else:
        d = lim

    x, y = xycoord(n, ps)

    # unpacking parameters
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fprim0 = 1.-fsec0-fbg0
    rM = pars['rM']
    xsec = pars['offsetx']
    ysec = pars['offsety']
    primD = pars['primD']
    secD = pars['secD']

    xprim = -1.0 * rM * xsec
    yprim = -1.0 * rM * ysec

    img = fbg0 * np.ones((n, n)) / (n*n)

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow( img, extent=[fov, -fov, -fov, fov], cmap='hot')

    if ( (fprim0 > 0) & (fprim0 <= 1) ):
        try:
            circle = plt.Circle((xprim, yprim), primD/2., color='red', fill=True)
            ax.add_artist(circle)
        except:
            plt.plot(xprim, yprim, 'r*')
    if ( (fsec0 > 0) & (fsec0 <= 1) ):
        try:
            circle2 = plt.Circle((xsec, ysec), secD/2., color='green', fill=True)
            ax.add_artist(circle2)
        except:
            plt.plot(xsec, ysec, 'g*')

    plt.axis([d, -d, -d, d])

    if save:
        plt.savefig(dir + name + '_image.pdf')

    # plt.show()

    plt.close()


def modbinbg_model_chrom(pars, u, v, wave, wave0, wavespec=[], starspec=[]):

    # unpacking parameters
    # print(pars['T'])
    # T = np.array([pars['T']])
    fsec0 = pars['fsec0']
    fbg0 = pars['fbg0']
    fprim0 = 1.-fsec0-fbg0
    dprim = pars['dprim']
    dsec = pars['dsec']
    # Tsec = np.array([pars['Tsec']])
    dback = pars['dback']
    rM = pars['rM']
    #d = pars['d']
    x2 = pars['offsetx']
    y2 = pars['offsety']
    primD = pars['primD']
    secD = pars['secD']

    x1 = -1.0 * x2 * rM
    y1 = -1.0 * y2 * rM

    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    primVis = offsetstar_model(u, v, x1, y1, primobj)
    secVis = offsetstar_model(u, v, x2, y2, secobj)

    # the blackbody flux scaling
    if starspec == []:
        fprim = fprim0 * (wave[0]/wave0)**dprim
        fprimCP = fprim0 * (wave[1]/wave0)**dprim
        fbg = fbg0 * (wave[0]/wave0)**dback
        fbgCP = fbg0 * (wave[1]/wave0)**dback
        fsec = fsec0 * (wave[0]/wave0)**dsec
        fsecCP = fsec0 * (wave[1]/wave0)**dsec
    else:
        fprim = fprim0 * np.interp( wave[0], wavespec, starspec)
        fprimCP = fprim0 * np.interp( wave[1], wavespec, starspec)
        if dback == -4:
            print('good')
            fbg = fbg0 * np.interp( wave[0], wavespec, starspec)
            fbgCP = fbg0 * np.interp( wave[1], wavespec, starspec)
        else:
            fbg = fbg0 * (wave[0]/wave0)**dback
            fbgCP = fbg0 * (wave[1]/wave0)**dback
        if dsec == -4:
            fsec = fsec0 * np.interp( wave[0], wavespec, starspec)
            fsecCP = fsec0 * np.interp( wave[1], wavespec, starspec)
        else:
            fsec = fsec0 * (wave[0]/wave0)**dsec
            fsecCP = fsec0 * (wave[1]/wave0)**dsec


    # the total visibilities and bispectrum -> closure phase
    totalVis = fprim * primVis[0] + fsec * secVis[0]
    totalVis /= fprim + fsec + fbg

    totalVis1 = fprimCP * primVis[1] + fsecCP * secVis[1]
    totalVis1 /= fprimCP + fsecCP + fbgCP

    totalVis2 = fprimCP * primVis[2] + fsecCP * secVis[2]
    totalVis2 /= fprimCP + fsecCP + fbgCP

    totalVis3 = fprimCP * primVis[3] + fsecCP * secVis[3]
    totalVis3 /= fprimCP + fsecCP + fbgCP

    V2 = abs(totalVis)**2

    bispectrum = totalVis1*totalVis2*totalVis3
    CP = np.angle(bispectrum, deg=True)  # ,deg=True)

    return V2, CP


def chi2_modbinbg(pars, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=[], starspec=[]):

    nV2 = len(vis2)
    nCP = len(cp)
    nTot = nV2 + nCP

    V2mod, CPmod = modbinbg_model_chrom(pars, u, v, wave, wave0, wavespec=wavespec, starspec=starspec)

    c2v2 = (vis2 - V2mod)**2
    c2v2 /= vis2err**2
    c2v2 = np.sum(c2v2) / nV2

    c2cp = (cp - CPmod)**2
    c2cp /= cperr**2
    c2cp = np.sum(c2cp) / nCP

    c2tot = nV2 * c2v2 + nCP * c2cp
    c2tot /= nTot

    return c2v2, c2cp, c2tot


def fitModRing(data, initpars, scalepars, fixedpars, labels, nrWalkers=10, target='anonymous', dir='./', acceptance=0.15, steps=1000):

    #
    # fit data of object with modulated Gaussian ring model
    #

    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    wave0 = data['wave0']

    # Disk fitting
    # 1) fit ring_model
    # initialize pars
    parNames = initpars.keys()
    modRing_0 = np.array([])
    modRing_scale = np.array([])
    allpars = fixedpars.copy()
    allpars.update(initpars)
    fitpars = {}
    selectedlabels = []
    parhdu = ''

    for i, parName in enumerate(parNames):
        modRing_0 = np.append(np.array([initpars[parName]]), modRing_0)
        modRing_scale = np.append(np.array([scalepars[parName]]), modRing_scale)
        fitpars[parName] = i
        selectedlabels.append(labels[parName])
        parhdu += parName + '        '

    ndim = len(modRing_0)
    ringobj = fitRoutines.ModulatedGaussianRing(allpars['rD'], allpars['rW'], allpars['c1'], allpars['c2'], allpars['c3'], allpars['s1'], allpars['s2'], allpars['s3'], allpars['PA'], allpars['inc'])
    primobj = fitRoutines.UD(allpars['primD'], X=0.0, Y=0.0)
    secobj = fitRoutines.UD(allpars['secD'], X=0.0, Y=0.0)

    modRing = [np.random.normal(modRing_0[::-1], np.abs(modRing_scale[::-1])) for i in range(nrWalkers)]

    try:
        wavespec, starspec = StarSpectrumSED(target)
    except:
        wavespec, starspec = [], []

    samplerRing = emcee.EnsembleSampler(nrWalkers, ndim, lnprob_binarymodring, args=(u, v, wave, vis2, vis2err, cp, cperr, wave0, ringobj, primobj, secobj, fitpars, fixedpars, wavespec, starspec))

    uuR = samplerRing.run_mcmc(modRing, steps)
    accfrac = samplerRing.acceptance_fraction
    fig = pl.figure()
    pl.hist(accfrac)
    pl.savefig(dir+target+'_acceptancefractions.png')
    print ("Mean,std,min,max acceptance fraction: ", np.mean(accfrac), np.std(accfrac), np.min(accfrac), np.max(accfrac))
    # print "Autocorrelation time: ",samplerRing.acor
    acceptedWalkers = (accfrac > acceptance)
    print (acceptedWalkers.sum(), len(accfrac)-acceptedWalkers.sum())
    # print(accfrac)

    # 2) fit results, error bars

    samplesRing1 = samplerRing.chain[acceptedWalkers, -500:, :].reshape((-1, ndim))
    # samplesRing2 = samplerRing.chain[: , -500:, :].reshape((-1,ndim))

    del samplerRing
    gc.collect()

    try:
        perc = np.transpose(np.percentile(samplesRing1, [16, 50, 84], axis=0))
        indices = np.random.randint(0, samplesRing1.shape[0], 200)

        fig = triangle.corner(samplesRing1, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        fig.savefig(dir+target+'_cornerplot1.png')

        #fig = triangle.corner(samplesRing2, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        #fig.savefig(dir+target+'_cornerplot2.png')
        # fig.close()

        comment = '#' + '        '.join(selectedlabels)
        fileIO.writeArray(perc.T, dir+target+'_quantiles.txt', separator='        ', openingcomment=comment)
        fileIO.writeArray(samplesRing1[indices, :], dir+target+'_samples.txt', separator='   ')

        del samplesRing1
        gc.collect()

        values = perc.T[1, :]
        # print(values)
        values = np.array(values)
        for obj in fixedpars:
            values = np.append(values, np.array(fixedpars[obj]))
            parhdu += obj + '        '

        bestfit = {}
        pars = parhdu.strip()
        pars = pars.split()
        for k in np.arange(len(values)):
            bestfit[pars[k]] = values[k]

        c2v2, c2cp, c2tot = chi2_binarymodring(bestfit, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=wavespec, starspec=starspec)


        f = open(dir+target+'_bestfit.txt', 'w')
        f.write('chi2        chi2v2        chi2cp        ')
        f.write(parhdu + ' \n')
        f.write(str(c2tot)+'        '+str(c2v2)+'        '+str(c2cp)+'        ')
        for obj in values:
            f.write(str(obj)+'        ')
        f.close()

        log('chi2 = '+ '%.2f' % c2tot, dir)
        chain = ''
        for key in initpars.keys():
            chain += '  ' + key + '=' + '%.2f' % allpars[key]
        log(chain, dir)

        modringImage(allpars, 512, 0.3, dir=dir, save=True, name=target)

        PlotBestModelMCMC(data, dir+target+'_bestfit.txt', save=True, name=target, dir=dir)
    except IndexError:
        warn('not enough accepted walkers to create output for'+target+'...\nGo to next star...')


def fitModBinbg(data, initpars, scalepars, fixedpars, labels, nrWalkers=10, target='anonymous', dir='./', acceptance=0.15, steps=1000):

    #
    # fit data of object with modulated Gaussian ring model
    #

    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    wave0 = data['wave0']

    # Disk fitting
    # 1) fit ring_model
    # initialize pars
    parNames = initpars.keys()
    modRing_0 = np.array([])
    modRing_scale = np.array([])
    allpars = fixedpars.copy()
    allpars.update(initpars)
    fitpars = {}
    selectedlabels = []
    parhdu = ''


    for i, parName in enumerate(parNames):
        modRing_0 = np.append(np.array([initpars[parName]]), modRing_0)
        modRing_scale = np.append(np.array([scalepars[parName]]), modRing_scale)
        fitpars[parName] = i
        selectedlabels.append(labels[parName])
        parhdu += parName + '        '

    ndim = len(modRing_0)
    primobj = fitRoutines.UD(allpars['primD'], X=0.0, Y=0.0)
    secobj = fitRoutines.UD(allpars['secD'], X=0.0, Y=0.0)

    modRing = [np.random.normal(modRing_0[::-1], np.abs(modRing_scale[::-1])) for i in range(nrWalkers)]


    try:
        wavespec, starspec = StarSpectrumSED(target)
    except:
        wavespec, starspec = [], []

    samplerRing = emcee.EnsembleSampler(nrWalkers, ndim, lnprob_binarymodbg, args=(u, v, wave, vis2, vis2err, cp, cperr, wave0, primobj, secobj, fitpars, fixedpars, wavespec, starspec))

    uuR = samplerRing.run_mcmc(modRing, steps)
    accfrac = samplerRing.acceptance_fraction
    fig = pl.figure()
    pl.hist(accfrac)
    pl.savefig(dir+target+'_acceptancefractions.png')
    print ("Mean,std,min,max acceptance fraction: ", np.mean(accfrac), np.std(accfrac), np.min(accfrac), np.max(accfrac))
    # print "Autocorrelation time: ",samplerRing.acor
    acceptedWalkers = (accfrac > acceptance)
    print (acceptedWalkers.sum(), len(accfrac)-acceptedWalkers.sum())
    # print(accfrac)

    # 2) fit results, error bars

    samplesRing1 = samplerRing.chain[acceptedWalkers, -500:, :].reshape((-1, ndim))
    # samplesRing2 = samplerRing.chain[: , -500:, :].reshape((-1,ndim))

    del samplerRing
    gc.collect()

    try:
        perc = np.transpose(np.percentile(samplesRing1, [16, 50, 84], axis=0))
        indices = np.random.randint(0, samplesRing1.shape[0], 200)

        fig = triangle.corner(samplesRing1, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        fig.savefig(dir+target+'_cornerplot1.png')

        #fig = triangle.corner(samplesRing2, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        #fig.savefig(dir+target+'_cornerplot2.png')
        # fig.close()

        comment = '#' + '        '.join(selectedlabels)
        fileIO.writeArray(perc.T, dir+target+'_quantiles.txt', separator='        ', openingcomment=comment)
        fileIO.writeArray(samplesRing1[indices, :], dir+target+'_samples.txt', separator='   ')

        del samplesRing1
        gc.collect()

        values = perc.T[1, :]
        # print(values)
        values = np.array(values)
        for obj in fixedpars:
            values = np.append(values, np.array(fixedpars[obj]))
            parhdu += obj + '        '

        bestfit = {}
        pars = parhdu.strip()
        pars = pars.split()
        for k in np.arange(len(values)):
            bestfit[pars[k]] = values[k]

        c2v2, c2cp, c2tot = chi2_modbinbg(bestfit, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=wavespec, starspec=starspec)


        f = open(dir+target+'_bestfit.txt', 'w')
        f.write('chi2        chi2v2        chi2cp        ')
        f.write(parhdu + ' \n')
        f.write(str(c2tot)+'        '+str(c2v2)+'        '+str(c2cp)+'        ')
        for obj in values:
            f.write(str(obj)+'        ')
        f.close()

        log('chi2 = '+ '%.2f' % c2tot, dir)
        chain = ''
        for key in initpars.keys():
            chain += '  ' + key + '=' + '%.2f' % allpars[key]
        log(chain, dir)

        modbinbgImage(allpars, 512, 0.3, dir=dir, save=True, name=target)

        PlotBestModelMCMCbin(data, dir+target+'_bestfit.txt', save=True, name=target, dir=dir)
    except IndexError:
        warn('not enough accepted walkers to create output for'+target+'...\nGo to next star...')


def fitModRingshift(data, initpars, scalepars, fixedpars, labels, nrWalkers=10, target='anonymous', dir='./', acceptance=0.15, steps=1000):

    #
    # fit data of object with modulated Gaussian ring model
    #

    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    wave0 = data['wave0']

    # Disk fitting
    # 1) fit ring_model
    # initialize pars
    parNames = initpars.keys()
    modRing_0 = np.array([])
    modRing_scale = np.array([])
    allpars = fixedpars.copy()
    allpars.update(initpars)
    fitpars = {}
    selectedlabels = []
    parhdu = ''

    for i, parName in enumerate(parNames):
        modRing_0 = np.append(np.array([initpars[parName]]), modRing_0)
        modRing_scale = np.append(np.array([scalepars[parName]]), modRing_scale)
        fitpars[parName] = i
        selectedlabels.append(labels[parName])
        parhdu += parName + '        '

    ndim = len(modRing_0)
    ring1obj = fitRoutines.ModulatedGaussianRingShift(allpars['rD1'], allpars['rW1'], allpars['c11'], allpars['c12'], allpars['c13'], allpars['s11'], allpars['s12'], allpars['s13'], allpars['PA'], allpars['inc'], 0, 0)
    ring2obj = fitRoutines.ModulatedGaussianRingShift(allpars['rD2'], allpars['rW2'], allpars['c21'], allpars['c22'], allpars['c23'], allpars['s21'], allpars['s22'], allpars['s23'], allpars['PA'], allpars['inc'], allpars['xring2'], allpars['yring2'])
    primobj = fitRoutines.UD(allpars['primD'], X=0.0, Y=0.0)
    secobj = fitRoutines.UD(allpars['secD'], X=0.0, Y=0.0)

    modRing = [np.random.normal(modRing_0[::-1], np.abs(modRing_scale[::-1])) for i in range(nrWalkers)]

    try:
        wavespec, starspec = StarSpectrumSED(target)
    except:
        wavespec, starspec = [], []

    samplerRing = emcee.EnsembleSampler(nrWalkers, ndim, lnprob_binarymodringshift, args=(u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, fitpars, fixedpars, wavespec, starspec))

    uuR = samplerRing.run_mcmc(modRing, steps)
    accfrac = samplerRing.acceptance_fraction
    fig = pl.figure()
    pl.hist(accfrac)
    pl.savefig(dir+target+'_acceptancefractions.png')
    print ("Mean,std,min,max acceptance fraction: ", np.mean(accfrac), np.std(accfrac), np.min(accfrac), np.max(accfrac))
    # print "Autocorrelation time: ",samplerRing.acor
    acceptedWalkers = (accfrac > acceptance)
    print (acceptedWalkers.sum(), len(accfrac)-acceptedWalkers.sum())
    # print(accfrac)

    # 2) fit results, error bars

    samplesRing1 = samplerRing.chain[acceptedWalkers, -500:, :].reshape((-1, ndim))
    # samplesRing2 = samplerRing.chain[: , -500:, :].reshape((-1,ndim))

    del samplerRing
    gc.collect()

    try:
        perc = np.transpose(np.percentile(samplesRing1, [16, 50, 84], axis=0))
        indices = np.random.randint(0, samplesRing1.shape[0], 200)

        fig = triangle.corner(samplesRing1, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        fig.savefig(dir+target+'_cornerplot1.png')

        #fig = triangle.corner(samplesRing2, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        #fig.savefig(dir+target+'_cornerplot2.png')
        # fig.close()

        comment = '#' + '        '.join(selectedlabels)
        fileIO.writeArray(perc.T, dir+target+'_quantiles.txt', separator='        ', openingcomment=comment)
        fileIO.writeArray(samplesRing1[indices, :], dir+target+'_samples.txt', separator='   ')

        del samplesRing1
        gc.collect()

        values = perc.T[1, :]
        # print(values)
        values = np.array(values)
        for obj in fixedpars:
            values = np.append(values, np.array(fixedpars[obj]))
            parhdu += obj + '        '

        bestfit = {}
        pars = parhdu.strip()
        pars = pars.split()
        for k in np.arange(len(values)):
            bestfit[pars[k]] = values[k]

        c2v2, c2cp, c2tot = chi2_binarymodringshift(bestfit, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=wavespec, starspec=starspec)


        f = open(dir+target+'_bestfit.txt', 'w')
        f.write('chi2        chi2v2        chi2cp        ')
        f.write(parhdu + ' \n')
        f.write(str(c2tot)+'        '+str(c2v2)+'        '+str(c2cp)+'        ')
        for obj in values:
            f.write(str(obj)+'        ')
        f.close()

        log('chi2 = '+ '%.2f' % c2tot, dir)
        chain = ''
        for key in initpars.keys():
            chain += '  ' + key + '=' + '%.2f' % allpars[key]
        log(chain, dir)

        modringshiftImage(allpars, 512, 0.3, dir=dir, save=True, name=target)

        PlotBestModelMCMCshift(data, dir+target+'_bestfit.txt', save=True, name=target, dir=dir)
    except IndexError:
        warn('not enough accepted walkers to create output for'+target+'...\nGo to next star...')


def fitModRingshift2(data, initpars, scalepars, fixedpars, labels, nrWalkers=10, target='anonymous', dir='./', acceptance=0.15, steps=1000):

    #
    # fit data of object with modulated Gaussian ring model
    #

    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    vis2, vis2err = data['v2']
    cp, cperr = data['cp']
    wave0 = data['wave0']

    # Disk fitting
    # 1) fit ring_model
    # initialize pars
    parNames = initpars.keys()
    modRing_0 = np.array([])
    modRing_scale = np.array([])
    allpars = fixedpars.copy()
    allpars.update(initpars)
    fitpars = {}
    selectedlabels = []
    parhdu = ''

    for i, parName in enumerate(parNames):
        modRing_0 = np.append(np.array([initpars[parName]]), modRing_0)
        modRing_scale = np.append(np.array([scalepars[parName]]), modRing_scale)
        fitpars[parName] = i
        selectedlabels.append(labels[parName])
        parhdu += parName + '        '

    ndim = len(modRing_0)
    ring1obj = fitRoutines.ModulatedGaussianRingShift(allpars['rD1'], allpars['rW1'], allpars['c11'], allpars['c12'], allpars['c13'], allpars['s11'], allpars['s12'], allpars['s13'], allpars['PA'], allpars['inc'], 0, 0)
    xring = allpars['Rring'] * np.sin((allpars['PA']-90)*np.pi/180.)
    yring = allpars['Rring'] * np.cos((allpars['PA']-90)*np.pi/180.)
    ring2obj = fitRoutines.ModulatedGaussianRingShift(allpars['rD2'], allpars['rW2'], allpars['c21'], allpars['c22'], allpars['c23'], allpars['s21'], allpars['s22'], allpars['s23'], allpars['PA'], allpars['inc'], xring, yring)
    primobj = fitRoutines.UD(allpars['primD'], X=0.0, Y=0.0)
    secobj = fitRoutines.UD(allpars['secD'], X=0.0, Y=0.0)

    modRing = [np.random.normal(modRing_0[::-1], np.abs(modRing_scale[::-1])) for i in range(nrWalkers)]

    try:
        wavespec, starspec = StarSpectrumSED(target)
    except:
        wavespec, starspec = [], []

    samplerRing = emcee.EnsembleSampler(nrWalkers, ndim, lnprob_binarymodringshift2, args=(u, v, wave, vis2, vis2err, cp, cperr, wave0, ring1obj, ring2obj, primobj, secobj, fitpars, fixedpars, wavespec, starspec))

    uuR = samplerRing.run_mcmc(modRing, steps)
    accfrac = samplerRing.acceptance_fraction
    fig = pl.figure()
    pl.hist(accfrac)
    pl.savefig(dir+target+'_acceptancefractions.png')
    print ("Mean,std,min,max acceptance fraction: ", np.mean(accfrac), np.std(accfrac), np.min(accfrac), np.max(accfrac))
    # print "Autocorrelation time: ",samplerRing.acor
    acceptedWalkers = (accfrac > acceptance)
    print (acceptedWalkers.sum(), len(accfrac)-acceptedWalkers.sum())
    # print(accfrac)

    # 2) fit results, error bars

    samplesRing1 = samplerRing.chain[acceptedWalkers, -500:, :].reshape((-1, ndim))
    # samplesRing2 = samplerRing.chain[: , -500:, :].reshape((-1,ndim))

    del samplerRing
    gc.collect()

    try:
        perc = np.transpose(np.percentile(samplesRing1, [16, 50, 84], axis=0))
        indices = np.random.randint(0, samplesRing1.shape[0], 200)

        fig = triangle.corner(samplesRing1, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        fig.savefig(dir+target+'_cornerplot1.png')

        #fig = triangle.corner(samplesRing2, labels=selectedlabels, quantiles=[0.16, 0.5, 0.84])
        #fig.savefig(dir+target+'_cornerplot2.png')
        # fig.close()

        comment = '#' + '        '.join(selectedlabels)
        fileIO.writeArray(perc.T, dir+target+'_quantiles.txt', separator='        ', openingcomment=comment)
        fileIO.writeArray(samplesRing1[indices, :], dir+target+'_samples.txt', separator='   ')

        del samplesRing1
        gc.collect()

        values = perc.T[1, :]
        # print(values)
        values = np.array(values)
        for obj in fixedpars:
            values = np.append(values, np.array(fixedpars[obj]))
            parhdu += obj + '        '

        bestfit = {}
        pars = parhdu.strip()
        pars = pars.split()
        for k in np.arange(len(values)):
            bestfit[pars[k]] = values[k]

        c2v2, c2cp, c2tot = chi2_binarymodringshift2(bestfit, u, v, wave, vis2, vis2err, cp, cperr, wave0, wavespec=wavespec, starspec=starspec)


        f = open(dir+target+'_bestfit.txt', 'w')
        f.write('chi2        chi2v2        chi2cp        ')
        f.write(parhdu + ' \n')
        f.write(str(c2tot)+'        '+str(c2v2)+'        '+str(c2cp)+'        ')
        for obj in values:
            f.write(str(obj)+'        ')
        f.close()

        log('chi2 = '+ '%.2f' % c2tot, dir)
        chain = ''
        for key in initpars.keys():
            chain += '  ' + key + '=' + '%.2f' % allpars[key]
        log(chain, dir)

        modringshift2Image(allpars, 512, 0.3, dir=dir, save=True, name=target)

        PlotBestModelMCMCshift2(data, dir+target+'_bestfit.txt', save=True, name=target, dir=dir)
    except IndexError:
        warn('not enough accepted walkers to create output for'+target+'...\nGo to next star...')
