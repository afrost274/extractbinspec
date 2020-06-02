# File to read GRAVITY oifits
import numpy as np
import JKmodRingFunctions as mrf
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import fnmatch
import math
import gc
import scipy
from synphot import observation #pysynphot
from synphot import spectrum
from scipy.optimize import curve_fit
#from PyAstronomy import pyasl


def shiftSpec(x, shift):

    flux = x[0]
    wave = x[1]
    wave0 = x[2]
    fl, wl = pyasl.dopplerShift(wave, flux, shift, edgeHandling="firstlast")
    flux = np.interp(wave0, wl, fl)
    return flux


def shiftScience2Template(fluxSCI, fluxCAL, fluxTEMP, waveSCI, waveTEMP):

    popt, pcov = curve_fit(shiftSpec, (fluxCAL, waveSCI, waveTEMP), fluxTEMP, p0=[60], bounds=(-500, 500), sigma = 1/(waveSCI>2.28e-6)+1e-16)
    print ('Need to shift by {}km/s'.format(popt[0]))
    fluxCAL, wave = pyasl.dopplerShift(waveSCI, fluxCAL, popt[0], edgeHandling="firstlast")
    fluxSCI, wave = pyasl.dopplerShift(waveSCI, fluxSCI, popt[0], edgeHandling="firstlast")

    return wave, fluxCAL, fluxSCI


def rebin_spec(wave, specin, wavnew):
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='micron')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux


def readIRTFTemplate(file):
    hdul = fits.open(file)
    stuff = hdul[0].data
    flux = stuff[1]
    wave = stuff[0]
    #fig, ax = plt.subplots()
    #ax.plot(wave, flux)
    #plt.show()
    #plt.close()
    return flux, wave


def FluxPlot(fileSCI, fileCAL):

    dataSCI, dataSCSCI, dataFTSCI = readGRAVITY(fileSCI)
    dataCAL, dataSCCAL, dataFTCAL = readGRAVITY(fileCAL)

    FluxSCI = dataSCSCI['flux']
    eFluxSCI = dataSCSCI['fluxerr']
    Wave = dataSCSCI['fluxwave']

    FluxCAL = dataSCCAL['flux']
    eFluxCAL = dataSCCAL['fluxerr']
    Wave2 = dataSCCAL['fluxwave']

    #if Wave == Wave2:
    Flux = FluxSCI/FluxCAL
    Flux = Flux.reshape(8,210)
    Flux = np.median(Flux, axis=0)
    Wave = Wave.reshape(8, 210)
    Wave = np.median(Wave, axis=0)
    print(len(Flux))
    #else:
    #    mrf.warn('Not the same spectra...')
    #    Flux = FluxSCI/FluxCAL

    fig, ax = plt.subplots()
    ax.plot(Wave, Flux)
    ax.scatter(Wave, Flux, c=Wave, cmap='gist_rainbow_r')
    ax.set_xlim(2.0e-6,2.4e-6)
    fig.tight_layout()
    plt.savefig('Flux.pdf')
    plt.show()
    plt.close()


def ReadFilesGRAVITY(dir, files):

    listOfFiles = os.listdir(dir)
    pattern = files
    i = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            i += 1
            print ('Reading '+entry+'...')
            if i == 1:
                data, dataSC, dataFT = readGRAVITY(dir+entry)
            else:
                datatmp, dataSCtmp, dataFTtmp = readGRAVITY(dir+entry)
                # Appending all the stuff together
                data['nB'], data['nW'], data['nCP'] = data['nB']+datatmp['nB'], data['nW']+datatmp['nW'], data['nCP']+datatmp['nCP']
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
                # Visibility amp
                uvt = datatmp['uvis']
                vvt = datatmp['vvis']
                uv = data['uvis']
                vv = data['vvis']
                uv = np.append(uv, uvt)
                vv = np.append(vv, vvt)
                data['uvis'] = uv
                data['vvis'] = vv
                lvt = datatmp['vwave']
                lv = data['vwave']
                lv = np.append(lv, lvt)
                data['vwave'] = lv
                vt, vet = datatmp['visamp']
                v, ve = data['visamp']
                v = np.append(v, vt)
                ve = np.append(ve, vet)
                data['visamp'] = (v, ve)
                # Visibility phase
                vpt, vpet = datatmp['visphi']
                vp, vpe = data['visphi']
                vp = np.append(vp, vpt)
                vpe = np.append(vpe, vpet)
                data['visphi'] = (vp, vpe)
                # closure phases
                cpt, cpet = datatmp['cp']
                cp, cpe = data['cp']
                cp = np.append(cp, cpt)
                cpe = np.append(cpe, cpet)
                data['cp'] = (cp, cpe)
                # fluxes
                fluxt = datatmp['flux']
                fluxet = datatmp['fluxerr']
                wavet = datatmp['fluxwave']
                flux, wave, fluxe = data['flux'], data['fluxwave'], data['fluxerr']
                fluxt = np.interp(wave, wavet, fluxt)
                fluxet = np.interp(wave, wavet, fluxet)
                #fluxt = list([fluxt])
                #fluxet = list([fluxet])
                #wavet = list(wavet)
                flux = list([flux])
                fluxe = list([fluxe])
                wave = list(wave)
                flux.append( fluxt)
                fluxe.append( fluxet)
                flux = np.array(flux)
                fluxe = np.array(fluxe)
                flux = np.median(flux, axis=0)
                fluxe = np.median(fluxe, axis=0)
                #fig, ax = plt.subplots()
                #ax.plot(wave, flux)
                #plt.show()
                #plt.close()
                data['flux'] = flux
                data['fluxerr'] = fluxe
                data['fluxwave'] = wave

    return data


def readGRAVITY(file):

    mrf.inform2('Opening the following file: '+file)
    dataSC = {}  # dicoinit()
    dataFT = {}  # dicoinit()
    hdul = fits.open(file)
    err = False
    i = 0
    while err == False:
        i += 1
        try:
            extname = hdul[i].header['EXTNAME']
            print ('Reading '+extname)
        #if extname == 'OI_ARRAY':
        #    if insname == 'GRAVITY_SC':
        #        dataSC = readARRAY(hdul[i], dataSC)
        #    else if insname == 'GRAVITY_FT':
        #        dataFT = readARRAY(hdul[i], dataFT)
        #else if extname == 'OI_TARGET':
        #    if insname == 'GRAVITY_SC':
        #        dataSC = readTARGET(hdul[i], dataSC)
        #    else if insname == 'GRAVITY_FT':
        #        dataFT = readTARGET(hdul[i], dataFT)
            if extname == 'OI_WAVELENGTH':
                insname = hdul[i].header['INSNAME']
                if insname[:10] == 'GRAVITY_SC':
                    dataSC = readWAVE(hdul[i], dataSC)
                elif insname[:10] == 'GRAVITY_FT':
                    dataFT = readWAVE(hdul[i], dataFT)
            elif extname == 'OI_VIS':
                insname = hdul[i].header['INSNAME']
                if insname[:10] == 'GRAVITY_SC':
                    dataSC = readVIS(hdul[i], dataSC)
                elif insname[:10] == 'GRAVITY_FT':
                    dataFT = readVIS(hdul[i], dataFT)
            elif extname == 'OI_VIS2':
                insname = hdul[i].header['INSNAME']
                if insname[:10] == 'GRAVITY_SC':
                    dataSC = readVIS2(hdul[i], dataSC)
                elif insname[:10] == 'GRAVITY_FT':
                    dataFT = readVIS2(hdul[i], dataFT)
            elif extname == 'OI_T3':
                insname = hdul[i].header['INSNAME']
                if insname[:10] == 'GRAVITY_SC':
                    dataSC = readT3(hdul[i], dataSC)
                elif insname[:10] == 'GRAVITY_FT':
                    dataFT = readT3(hdul[i], dataFT)
            elif extname == 'OI_FLUX':
                insname = hdul[i].header['INSNAME']
                if insname[:10] == 'GRAVITY_SC':
                    dataSC = readFLUX(hdul[i], dataSC)
                elif insname[:10] == 'GRAVITY_FT':
                    dataFT = readFLUX(hdul[i], dataFT)
        except IndexError:
            err = True

    data = {}
    data['u'] = (dataSC['u'], dataSC['u1'], dataSC['u2'], dataSC['u3'])
    data['v'] = (dataSC['v'], dataSC['v1'], dataSC['v2'], dataSC['v3'])
    data['wave'] = (dataSC['v2wave'], dataSC['wavecp'])
    data['v2'] = (dataSC['vis2'], dataSC['vis2err'])
    data['cp'] = (dataSC['t3'], dataSC['t3err'])
    data['flux'] = dataSC['flux']
    data['fluxerr'] = dataSC['fluxerr']
    data['fluxwave'] = dataSC['fluxwave']
    data['visamp'] = (dataSC['visamp'], dataSC['visamperr'])
    data['visphi'] = (dataSC['visphi'], dataSC['visphierr'])
    data['uvis'] = dataSC['uvis']
    data['vvis'] = dataSC['vvis']
    data['vwave'] = dataSC['vwave']

    nV2 = dataFT['vis2'].size
    nCP = dataFT['t3'].size
    nWave = dataFT['effwave'].size
    maxW = np.amax(dataFT['effwave'])
    minW = np.amin(dataFT['effwave'])
    mrf.inform2('Fringe tracker:')
    mrf.inform('Number of V2: {}    Number of CP: {}'.format(nV2, nCP))
    mrf.inform('{0} channels from {1:.2e}m to {2:.2e}m'.format(nWave, minW, maxW))

    nV2 = dataSC['vis2'].size
    nCP = dataSC['t3'].size
    nWave = dataSC['effwave'].size
    maxW = np.amax(dataSC['effwave'])
    minW = np.amin(dataSC['effwave'])
    mrf.inform2('Science:')
    mrf.inform('Number of V2: {}    Number of CP: {}'.format(nV2, nCP))
    mrf.inform('{0} channels from {1:.2e}m to {2:.2e}m'.format(nWave, minW, maxW))

    data['nB'], data['nW'], data['nCP'] = nV2/nWave, nWave, nCP

    return data, dataSC, dataFT


def readPIinfoMulti(dir, files):

    listOfFiles = os.listdir(dir)
    pattern = files
    i = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            i += 1
            print ('Reading '+entry+'...')
            if i == 1:
                data = readPIinfo(dir+entry)
            else:
                datatmp = readPIinfo(dir+entry)
                # Appending all the stuff together
                # Starting with u coordinates
                prog2 = datatmp['prog']
                prog = data['prog']
                data['prog'] = np.array(np.append(prog, prog2))
                date2 = datatmp['date']
                date = data['date']
                data['date'] = np.array(np.append(date, date2))
                MJD2 = datatmp['MJD']
                MJD = data['MJD']
                data['MJD'] = np.array(np.append(MJD, MJD2))
                AT12 = datatmp['AT1']
                AT1 = data['AT1']
                data['AT1'] = np.array(np.append(AT1, AT12))
                AT22 = datatmp['AT2']
                AT2 = data['AT2']
                data['AT2'] = np.append(AT2, AT22)
                AT32 = datatmp['AT3']
                AT3 = data['AT3']
                data['AT3'] = np.append(AT3, AT32)
                AT42 = datatmp['AT4']
                AT4 = data['AT4']
                data['AT4'] = np.append(AT4, AT42)

    return data



def readPIinfoMultiSimple(dir, files):

    listOfFiles = os.listdir(dir)
    pattern = files
    i = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            print(entry)
            i += 1
            print ('Reading '+entry+'...')
            if i == 1:
                data = readPIinfoSimple(dir+entry)
            else:
                datatmp = readPIinfoSimple(dir+entry)
                # Appending all the stuff together
                # Starting with u coordinates
                date2 = datatmp['date']
                date = data['date']
                data['date'] = np.array(np.append(date, date2))
                MJD2 = datatmp['MJD']
                MJD = data['MJD']
                data['MJD'] = np.array(np.append(MJD, MJD2))
                AT12 = datatmp['AT1']
                AT1 = data['AT1']
                data['AT1'] = np.array(np.append(AT1, AT12))
                AT22 = datatmp['AT2']
                AT2 = data['AT2']
                data['AT2'] = np.append(AT2, AT22)
                AT32 = datatmp['AT3']
                AT3 = data['AT3']
                data['AT3'] = np.append(AT3, AT32)
                AT42 = datatmp['AT4']
                AT4 = data['AT4']
                data['AT4'] = np.append(AT4, AT42)

    return data


def readPIinfo(file):

    mrf.inform2('Opening the following file: '+file)
    data = {}  # dicoinit()
    hdul = fits.open(file)
    i = 0

    data = {}
    data['prog'] = np.array(hdul[i].header['HIERARCH ESO OBS PROG ID'])
    data['date'] = np.array(hdul[i].header['DATE-OBS'])
    data['MJD'] = np.array(hdul[i].header['MJD-OBS'])
    data['AT1'] = np.array(hdul[i].header['HIERARCH ESO ISS CONF STATION1'])
    data['AT2'] = np.array(hdul[i].header['HIERARCH ESO ISS CONF STATION2'])
    data['AT3'] = np.array(hdul[i].header['HIERARCH ESO ISS CONF STATION3'])
    try:
        data['AT4'] = np.array(hdul[i].header['HIERARCH ESO ISS CONF STATION4'])
    except:
        data['AT4'] = np.array('')
    return data


def readPIinfoSimple(file):

    mrf.inform2('Opening the following file: '+file)
    data = {}  # dicoinit()
    hdul = fits.open(file)
    i = 0
    err = False
    i = 0
    while err == False:
        i += 1
        try:
            extname = hdul[i].header['EXTNAME']
            print ('Reading '+extname)
            if extname == 'OI_VIS2':
                data['date'] = hdul[i].header['DATE-OBS']
                data['MJD'] = hdul[i].data['MJD'][0]
            elif extname == 'OI_ARRAY':
                stations = hdul[i].data['STA_NAME']
                data['AT1'] = stations[0]
                data['AT2'] = stations[1]
                data['AT3'] = stations[2]
                try:
                    data['AT4'] = stations[3]
                except:
                    data['AT4'] = np.array('')
        except IndexError:
            err = True

    return data


def dicoinit():

    data = {}
    data['u'] = (np.array([]), np.array([]), np.array([]), np.array([]))
    data['v'] = (np.array([]), np.array([]), np.array([]), np.array([]))
    data['wave'] = (np.array([]), np.array([]))
    data['v2'] = (np.array([]), np.array([]))
    data['cp'] = (np.array([]), np.array([]))

    return data


def readARRAY(hdul, data):

    dat = {}

    return data

def readTARGET(hdul, data):

    dat = {}

    return data

def readWAVE(hdul, data):

    data['effwave'] = np.array(hdul.data['EFF_WAVE'])

    return data

def readVIS(hdul, data):

    wav = data['effwave']
    flag = hdul.data['FLAG']
    visamp, visamperr, visphi, visphierr, uvis, vvis, wave = [], [], [], [], [], [], []
    for i in range(len(hdul.data['VISAMP'])):
        VISAMP = hdul.data['VISAMP'][i]
        VISPHI = hdul.data['VISPHI'][i]
        VISAMPERR = hdul.data['VISAMPERR'][i]
        VISPHIERR = hdul.data['VISPHIERR'][i]
        UVIS = hdul.data['UCOORD'][i]
        VVIS = hdul.data['VCOORD'][i]
        for j in range(len(VISAMP)):
            if flag[i][j] == False:
                visamp = np.append(visamp, VISAMP[j])
                visphi = np.append(visphi, VISPHI[j])
                visamperr = np.append(visamperr, VISAMPERR[j])
                visphierr = np.append(visphierr, VISPHIERR[j])
                uvis = np.append(uvis, UVIS/wav[j])
                vvis = np.append(vvis, VVIS/wav[j])
                wave = np.append(wave, wav[j])

    visamp = np.array(visamp)
    visphi = np.array(visphi)
    visamperr = np.array(visamperr)
    visphierr = np.array(visphierr)
    uvis = np.array(uvis)
    vvis = np.array(vvis)
    wave = np.array(wave)

    data['visamp'] = visamp
    data['visamperr'] = visamperr
    data['visphi'] = visphi
    data['visphierr'] = visphierr
    data['uvis'] = uvis
    data['vvis'] = vvis
    data['vwave'] = wave

    return data


def readVIS2(hdul, data):

    wav = data['effwave']
    flag = hdul.data['FLAG']
    vis2, vis2err, uvis, vvis, wave = [], [], [], [], []
    for i in range(len(hdul.data['VIS2DATA'])):
        VIS2 = hdul.data['VIS2DATA'][i]
        VIS2ERR = hdul.data['VIS2ERR'][i]
        UVIS = hdul.data['UCOORD'][i]
        VVIS = hdul.data['VCOORD'][i]
        for j in range(len(VIS2)):
            if flag[i][j] == False:
                vis2 = np.append(vis2, VIS2[j])
                vis2err = np.append(vis2err, VIS2ERR[j])
                uvis = np.append(uvis, UVIS/wav[j])
                vvis = np.append(vvis, VVIS/wav[j])
                wave = np.append(wave, wav[j])

    vis2 = np.array(vis2)
    vis2err = np.array(vis2err)
    uvis = np.array(uvis)
    vvis = np.array(vvis)
    wave = np.array(wave)

    data['vis2'] = vis2
    data['vis2err'] = vis2err
    data['u'] = uvis
    data['v'] = vvis
    data['v2wave'] = wave

    return data



def readT3(hdul, data):

    wav = data['effwave']
    flag = hdul.data['FLAG']
    t3, t3err, u1, u2, u3, v1, v2, v3, wavecp = [], [], [], [], [], [], [], [], []
    for i in range(len(hdul.data['T3PHI'])):
        T3 = hdul.data['T3PHI'][i]
        T3ERR = hdul.data['T3PHIERR'][i]
        U1 = hdul.data['U1COORD'][i]
        V1 = hdul.data['V1COORD'][i]
        U2 = hdul.data['U2COORD'][i]
        V2 = hdul.data['V2COORD'][i]
        for j in range(len(T3)):
            if flag[i][j] == False:
                t3 = np.append(t3, T3[j])
                t3err = np.append(t3err, T3ERR[j])
                u1 = np.append(u1, U1/wav[j])
                v1 = np.append(v1, V1/wav[j])
                u2 = np.append(u2, U2/wav[j])
                v2 = np.append(v2, V2/wav[j])
                u3 = np.append(u3, (U1+U2)/wav[j])
                v3 = np.append(v3, (V2+V2)/wav[j])
                wavecp = np.append(wavecp, wav[j])

    t3 = np.array(t3)
    t3err = np.array(t3err)
    u1 = np.array(u1)
    v1 = np.array(v1)
    u2 = np.array(u2)
    v2 = np.array(v2)
    u3 = np.array(u3)
    v3 = np.array(v3)
    wavecp = np.array(wavecp)

    data['t3'] = t3
    data['t3err'] = t3err
    data['u1'] = u1
    data['v1'] = v1
    data['u2'] = u2
    data['v2'] = v2
    data['u3'] = u3
    data['v3'] = v3
    data['wavecp'] = wavecp

    return data


def multflux(flux, k):
    return k*flux


def NormalizeFlux(fluxCAL, fluxSCI, fluxerrSCI):
    popt, pcov = curve_fit(multflux, fluxCAL, fluxSCI, p0=[1], sigma = fluxerrSCI, bounds=(0,np.inf))

    return multflux(fluxCAL, popt)


def readFLUX(hdul, data):

    wav = data['effwave']
    flag = hdul.data['FLAG']
    flux, fluxerr, wave = [], [], []

    for i in range(len(hdul.data['FLUX'])):
        FLUX = hdul.data['FLUX'][i]
        FLUXERR = hdul.data['FLUXERR'][i]
        medferr = np.median(FLUXERR)
        fluxp, fluxperr, wavep = [], [], []
        for j in range(len(FLUX)):
            if flag[i][j] == False and FLUXERR[j] < 2*medferr:
                fluxp.append(FLUX[j])
                fluxperr.append(FLUXERR[j])
                wavep.append( wav[j])
        fluxp2 = np.interp(wav, wavep, fluxp)
        fluxperr2 = np.interp(wav, wavep, fluxperr)
        flux.append(fluxp2)
        fluxerr.append(fluxperr2)

    #flux = np.array(flux)
    #fluxerr = np.array(fluxerr)
    #wave = np.array(wav)

    flux = np.median(flux, axis=0)
    fluxerr = np.median(fluxerr, axis=0)

    # print flux
    #print np.mean(flux, axis=1)

    data['flux'] = flux
    data['fluxerr'] = fluxerr
    data['fluxwave'] = wav

    return data
