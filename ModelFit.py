import numpy as np
import JKmodRingFunctions as mrf
import readGravity as rg
import fitRoutines
import matplotlib.pyplot as plt
import copy


def Bin2V2CP(u, v, x, y, primD, secD,  frv2, frcp):

    # Tsec = np.array([pars['Tsec']])
    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    primVis = mrf.offsetstar_model(u, v, 0, 0, primobj)
    secVis = mrf.offsetstar_model(u, v, x, y, secobj)

    # the total visibilities and bispectrum -> closure phase
    totalVis = frv2 * primVis[0] + (1-frv2) * secVis[0]

    totalVis1 = frcp * primVis[1] + (1-frcp) * secVis[1]
    totalVis2 = frcp * primVis[2] + (1-frcp) * secVis[2]
    totalVis3 = frcp * primVis[3] + (1-frcp) * secVis[3]

    V2 = abs(totalVis)**2

    bispectrum = totalVis1*totalVis2*totalVis3
    CP = np.angle(bispectrum, deg=True)  # ,deg=True)

    return V2, CP

def Bin2VisPhi(u, v, wave, x, y, primD, secD,  frv):

    # Tsec = np.array([pars['Tsec']])
    primobj = fitRoutines.UD(primD, X=0.0, Y=0.0)
    secobj = fitRoutines.UD(secD, X=0.0, Y=0.0)

    primVis = mrf.offsetstar_modelVis(u, v, 0, 0, primobj)
    secVis = mrf.offsetstar_modelVis(u, v, x, y, secobj)

    #print(primVis, secVis)
    #fig, ax = plt.subplots()
    #ax.plot(np.real(primVis), np.imag(primVis), '+')
    #ax.plot(np.real(secVis), np.imag(secVis), 'o')
    #plt.show()
    # the total visibilities and bispectrum -> closure phase

    totalVis = frv * primVis[0] + (1-frv) * secVis[0]

    #totalVis1 = frcp * primVis[1] + (1-frcp) * secVis[1]
    #totalVis2 = frcp * primVis[2] + (1-frcp) * secVis[2]
    #totalVis3 = frcp * primVis[3] + (1-frcp) * secVis[3]

    Vis = abs(totalVis)
    Phi = np.arctan2(np.imag(totalVis), np.real(totalVis))*180/np.pi
    #print(len(Phi))
    Vis, Phi, wave = CutVis(Vis, Phi, u, v, wave)

    #fig, ax = plt.subplots()

    #for i in np.arange(12):
    #    ax.plot(wave[i], Phi[i], '-')
    #plt.show()

    nbase = len(Vis)
    dPhi = []
    for i in np.arange(nbase):
        dPhiB = diffPhi(Phi[i], wave[i])
        #dVisB = diffVis(Vis[i])
        dPhi.extend(dPhiB)

    #dPhi = np.array(dPhi)
    #print(dPhi.shape)
    dVis = np.array(Vis)
    #print(dVis.shape)

    return FlattenList(dVis), dPhi


def invlist(my_list):
    return [ x**-1 for x in my_list ]


def diffPhi(Phi, wave):
    #k = invlist(wave)
    wave = np.array(wave)
    k = wave**-1
    y = Phi
    x = k
    a = np.sum(y-np.average(y)) * np.sum(x-np.average(x)) / (np.sum(x-np.average(x)))**2
    b = np.average(y)

    dPhi = Phi - a*x -b
    #fig, ax = plt.subplots()
    #ax.plot(wave, Phi, '-')
    #ax.plot(wave, a*x+b, ':')
    #ax.plot(wave, dPhi, '--')
    #plt.show()

    return dPhi

#def BoundsDef(nwave):

#    boundsmin = np.array([-20, -20])
#    boundsmax = np.array([20, 20])
#    frmin = np.zeros(nwave)
#    frmax = np.ones(nwave)

#    boundsmin = np.append(boundsmin, frmin)
#    boundsmax = np.append(boundsmax, frmax)

#    return boundsmin, boundsmax


def BoundsDef(nwave):

    boundsmin = np.array([-10, -10])
    boundsmax = np.array([10, 10])
    frmin = np.zeros(nwave)
    frmax = np.ones(nwave)
    boundsmin = np.append(boundsmin, frmin)
    boundsmax = np.append(boundsmax, frmax)

    bounds = []
    for i in np.arange(nwave+2):
        bounds.append((boundsmin[i], boundsmax[i]))



    return bounds


def GenerateDataVector(data):

    V2, V2err = data['v2']
    CP, CPerr = data['cp']

    datay = V2
    datay = np.append(V2, CP)

    datayerr = V2err
    datayerr = np.append(V2err, CPerr)

    return datay, datayerr


def GenerateDataVectorVis(data):

    V2, V2err = data['v2']
    CP, CPerr = data['cp']
    Vis, Viserr = data['visamp']
    Phi, Phierr = data['visphi']

    datay = V2
    datay = np.append(V2, CP)
    datay = np.append(datay, Vis)
    datay = np.append(datay, Phi)

    datayerr = V2err
    datayerr = np.append(V2err, CPerr)
    datayerr = np.append(datayerr, Viserr)
    datayerr = np.append(datayerr, Phierr)

    return datay, datayerr


def DataUnpack(data):

    # Unpacking the data
    u = data['u']
    v = data['v']
    wave = data['wave']
    # vis2, vis2err = data['v2']
    # cp, cperr = data['cp']
    wavev2, wavecp = wave

    return u, v, wavev2, wavecp


def DataUnpackVis(data):

    # Unpacking the data
    u = data['uvis']
    v = data['vvis']
    wave = data['vwave']

    return u, v, wave


def ParamUnpack(params):

    # unpacking parameters
    xsec = params[0]
    ysec = params[1]
    fratios = params[2:]
    primD = 0.001
    secD = 0.001

    return xsec, ysec, fratios, primD, secD


def GenParamVector(x, y, fr, nwave):

    params = np.array([x, y])
    fratios = np.ones((nwave))
    fratios *= fr
    params = np.append(params, fratios)

    return params


def ModelFitBin(data, params):

    u, v, wavev2, wavecp = DataUnpack(data)

    xsec, ysec, fratios, primD, secD = ParamUnpack(params)

    # interpolating the spectrum to data wavelengths
    wavefr = np.linspace(min(wavev2), max(wavev2), num=len(fratios))

    frv2 = np.interp(wavev2, wavefr, fratios)
    frcp = np.interp(wavecp, wavefr, fratios)

    V2, CP = Bin2V2CP(u, v, xsec, ysec, primD, secD, frv2, frcp)

    ymodel = V2
    ymodel = np.append(ymodel, CP)

    return ymodel


def ModelFitBinVis(data, params):

    u, v, wavev2, wavecp = DataUnpack(data)
    uv, vv, wavev = DataUnpackVis(data)

    xsec, ysec, fratios, primD, secD = ParamUnpack(params)

    # interpolating the spectrum to data wavelengths
    wavefr = np.linspace(min(wavev2), max(wavev2), num=len(fratios))

    frv2 = np.interp(wavev2, wavefr, fratios)
    frcp = np.interp(wavecp, wavefr, fratios)
    frv = np.interp(wavev, wavefr, fratios)

    V2, CP = Bin2V2CP(u, v, xsec, ysec, primD, secD, frv2, frcp)
    Vis, Phi = Bin2VisPhi(uv, vv, wavev, xsec, ysec, primD, secD, frv)

    ymodel = V2
    ymodel = np.append(ymodel, CP)
    ymodel = np.append(ymodel, Vis)
    ymodel = np.append(ymodel, Phi)

    return ymodel


def GenerateData(data, params):

    model = copy.deepcopy(data)
    u, v, wavev2, wavecp = DataUnpack(data)

    xsec, ysec, fratios, primD, secD = ParamUnpack(params)

    # interpolating the spectrum to data wavelengths
    wavefr = np.linspace(min(wavev2), max(wavev2), num=len(fratios))

    frv2 = np.interp(wavev2, wavefr, fratios)
    frcp = np.interp(wavecp, wavefr, fratios)

    V2, CP = Bin2V2CP(u, v, xsec, ysec, primD, secD,  frv2, frcp)

    A, V2err = data['v2']
    B, CPerr = data['cp']

    model['v2'] = V2, V2err
    model['cp'] = CP, CPerr

    return model


def GiveDataValues(data):

    V2, V2err = data['v2']
    CP, CPerr = data['cp']

    return V2, V2err, CP, CPerr

def GiveDataValuesVis(data):

    Vis, Viserr = data['visamp']
    Phi, Phierr = data['visphi']

    return Vis, Viserr, Phi, Phierr


def CreateWaveGrid(wavev2, nfr):
    wavefr = np.linspace(min(wavev2), max(wavev2), num=nfr)
    return wavefr


def ModelChi2Bin(params, data):

    V2, V2err, CP, CPerr = GiveDataValues(data)

    u, v, wavev2, wavecp = DataUnpack(data)

    xsec, ysec, fratios, primD, secD = ParamUnpack(params)

    # interpolating the spectrum to data wavelengths
    wavefr = CreateWaveGrid(wavev2, len(fratios))


    frv2 = np.interp(wavev2, wavefr, fratios)
    frcp = np.interp(wavecp, wavefr, fratios)

    V2mod, CPmod = Bin2V2CP(u, v, xsec, ysec, primD, secD,  frv2, frcp)

    resV2 = (V2 - V2mod)**2 / V2err**2
    resCP = (CP - CPmod)**2 / CPerr**2
    # nV2 = len(V2)
    # nCP = len(CP)

    chi2 = np.sum(resV2) + np.sum(resCP)

    return chi2


def FlattenList(List):
    #print(List)
    flat_list = []
    for sublist in List:
        #print(sublist)
        for item in sublist:
            #print(item)
            flat_list.append(item)
    return flat_list


def ModelChi2BinVis(params, data):

    V2, V2err, CP, CPerr = GiveDataValues(data)
    Vis, Viserr, Phi, Phierr = GiveDataValuesVis(data)

    u, v, wavev2, wavecp = DataUnpack(data)
    uv, vv, wavev = DataUnpackVis(data)

    xsec, ysec, fratios, primD, secD = ParamUnpack(params)

    # interpolating the spectrum to data wavelengths
    wavefr = CreateWaveGrid(wavev2, len(fratios))


    frv2 = np.interp(wavev2, wavefr, fratios)
    frcp = np.interp(wavecp, wavefr, fratios)
    frvis = np.interp(wavev, wavefr, fratios)

    V2mod, CPmod = Bin2V2CP(u, v, xsec, ysec, primD, secD,  frv2, frcp)
    Vismod, Phimod = Bin2VisPhi(uv, vv, wavev, xsec, ysec, primD, secD,  frvis)

    # print(Vismod, Vis)

    resV2 = (V2 - V2mod)**2 / V2err**2
    resCP = (CP - CPmod)**2 / CPerr**2
    resVis = (Vis - Vismod)**2 / Viserr**2
    resPhi = (Phi - Phimod)**2 / Phierr**2
    # nV2 = len(V2)
    # nCP = len(CP)

    chi2 = np.sum(resV2) + np.sum(resCP) + np.sum(resVis) + np.sum(resPhi)

    return chi2


def plotFluxRatios(data, params, time=[], dirdat='.', name='OUTPUT'):

    fr = params[2:]
    u, v, wavev2, wavecp = DataUnpack(data)
    wave = CreateWaveGrid(wavev2, len(fr))
    fig, ax = plt.subplots()

    plt.plot(wave*1e6, fr)
    plt.text(2.1, 1.05, 'time={}'.format(time) )
    plt.text(2.2, 1.0, '$\Delta x$={}mas \t $\Delta y$={}mas'.format(params[0], params[1]) )
    ax.set_ylim((0, 1.2))
    plt.savefig(dirdat + name +'_fratios.pdf')
    plt.show()
    plt.close()


def ListV2 (data):

    u, v, wavev2, wavecp = DataUnpack(data)
    V2, V2err, CP, CPerr = GiveDataValues(data)

    nV2 = len(V2)

    u, u1, u2, u3 = u
    v, v1, v2, v3 = v

    u *= wavev2
    v *= wavev2

    ubase = u[0]
    vbase = v[0]
    newV2 = []
    newV2err = []
    newwave = []
    newu = []
    newv = []
    tmpV2 = []
    tmpV2err = []
    tmpwave = []
    tmpu = []
    tmpv = []
    for i in np.arange(nV2):
        #print (u[i] - ubase)
        if np.abs(u[i] - ubase) < 1e-5 and np.abs(v[i]- vbase) < 1e-5:
            tmpV2.append(V2[i])
            tmpV2err.append(V2err[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wavev2[i])
        else:
            newV2.extend([tmpV2])
            newV2err.extend([tmpV2err])
            newu.extend([tmpu])
            newv.extend([tmpv])
            newwave.extend([tmpwave])
            ubase = u[i]
            vbase = v[i]
            tmpV2 = []
            tmpV2err = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpV2.append(V2[i])
            tmpV2err.append(V2err[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wavev2[i])

    newV2.extend([tmpV2])
    newV2err.extend([tmpV2err])
    newu.extend([tmpu])
    newv.extend([tmpv])
    newwave.extend([tmpwave])

    return newV2, newV2err, newu, newv, newwave


def CutVis(Vis, Phi, u, v, wave):

    nV = len(Vis)
    u *= wave
    v *= wave

    ubase = u[0]
    vbase = v[0]
    newVis = []
    newPhi = []
    newwave = []
    newu = []
    newv = []
    tmpVis = []
    tmpPhi = []
    tmpwave = []
    tmpu = []
    tmpv = []
    for i in np.arange(nV):
        #print (u[i] - ubase)
        if np.abs(u[i] - ubase) < 1e-5 and np.abs(v[i]- vbase) < 1e-5:
            tmpVis.append(Vis[i])
            tmpPhi.append(Phi[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wave[i])
        else:
            newVis.extend([tmpVis])
            newPhi.extend([tmpPhi])
            newu.extend([tmpu])
            newv.extend([tmpv])
            newwave.extend([tmpwave])
            ubase = u[i]
            vbase = v[i]
            tmpVis = []
            tmpPhi = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpVis.append(Vis[i])
            tmpPhi.append(Phi[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wave[i])

    newVis.extend([tmpVis])
    newPhi.extend([tmpPhi])
    newu.extend([tmpu])
    newv.extend([tmpv])
    newwave.extend([tmpwave])

    return newVis, newPhi, newwave


def ListVis(data):

    u, v, wave = DataUnpackVis(data)
    Vis, Viserr, Phi, Phierr = GiveDataValuesVis(data)

    nV = len(Vis)

    u *= wave
    v *= wave

    ubase = u[0]
    vbase = v[0]
    newVis = []
    newViserr = []
    newPhi = []
    newPhierr = []
    newwave = []
    newu = []
    newv = []
    tmpVis = []
    tmpViserr = []
    tmpPhi = []
    tmpPhierr = []
    tmpwave = []
    tmpu = []
    tmpv = []
    for i in np.arange(nV):
        #print (u[i] - ubase)
        if np.abs(u[i] - ubase) < 1e-5 and np.abs(v[i]- vbase) < 1e-5:
            tmpVis.append(Vis[i])
            tmpViserr.append(Viserr[i])
            tmpPhi.append(Phi[i])
            tmpPhierr.append(Phierr[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wave[i])
        else:
            newVis.extend([tmpVis])
            newViserr.extend([tmpViserr])
            newPhi.extend([tmpPhi])
            newPhierr.extend([tmpPhierr])
            newu.extend([tmpu])
            newv.extend([tmpv])
            newwave.extend([tmpwave])
            ubase = u[i]
            vbase = v[i]
            tmpVis = []
            tmpViserr = []
            tmpPhi = []
            tmpPhierr = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpVis.append(Vis[i])
            tmpViserr.append(Viserr[i])
            tmpPhi.append(Phi[i])
            tmpPhierr.append(Phierr[i])
            tmpu.append(u[i])
            tmpv.append(v[i])
            tmpwave.append(wave[i])

    newVis.extend([tmpVis])
    newViserr.extend([tmpViserr])
    newPhi.extend([tmpPhi])
    newPhierr.extend([tmpPhierr])
    newu.extend([tmpu])
    newv.extend([tmpv])
    newwave.extend([tmpwave])

    return newVis, newViserr, newPhi, newPhierr, newu, newv, newwave


def ListCP (data):

    u, v, wavev2, wavecp = DataUnpack(data)
    V2, V2err, CP, CPerr = GiveDataValues(data)

    nCP = len(CP)

    u, u1, u2, u3 = u
    v, v1, v2, v3 = v

    u1 *= wavecp
    v1 *= wavecp
    u2 *= wavecp
    v2 *= wavecp
    u3 *= wavecp
    v3 *= wavecp

    u1base = u1[0]
    v1base = v1[0]
    u2base = u2[0]
    v2base = v2[0]
    newCP = []
    newCPerr = []
    newwave = []
    newu1 = []
    newv1 = []
    newu2 = []
    newv2 = []
    newu3 = []
    newv3 = []
    tmpCP = []
    tmpCPerr = []
    tmpwave = []
    tmpu1 = []
    tmpv1 = []
    tmpu2 = []
    tmpv2 = []
    tmpu3 = []
    tmpv3 = []
    for i in np.arange(nCP):
        #print (u[i] - ubase)
        if np.abs(u1[i] - u1base) < 1e-5 and np.abs(v1[i]- v1base) < 1e-5 and np.abs(u2[i] - u2base) < 1e-5 and np.abs(v2[i]- v2base) < 1e-5:
            tmpCP.append(CP[i])
            tmpCPerr.append(CPerr[i])
            tmpu1.append(u1[i])
            tmpv1.append(v1[i])
            tmpu2.append(u2[i])
            tmpv2.append(v2[i])
            tmpu3.append(u3[i])
            tmpv3.append(v3[i])
            tmpwave.append(wavecp[i])
        else:
            newCP.extend([tmpCP])
            newCPerr.extend([tmpCPerr])
            newu1.extend([tmpu1])
            newv1.extend([tmpv1])
            newu2.extend([tmpu2])
            newv2.extend([tmpv2])
            newu3.extend([tmpu3])
            newv3.extend([tmpv3])
            newwave.extend([tmpwave])
            u1base = u1[i]
            v1base = v1[i]
            u2base = u2[i]
            v2base = v2[i]
            tmpCP = []
            tmpCPerr = []
            tmpwave = []
            tmpu = []
            tmpv = []
            tmpCP.append(CP[i])
            tmpCPerr.append(CPerr[i])
            tmpu1.append(u1[i])
            tmpv1.append(v1[i])
            tmpu2.append(u2[i])
            tmpv2.append(v2[i])
            tmpu3.append(u3[i])
            tmpv3.append(v3[i])
            tmpwave.append(wavecp[i])

    newCP.extend([tmpCP])
    newCPerr.extend([tmpCPerr])
    newu1.extend([tmpu1])
    newv1.extend([tmpv1])
    newu2.extend([tmpu2])
    newv2.extend([tmpv2])
    newu3.extend([tmpu3])
    newv3.extend([tmpv3])
    newwave.extend([tmpwave])

    return newCP, newCPerr, newu1, newv1, newu2, newv2, newu3, newv3, newwave



def giveDataModelChi2(data, model, dirdat='.', name='OUTPUT'):

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


    base, Bmax = mrf.Bases(data)

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
    ax11.set_ylim((0, 1.2))
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
