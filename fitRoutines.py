# -*- coding: utf-8 -*-
"""
This module contains some general tools that can be used to fit simple mathematical
functions to a dataset. The functions that can be fitted are:
- a sine with a combination of frequency,amplitude,phase and offset as fitparameters
- a cosine ...
- a gaussian with a combination of amplitude,mean and sigma as fitparameters
- a straight line with a combination of 'rico' and offset as fitparameters
- ...
Finally there is also the possibility to make phasediagrams.
"""
import pylab as pl
from numpy import *
from copy import copy,deepcopy
from scipy.optimize import leastsq,fmin #,fminbound
from scipy.special import j0,j1,jv

mas = pi/3600/1000/180.

#############################################################################################
# First define some classes to represent simple parametric (interferometric) models as objects
#############################################################################################

class UD(object):
     """To easily fit a UD."""

     attributeNames = array(["diameter"])

     def __init__(self,diameter,free=array([True]),evaluationfunction='V**2',X=0.0,Y=0.0):
           """
           Initialize the fitting parameter.

           diameter	- the diameter
           free		- an array of booleans to decide if the diameter is free to be fitted
           """
           self._diameter = abs(diameter)
           self.free = free
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1
           self._x = X*mas
           self._y = Y*mas

     def setDiameter(self,diameter):
           self._diameter = abs(diameter)
     def getDiameter(self):
           return self._diameter

     diameter = property(getDiameter,setDiameter)

     def setEvaluation(self,evaluationfunction):
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1
     def getEvaluation(self):
           if self._evaluationfunction == 0:
             return 'V'
           elif self._evaluationfunction == 0:
             return 'V**2'

     evaluationfunction = property(getEvaluation,setEvaluation)

     def setX(self,X):
           self._x = X*mas
     def getX(self):
           return self._x

     x = property(getX,setX)

     def setY(self,Y):
           self._y = Y*mas
     def getY(self):
           return self._y

     y = property(getY,setY)

     def getFreeIndices(self):
           """
           Return the indices of the parameters that are put on 'free'.
           """
           return where(self.free)[0]

     def getFixedIndices(self):
           """
           Return the indices of the parameters that are put on 'fixed'.
           """
           return where(self.free == False)[0]

     def evaluate(self,x):
           arg = pi*self._diameter*mas*x
           if self._evaluationfunction:
             return where(arg==0.,1.,abs(2*j1(arg)/arg)**2)
           else:
             return where(arg==0.,1.,abs(2*j1(arg)/arg)) #,angle(2*j1(arg)/arg,deg=True))

     def evaluatePhase(self,x):
           V = self.evaluateComplex(x)
           return angle(V,deg=True)

     def evaluateComplex(self,x):
           if type(x)!=tuple: #((self.x == 0.0) and (self.y == 0.0)):
               X = x
               arg = pi*mas*self._diameter*X
               phasearg = 0.
           else:
               u,v = x[0],x[1] #(where(((x[0] == 0.) & (x[1] == 0.)),0.00000001,x[0]),where(((x[0] == 0.) & (x[1] == 0.)),0.00000001,x[1]))
               X = sqrt(u**2+v**2)
               arg = pi*mas*self._diameter*X
               phasearg = u*self._x+v*self._y
           return where(arg==0.,1.,(2*j1(arg)/arg))*exp(-1j*2*pi*phasearg)

     #def evaluatePhase(self,x):
           #arg = pi*1e-3*self._diameter*where(x == 0.,0.000000001,x)
           #return angle(2*j1(arg)/arg,deg=True)

     def __call__(self):
           return [self._diameter]

     def __str__(self):
           return "diameter = %.5f"%(self._diameter)



# The following class needs to be adapted and tested still!!!!!!!!!!!!
class Ellipse(UD):
     """To easily fit an ellipse (to visibilities in the first place)."""

     attributeNames = array(["diameter","psi","f"])

     def __init__(self,diameter,psi,f,free=array([True,True,True]),evaluationfunction='V**2'):
           """
           Initialize the fitting parameter.

           diameter	- the diameter
           free		- an array of booleans to decide if the diameter is free to be fitted
           """
           super(Ellipse,self).__init__(diameter,evaluationfunction)
           self._psi = psi
           self._f = abs(f)
           self.free = free

     def setPsi(self,psi):
           self._psi = psi
     def getPsi(self):
           return self._psi

     psi = property(getPsi,setPsi)

     def setF(self,f):
           self._f = abs(f)
     def getF(self):
           return self._f

     f = property(getF,setF)

     def evaluate(self,x):
           uprime = x[:,0]*cos(self._psi*pi/180) + x[:,1]*sin(self._psi*pi/180)
           vprime = -x[:,0]*sin(self._psi*pi/180) + x[:,1]*cos(self._psi*pi/180)
           r = sqrt(uprime**2 + (self._f*vprime)**2)
           arg = pi*mas*self._diameter*r
           if self._evaluationfunction:
             return abs(2*j1(arg)/arg)**2
           else:
             return abs(2*j1(arg)/arg) #,angle(2*j1(arg)/arg,deg=True))

     def evaluatePhase(self,x):
           uprime = x[:,0]*cos(self._psi*pi/180) + x[:,1]*sin(self._psi*pi/180)
           vprime = -x[:,0]*sin(self._psi*pi/180) + x[:,1]*cos(self._psi*pi/180)
           r = sqrt(uprime**2 + (self._f*vprime)**2)
           arg = pi*mas*self._diameter*r
           return angle(2*j1(arg)/arg,deg=True)

     def __call__(self):
           return [self._diameter,self._psi,self._f]

     def __str__(self):
           return "diameter = %.5f | psi = %.3f | f = %.3f"%(self._diameter,self._psi,self._f)

class Ring(UD):
     """A representation of a ring model. """
     attributeNames = array(["diameter","width"])

     def __init__(self,diameter,width,free=array([True,True]),evaluationfunction='V**2',X=0.,Y=0.):
           """
           Initialize the fitting parameter.

           diameter     - the diameter of the ring in mas
           width        - the width of the ring in mas
           free         - an array of booleans to decide if the diameter is free to be fitted
           """
           try:
               self._f = width/diameter*2.
               super(Ring,self).__init__(diameter,evaluationfunction=evaluationfunction,X=X,Y=Y)
           except ZeroDivisionError:
               self._f = width/1e-15*2.
               super(Ring,self).__init__(1e-15,evaluationfunction=evaluationfunction,X=X,Y=Y)

           self._width = width
           self.free = free

     def setWidth(self,width):
           self._width = width
     def getWidth(self):
           return self._width

     width = property(getWidth,setWidth)

     def evaluate(self,x):
           V = self.evaluateComplex(x)
           if self._evaluationfunction:
             return abs((2./arg/(2*self._f+self._f**2))*((1+self._f)*j1((1+self._f)*arg) - j1(arg)))**2
           else:
             return abs((2./arg/(2*self._f+self._f**2))*((1+self._f)*j1((1+self._f)*arg) - j1(arg)))

     def evaluatePhase(self,x):
           V = self.evaluateComplex(x)
           return angle(V,deg=True)

     def evaluateComplex(self,x):
           self._f = self._width/self._diameter*2.
           if type(x)!=tuple:
               X = x
               arg = pi*mas*self._diameter*X
               phasearg = 0.
           else:
               u,v = x[0],x[1]
               X = sqrt(u**2+v**2)
               arg = pi*mas*self._diameter*X
               phasearg = u*self._x+v*self._y
           return where(arg==0.,1.,(2./(arg*(2*self._f+self._f**2)))*((1+self._f)*j1((1+self._f)*arg) - j1(arg))*exp(-1j*2*pi*phasearg))

     def __call__(self):
           return [self._diameter,self._width]

     def __str__(self):
           return "diameter = %.5f | width = %.3f"%(self._diameter,self._width)

class ModulatedGaussianRing(UD):
     """A representation of an azimuthally modulated ring."""
     attributeNames = array(["diameter","width","c1","c2","c3","s1","s2","s3"])

     def __init__(self,diameter,width,c1,c2,c3,s1,s2,s3,PA,inclination,free=array([True,True,True,True,True,True,True,True]),evaluationfunction='V**2',X=0.0,Y=0.0):
           """
           Initialize the object with fitting parameters:

           diameter     - the central diameter of the ring
           width        - the Gaussian FWHM of the ring, normalized to the ring diameter
           c1           - the cosine-amplitude of the first order azimuthal modulation
           c2           - the cosine-amplitude of the second order azimuthal modulation
           c3           - the cosine-amplitude of the third order azimuthal modulation
           s1           - the sine-amplitude of the first order azimuthal modulation
           s2           - the sine-amplitude of the second order azimuthal modulation
           s3           - the sine-amplitude of the third order azimuthal modulation
           PA           - PA of the major axis, E of N, in degrees
           inclination  - inclination of the ring, in degrees
           """
           super(ModulatedGaussianRing,self).__init__(diameter,evaluationfunction=evaluationfunction,X=X,Y=Y)
           self._width = width
           self._c1 = c1
           self._c2 = c2
           self._c3 = c3
           self._s1 = s1
           self._s2 = s2
           self._s3 = s3
           self._PA = mod(90.-PA,360.)
           self._inclination = mod(inclination,180.)
           self.free = free

     def setWidth(self,width):
           self._width = width
     def getWidth(self):
           return self._width

     width = property(getWidth,setWidth)

     def setC1(self,c1):
           self._c1 = c1

     def getC1(self):
           return self._c1

     c1 = property(getC1,setC1)

     def setC2(self,c2):
           self._c2 = c2

     def getC2(self):
           return self._c2

     c2 = property(getC2,setC2)

     def setC3(self,c3):
           self._c3 = c3

     def getC3(self):
           return self._c3

     c3 = property(getC3,setC3)

     def setS1(self,s1):
           self._s1 = s1

     def getS1(self):
           return self._s1

     s1 = property(getS1,setS1)

     def setS2(self,s2):
           self._s2 = s2

     def getS2(self):
           return self._s2

     s2 = property(getS2,setS2)

     def setS3(self,s3):
           self._s3 = s3

     def getS3(self):
           return self._s3

     s3 = property(getS3,setS3)

     def setPA(self,PA):
           self._PA = mod(90.-PA,360.)                   # the -90 is to correct from E of N to the here adopted reference frame

     def getPA(self):
           return self._PA

     PA = property(getPA,setPA)

     def setInclination(self,inclination):
           self._inclination = mod(inclination,180.)

     def getInclination(self):
           return self._inclination

     inclination = property(getInclination,setInclination)

     def rotateUV(self,u,v):
           """
           To do a rotation in the image plane, perform a rotation in the uv plane.
           """
           newu =  u*cos(self._PA*pi/180.) + v*sin(self._PA*pi/180.)
           newv = -u*sin(self._PA*pi/180.) + v*cos(self._PA*pi/180.)
           return newu,newv

     def inclineUV(self,u,v):
           """
           To incline in the image plane, do a scaling in the uv plane.
           """
           newu = u
           newv = v*abs(cos(self._inclination*pi/180.))
           return newu,newv

     def evaluate(self,x):
           Vring = self.evaluateComplex(x)
           if self._evaluationfunction:
               return abs(Vring)**2
           else:
               return abs(Vring)

     def evaluatePhase(self,x):
           Vring = self.evaluateComplex(x)
           return angle(Vring,deg=True)

     def evaluateComplex(self,x):
           """
           Returns the complex visibility.
           """
           u,v = x[0],x[1]
           rotu,rotv = self.rotateUV(u,v)
           scaledu,scaledv = self.inclineUV(rotu,rotv)
           rotB = sqrt(scaledu**2+scaledv**2)
           modAngle = arctan2(scaledv,scaledu)

           Vring = (j0(pi*mas*self._diameter*rotB) \
                   - 1j*(self._c1*cos(modAngle) + self._s1*sin(modAngle))*j1(pi*mas*self._diameter*rotB) \
                   -(self._c2*cos(2*modAngle) + self._s2*sin(2*modAngle))*jv(2,pi*mas*self._diameter*rotB) \
                   - 1j*(self._c3*cos(3*modAngle) + self._s3*sin(3*modAngle))*jv(3,pi*mas*self._diameter*rotB)) \
                   *exp(-(pi*mas*self._diameter/2.*self._width*rotB)**2/(4*log(2)))
           #Vring -= 1j*(self._c1*cos(modAngle) + self._s1*sin(modAngle))*j1(pi*self._diameter*rotB)         # azimuthal modulation of the first order
           #Vring += -(self._c2*cos(2*modAngle) + self._s2*sin(2*modAngle))*jv(2,pi*self._diameter*rotB)      # azimuthal modulation of the second order
           #Vring *= exp(-(pi*self._diameter/2.*self._width*rotB)**2/(4*log(2)))                              # the Gaussian width
           phasearg = u*self._x+v*self._y
           Vring = Vring*exp(-1j*2*pi*phasearg)
           #print 'Boum!'
           return Vring

     def makeImage(self,image,fluxratio):
           """
           Returns the image.
           """
           #x,y,img,ps = image[0],image[1],image[2],image[3]
           x,y,img = image[0],image[1],image[2] #,image[3]
           thetar2 = -self._PA*pi/180.

           #rotx = (x*cos(thetar2) + y*sin(thetar2))*ps
           #roty = (y*cos(thetar2) + x*sin(thetar2))*ps/cos(self._inclination*pi/180.)
           rotx = (x*cos(thetar2) + y*sin(thetar2))
           roty = (-y*cos(thetar2) + x*sin(thetar2))/cos(self._inclination*pi/180.)
           modAngle = arctan2(roty,rotx)

           sig = self._width*self._diameter/2/(2*sqrt(2*log(2)))
           tempimg = 1./(sig*sqrt(2*pi))*exp(-(sqrt(rotx**2+roty**2)-self._diameter/2)**2/(2*sig**2)) \
                     *(1+self._c1*cos(modAngle)+self._s1*sin(modAngle)+self._c2*cos(2*modAngle)+self._s2*sin(2*modAngle)+self._c3*cos(3*modAngle)+self._s3*sin(3*modAngle))
           print tempimg.sum(),fluxratio
           img += fluxratio*tempimg/tempimg.sum()

           return img

     def __call__(self):
           return [self._diameter,self._width,self._c1,self._c2,self._c3,self._s1,self._s2,self._s3,self._PA,self._inclination]

     def __str__(self):
           return "diameter = %.5f etc."%(self._diameter)


class ModulatedGaussianRingShift(UD):
     """A representation of an azimuthally modulated ring."""
     attributeNames = array(["diameter","width","c1","c2","c3","s1","s2","s3","x0","y0"])

     def __init__(self,diameter,width,c1,c2,c3,s1,s2,s3,PA,inclination,x0,y0,free=array([True,True,True,True,True,True,True,True,True,True]),evaluationfunction='V**2',X=0.0,Y=0.0):
           """
           Initialize the object with fitting parameters:

           diameter     - the central diameter of the ring
           width        - the Gaussian FWHM of the ring, normalized to the ring diameter
           c1           - the cosine-amplitude of the first order azimuthal modulation
           c2           - the cosine-amplitude of the second order azimuthal modulation
           c3           - the cosine-amplitude of the third order azimuthal modulation
           s1           - the sine-amplitude of the first order azimuthal modulation
           s2           - the sine-amplitude of the second order azimuthal modulation
           s3           - the sine-amplitude of the third order azimuthal modulation
           PA           - PA of the major axis, E of N, in degrees
           inclination  - inclination of the ring, in degrees
           x0           - Shift in RA (mas)
           y0           - Shift in DEC (mas)
           """
           super(ModulatedGaussianRingShift,self).__init__(diameter,evaluationfunction=evaluationfunction,X=X,Y=Y)
           self._width = width
           self._c1 = c1
           self._c2 = c2
           self._c3 = c3
           self._s1 = s1
           self._s2 = s2
           self._s3 = s3
           self._PA = mod(90.-PA,360.)
           self._inclination = mod(inclination,180.)
           self._x0 = x0
           self._y0 = y0
           self.free = free

     def setWidth(self,width):
           self._width = width
     def getWidth(self):
           return self._width

     width = property(getWidth,setWidth)

     def setC1(self,c1):
           self._c1 = c1

     def getC1(self):
           return self._c1

     c1 = property(getC1,setC1)

     def setC2(self,c2):
           self._c2 = c2

     def getC2(self):
           return self._c2

     c2 = property(getC2,setC2)

     def setC3(self,c3):
           self._c3 = c3

     def getC3(self):
           return self._c3

     c3 = property(getC3,setC3)

     def setS1(self,s1):
           self._s1 = s1

     def getS1(self):
           return self._s1

     s1 = property(getS1,setS1)

     def setS2(self,s2):
           self._s2 = s2

     def getS2(self):
           return self._s2

     s2 = property(getS2,setS2)

     def setS3(self,s3):
           self._s3 = s3

     def getS3(self):
           return self._s3

     s3 = property(getS3,setS3)

     def setPA(self,PA):
           self._PA = mod(90.-PA,360.)                   # the -90 is to correct from E of N to the here adopted reference frame

     def getPA(self):
           return self._PA

     PA = property(getPA,setPA)

     def setInclination(self,inclination):
           self._inclination = mod(inclination,180.)

     def getInclination(self):
           return self._inclination

     inclination = property(getInclination,setInclination)

     def setX0(self,x0):
           self._x0 = x0

     def getX0(self):
           return self._x0

     x0 = property(getX0,setX0)

     def setY0(self,y0):
           self._y0 = y0

     def getY0(self):
           return self._y0

     y0 = property(getY0,setY0)

     def rotateUV(self,u,v):
           """
           To do a rotation in the image plane, perform a rotation in the uv plane.
           """
           newu =  u*cos(self._PA*pi/180.) + v*sin(self._PA*pi/180.)
           newv = -u*sin(self._PA*pi/180.) + v*cos(self._PA*pi/180.)
           return newu,newv

     def inclineUV(self,u,v):
           """
           To incline in the image plane, do a scaling in the uv plane.
           """
           newu = u
           newv = v*abs(cos(self._inclination*pi/180.))
           return newu,newv

     def evaluate(self,x):
           Vring = self.evaluateComplex(x)
           if self._evaluationfunction:
               return abs(Vring)**2
           else:
               return abs(Vring)

     def evaluatePhase(self,x):
           Vring = self.evaluateComplex(x)
           return angle(Vring,deg=True)

     def evaluateComplex(self,x):
           """
           Returns the complex visibility.
           """
           u,v = x[0],x[1]
           rotu,rotv = self.rotateUV(u,v)
           scaledu,scaledv = self.inclineUV(rotu,rotv)
           rotB = sqrt(scaledu**2+scaledv**2)
           modAngle = arctan2(scaledv,scaledu)

           Vring = (j0(pi*mas*self._diameter*rotB) \
                   - 1j*(self._c1*cos(modAngle) + self._s1*sin(modAngle))*j1(pi*mas*self._diameter*rotB) \
                   -(self._c2*cos(2*modAngle) + self._s2*sin(2*modAngle))*jv(2,pi*mas*self._diameter*rotB) \
                   - 1j*(self._c3*cos(3*modAngle) + self._s3*sin(3*modAngle))*jv(3,pi*mas*self._diameter*rotB)) \
                   *exp(-(pi*mas*self._diameter/2.*self._width*rotB)**2/(4*log(2)))
           #Vring -= 1j*(self._c1*cos(modAngle) + self._s1*sin(modAngle))*j1(pi*self._diameter*rotB)         # azimuthal modulation of the first order
           #Vring += -(self._c2*cos(2*modAngle) + self._s2*sin(2*modAngle))*jv(2,pi*self._diameter*rotB)      # azimuthal modulation of the second order
           #Vring *= exp(-(pi*self._diameter/2.*self._width*rotB)**2/(4*log(2)))                              # the Gaussian width
           phasearg = u*self._x0*pi/180./3600./1000.+v*self._y0*pi/180./3600./1000.
           Vring = Vring*exp(-1j*2.*pi*phasearg)
           return Vring

     def makeImage(self,image,fluxratio):
           """
           Returns the image.
           """
           #x,y,img,ps = image[0],image[1],image[2],image[3]
           x,y,img = image[0],image[1],image[2] #,image[3]
           thetar2 = -self._PA*pi/180.

           #rotx = (x*cos(thetar2) + y*sin(thetar2))*ps
           #roty = (y*cos(thetar2) + x*sin(thetar2))*ps/cos(self._inclination*pi/180.)
           rotx = (x*cos(thetar2) + y*sin(thetar2))
           roty = (-y*cos(thetar2) + x*sin(thetar2))/cos(self._inclination*pi/180.)
           modAngle = arctan2(roty,rotx)

           sig = self._width*self._diameter/2/(2*sqrt(2*log(2)))
           tempimg = 1./(sig*sqrt(2*pi))*exp(-(sqrt(rotx**2+roty**2)-self._diameter/2)**2/(2*sig**2)) \
                     *(1+self._c1*cos(modAngle)+self._s1*sin(modAngle)+self._c2*cos(2*modAngle)+self._s2*sin(2*modAngle)+self._c3*cos(3*modAngle)+self._s3*sin(3*modAngle))
           print tempimg.sum(),fluxratio
           img += fluxratio*tempimg/tempimg.sum()

           return img

     def __call__(self):
           return [self._diameter,self._width,self._c1,self._c2,self._c3,self._s1,self._s2,self._s3,self._PA,self._inclination,self._x0,self._y0]

     def __str__(self):
           return "diameter = %.5f etc."%(self._diameter)


# The following class needs to be adapted and tested still!!!!!!!!!!!!
class Gaussian(object):
     """A representation of a Gaussian model."""
     attributeNames = array(["fwhm"])

     def __init__(self,fwhm,free=array([True]),evaluationfunction='V**2'):
           """
           Initialize the fitting parameter.

           fwhm         - the fwhm
           free         - an array of booleans to decide if the fwhm is free to be fitted
           """
           self._fwhm = abs(fwhm)
           self.free = free
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1

     def setFWHM(self,fwhm):
           self._fwhm = abs(fwhm)
     def getFWHM(self):
           return self._fwhm

     width = property(getFWHM,setFWHM)

     def evaluate(self,x):
           arg = pi*mas*self._fwhm*where(x == 0.,0.000000001,x)
           if self._evaluationfunction:
             return abs(exp(-(arg**2)/4./log(2.)))**2
           else:
             return abs(exp(-(arg**2)/4./log(2.)))

     def evaluatePhase(self,x):
           arg = pi*mas*self._fwhm*where(x == 0.,0.000000001,x)
           return angle(exp(-(arg**2)/4./log(2.)),deg=True)

     def __call__(self):
           return [self._fwhm]

     def __str__(self):
           return "fwhm = %.5f"%(self._fwhm)

# The following class needs to be adapted and tested still!!!!!!!!!!!!
class Binary(object):
     """To easily fit a binary model (to visibilities in the first place). """
     attributeNames = array(["diameter1","diameter2","q","x","y"])

     def __init__(self,diameter1,diameter2,q,x1,y1,x2,y2,free=array([True,True,True,True,True]),evaluationfunction='V**2'):
           """
           Initialize the fitting parameter.

           diameter     - the diameter
           free         - an array of booleans to decide if the diameter is free to be fitted
           """
           self.ud1 = UD(diameter1,evaluationfunction='V',X=x1,Y=y1)
           self.ud2 = UD(diameter2,evaluationfunction='V',X=x2,Y=y2)
           self._q = abs(q)
           self.free = free
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1

     def setDiameter1(self,diameter):
           self.ud1.diameter = abs(diameter)
     def getDiameter1(self):
           return self.ud1()[0]

     diameter1 = property(getDiameter1,setDiameter1)

     def setDiameter2(self,diameter):
           self.ud2.diameter = abs(diameter)
     def getDiameter2(self):
           return self.ud2()[0]

     diameter2 = property(getDiameter2,setDiameter2)

     def setQ(self,q):
           self._q = abs(q)
     def getQ(self):
           return self._q

     q = property(getQ,setQ)

     def setX1(self,x):
           self.ud1.x = x
     def getX1(self):
           return self.ud1.x

     x1 = property(getX1,setX1)

     def setY1(self,y):
           self.ud1.y = y
     def getY1(self):
           return self.ud1.y

     y1 = property(getY1,setY1)

     def setX2(self,x):
           self.ud2.x = x
     def getX2(self):
           return self.ud2.x

     x2 = property(getX2,setX2)

     def setY2(self,y):
           self.ud2.y = y
     def getY2(self):
           return self.ud2.y

     y2 = property(getY2,setY2)

     def setEvaluation(self,evaluationfunction):
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1
     def getEvaluation(self):
           if self._evaluationfunction == 0:
             return 'V'
           elif self._evaluationfunction == 0:
             return 'V**2'

     evaluationfunction = property(getEvaluation,setEvaluation)

     def getFreeIndices(self):
           """
           Return the indices of the parameters that are put on 'free'.
           """
           return where(self.free)[0]

     def getFixedIndices(self):
           """
           Return the indices of the parameters that are put on 'fixed'.
           """
           return where(self.free == False)[0]

     def evaluate(self,x):
           V = self.evaluateVis(x)
           if self._evaluationfunction:
             return abs(V)**2
           else:
             return abs(V)

     def _calculateVis(self,u,v):
           r = sqrt(u**2+v**2)
           V1 = self.ud1.evaluate(r)*exp(1j*self.ud1.evaluatePhase((u,v))*pi/180)
           V2 = self.ud2.evaluate(r)*exp(1j*self.ud2.evaluatePhase((u,v))*pi/180)
           return 1./(1.+self._q)*(V1 + V2*self._q)

     #def evaluateVis(self,x):
     #      u,v = (where(((x[0] == 0.) & (x[1] == 0.)),0.00000001,x[0]),where(((x[0] == 0.) & (x[1] == 0.)),0.00000001,x[1]))
     #      V = self._calculateVis(u,v)
     #      return V

     def evaluatePhase(self,x):
           V = self.evaluateVis(x)
           return angle(V,deg=True)

     def __call__(self):
           return [self.ud1.diameter,self.ud2.diameter,self._q,self.ud1.x,self.ud1.y,self.ud2.x,self.ud2.y]

     def __str__(self):
           return "diameter1 = %.5f | diameter2 = %.5f | q = %.5f | x = %.5f | y = %.5f"%(self.ud1.diameter,self.ud2.diameter,self._q,self.ud1.x,self.ud1.y,self.ud2.x,self.ud2.y)

# The following class needs to be tested still!!!!!!!!!!!!
class RingStar(object):
     """To easily fit a binary model (to visibilities in the first place). """
     attributeNames = array(["diameter1","diameter2","width","q","x","y"])

     def __init__(self,diameter1,diameter2,width,q,x,y,free=array([True,True,True,True,True]),evaluationfunction='V**2'):
           """
           Initialize the fitting parameter.

           diameter1     - the diameter of the uniform disk
           diameter2     - the inner diameter of the ring
           width         - the width of the ring
           q             - the flux ratio Fring/Fstar
           x             - the relative x position of the center of the ring versus the star
           y             - the relative y position of the center of the ring versus the star
           free          - an array of booleans to decide if the diameter is free to be fitted
           """
           self.ud = UD(diameter1,evaluationfunction='V',X=x,Y=y)
           self.ring = Ring(diameter2,width,evaluationfunction='V')
           self._q = abs(q)
           #self._x = x
           #self._y = y
           self.free = free
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1

     def setDiameter1(self,diameter):
           self.ud.diameter = abs(diameter)
     def getDiameter1(self):
           return self.ud()[0]

     diameter1 = property(getDiameter1,setDiameter1)

     def setDiameter2(self,diameter):
           self.ring.diameter = abs(diameter)
     def getDiameter2(self):
           return self.ring()[0]

     diameter2 = property(getDiameter2,setDiameter2)

     def setWidth(self,width):
           self.ring.width = abs(width)
     def getWidth(self):
           return self.ring()[1]

     width = property(getWidth,setWidth)

     def setQ(self,q):
           self._q = abs(q)
     def getQ(self):
           return self._q

     q = property(getQ,setQ)

     def setX(self,x):
           self.ud.x = x
     def getX(self):
           return self.ud.x

     x = property(getX,setX)

     def setY(self,y):
           self.ud.y = y
     def getY(self):
           return self.ud.y

     y = property(getY,setY)

     def setEvaluation(self,evaluationfunction):
           if evaluationfunction == 'V':
             self._evaluationfunction = 0
           elif (evaluationfunction == 'V**2') or (evaluationfunction == 'V^2'):
             self._evaluationfunction = 1
     def getEvaluation(self):
           if self._evaluationfunction == 0:
             return 'V'
           elif self._evaluationfunction == 0:
             return 'V**2'

     evaluationfunction = property(getEvaluation,setEvaluation)

     def getFreeIndices(self):
           """
           Return the indices of the parameters that are put on 'free'.
           """
           return where(self.free)[0]

     def getFixedIndices(self):
           """
           Return the indices of the parameters that are put on 'fixed'.
           """
           return where(self.free == False)[0]

     def evaluate(self,x):
           V = self.evaluateComplex(x)
           if self._evaluationfunction:
             return abs(V)**2
           else:
             return abs(V)

     def evaluateComplex(self,x):
           u,v = x[0],x[1]
           V1 = self.ud.evaluateComplex((u,v))
           V2 = self.ring.evaluateComplex((u,v)) #(r)*exp(1j*self.ring.evaluatePhase(r)*pi/180)
           #return 1./(1.+self._q)*(V1 + V2*exp(arg)*self._q)
           V = 1./(1.+self._q)*(V1 + V2*self._q)
           return V

     def evaluatePhase(self,x):
           V = self.evaluateComplex(x)
           return angle(V,deg=True)

     def __call__(self):
           return [self.diameter1,self.diameter2,self.width,self.q,self.x,self.y]

     def __str__(self):
           return "diameter1 = %.5f | diameter2 = %.5f | width = %.5f | q = %.5f | x = %.5f | y = %.5f"%(self.diameter1,self.diameter2,self.width,self.q,self.x,self.y)

#############################################################################################
# Now define some classes to represent simple mathematical functions as objects
#############################################################################################

class Parameter(object):
     """
     A class to represent the parameters of a mathematical function to be fitted to a dataset
     in an object. Make sure to enter a correct 'free-array' to make clear
     which parameters are fitted.
     """

     attributeNames = array(["amplitude","offset"])

     def __init__(self,amplitude,offset,free=array([True,True])):
           """
           Initialize the fitting parameters.

           amplitude    - the amplitude
           offset       - the 'base level'
           free         - an array of booleans to decide which parameters are free to be fitted.
           """
           self._amplitude = amplitude
           self.offset = offset
           self.free = free

     def setAmplitude(self,amplitude):
           self._amplitude = amplitude
     def getAmplitude(self):
           return self._amplitude

     amplitude = property(getAmplitude,setAmplitude)

     def getFreeIndices(self):
           """
           Return the indices of the parameters that are put on 'free'.
           """
           return where(self.free)[0]

     def getFixedIndices(self):
           """
           Return the indices of the parameters that are put on 'fixed'.
           """
           return where(self.free == False)[0]

     def evaluate(self,x):
           """
           As the base-class of other objects, it evaluates into a straight line.
           """
           return self._amplitude * x + self.offset

     def __call__(self):
           return [self._amplitude,self.offset]

     def __str__(self):
           return "amplitude = %.4f | offset = %.4f"%(self._amplitude,self.offset)

class Parabola(Parameter):
     """
     A class to represent the parameters of a mathematical function to be fitted to a dataset
     in an object. Make sure to enter a correct 'free-array' to make clear
     which parameters are fitted.
     """

     attributeNames = array(["amplitude1","amplitude","offset"])

     def __init__(self,amplitude1,amplitude,offset,free=array([True,True,True])):
           """
           Initialize the fitting parameters.

           amplitude1   - the coefficient of second degree
           amplitude    - the coefficient of first degree
           offset       - the 'base level'
           free         - an array of booleans to decide which parameters are free to be fitted.
           """
           super(Parabola,self).__init__(amplitude,offset,free)
           self._amplitude1 = amplitude1

     def setAmplitude1(self,amplitude1):
           self._amplitude1 = amplitude1
     def getAmplitude1(self):
           return self._amplitude1

     amplitude1 = property(getAmplitude1,setAmplitude1)

     def getFreeIndices(self):
           """
           Return the indices of the parameters that are put on 'free'.
           """
           return where(self.free)[0]

     def getFixedIndices(self):
           """
           Return the indices of the parameters that are put on 'fixed'.
           """
           return where(self.free == False)[0]

     def evaluate(self,x):
           """
           As the base-class of other objects, it evaluates into a straight line.
           """
           return self._amplitude1 * x**2 + self._amplitude * x + self.offset

     def __call__(self):
           return [self._amplitude1,self._amplitude,self.offset]

     def __str__(self):
           return "amplitude1 = %.4f | amplitude = %.4f | offset = %.4f"%(self._amplitude1,self._amplitude,self.offset)

class GaussianParameter(Parameter):
     """A class to make objects of parameters."""

     attributeNames = array(["amplitude","offset","sigma","mu"])

     def __init__(self,amplitude,offset,sigma,mu,free=array([True,True,True,True])):
           """
           Initialize the fitting parameters.

           amplitude    - the amplitude
           offset       - the 'base level'
           sigma        - the standard deviation
           mu           - the mean value
           free         - an array of booleans to decide which parameters are free to be fitted.
           """
           super(GaussianParameter,self).__init__(amplitude,offset,free)
           self.sigma = sigma
           self.mu = mu

     def evaluate(self,x):
           return self._amplitude * exp( -(x-self.mu)**2 / (2.*self.sigma**2)) + self.offset

     def __call__(self):
           return [self._amplitude,self.offset,self.sigma,self.mu]

     def __str__(self):
           return super(GaussianParameter,self).__str__() + " | sigma = %.4f | mu = %.4f"%(self.sigma,self.mu)

class SineParameter(Parameter):
     """A class to make objects of cyclic parameters. In this case a sine."""

     attributeNames = array(["amplitude","offset","frequency","phase"])

     def __init__(self,amplitude,offset,frequency=1,phase=0,free=array([True,True,True,True])):
           """
           Initialize the fitting parameters.

           amplitude    - the amplitude
           offset       - the 'base level'
           frequency    - the frequency
           phase        - the phase as a number between 0 and 1
           free         - an array of booleans to decide which parameters are free to be fitted.
           """
           super(SineParameter,self).__init__(abs(amplitude),offset,free)
           self._frequency = abs(frequency)
           self._phase = phase < 0 and fmod(phase,1) + 1 or fmod(phase,1)


     def setAmplitude(self,amplitude):
           super(SineParameter,self).setAmplitude(abs(amplitude))

     amplitude = property(Parameter.getAmplitude,setAmplitude)

     def setFrequency(self,frequency):
           self._frequency = abs(frequency)

     def getFrequency(self):
           return self._frequency

     frequency = property(getFrequency,setFrequency)

     def setPhase(self,phase):
           self._phase = phase < 0 and fmod(phase,1) + 1 or fmod(phase,1)
     def getPhase(self):
           return self._phase

     phase = property(getPhase,setPhase)

     def evaluate(self,x):
           return self._amplitude * sin(2*pi*(self._frequency * x - self._phase)) + self.offset

     def __call__(self):
           return [self._amplitude,self.offset,self._frequency,self._phase]

     def __str__(self):
           return super(SineParameter,self).__str__() + " | frequency = %.5f | phase = %.4f"%(self.frequency,self.phase)

class CosineParameter(SineParameter):
     """The same as SineParameter but evaluated as a cosine instead of a sine."""

     attributeNames = array(["amplitude","offset","frequency","phase"])

     def __init__(self,amplitude,offset,frequency=1,phase=0,free=array([True,True,True,True])):
           """
           Initialize the fitting parameters.

           amplitude    - the amplitude
           offset       - the 'base level'
           frequency    - the frequency
           phase        - the phase as a number between 0 and 1
           free         - an array of booleans to decide which parameters are free to be fitted.
           """
           super(CosineParameter,self).__init__(amplitude,offset,frequency,phase,free)

     def evaluate(self,x):
           return self._amplitude * cos(2*pi*(self.frequency * x - self._phase)) + self.offset


#################################################################################################
# Now define a class to hold a number of these objects in a list
#################################################################################################

class ParameterList(object):
     """
     A class to represent a list of Parameter-type (and subtypes) objects.
     """
     def __init__(self,classType,initialvalues,free = None,evaluationfunction = 'V**2'):
           """
           Make a list of objects of (a sub)class (of) Parameter.

           classType     - the class of which objects should be made
           initialvalues - a numpy array containing the initial values: the i'th row contains the initial values for the parameters of the i'th object
           free          - an optional boolean array to define the free parameters
           NOTE: the initialvalues array should be 2d, with the first dimension equal to the number of objects in the list.
           """
           if initialvalues.ndim == 1: initialvalues = initialvalues[newaxis,:]

           self._objectlist = []
           self._len = initialvalues.shape[0]

           if classType == "Parameter":
             if free == None: free = array([True,True])
             self._objectlist = map(Parameter,initialvalues[:,0],initialvalues[:,1],[free]*self._len)
           elif classType == "GaussianParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = map(GaussianParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "SineParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = map(SineParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "CosineParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = map(CosineParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "UD":
             if free == None: free = array([True])
             self._objectlist = map(UD,initialvalues[:,0],[free]*self._len,[evaluationfunction]*self._len)

     def getObjectlist(self):
           return copy(self._objectlist)

     def getObject(self,index):
           return self._objectlist[index]

     def setObjects(self,attribute,attributeValues):
           for i,parameterObject in enumerate(self._objectlist):
               setattr(parameterObject,attribute,attributeValues[i])

     def addObjects(self,classType,initialvalues,free = None):
           """
           To add new objects to the current _objectlist.
           """
           if initialvalues.ndim == 1: initialvalues = initialvalues[newaxis,:]

           self._len = self._len + initialvalues.shape[0]

           if classType == "Parameter":
             if free == None: free = array([True,True])
             self._objectlist = self._objectlist + map(Parameter,initialvalues[:,0],initialvalues[:,1],[free]*self._len)
           elif classType == "GaussianParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = self._objectlist + map(GaussianParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "SineParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = self._objectlist + map(SineParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "CosineParameter":
             if free == None: free = array([True,True,True,True])
             self._objectlist = self._objectlist + map(CosineParameter,initialvalues[:,0],initialvalues[:,1],initialvalues[:,2],initialvalues[:,3],[free]*self._len)
           elif classType == "UD":
             if free == None: free = array([True])
             self._objectlist = self._objectlist + map(UD,initialvalues[:,0],[free]*self._len)

     def removeObjects(self,indices):
           """
           Remove the objects at indexes = indices.
           MAKE SURE TO ENTER THE INDEXES IN SORTED ORDER FROM HIGH TO LOW!!!!!
           """
           self._len = self._len - len(indices)
           map(self._objectlist.pop,indices)

     def fitLSObject(self,index,y,x = None):
           """
           """
           fitLS(self._objectlist[index],y,x)

     def fitLSObjects(self,y,x = None):
           """
           Use fitLS to fit the data in y with the objects in the objectlist.

           y            - the data to be fitted
           x            - the domain(/dependent variable) of the data to be fitted
           """
           if(y.shape[0] > self._len): raise IndexError, "Number of datasets to be fitted larger than number of objects in the objectlist!"
           if x != None and x.shape == y.shape:
             for i in range(self._len):
               fitLS(self._objectlist[i],y[i,:],x[i,:])
           else:
             for i in range(self._len):
               fitLS(self._objectlist[i],y[i,:],x)

     def fitCSObject(self,index,y,yerr,x = None):
           """
           """
           fitCS(self._objectlist[index],y,yerr,x)

     def fitCSObjects(self,y,yerr,x = None):
           """
           Use fitCS to fit the data in y with the objects in the objectlist.

           y            - a list of arrays containing the data to be fitted: the length of the list should correspond to the number of objects in the list
           yerr         - similar to y but with errors
           x            - a list of arrays containing the domain(/dependent variable) of the data to be fitted
                        OR one array containing the dependent variable of each array in the y-list

           """
           if(len(y) > self._len): raise IndexError, "Number of datasets to be fitted larger than number of objects in the objectlist!"
           if x != None and len(x) == len(y):
             for i in range(self._len):
               fitCS(self._objectlist[i],y[i],yerr[i],x[i],display=0)
           else:
             for i in range(self._len):
               fitCS(self._objectlist[i],y[i,:],yerr[i,:],x,display=0)

     def evaluateObjects(self,x):
           """
           Evaluate the objects at the positions in x.

           x            - domain of evaluation: if 1d array --> evaluate every object at these points
                                                   nd array --> evaluate every object at corresponding column of points (so that the first dimension corresponds to the number of evaluations)
           """
           evaluation = empty((x.shape[0],self._len))
           if x.ndim == 1:
             for i in range(self._len):
               evaluation[:,i] = self._objectlist[i].evaluate(x)
           elif x.shape[1] == self._len:
             for i in range(self._len):
               evaluation[:,i] = self._objectlist[i].evaluate(x[:,i])
           else:
              raise IndexError, 'Dimension of domain x is not valid: too many or too few columns!'
           return evaluation

     def evaluateObjects2(self,x):
           """
           Evaluate the objects at the positions in x.

           x            - domain of evaluation: if 1d array --> evaluate every object at these points
                                                   nd array --> evaluate every object at corresponding column of points (so that the first dimension corresponds to the number of evaluations)
           """
           evaluation = empty(x.shape[0])
           if x.shape[0] == self._len:
             for i in range(self._len):
               evaluation[i] = self._objectlist[i].evaluate(x)[0]
           else:
              raise IndexError, 'Dimension of x does not equal the number of objects in the list! Use other evaluation method!'
           return evaluation

     def __len__(self):
           return self._len

     def __call__(self):
           output = []
           for instance in self._objectlist:
             output.append(instance())
           return output

     def __str__(self):
           output = ""
           for instance in self._objectlist:
             output = output + str(instance) + "\n"
           return output


#################################################################################################
# Some auxiliary functions: if one wishes to compute residuals explicitly, etc.
#################################################################################################

def residuals(x,y,parameter_object):
    """
    Residuals from a gaussian fit.

    @parameter x: domain of evaluation/fit
    @type x: numpy array
    @parameter y: 'measured' values
    @type y: numpy array
    @parameter parameter_object: object with attributes containing values of the necessary parameters
    @type parameter_object: instance of (sub)class (of) Parameter
    """
    fit = parameter_object.evaluate(x)
    return (y-fit)

def chisquare(x,y,parameter_object,yerr):
    """
    Residuals from a gaussian fit.

    @parameter x: domain of evaluation/fit
    @type x: numpy array
    @parameter y: 'measured' values
    @type y: numpy array
    @parameter parameter_object: object with attributes containing values of the necessary parameters
    @type parameter_object: instance of (sub)class (of) Parameter
    """
    fit = parameter_object.evaluate(x)
    return sum(((y-fit)/yerr)**2)

##########

def plot_fit(x,y,parameter_object,yerr=None):
    """
    Make a plot of the fit to the data.

    @parameter x: domain of evaluation/fit
    @type x: numpy array
    @parameter y: 'measured' values
    @type y: numpy array
    @parameter parameter_object: object containing attributes with the necessary parameters and methods
    @type parameter_object: instance of (sub)class (of) Parameter
    """
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    x_range = xmax - xmin
    y_range = ymax - ymin

    try:
       fitSampling = 1./(200*parameter_object.frequency)
    except AttributeError:
       fitSampling = (parameter_object.sigma)/20.

    fitted_x = arange(xmin - 0.1*x_range, xmax + 0.1*x_range,fitSampling)
    fitted_y = parameter_object.evaluate(fitted_x)

    pl.plot(fitted_x,fitted_y,'b-')

    if(yerr == None):
      pl.plot(x,y,'k.')
    else:
      pl.errorbar(x,y,yerr,ls='',marker = '+',color = 'k')

    pl.title("Fit to data")
    pl.axis([xmin - 0.05*x_range, xmax + 0.05*x_range, min(ymin*(3./4),ymin*(4./3)), max(ymax*(3./4),ymax*(4./3))])

    ax = pl.gca()
    for label in ax.xaxis.get_ticklabels():
       label.set_fontsize(16)
    for label in ax.yaxis.get_ticklabels():
       label.set_fontsize(16)
    pl.axes(ax)
    pl.show()

###########

def plot_list(parameter_list,t):
    """
    Make a plot of the evaluation of each object in the parameter_list at times t.
    """
    evaluatedObjects = parameter_list.evaluateObjects(t)

    pl.plot(t,evaluatedObjects)

    ax = pl.gca()
    for label in ax.xaxis.get_ticklabels():
       label.set_fontsize(16)
    for label in ax.yaxis.get_ticklabels():
       label.set_fontsize(16)
    pl.axes(ax)
    pl.show()


##############################################################################################
# Define different ways to "minimize the residuals".
##############################################################################################

def fitLS(parameterObject, y, x = None):
    """
    Fit a function to the data with the least-squares method.

    Example usage:

    Construct a sine wave and add some noise, Plot input in blue and
    noisy data in black:
        >>> xvalues = arange(0,100,1)
        >>> input_p = SineParameter(5.0,0.0,0.025,0.5,array([False,False,False,False]))
        >>> yvalues = input_p.evaluate(xvalues)
        >>> noise = random.normal(size=len(xvalues),scale=0.5)
        >>> z = yvalues+noise
        >>> pl.plot(xvalues,z,'k.')
        >>> pl.plot(xvalues,yvalues,'b-',lw=2)
        >>> pl.show()

    Now we try to fit the noisy data by first creating an object containing our initial guess for the
    different parameters:
        >>> initialFit = SineParameter(1.0,1.0,0.025,0.2,array([True,True,False,True]))

    Now fit the data:
        >>> fitLS(initialFit,z,xvalues)
        >>> plot_fit(xvalues,z,initialFit)

    Compare input with fitted parameters
        >>> print input_p
        amplitude = 5.0 | offset = 0.0 | frequency = 0.025 | phase = 0.5
        >>> print initialFit
        amplitude = 5.0 | offset = 0.0 | frequency = 0.025 | phase = 0.5
    """
    allParNames = parameterObject.attributeNames
    indices = parameterObject.getFreeIndices()
    fitPars = allParNames[indices]
    def f(params):
         """
         The function that changes the values of the "free" parameters during the fitting procedure and
         returns the residuals.
         """
         for i in range(len(fitPars)):
            setattr(parameterObject,fitPars[i],params[i])
         return y - parameterObject.evaluate(x)

    if x is None: x = arange(y.shape[0])
    p = parameterObject()
    p = array(p)[indices]

    leastsq(f, p)


def fitCS(parameterObject, y, yerr, x = None, display = 1):
    """Fit a function to the data with the chi-square method (using fmin)."""
    N = prod(y.shape)
    allParNames = parameterObject.attributeNames
    indices = parameterObject.getFreeIndices()
    fitPars = allParNames[indices]
    if N <= len(fitPars):
        print 'Number of data points too small!'
        norm = 1.
    else:
        norm = 1./(N-len(fitPars))
    def f(params):
         """The function that changes the values of the "free" parameters during the fitting procedure and
            returns the residuals.
         """
         for i in range(len(fitPars)):
            setattr(parameterObject,fitPars[i],params[i])
            #print params,norm*sum(((y - parameterObject.evaluate(x))/yerr)**2)
         return norm*sum(((y - parameterObject.evaluate(x))/yerr)**2)

    if x is None: x = arange(y.shape[0])
    p = parameterObject()
    p = array(p)[indices]

    fmin(f, p, disp=display)


def fitCS_indirect(parameterObject, x, y_indirect, yerr_indirect, x_indirect, object_indirect, display = 1):
    """Fit a function to data through an intermediate step with the chi-square method (using fmin)."""
    N = len(y_indirect)
    allParNames = parameterObject.attributeNames
    indices = parameterObject.getFreeIndices()
    fitPars = allParNames[indices]
    norm = 1./(N-len(fitPars))
    classtype = str(object_indirect.__class__)[20:-2]
    initials = outer(ones_like(x),array(object_indirect()))
    indirect_list = ParameterList(classtype,initials)
    attributeName = (object_indirect.attributeNames)[object_indirect.getFreeIndices()[0]]

    def f(params):
         """The function that changes the values of the "free" parameters during the fitting procedure and
            returns the residuals.
         """
         for i in range(len(fitPars)):
            setattr(parameterObject,fitPars[i],params[i])
         indirect_list.setObjects(attributeName,parameterObject.evaluate(x))
         #print parameterObject(),norm*sum(((y_indirect - indirect_list.evaluateObjects2(x_indirect))/yerr_indirect)**2)
         return norm*sum(((y_indirect - indirect_list.evaluateObjects2(x_indirect))/yerr_indirect)**2)

    if x is None: x = arange(y.shape[0])
    p = parameterObject()
    p = array(p)[indices]

    fmin(f, p, disp=display)

    return norm*sum(((y_indirect - indirect_list.evaluateObjects2(x_indirect))/yerr_indirect)**2)




##############################################################################################
# A function to determine errorbars through a Monte Carlo simulation
##############################################################################################

def montecarlo(parameterListObject, y, yerr, x):
    """
    """
    firstObject = parameterListObject.getObject(0)
    nrDist = len(parameterListObject)
    N = prod(y.shape)
    allParNames = firstObject.attributeNames
    indices = firstObject.getFreeIndices()
    fitPars = allParNames[indices]
    norm = 1./(N-len(fitPars))

    #parameterListObject.fitCSObject(0,y,yerr,x)
    fitCS(firstObject,y,yerr,x,display=0)
    chi2 = norm*chisquare(x,y,firstObject,yerr)
    parameterValues = array(firstObject())

    newy = pl.normal(loc=multiply.outer(ones(nrDist),y),scale=multiply.outer(ones(nrDist),abs(yerr*sqrt(chi2))))
    newyerr = multiply.outer(ones(nrDist),yerr)

    parameterListObject.fitCSObjects(newy,newyerr,x)
    allValues = array(parameterListObject())
    parameterValueErrors = allValues.std(axis=0)
    print allValues.mean(axis=0),median(allValues,axis=0),allValues.std(axis=0)
    #pl.hist(allValues)
    #pl.xlim((0.0,1.2))
    #pl.show()
    return (parameterValues,parameterValueErrors,chi2)



##############################################################################################
# A function to make phasediagrams
##############################################################################################

def phasediagram(times, signal, nu0, D = 0.0, errsignal=None, t0=None, return_indices=False):
    """
    Construct a phasediagram, using frequency nu0.

    Possibility to include a frequency shift D and the zero time point t0.

    If wanted, the indices can be returned. The phased errors are returned
    if a correct errorsignal is given as input.

    Example usage:

        >>> xvalues = arange(0,100,1)
        >>> input_p = SineParameter(5.0,0.0,0.025,0.5,array([False,False,False,False]))
        >>> yvalues = input_p.evaluate(xvalues)
        >>> noise = random.normal(size=len(xvalues),scale=0.5)
        >>> z = yvalues+noise
        >>> phasedx,phasedz = phasediagram(xvalues,z,0.025,t0=10)

    It is essential that all input arrays have the same length along the zeroth axis!! No more than 2 dimensions are allowed for signal and errsignal!
    """
    # the following code checks whether the input is correct: the code is triple-checked and proven to be correct!
    if (times.ndim == signal.ndim):
      if (times.shape != signal.shape):
        raise IndexError, 'Shape mismatch between "times" and "signal".'
    else:
      if (times.shape[0] != signal.shape[0]):
        raise IndexError, 'Shape mismatch between "times" and "signal" along axis 0.'
      times = multiply.outer(times,ones(signal.shape[1]))
    try:
      if (times.shape != errsignal.shape):
        raise IndexError, 'Shape mismatch between "signal" and "errsignal".'
    except AttributeError:
      pass


    if (t0 is None): t0 = times[0]

    phase = fmod(nu0*(times-t0) + D/2.*(times-t0)**2,1.0)
    phase = where(phase<0,phase+1,phase)
    indices_d0 = phase.argsort(axis=0)
    indices_d1 = indices(signal.shape)[1]

    if not return_indices and errsignal != None:
        return phase[indices_d0,indices_d1], signal[indices_d0,indices_d1],errsignal[indices_d0,indices_d1]
    elif errsignal != None:
        return phase[indices_d0,indices_d1], signal[indices_d0,indices_d1],errsignal[indices_d0,indices_d1],indices_d0
    elif return_indices != False:
        return phase[indices_d0,indices_d1], signal[indices_d0,indices_d1],indices_d0
    else:
        return phase[indices_d0,indices_d1], signal[indices_d0,indices_d1]


##############################################################################################
# A function to determine phases with respect to times of maxima that are given
##############################################################################################

def phase(times, maxima):
    """
    """
    N = len(times)
    M = len(maxima)

    if maxima[-1] < times.max():
            raise ValueError, 'Latest maximum should be larger than times of interest.'

    newtimes = outer(ones(M),times)
    newmaxima = outer(maxima,ones(N))

    verschil = newtimes - newmaxima
    replacement = verschil.max()
    verschil = where(verschil < 0.0,replacement,verschil)

    minverschilindex = verschil.argmin(axis=0)
    minverschil = verschil.min(axis=0)

    binwidths = maxima[1:] - maxima[:-1]

    phases = minverschil/binwidths[minverschilindex]

    return phases


###########

def test():
    """
    >>> p = show()
    """
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    test()
