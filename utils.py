import pdb
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
pi=3.141592653589793

## B
def black(nu_in, T):
  #h = 6.623e-34     ; Joule*s
  #k = 1.38e-23      ; Joule/K
  #c = 3e8           ; m/s
  # (2*h*nu_in^3/c^2)*(1/( exp(h*nu_in/k*T) - 1 )) * 10^29

  a0 = 1.4718e-21   # 2*h*10^29/c^2
  a1 = 4.7993e-11   # h/k

  num = a0 * nu_in**3.0
  den = np.exp(a1 * np.outer(1.0/T,nu_in)) - 1.0 
  ret = num / den                 
  
  return ret

## C
def clean_nans(dirty_array, replacement_char=0.0):
  clean_array = dirty_array
  clean_array[np.isnan(dirty_array)] = replacement_char
  clean_array[np.isinf(dirty_array)] = replacement_char

  return clean_array

def comoving_distance(z,h=0.6774,OmM=0.3089,OmL=0.6911,Omk=0.0,dz=0.001,inverse_h=None):
  #Defaults to Planck 2015 cosmology
  H0 = 100. * h #km / s / Mpc
  D_hubble = 3000. / h # h^{-1} Mpc = 9.26e25 / h; (meters)
  cosmo = FlatLambdaCDM(H0 = H0 * u.km / u.s / u.Mpc, Om0 = OmM)
  n_z = z/dz 
  i_z = np.arange(n_z)*dz
  D_c = 0.0
  for i in i_z:
    E = np.sqrt(OmM*(1.+i)**3. + OmL)
    D_c += D_hubble * dz / E

    return D_c

def comoving_volume(zed1, zed2, mpc=None):
  if zed1 < zed2:
    z1 = zed1
    z2 = zed2
  else:
    z1 = zed2
    z2 = zed1
  comovo=1e-9* 4./3.* pi * (comoving_distance(z2)**3. - comoving_distance(z1)**3.)
  if mpc != None:
    comovo *= 1e3**3.0

  return comovo

def comoving_number_density(number, area, z1, z2, ff=1.0, mpc=None, verbose=None):
  #if z2 != None: zin2 = 0.0
  vol = comoving_volume(z2,z1,mpc=1)
  num = (number/(area*ff)) * (180.0/pi)**2.0 * 4.0 * pi
  comovnumden=num/vol

  return comovnumden

def comoving_volume_given_area(area, zz1, zz2, mpc=None, arcmin=None):
  if arcmin != None: 
      ff = 3600. 
  else:
    ff=1. 
  vol0=comoving_volume(zz1,zz2,mpc=mpc)
  vol=((area*ff)/(180./pi)**2.)/(4.*pi)*vol0

  return vol

## D
def dist_idl(n1,m1=None):
  ''' Copy of IDL's dist.pro
  Create a rectangular array in which each element is 
  proportinal to its frequency'''

  if m1 == None:
    m1 = n1

  x = np.arange(float(n1))
  for i in range(len(x)): x[i]= min(x[i],(n1 - x[i])) ** 2.

  a = np.zeros([float(n1),float(m1)])

  i2 = m1/2 + 1

  for i in np.arange(i2):
    y = np.sqrt(x + i ** 2.)
    a[:,i] = y
    if i != 0:
      a[:,m1-i]=y 

  return a

## G
def gauss_kern(fwhm, side, pixsize):
  ''' Create a 2D Gaussian (size= side x side)'''
  from scipy.ndimage.filters import gaussian_filter
  from shift import shift_twod
  import numpy as np
  from numpy import zeros
  from numpy import shape

  sig = fwhm / 2.355 / pixsize
  delt = zeros([side,side])
  delt[0,0]=1.0
  ms = shape(delt)
  delt = shift_twod(delt, ms[0] / 2, ms[1] / 2)
  kern = delt
  gaussian_filter(delt, sig, output= kern)
  kern /= np.max(kern)

  return kern

## L
def lambda_to_ghz(lam):
  c  = 3e8
  hz=c/(lam*1e-6)
  ghz = 1e-9*hz
  return ghz

def loggen(minval, maxval, npoints, linear = None):
  points = np.arange(npoints)/(npoints - 1)
  if (linear != None):
    return (maxval - minval)*points + minval
  else:
    return 10.0 ** ( (np.log10(maxval/minval)) * points + np.log10(minval) )

## P
def pad_and_smooth_psf(mapin, psfin):

  s = np.shape(mapin)
  mnx = s[0]
  mny = s[1]

  s = np.shape(psfin)
  pnx = s[0]
  pny = s[1]

  psf_x0 = pnx/2
  psf_y0 = pny/2
  psf = psfin
  px0 = psf_x0
  py0 = psf_y0

  # pad psf
  psfpad = np.zeros([mnx, mny])
  psfpad[0:pnx,0:pny] = psf

  # shift psf so that centre is at (0,0)
  psfpad = shift_twod(psfpad, -px0, -py0)
  smmap = np.real( np.fft.ifft2( np.fft.fft2(zero_pad(mapin) ) *
    np.fft.fft2(zero_pad(psfpad)) ) ) 

  return smmap[0:mnx,0:mny]

def planck(wav, T): 
  #nuvector = c * 1.e6 / lambdavector # Hz from microns??
  h = 6.626e-34
  c = 3.0e+8
  k = 1.38e-23
  a = 2.0 * h * c**2
  b = h * c / (wav * k * T)
  intensity = a / ( (wav**5) * (np.exp(b) - 1.0) )

  return intensity

## S
def schecter(X,P,exp=None,plaw=None):
  '''# X is alog10(M)
  # P[0]=alpha, P[1]=M*, P[2]=phi*
  # the output is in units of [Mpc^-3 dex^-1] ???
  '''  
  if exp != None: 
    return np.log(10.) * P[2] * np.exp(-10**(X - P[1]))
  if plaw != None: 
    return np.log(10.) * P[2] * (10**((X - P[1])*(1+P[0])))
  return np.log(10.) * P[2] * (10.**((X-P[1])*(1.0+P[0]))) * np.exp(-10.**(X-P[1]))

def shift(seq, x):
  from numpy import roll
  out = roll(seq, int(x))
  return out 

def shift_twod(seq, x, y):
  from numpy import roll
  out = roll(roll(seq, int(x), axis = 1), int(y), axis = 0)
  return out 

def shift_bit_length(x):
  return 1<<(x-1).bit_length()

def smooth_psf(mapin, psfin):

  s = np.shape(mapin)
  mnx = s[0]
  mny = s[1]

  s = np.shape(psfin)
  pnx = s[0]
  pny = s[1]

  psf_x0 = pnx/2
  psf_y0 = pny/2
  psf = psfin
  px0 = psf_x0
  py0 = psf_y0

  # pad psf
  psfpad = np.zeros([mnx, mny])
  psfpad[0:pnx,0:pny] = psf

  # shift psf so that centre is at (0,0)
  psfpad = shift_twod(psfpad, -px0, -py0)
  smmap = np.real( np.fft.ifft2( np.fft.fft2(mapin) *
    np.fft.fft2(psfpad))
    ) 

  return smmap

## Z
def zero_pad(cmap,l2=0):
  ms=np.shape(cmap)
  if l2 == 0:
    l2 = max([shift_bit_length(ms[0]),shift_bit_length(ms[1])])
  if ms[0] <= l2 and ms[1] <=l2:
    zmap=np.zeros([l2,l2])
    zmap[:ms[0],:ms[1]]=cmap 
  else:
    zmap=cmap
  return zma
