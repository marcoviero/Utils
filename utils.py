import pdb
import numpy as np
import cPickle as pickle
from numpy import zeros
from numpy import shape
#from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, minimize, fit_report
pi=3.141592653589793
L_sun = 3.839e26 # W
c = 299792458.0 # m/s
conv_sfr=1.728e-10/10 ** (.23)

## B
def bin_ndarray(ndarray, new_shape, operation='sum'):
  """
  Bins an ndarray in all axes based on the target shape, by summing or
    averaging.

  Number of output dimensions must match number of input dimensions.

  Example
  -------
  >>> m = np.arange(0,100,1).reshape((10,10))
  >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
  >>> print(n)

  [[ 22  30  38  46  54]
   [102 110 118 126 134]
   [182 190 198 206 214]
   [262 270 278 286 294]
   [342 350 358 366 374]]
  """
  if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
    raise ValueError("Operation {} not supported.".format(operation))
  if ndarray.ndim != len(new_shape):
    raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,new_shape))

  compression_pairs = [(d, c//d) for d, c in zip(new_shape,ndarray.shape)]

  flattened = [l for p in compression_pairs for l in p]
  ndarray = ndarray.reshape(flattened)
  for i in range(len(new_shape)):
    if operation.lower() == "sum":
      ndarray = ndarray.sum(-1*(i+1))
    elif operation.lower() in ["mean", "average", "avg"]:
      ndarray = ndarray.mean(-1*(i+1))
  return ndarray

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
def calzetti(lam,A_v,A_lam):
  ''' Calzetti+ 2000 reddening law, see their eqs. 2-4.  Input parameters
    are a vector lam (angstroms) and a scalar A_v.  The output is
    A_lam (the extinction in magnitudes for each value in lam).
    Adapted from the calz_unred procedure in the IDL astro lib.
  '''
  R_V = 4.05             # default value from Calzetti
  ebv = A_v / R_v
  w1 = np.where((lam >= 6300) & (lam <= 22000))
  w2 = np.where((lam >=  912) & (lam <  6300))
  c1 = len(w1)
  c2 = len(w2)
  x  = 10000.0/lam      # Wavelength in inverse microns

  if((c1 + c2) != N_elements(lam)): 
    'Warning - some elements of wavelength vector outside valid domain'

  klam = np.zeros(len(lam))

  if(c1 > 0):
    klam[w1] = 2.659*(-1.857 + 1.040 * x[w1]) + R_V

  if(c2 > 0):
    klam[w2] = 2.659*(poly(x[w2], [-2.156, 1.509, -0.198, 0.011])) + R_V

  A_lam = klam * ebv

  return A_lam

def circle_mask(pixmap,radius_in,pixres):
  ''' Makes a 2D circular image of zeros and ones'''

  radius=radius_in/pixres
  xy = np.shape(pixmap)
  xx = xy[0]
  yy = xy[1]
  beforex = np.log2(xx)
  beforey = np.log2(yy)
  if beforex != beforey:
    if beforex > beforey:
      before = beforex 
    else:
      before = beforey
  else: before = beforey
  l2 = np.ceil(before)
  pad_side = int(2.0 ** l2)
  outmap = np.zeros([pad_side, pad_side])
  outmap[:xx,:yy] = pixmap

  dist_array = shift_twod(dist_idl(pad_side, pad_side), pad_side/2, pad_side/2)
  circ = np.zeros([pad_side, pad_side])
  ind_one = np.where(dist_array <= radius)
  circ[ind_one] = 1.
  mask  = np.real( np.fft.ifft2( np.fft.fft2(circ) *
          np.fft.fft2(outmap)) 
          ) * pad_side * pad_side
  mask = np.round(mask)
  ind_holes = np.where(mask >= 1.0)
  mask = mask * 0.
  mask[ind_holes] = 1.
  maskout = shift_twod(mask, pad_side/2, pad_side/2)

  return maskout[:xx,:yy]

def clean_args(dirty_args):
  return dirty_args.replace('.','p').replace('-','_')

def clean_nans(dirty_array, replacement_char=0.0):
  clean_array = dirty_array
  clean_array[np.isnan(dirty_array)] = replacement_char
  clean_array[np.isinf(dirty_array)] = replacement_char

  return clean_array

def comoving_distance(z,h=cosmo.h,OmM=cosmo.Om0,OmL=cosmo.Ode0,Omk=cosmo.Ok0,dz=0.001,inverse_h=None):
  #Defaults to Planck 2015 cosmology
  H0 = cosmo.H0.value #km / s / Mpc
  D_hubble = 3000. / h # h^{-1} Mpc = 9.26e25 / h; (meters)
  #cosmo = FlatLambdaCDM(H0 = H0 * u.km / u.s / u.Mpc, Om0 = OmM)
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

def cumulative_number_density(z,Mass=np.linspace(9,13,100),sfg=2):
  dM = Mass[1] - Mass[0]
  smf = dM * leja_mass_function(z,Mass=Mass,sfg=sfg)
  return np.cumsum(smf[::-1])[::-1] 

## D
def dist_idl(n1,m1=None):
  ''' Copy of IDL's dist.pro
  Create a rectangular array in which each element is 
  proportinal to its frequency'''

  if m1 == None:
    m1 = int(n1)

  x = np.arange(float(n1))
  for i in range(len(x)): x[i]= min(x[i],(n1 - x[i])) ** 2.

  a = np.zeros([int(n1),int(m1)])

  i2 = m1/2 + 1

  for i in range(i2):
    y = np.sqrt(x + i ** 2.)
    a[:,i] = y
    if i != 0:
      a[:,m1-i]=y 

  return a

## F
def fast_Lir(m,zin): #Tin,betain,alphain,z):
  '''I dont know how to do this yet'''
  #wavelength_range = np.logspace(np.log10(8.),np.log10(1000.),10000.)
  wavelength_range = np.linspace(8.,1000.,10.*992.) 
  model_sed = fast_sed(m,wavelength_range)

  nu_in = c * 1.e6 / wavelength_range
  ns = len(nu_in)
  
  dnu = nu_in[0:ns-1] - nu_in[1:ns]
  dnu = np.append(dnu[0],dnu)
  Lir = np.sum(model_sed * dnu, axis=1)
  conversion = 4.0 * np.pi *(1.0E-13 * cosmo.luminosity_distance(zin) * 3.08568025E22)**2.0 / L_sun # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

  Lrf = Lir * conversion # Jy x Hz
  return Lrf

def fast_double_Lir(m,zin): #Tin,betain,alphain,z):
  '''I dont know how to do this yet'''
  wavelength_range = np.linspace(8.,1000.,10.*992.) 

  v = m.valuesdict()
  betain = np.asarray(v['beta'])
  alphain = np.asarray(v['alpha'])
  A_hot= np.asarray(v['A_hot'])
  A_cold= np.asarray(v['A_cold'])
  T_hot = np.asarray(v['T_hot'])
  T_cold = np.asarray(v['T_cold'])

  #Hot
  p_hot = Parameters()
  p_hot.add('A', value = A_hot, vary = True)
  p_hot.add('T_observed', value = T_hot, vary = True)
  p_hot.add('beta', value = betain, vary = False)
  p_hot.add('alpha', value = alphain, vary = False)
  hot_sed = fast_sed(p_hot,wavelength_range)

  #Hot
  p_cold = Parameters()
  p_cold.add('A', value = A_cold, vary = True)
  p_cold.add('T_observed', value = T_cold, vary = True)
  p_cold.add('beta', value = betain, vary = False)
  p_cold.add('alpha', value = alphain, vary = False)
  cold_sed = fast_sed(p_cold,wavelength_range)

  nu_in = c * 1.e6 / wavelength_range
  ns = len(nu_in)
  
  dnu = nu_in[0:ns-1] - nu_in[1:ns]
  dnu = np.append(dnu[0],dnu)
  Lir_hot = np.sum(hot_sed * dnu, axis=1)
  Lir_cold = np.sum(cold_sed * dnu, axis=1)
  conversion = 4.0 * np.pi *(1.0E-13 * cosmo.luminosity_distance(zin) * 3.08568025E22)**2.0 / L_sun # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

  Lrf_hot = Lir_hot * conversion # Jy x Hz
  Lrf_cold = Lir_cold * conversion # Jy x Hz
  return [Lrf_hot, Lrf_cold]

def fast_sed_fitter(wavelengths, fluxes, covar, betain = 1.8):

  fit_params = Parameters()
  fit_params.add('A', value = 1e-32, vary = True)
  fit_params.add('T_observed', value = 24.0, vary = True)
  fit_params.add('beta', value = betain, vary = False)
  fit_params.add('alpha', value = 2.0, vary = False)

  #nu_in = c * 1.e6 / wavelengths

  sed_params = minimize(find_sed_min,fit_params,
    args=(np.ndarray.flatten(wavelengths),),
    kws={'fluxes':fluxes,'covar':covar})

  m = sed_params.params

  return m

def fast_double_sed_fitter(wavelengths, fluxes, covar, T_cold=15.0, T_hot=30.0):

  fit_params = Parameters()
  fit_params.add('A_hot', value = 1e-40, vary = True)#, min = 0.)
  fit_params.add('A_cold', value = 1e-35, vary = True)#, min = 0.)
  fit_params.add('T_hot', value = T_hot, vary = False, min = 9.0, max = 150.0)
  fit_params.add('T_cold', value = T_cold, vary = False, min = 1.0, max = 20.0)
  fit_params.add('beta', value = 1.80, vary = False)
  fit_params.add('alpha', value = 2.0, vary = False)

  #nu_in = c * 1.e6 / wavelengths

  sed_params = minimize(find_double_sed_min,fit_params,
    args=(np.ndarray.flatten(wavelengths),),
    kws={'fluxes':fluxes,'covar':covar})

  m = sed_params.params

  return m

def find_sed_min(p, wavelengths, fluxes, covar):

  graybody = fast_sed(p,wavelengths) 

  return (fluxes - graybody) / covar
  #return (fluxes - graybody) # np.invert(covar) # (fluxes - graybody) 

def find_double_sed_min(p, wavelengths, fluxes, covar):

  graybody_hot = fast_hot_sed(p,wavelengths) 
  graybody_cold = fast_cold_sed(p,wavelengths) 
  graybody = graybody_hot+graybody_cold

  return (fluxes - graybody) / covar

def fast_hot_sed(m,wavelengths):
  nu_in = c * 1.e6 / wavelengths

  v = m.valuesdict()
  A= np.asarray(v['A_hot'])
  T = np.asarray(v['T_hot'])
  betain = np.asarray(v['beta'])
  alphain = np.asarray(v['alpha'])
  ng = np.size(A)

  ns = len(nu_in)
  base = 2.0 * (6.626)**(-2.0 - betain - alphain) * (1.38)**(3. + betain + alphain) / (2.99792458)**2.0
  expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
  K = base * 10.0**expo
  w_num = A * K * (T * (3.0 + betain + alphain))**(3.0 + betain + alphain) 
  w_den = (np.exp(3.0 + betain + alphain) - 1.0)
  w_div = w_num/w_den 
  nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T

  graybody = np.reshape(A,(ng,1)) * nu_in**np.reshape(betain,(ng,1)) * black(nu_in, T) / 1000.0 
  powerlaw = np.reshape(w_div,(ng,1)) * nu_in**np.reshape(-1.0 * alphain,(ng,1))
  graybody[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]=powerlaw[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]

  return graybody

def fast_cold_sed(m,wavelengths):
  nu_in = c * 1.e6 / wavelengths

  v = m.valuesdict()
  A= np.asarray(v['A_cold'])
  T = np.asarray(v['T_cold'])
  betain = np.asarray(v['beta'])
  alphain = np.asarray(v['alpha'])
  ng = np.size(A)

  ns = len(nu_in)
  base = 2.0 * (6.626)**(-2.0 - betain - alphain) * (1.38)**(3. + betain + alphain) / (2.99792458)**2.0
  expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
  K = base * 10.0**expo
  w_num = A * K * (T * (3.0 + betain + alphain))**(3.0 + betain + alphain) 
  w_den = (np.exp(3.0 + betain + alphain) - 1.0)
  w_div = w_num/w_den 
  nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T

  graybody = np.reshape(A,(ng,1)) * nu_in**np.reshape(betain,(ng,1)) * black(nu_in, T) / 1000.0 
  powerlaw = np.reshape(w_div,(ng,1)) * nu_in**np.reshape(-1.0 * alphain,(ng,1))
  graybody[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]=powerlaw[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]

  return graybody

def fast_double_sed(m,wavelengths):
  nu_in = c * 1.e6 / wavelengths

  v = m.valuesdict()
  A_hot= np.asarray(v['A_hot'])
  A_cold= np.asarray(v['A_cold'])
  T_hot = np.asarray(v['T_hot'])
  T_cold = np.asarray(v['T_cold'])
  betain = np.asarray(v['beta'])
  alphain = np.asarray(v['alpha'])
  ng_hot = np.size(A_hot)
  ng_cold = np.size(A_cold)

  ns = len(nu_in)
  base = 2.0 * (6.626)**(-2.0 - betain - alphain) * (1.38)**(3. + betain + alphain) / (2.99792458)**2.0
  expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
  K = base * 10.0**expo

  #Hot
  w_num_hot = A_hot * K * (T_hot * (3.0 + betain + alphain))**(3.0 + betain + alphain) 
  w_den_hot = (np.exp(3.0 + betain + alphain) - 1.0)
  w_div_hot = w_num_hot/w_den_hot 
  nu_cut_hot= (3.0 + betain + alphain) * 0.208367e11 * T_hot
  graybody_hot = np.reshape(A_hot,(ng_hot,1)) * nu_in**np.reshape(betain,(ng_hot,1)) * black(nu_in, T_hot) / 1000.0 
  powerlaw_hot = np.reshape(w_div_hot,(ng_hot,1)) * nu_in**np.reshape(-1.0 * alphain,(ng_hot,1))
  graybody_hot[np.where(nu_in >= np.reshape(nu_cut_hot,(ng_hot,1)))]=powerlaw_hot[np.where(nu_in >= np.reshape(nu_cut_hot,(ng_hot,1)))]

  #Cold
  w_num_cold = A_cold * K * (T_cold * (3.0 + betain + alphain))**(3.0 + betain + alphain) 
  w_den_cold = (np.exp(3.0 + betain + alphain) - 1.0)
  w_div_cold = w_num_cold/w_den_cold 
  nu_cut_cold = (3.0 + betain + alphain) * 0.208367e11 * T_cold
  graybody_cold = np.reshape(A_cold,(ng_cold,1)) * nu_in**np.reshape(betain,(ng_cold,1)) * black(nu_in, T_cold) / 1000.0 
  powerlaw_cold = np.reshape(w_div_cold,(ng_cold,1)) * nu_in**np.reshape(-1.0 * alphain,(ng_cold,1))
  graybody_cold[np.where(nu_in >= np.reshape(nu_cut_cold,(ng_cold,1)))]=powerlaw_cold[np.where(nu_in >= np.reshape(nu_cut_cold,(ng_cold,1)))]

  return graybody_hot+graybody_cold

def fast_sed(m,wavelengths):
  nu_in = c * 1.e6 / wavelengths

  v = m.valuesdict()
  A= np.asarray(v['A'])
  T = np.asarray(v['T_observed'])
  betain = np.asarray(v['beta'])
  alphain = np.asarray(v['alpha'])
  ng = np.size(A)

  ns = len(nu_in)
  base = 2.0 * (6.626)**(-2.0 - betain - alphain) * (1.38)**(3. + betain + alphain) / (2.99792458)**2.0
  expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
  K = base * 10.0**expo
  w_num = A * K * (T * (3.0 + betain + alphain))**(3.0 + betain + alphain) 
  w_den = (np.exp(3.0 + betain + alphain) - 1.0)
  w_div = w_num/w_den 
  nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T

  graybody = np.reshape(A,(ng,1)) * nu_in**np.reshape(betain,(ng,1)) * black(nu_in, T) / 1000.0 
  powerlaw = np.reshape(w_div,(ng,1)) * nu_in**np.reshape(-1.0 * alphain,(ng,1))
  graybody[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]=powerlaw[np.where(nu_in >= np.reshape(nu_cut,(ng,1)))]

  return graybody

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    
    return array[idx]

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    
    return idx

## G
def gauss_kern(fwhm, side, pixsize):
  ''' Create a 2D Gaussian (size= side x side)'''

  sig = fwhm / 2.355 / pixsize
  delt = zeros([int(side),int(side)])
  delt[0,0]=1.0
  ms = shape(delt)
  delt = shift_twod(delt, ms[0] / 2, ms[1] / 2)
  kern = delt
  gaussian_filter(delt, sig, output= kern)
  kern /= np.max(kern)

  return kern

def get_stellar_mass_at_number_density(zeds,nden,sfg=2):
  nz = np.shape(zeds)[0]
  nn = np.shape(nden)[0]
  sm = np.zeros([nz,nn])
  Mass = np.linspace(8,14,10000) 
  for iz in range(nz):
    cnd = cumulative_number_density(zeds[iz],Mass=Mass,sfg=sfg)
    for jn in range(nn):
      sm[iz,jn] = Mass[find_nearest_index(cnd,10**nden[jn])]

  return sm

def ghz_to_lambda(ghz):
  hz = 1e9*ghz
  c  = 3e8
  lam=c/hz * 1e6
  return lam

## L
def lambda_to_ghz(lam):
  c  = 3e8
  hz=c/(lam*1e-6)
  ghz = 1e-9*hz
  return ghz

def leja_mass_function(z,Mass=np.linspace(9,13,100),sfg=2):
  #sfg = 0  -  Quiescent 
  #sfg = 1  -  Star Forming
  #sfg = 2  -  All

  nz=np.shape(z)

  a1= [-0.10,-0.97,-0.39]
  a2= [-1.69,-1.58,-1.53]
  p1a=[-2.51,-2.88,-2.46]
  p1b=[-0.33, 0.11, 0.07]
  p1c=[-0.07,-0.31,-0.28]
  p2a=[-3.54,-3.48,-3.11]
  p2b=[-2.31, 0.07,-0.18]
  p2c=[ 0.73,-0.11,-0.03]
  ma= [10.70,10.67,10.72]
  mb= [ 0.00,-0.02,-0.13]
  mc= [ 0.00, 0.10, 0.11]

  aone=a1[sfg]+np.zeros(nz)
  atwo=a2[sfg]+np.zeros(nz)
  phione=10**(p1a[sfg] + p1b[sfg]*z + p1c[sfg]*z**2)
  phitwo=10**(p2a[sfg] + p2b[sfg]*z + p2c[sfg]*z**2)
  mstar = ma[sfg] + mb[sfg]*z + mc[sfg]*z**2

  #P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
  P = np.array([aone,mstar,phione,atwo,mstar,phitwo])
  return dschecter(Mass,P)

def loggen(minval, maxval, npoints, linear = None):
  points = np.arange(npoints)/(npoints - 1)
  if (linear != None):
    return (maxval - minval)*points + minval
  else:
    return 10.0 ** ( (np.log10(maxval/minval)) * points + np.log10(minval) )

def L_fun(p,zed):
  '''Luminosities in log(L)'''
  v = p.valuesdict()
  lum = v["s0"] - (1.+(zed/v["zed0"])**(-1.0*v["gamma"]))
  return lum

def L_fit(p, zed, L, Lerr):
  '''Luminosities in log(L)'''
  lum = L_fun(p,zed)
  return (L - lum)/Lerr

## M

def moster_shm(z, Mh): # = 0, nm  =100.0, mmin = 10.0, mmax = 15.0):

  #if Mh == 0:
  #  Mh=np.log10(loggen(10 ** mmin,10 ** mmax,nm))

  #Moster 2013 eqn 2

  M_10=11.590
  M_11=1.195
  N_10=0.0351
  N_11=-0.0247
  b_10=1.376
  b_11=-0.826
  g_10=0.608
  g_11=0.329

  M_1 = 10.0 ** (M_10 + M_11 * (z / (z+1.0) ))
  N   = N_10 + N_11 * (z/ (z+1.0) )
  b   = b_10 + b_11 * (z/ (z+1.0) )
  gam = g_10 + g_11 * (z/ (z+1.0) )

  m_over_M=2.*N/( (Mh/M_1) ** (-1.0*b) + (Mh/M_1) ** (gam)  )

  m_max=M_1*(b/gam) ** (1./(b+gam))

  return m_over_M

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

def poly(X,C):
  n = len(C) - 1 # Find degree of polynomial
  if (n == 0):
    return x*0.0 + c[0]
  else:
    y = c[n]
    for i in range(n-1)[::-1]:
      y = y * x + c[i]
    
    return y

## R
#def round_sig(x, sig=2):
#  return np.round(x, sig-int(np.floor(np.log10(x)))-1)

## S

def dschecter(X,P):
  '''Fits a double Schechter function but using the same M*
     X is alog10(M)
     P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
  '''
  rsch1 = np.log(10.) * P[2] * (10.**((X-P[1])*(1+P[0]))) * np.exp(-10.**(X-P[1]))
  rsch2 = np.log(10.) * P[5] * (10.**((X-P[4])*(1+P[3]))) * np.exp(-10.**(X-P[4]))

  return rsch1+rsch2

def schecter(X,P,exp=None,plaw=None):
  ''' X is alog10(M)
      P[0]=alpha, P[1]=M*, P[2]=phi*
      the output is in units of [Mpc^-3 dex^-1] ???
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

def string_is_true(sraw):
    """Is string true? Returns boolean value.
    """
    s       = sraw.lower() # Make case-insensitive

    # Lists of acceptable 'True' and 'False' strings
    true_strings    = ['true', 't', 'yes', 'y', '1']
    false_strings    = ['false', 'f', 'no', 'n', '0']
    if s in true_strings:
        return True
    elif s in false_strings:
        return False
    else:
        logging.warning("Input not recognized for parameter: %s" % (key))
        logging.warning("You provided: %s" % (sraw))
        raise
        
def solid_angle_from_fwhm(fwhm_arcsec):
  sa = np.pi*(fwhm_arcsec / 3600.0 * np.pi / 180.0)**2.0 / (4.0 * np.log(2.0))
  return sa

## T
def T_fun(p,zed):
  z_T = 1.0
  v = p.valuesdict()
  T = v['T_0'] * ((1+zed)/(1+z_T))**(v['epsilon_T'])
  return T

def T_fit(p, zed, T, Terr):
  Temp = T_fun(p,zed)
  return (T - Temp)/Terr

## V

def viero_2013_luminosities(z,mass,sfg=1):
  import numpy as np
  y = np.array([[-7.2477881 , 3.1599509  , -0.13741485],
               [-1.6335178 , 0.33489572 , -0.0091072162],
               [-7.7579780 , 1.3741780  , -0.061809584 ]])
  ms=np.shape(y)
  npp=ms[0]
  nz=len(z)
  nm=len(mass)

  ex=np.zeros([nm,nz,npp])
  logl=np.zeros([nm,nz])

  for iz in range(nz):
    for im in range(nm):
      for ij in range(npp):
         for ik in range(npp):
          ex[im,iz,ij] += y[ij,ik] * mass[im]**(ik)
      for ij in range(npp):
        logl[im,iz] += ex[im,iz,ij] * z[iz]**(ij)

  T_0 = 27.0
  z_T = 1.0
  epsilon_T = 0.4
  Tdust = T_0 * ((1+np.array(z))/(1.0+z_T))**(epsilon_T)

  return [logl,Tdust]

def viero_2013_luminosities_fast(z,mass,sfg=1):
  import numpy as np
  y = np.array([[-7.2477881 , 3.1599509  , -0.13741485],
               [-1.6335178 , 0.33489572 , -0.0091072162],
               [-7.7579780 , 1.3741780  , -0.061809584 ]])
  ms=np.shape(y)
  npp=ms[0]
  nz=len(z)
  nm=len(mass)

  ex=np.zeros([nm,nz,npp])
  logl=np.zeros([nm,nz])

  for ij in range(npp):
    pdb.set_trace()
    for ik in range(npp):
      ex[:,:,ij] += y[ij,ik] * mass**(ik)
      pdb.set_trace()
    for ij in range(npp):
      logl += ex[:,:,ij] * z**(ij)
      pdb.set_trace()

  T_0 = 27.0
  z_T = 1.0
  epsilon_T = 0.4
  Tdust = T_0 * ((1+np.array(z))/(1.0+z_T))**(epsilon_T)

  return [logl,Tdust]
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
  return zmap

def viero_2013_luminosities_neural_net(z,mass,sfg=1, wpath = '/data/simstack/pickles/', wfile = 'SFR_Mz_Jason_weights_from_neural_network_300layers_N201_SFG.p' ):
  '''
  First attempt at fitting the LMz relation with a neural network.  Not optimized yet.  Also not done for quiescent galaxies.  
  '''

  reg_sfg = pickle.load( open( wpath + wfile, "rb" ) )
  rearrange_x = np.transpose(np.array([mass, z]))

  #pdb.set_trace()
  return np.log10(10**reg_sfg.predict(rearrange_x) / conv_sfr)
