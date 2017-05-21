import pdb
import gc
import numpy as np
from numpy import zeros
from numpy import shape
import cPickle as pickle
#from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from scipy.ndimage.filters import gaussian_filter
import scipy.io
from scipy import fftpack
from lmfit import Parameters, minimize, fit_report
from radial_data import radial_data
pi=3.141592653589793
L_sun = 3.839e26 # W
c = 299792458.0 # m/s
conv_sfr=1.728e-10/10 ** (.23)
gc.enable()

def simple_power_law_fitter(redshifts, lir, additional_features = {}, covar=None):
    fit_params = Parameters()
    fit_params.add('L0',value=9.5)
    fit_params.add('gm_z',value= 1.8, vary = True, min = 0.001, max=2.5)
    kws = {}
    if covar != None: kws['covar']=covar
    kws['lir'] = lir
    j=1
    for i in additional_features:
        fit_params.add('gm_'+i, value= 1.0, vary = True,min=1e-5)
        kws['feature'+str(j)] = additional_features[i]
        j+=1

    LMZpl_params = minimize(find_simple_power_law_fit,fit_params,
        args = (np.ndarray.flatten(redshifts),),
        kws  = kws)

    m = LMZpl_params

    return m

def simple_power_law_fit(p, redshifts, lir, feature1=None, feature2=None, feature3=None,  feature4=None, covar = None):
    v = p.valuesdict()
    A= np.asarray(v['L0'])

    powerlaw = A + np.asarray(v['gm_z'])*np.log10(redshifts)

    if feature4 != None:
        powerlaw += v['gm_'+feature4.keys()[0]] * np.log10(feature4.values()[0])
    if feature3 != None:
        powerlaw += v['gm_'+feature3.keys()[0]] * np.log10(feature3.values()[0])
    if feature2 != None:
        powerlaw += v['gm_'+feature2.keys()[0]] * np.log10(feature2.values()[0])
    if feature1 != None:
        powerlaw += v['gm_'+feature1.keys()[0]] * np.log10(feature1.values()[0])

    ind = np.where(clean_nans(powerlaw) > 0)
    if covar == None:
        return (np.log10(lir[ind])- powerlaw[ind])
    else:
        return (np.log10(lir[ind]) - powerlaw[ind]) / np.log10(covar[ind])
