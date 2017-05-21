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

    LMZpl_params = minimize(fit_simple_power_law,fit_params,
        args = (np.ndarray.flatten(redshifts),),
        kws  = kws)

    m = LMZpl_params

    return m

def fit_simple_power_law(p, redshifts, lir, feature1=None, feature2=None, feature3=None,  feature4=None, covar = None):
    v = p.valuesdict()
    A= np.asarray(v['L0'])

    powerlaw = A + np.asarray(v['gm_z'])*np.log10(redshifts)

    if feature5 != None:
        powerlaw += v['gm_'+feature5.keys()[0]] * np.log10(feature5.values()[0])
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

def build_power_laws(fn,exes):
    pl = fn['M0']
    weights = assign_weights(fn)
    for key in weights:
        pl += weights[key] * exes[key]
    return pl

def assign_weights(fn):
    print fn.var_names
    print 'L0=' + str(fn.params['L0'].value)
    print 'gm_z=' + str(fn.params['gm_z'].value)

    weights = {}

    weights['M0'] = fn.params['L0'].value
    weights['gm_z'] = fn.params['gm_z'].value

    try:
        print 'gm_stellar_mass = ' +str(fn.params['gm_stellar_mass'].value)
        weights['gm_mass'] = fn.params['gm_stellar_mass'].value
        weights['gm_z_mass'] = 0.0
    except:
        weights['gm_mass'] = 0.0
        weights['gm_z_mass'] = 0.0
    try:
        print 'gm_ahat = ' +str(fn.params['gm_a_hat'].value)
        weights['gm_ahat'] = fn.params['gm_a_hat'].value
        weights['gm_z_ahat'] = 0.0
    except:
        weights['gm_ahat'] = 0.0
        weights['gm_z_ahat'] = 0.0
    try:
        print 'gm_fratio = ' +str(fn.params['gm_f_ratio'].value)
        weights['gm_fratio'] = fn.params['gm_f_ratio'].value
        weights['gm_z_fratio'] = 0.0
    except:
        weights['gm_fratio'] = 0.0
        weights['gm_z_fratio'] = 0.0
    try:
        print 'gm_uvj = ' + str(fn.params['gm_uvj'].value)
        weights['gm_uvj'] = fn.params['gm_uvj'].value
        weights['gm_z_uvj'] = 0.0
    except:
        weights['gm_uvj'] = 0.0
        weights['gm_z_uvj'] = 0.0
    try:
        print 'gm_uv = ' + str(fn.params['gm_uv'].value)
        weights['gm_uv'] = fn.params['gm_uv'].value
        weights['gm_z_uv'] = 0.0
    except:
        weights['gm_uv'] = 0.0
        weights['gm_z_uv'] = 0.0
    try:
        print 'gm_vj = ' + str(fn.params['gm_vj'].value)
        weights['gm_vj'] = fn.params['gm_uvj'].value
        weights['gm_z_vj'] = 0.0
    except:
        weights['gm_vj'] = 0.0
        weights['gm_z_vj'] = 0.0
    try:
        print 'gm_a2t = ' + str(fn.params['gm_a2t'].value)
        weights['gm_a2t'] = fn.params['gm_a2t'].value
        weights['gm_z_a2t'] = 0.0
    except:
        weights['gm_a2t'] = 0.0
        weights['gm_z_a2t'] = 0.0
    try:
        print 'gm_age = ' + str(fn.params['gm_age'].value)
        weights['gm_age'] = fn.params['gm_age'].value
        weights['gm_z_age'] = 0.0
    except:
        weights['gm_age'] = 0.0
        weights['gm_z_age'] = 0.0
    try:
        print 'gm_tau = ' + str(fn.params['gm_tau'].value)
        weights['gm_tau'] = fn.params['gm_tau'].value
        weights['gm_z_tau'] = 0.0
    except:
        weights['gm_tau'] = 0.0
        weights['gm_z_tau'] = 0.0
    try:
        print 'gm_Av = ' + str(fn.params['gm_Av'].value)
        weights['gm_Av'] = fn.params['gm_Av'].value
        weights['gm_z_Av'] = 0.0
    except:
        weights['gm_Av'] = 0.0
        weights['gm_z_Av'] = 0.0

    return weights
