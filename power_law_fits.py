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
from utils import clean_nans
from utils import clean_args
from utils import completeness_flag_neural_net

pi = 3.141592653589793
L_sun = 3.839e26 # W
c = 299792458.0 # m/s
conv_sfr=1.728e-10/10 ** (.23)
gc.enable()

def fit_simple_power_law(p, redshifts, lir, feature1=None, feature2=None, feature3=None,  feature4=None, feature5=None, covar = None):
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
        return lir[ind]- 10**powerlaw[ind]
    else:
        return (lir[ind] - 10**powerlaw[ind]) / covar[ind]

def simple_power_law_fitter(redshifts, lir, additional_features = {}, covar=None):
    fit_params = Parameters()
    fit_params.add('L0',value=9.5)
    fit_params.add('gm_z',value= 1.8, vary = True) #, min = 0.0001, max=3.0)
    kws = {}
    if covar != None: kws['covar']=covar
    kws['lir'] = lir
    j=1
    for i in additional_features:
        fit_params.add('gm_'+i, value= 1.0, vary = True)
        kws['feature'+str(j)] = additional_features[i]
        j+=1

    LMZpl_params = minimize(fit_simple_power_law,fit_params,
        args = (np.ndarray.flatten(redshifts),),
        kws  = kws)

    m = LMZpl_params

    return m

def find_variable_power_law_fit(p, redshifts, lir, stellar_mass, feature1=None, feature2=None, feature3=None,  feature4=None, covar = None):

    v = p.valuesdict()
    A= np.asarray(v['L0'])
    gamma_z = np.asarray(v['gm_z']) * np.ones(len(stellar_mass[0]))

    powerlaw = A +  v['gm_stellar_mass']*np.log10(stellar_mass[0])

    if feature4 != None:
        gamma_z += v['gm_z_'+feature4.keys()[0]] * np.log10(feature4.values()[0])
        powerlaw += v['gm_'+feature4.keys()[0]] * np.log10(feature4.values()[0])
    if feature3 != None:
        gamma_z += v['gm_z_'+feature3.keys()[0]] * np.log10(feature3.values()[0])
        powerlaw += v['gm_'+feature3.keys()[0]] * np.log10(feature3.values()[0])
    if feature2 != None:
        gamma_z += v['gm_z_'+feature2.keys()[0]] * np.log10(feature2.values()[0])
        powerlaw += v['gm_'+feature2.keys()[0]] * np.log10(feature2.values()[0])
    if feature1 != None:
        #pdb.set_trace()
        gamma_z += v['gm_z_'+feature1.keys()[0]] * np.log10(feature1.values()[0])
        powerlaw += v['gm_'+feature1.keys()[0]] * np.log10(feature1.values()[0])

    powerlaw += gamma_z*np.log10(redshifts)

    ind = np.where(clean_nans(powerlaw) > 0)

    if covar == None:
        return lir[ind]- 10**powerlaw[ind]
    else:
        return (lir[ind] - 10**powerlaw[ind]) / covar[ind]

def fast_variable_power_law_fitter(redshifts, lir, additional_features = {}, covar=None):
    fit_params = Parameters()
    fit_params.add('L0',value=9.5)
    fit_params.add('gm_z',value= 1.8, vary = True) #, min = 0.001, max=2.5)
    kws = {}
    if covar != None: kws['covar']=covar
    kws['lir'] = lir
    j=1
    for i in additional_features:
        if i == 'stellar_mass':
            fit_params.add('gm_z_stellar_mass', value= 1e-5, vary = True)
            fit_params.add('gm_stellar_mass', value= 0.1, vary = True)
            kws['stellar_mass'] = additional_features[i].values()
        else:
            fit_params.add('gm_z_'+i, value= 1e-5, vary = True)
            fit_params.add('gm_'+i, value= 0.1, vary = True)
            kws['feature'+str(j)] = additional_features[i]
            j+=1

    LMZpl_params = minimize(find_variable_power_law_fit,fit_params,
        args = (np.ndarray.flatten(redshifts),),
        kws  = kws)

    m = LMZpl_params

    return m

def assign_weights(fn,silent=True):
    if silent == True:
        print fn.var_names
        print 'L0=' + str(fn.params['L0'].value)

    weights = {}

    weights['L0'] = fn.params['L0'].value
    #weights['gm_z'] = fn.params['gm_z'].value

    try:
        if silent == True:
            print 'gm_z = ' +str(fn.params['gm_z'].value)
        weights['gm_z'] = fn.params['gm_z'].value
        try:
            weights['gm_z_z'] =  fn.params['gm_z_z'].value
        except:
            weights['gm_z_z'] = 0.0
    except:
        weights['gm_z'] = 0.0
        weights['gm_z_z'] = 0.0
    try:
        if silent == True:
            print 'gm_mass = ' +str(fn.params['gm_mass'].value)
        weights['gm_mass'] = fn.params['gm_mass'].value
        try:
            weights['gm_z_mass'] =  fn.params['gm_z_mass'].value
        except:
            weights['gm_z_mass'] = 0.0
    except:
        weights['gm_mass'] = 0.0
        weights['gm_z_mass'] = 0.0
    try:
        if silent == True:
            print 'gm_ahat = ' +str(fn.params['gm_a_hat'].value)
        weights['gm_ahat'] = fn.params['gm_a_hat'].value
        try:
            weights['gm_z_ahat'] =  fn.params['gm_z_ahat'].value
        except:
            weights['gm_z_ahat'] = 0.0
    except:
        weights['gm_ahat'] = 0.0
        weights['gm_z_ahat'] = 0.0
    try:
        if silent == True:
            print 'gm_fratio = ' +str(fn.params['gm_f_ratio'].value)
        weights['gm_fratio'] = fn.params['gm_f_ratio'].value
        try:
            weights['gm_z_fratio'] =  fn.params['gm_z_fratio'].value
        except:
            weights['gm_z_fratio'] = 0.0
    except:
        weights['gm_fratio'] = 0.0
        weights['gm_z_fratio'] = 0.0
    try:
        if silent == True:
            print 'gm_uvj = ' + str(fn.params['gm_uvj'].value)
        weights['gm_uvj'] = fn.params['gm_uvj'].value
        try:
            weights['gm_z_uvj'] =  fn.params['gm_z_uvj'].value
        except:
            weights['gm_z_uvj'] = 0.0
    except:
        weights['gm_uvj'] = 0.0
        weights['gm_z_uvj'] = 0.0
    try:
        if silent == True:
            print 'gm_uv = ' + str(fn.params['gm_uv'].value)
        weights['gm_uv'] = fn.params['gm_uv'].value
        try:
            weights['gm_z_uv'] =  fn.params['gm_z_uv'].value
        except:
            weights['gm_z_uv'] = 0.0
    except:
        weights['gm_uv'] = 0.0
        weights['gm_z_uv'] = 0.0
    try:
        if silent == True:
            print 'gm_vj = ' + str(fn.params['gm_vj'].value)
        weights['gm_vj'] = fn.params['gm_uvj'].value
        try:
            weights['gm_z_vj'] =  fn.params['gm_z_vj'].value
        except:
            weights['gm_z_vj'] = 0.0
    except:
        weights['gm_vj'] = 0.0
        weights['gm_z_vj'] = 0.0
    try:
        if silent == True:
            print 'gm_a2t = ' + str(fn.params['gm_a2t'].value)
        weights['gm_a2t'] = fn.params['gm_a2t'].value
        try:
            weights['gm_z_a2t'] =  fn.params['gm_z_a2t'].value
        except:
            weights['gm_z_a2t'] = 0.0
    except:
        weights['gm_a2t'] = 0.0
        weights['gm_z_a2t'] = 0.0
    try:
        if silent == True:
            print 'gm_age = ' + str(fn.params['gm_age'].value)
        weights['gm_age'] = fn.params['gm_age'].value
        try:
            weights['gm_z_age'] =  fn.params['gm_z_age'].value
        except:
            weights['gm_z_age'] = 0.0
    except:
        weights['gm_age'] = 0.0
        weights['gm_z_age'] = 0.0
    try:
        if silent == True:
            print 'gm_tau = ' + str(fn.params['gm_tau'].value)
        weights['gm_tau'] = fn.params['gm_tau'].value
        try:
            weights['gm_z_tau'] =  fn.params['gm_z_tau'].value
        except:
            weights['gm_z_tau'] = 0.0
    except:
        weights['gm_tau'] = 0.0
        weights['gm_z_tau'] = 0.0
    try:
        if silent == True:
            print 'gm_Av = ' + str(fn.params['gm_Av'].value)
        weights['gm_Av'] = fn.params['gm_Av'].value
        try:
            weights['gm_z_Av'] =  fn.params['gm_z_Av'].value
        except:
            weights['gm_z_Av'] = 0.0
    except:
        weights['gm_Av'] = 0.0
        weights['gm_z_Av'] = 0.0

    return weights

def build_power_laws(fn,exes):
    pl = fn.params['L0'].value
    weights = assign_weights(fn)
    for key in exes:
        pl += weights['gm_'+key] * np.log10(exes[key])#[key]
    return pl

def build_training_set(stacked_flux_densities, features_list, znodes, mnodes, knodes, Y_dict=None, k_init = 0, k_final = None, cc=0.5, ngal_cut=10):

    nz = len(znodes)-1
    nm = len(mnodes)-1
    nk = len(knodes)
    if k_final == None:
        k_final = len(knodes)+1
    training_set = {}
    if Y_dict != None:
        keys = [i for i in Y_dict.keys()]
        key = keys[keys != 'err']
        training_set[key] = []
        training_set['covar'] = []
    for ft in features_list:
        training_set[ft] = []

    for k in range(nk)[k_init:k_final]:
        for im in range(nm):
            mn = mnodes[im:im+2]
            m_suf = '{:.2f}'.format(mn[0])+'-'+'{:.2f}'.format(mn[1])
            for iz in range(nz):
                zn = znodes[iz:iz+2]
                z_suf = '{:.2f}'.format(zn[0])+'-'+'{:.2f}'.format(zn[1])

                arg = clean_args('z_'+z_suf+'__m_'+m_suf+'_'+knodes[k])
                if Y_dict != None:
                    training_set[key].append(Y_dict[key][arg][0])
                    training_set['covar'].append(Y_dict['err'][arg])
                ind = stacked_flux_densities.bin_ids[arg]
                ngal = len(ind)
                if ngal > ngal_cut:
                    for ft in features_list:
                        completeness_flag = completeness_flag_neural_net([np.mean(zn)],[np.mean(mn)],sfg=k, completeness_cut = cc)
                        if completeness_flag == True:
                            if ft in['LMASS','lmass','ltau','lage','a_hat_AGN','la2t','Av']:
                                training_set[ft].append(10**subset_averages_from_ids(uVista.table,ind,ft))
                            else:
                                training_set[ft].append(subset_averages_from_ids(uVista.table,ind,ft))
                        else:
                            pass

    return training_set
