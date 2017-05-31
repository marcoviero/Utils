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
from utils import subset_averages_from_ids

pi = 3.141592653589793
L_sun = 3.839e26 # W
c = 299792458.0 # m/s
conv_sfr=1.728e-10/10 ** (.23)
gc.enable()

def fast_variable_power_law_wrapper(feature_dict_in, redshifts=None, Y_array=None, covar=None):
    additional_features = {}
    feature_dict = feature_dict_in.copy()
    if redshifts == None:
        redshifts = feature_dict.pop('z_peak')
    if Y_array == None:
        Y_array = feature_dict.pop('Y_val')
    if covar == None:
        covar = feature_dict.pop('Y_err')
    for fkey in feature_dict:
        additional_features[fkey] = {fkey: np.ndarray.flatten(np.array(feature_dict[fkey]))}
    return fast_variable_power_law_fitter(np.ndarray.flatten(np.array(redshifts)), np.ndarray.flatten(np.array(Y_array)), additional_features = additional_features, covar=np.ndarray.flatten(np.array(covar)))

def build_power_laws(fn,exes):
    pl = fn.params['L0'].value
    weights = assign_weights_cat_names(fn)
    for key in exes:
        pl += weights['gm_'+key] * np.log10(exes[key])#[key]
    return pl

def build_training_set(db,stacked_flux_densities, features_list, znodes, mnodes, knodes, Y_dict=None, k_init = 0, k_final = None, cc=0.5, ngal_cut=10):

    nz = len(znodes)-1
    nm = len(mnodes)-1
    nk = len(knodes)
    if k_final == None:
        k_final = len(knodes)+1
    training_set = {}
    if Y_dict != None:
        keys = [i for i in Y_dict.keys()]
        key = keys[keys != 'Y_err']
        training_set[key] = []
        training_set['Y_err'] = []
    for ft in features_list:
        if ft in ['LMASS','lmass']:
            training_set['stellar_mass'] = []
        else:
            training_set[ft] = []

    for k in range(nk)[k_init:k_final]:
        for im in range(nm):
            mn = mnodes[im:im+2]
            m_suf = '{:.2f}'.format(mn[0])+'-'+'{:.2f}'.format(mn[1])
            for iz in range(nz):
                zn = znodes[iz:iz+2]
                z_suf = '{:.2f}'.format(zn[0])+'-'+'{:.2f}'.format(zn[1])

                arg = clean_args('z_'+z_suf+'__m_'+m_suf+'_'+knodes[k])
                ind = stacked_flux_densities.bin_ids[arg]
                ngal = len(ind)
                completeness_flag = completeness_flag_neural_net([np.mean(zn)],[np.mean(mn)],sfg=k, completeness_cut = cc)

                if ((ngal > ngal_cut) & (completeness_flag == True)):
                    if Y_dict != None:
                        training_set[key].append(Y_dict[key][arg][0])
                        training_set['Y_err'].append(Y_dict['Y_err'][arg])
                    for ft in features_list:
                        if ft in['ltau','lage','a_hat_AGN','la2t','Av']:
                            training_set[ft].append(10**subset_averages_from_ids(db.table,ind,ft))
                        elif ft in['LMASS','lmass']:
                            training_set['stellar_mass'].append(10**subset_averages_from_ids(db.table,ind,ft))
                        else:
                            training_set[ft].append(subset_averages_from_ids(db.table,ind,ft))

    return training_set

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
    if silent == False:
        #print fn.var_names
        print 'L0=' + str(fn.params['L0'].value)

    weights = {}

    weights['L0'] = fn.params['L0'].value
    #weights['gm_z'] = fn.params['gm_z'].value

    try:
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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
        if silent == False:
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

def assign_weights_cat_names(fn,silent=True):
    if silent == False:
        #print fn.var_names
        print 'L0=' + str(fn.params['L0'].value)

    weights = {}

    weights['L0'] = fn.params['L0'].value
    #weights['gm_z'] = fn.params['gm_z'].value

    try:
        if silent == False:
            print 'gm_z = ' +str(fn.params['gm_z'].value)
        weights['gm_z'] = fn.params['gm_z'].value
        try:
            if silent == False:
                print 'gm_z_z = ' +str(fn.params['gm_z_z'].value)
            weights['gm_z_z'] =  fn.params['gm_z_z'].value
        except:
            weights['gm_z_z'] = 0.0
    except:
        weights['gm_z'] = 0.0
        weights['gm_z_z'] = 0.0
    try:
        if silent == False:
            print 'gm_stellar_mass = ' +str(fn.params['gm_stellar_mass'].value)
        weights['gm_stellar_mass'] = fn.params['gm_stellar_mass'].value
        try:
            if silent == False:
                print 'gm_z_stellar_mass = ' +str(fn.params['gm_z_stellar_mass'].value)
            weights['gm_z_stellar_mass'] =  fn.params['gm_z_stellar_mass'].value
        except:
            weights['gm_z_stellar_mass'] = 0.0
    except:
        weights['gm_stellar_mass'] = 0.0
        weights['gm_z_stellar_mass'] = 0.0
    try:
        if silent == False:
            print 'gm_ahat_AGN = ' +str(fn.params['gm_a_hat_AGN'].value)
        weights['gm_ahat_AGN'] = fn.params['gm_a_hat_AGN'].value
        try:
            if silent == False:
                print 'gm_z_a_hat_AGN = ' +str(fn.params['gm_z_a_hat_AGN'].value)
            weights['gm_z_ahat_AGN'] =  fn.params['gm_z_ahat_AGN'].value
        except:
            weights['gm_z_ahat_AGN'] = 0.0
    except:
        weights['gm_ahat_AGN'] = 0.0
        weights['gm_z_ahat_AGN'] = 0.0
    try:
        if silent == False:
            print 'gm_F_ratio = ' +str(fn.params['gm_F_ratio'].value)
        weights['gm_F_ratio'] = fn.params['gm_F_ratio'].value
        try:
            if silent == False:
                print 'gm_z_F_ratio = ' +str(fn.params['gm_z_F_ratio'].value)
            weights['gm_z_F_ratio'] =  fn.params['gm_z_F_ratio'].value
        except:
            weights['gm_z_F_ratio'] = 0.0
    except:
        weights['gm_F_ratio'] = 0.0
        weights['gm_z_F_ratio'] = 0.0
    try:
        if silent == False:
            print 'gm_UVJ = ' + str(fn.params['gm_UVJ'].value)
        weights['gm_UVJ'] = fn.params['gm_UVJ'].value
        try:
            if silent == False:
                print 'gm_z_UVJ = ' +str(fn.params['gm_z_UVJ'].value)
            weights['gm_z_UVJ'] =  fn.params['gm_z_UVJ'].value
        except:
            weights['gm_z_UVJ'] = 0.0
    except:
        weights['gm_UVJ'] = 0.0
        weights['gm_z_UVJ'] = 0.0
    try:
        if silent == False:
            print 'gm_UV = ' + str(fn.params['gm_UV'].value)
        weights['gm_UV'] = fn.params['gm_UV'].value
        try:
            if silent == False:
                print 'gm_z_UV = ' +str(fn.params['gm_z_UV'].value)
            weights['gm_z_UV'] =  fn.params['gm_z_UV'].value
        except:
            weights['gm_z_UV'] = 0.0
    except:
        weights['gm_UV'] = 0.0
        weights['gm_z_UV'] = 0.0
    try:
        if silent == False:
            print 'gm_VJ = ' + str(fn.params['gm_VJ'].value)
        weights['gm_VJ'] = fn.params['gm_VJ'].value
        try:
            if silent == False:
                print 'gm_z_VJ = ' +str(fn.params['gm_z_VJ'].value)
            weights['gm_z_VJ'] =  fn.params['gm_z_VJ'].value
        except:
            weights['gm_z_VJ'] = 0.0
    except:
        weights['gm_VJ'] = 0.0
        weights['gm_z_VJ'] = 0.0
    try:
        if silent == False:
            print 'gm_la2t = ' + str(fn.params['gm_la2t'].value)
        weights['gm_la2t'] = fn.params['gm_la2t'].value
        try:
            if silent == False:
                print 'gm_z_la2t = ' +str(fn.params['gm_z_la2t'].value)
            weights['gm_z_la2t'] =  fn.params['gm_z_la2t'].value
        except:
            weights['gm_z_la2t'] = 0.0
    except:
        weights['gm_la2t'] = 0.0
        weights['gm_z_la2t'] = 0.0
    try:
        if silent == False:
            print 'gm_lage = ' + str(fn.params['gm_lage'].value)
        weights['gm_lage'] = fn.params['gm_lage'].value
        try:
            if silent == False:
                print 'gm_z_lage = ' +str(fn.params['gm_z_lage'].value)
            weights['gm_z_lage'] =  fn.params['gm_z_lage'].value
        except:
            weights['gm_z_lage'] = 0.0
    except:
        weights['gm_lage'] = 0.0
        weights['gm_z_lage'] = 0.0
    try:
        if silent == False:
            print 'gm_ltau = ' + str(fn.params['gm_ltau'].value)
        weights['gm_ltau'] = fn.params['gm_ltau'].value
        try:
            if silent == False:
                print 'gm_z_ltau = ' +str(fn.params['gm_z_ltau'].value)
            weights['gm_z_ltau'] =  fn.params['gm_z_ltau'].value
        except:
            weights['gm_z_ltau'] = 0.0
    except:
        weights['gm_ltau'] = 0.0
        weights['gm_z_ltau'] = 0.0
    try:
        if silent == False:
            print 'gm_Av = ' + str(fn.params['gm_Av'].value)
        weights['gm_Av'] = fn.params['gm_Av'].value
        try:
            if silent == False:
                print 'gm_z_Av = ' +str(fn.params['gm_z_Av'].value)
            weights['gm_z_Av'] =  fn.params['gm_z_Av'].value
        except:
            weights['gm_z_Av'] = 0.0
    except:
        weights['gm_Av'] = 0.0
        weights['gm_z_Av'] = 0.0

    return weights
