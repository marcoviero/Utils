import numpy as np
import os as os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, interp1d
import itertools

#%matplotlib inline

# Get the path of the master catalog file
#_HOME = os.environ.get('HOME')

rand = np.random.RandomState(42)

np.set_printoptions(precision=4,suppress=True)



###############     Mass Estimator (from R. Quadri)     ###############
# Data points for interpolation
zmeans = [0.200, 0.400, 0.600, 0.800, 1.050, 1.350, 1.650, 2.000, 2.400, 2.900, 3.500]
intercepts = [18.2842,18.9785,19.2706,19.1569,20.5633,21.5504,19.6128,19.8258,19.8795,23.1529,22.1678]
slopes = [-0.454737,-0.457170,-0.454706,-0.439577,-0.489793,-0.520825,-0.436967,-0.447071,-0.443592,-0.558047,-0.510875]
slopes_cols = [0.0661783,-0.0105074,0.00262891,0.140916,0.0321968,0.0601271,0.470524,0.570098,0.455855,0.0234542,0.0162301]

intercepts_b = [18.3347,18.9626,19.2789,19.6839,20.7085,21.8991,22.9160,24.1886,22.6673,23.1514,21.6482]
slopes_b = [-0.456550,-0.456620,-0.455029,-0.460626,-0.495505,-0.534706,-0.570496,-0.617651,-0.543646,-0.556633,-0.487324]

data_list = [zmeans, intercepts, slopes, slopes_cols, intercepts_b, slopes_b]


# Interpolate the data points, and force the extrapolation to asymptote the mean of data points. 
def intercept_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    # Interpolating data
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, intercepts, k=3)
    elif flag == 'magnitude_only':
        sm = InterpolatedUnivariateSpline(zmeans, intercepts_b, k=3)
    else:
        raise ValueError('Invalid flag name!')
    # Forcing extrapolation to asymptote
    ans = sm(z) * np.exp(-dist) + np.mean(intercepts) * (1.-np.exp(-dist))
    return ans

def slope_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, slopes, k=3)
    elif flag == 'magnitude_only':
        sm = InterpolatedUnivariateSpline(zmeans, slopes_b, k=3)
    else:
        raise ValueError('Invalid flag name!')
    ans = sm(z) * np.exp(-dist) + np.mean(slopes) * (1.-np.exp(-dist))
    return ans

def slope_col_full(z, flag='color_magnitude'):
    dist = np.maximum((z - 0.2)*(z - 3.5), 0)
    if flag == 'color_magnitude':
        sm = InterpolatedUnivariateSpline(zmeans, slopes_cols, k=3)
        ans = sm(z) * np.exp(-dist) + np.mean(slopes_cols) * (1.-np.exp(-dist))
        return ans
    elif flag == 'magnitude_only':
        return 0
    else:
        raise ValueError('Invalid flag name!')

func_list = [intercept_full, slope_full, slope_col_full]


# Define the mass estimation function
def Mass(K, JK, z):
    """ Return the mass estimate for the given K-band magnitude, J-K color and redshift """
    if (JK < 0.) or (JK > 5.):
        flag = 'magnitude_only'
    else:
        flag = 'color_magnitude'
    model = slope_full(z, flag) * K + intercept_full(z, flag) + slope_col_full(z, flag) * JK
    return model
Mass = np.vectorize(Mass)

# Flux to magnitude conversion adopted by the UltraVISTA catalog
def FluxToMagnitude(flux, ap_corr):
    return 25.0 - 2.5*np.log10(flux*ap_corr)
    


####################   Extended Bootstrapping Technique (EBT)   ####################
# --- EBT assembles new bins for stacking, rather than drawing from original bins
# Step 1: draw simulated redshifts from the photometric redshift probability distribution
# Step 2: estimate the mass using the perturbed redshift and observed K magnitude and J-K color
# Step 3: a simulated catalog is split up into (original) bins and calculate new stacked flux densities
# Step 4: repeat Step 1-3 many (>1000) times to complete the "bootstrapping"

n_bt = 100   # Number of bootstrapping

path_cat = 'C:/Users/Jsun1992/Desktop/stacking/data_collection/catalogs/UVISTA_DR2_master_v2.1_USE.csv'
#path_cat = _HOME + path_cat

col_to_read = ['ra','dec','z_peak','l68','u68','J','Ks','ap_corr','lmass','rf_U_V','rf_V_J']

df_cat_in = pd.read_csv(path_cat,usecols=col_to_read)
header_list = list(df_cat_in.columns.values)

cat_in = df_cat_in.as_matrix(columns=df_cat_in.columns)
n_sources = cat_in.shape[0]; n_params = cat_in.shape[1]
print 'Size Read-In: ', cat_in.shape

c_z_peak = header_list.index('z_peak')
c_z_l68 = header_list.index('l68')
c_z_u68 = header_list.index('u68')
c_J = header_list.index('J')
c_Ks = header_list.index('Ks')
c_ap_corr = header_list.index('ap_corr')
c_lmass = header_list.index('lmass')

path_EBT_zs = 'C:/Users/Jsun1992/Desktop/stacking/z_pert_array.dat'
with open(path_EBT_zs) as file:
    z_pert_arr = np.asarray([[float(digit) for digit in line.split()] for line in itertools.islice(file, 0, None)])

for i_bt in range(n_bt):
    short_catalog_arr = np.zeros((n_sources,n_params))
    for i in range(n_sources):
        short_catalog_arr[i,:] = cat_in[i,:]
        #mean_std = abs(cat_in[i,c_z_u68] - cat_in[i,c_z_l68])/2.
        #z_pert = rand.normal(cat_in[i,c_z_peak],mean_std,1)
        #d_z_l68 = np.maximum(cat_in[i,c_z_peak]-cat_in[i,c_z_l68], 1.0e-6)
        #d_z_u68 = np.maximum(cat_in[i, c_z_u68] - cat_in[i, c_z_peak], 1.0e-6)
        #print 'i=', i
        #print 'd_z_l68:', d_z_l68
        #print 'd_z_u68:', d_z_u68
        #z_pert_l = abs(rand.normal(0., d_z_l68 ,1)) * 0.01
        #z_pert_u = abs(rand.normal(0., d_z_u68, 1)) * 0.01
        #draw_pn = rand.normal(0.,1.,1)
        #if draw_pn > 0.:
        #    z_pert = cat_in[i,c_z_peak] + z_pert_u
        #else:
        #    z_pert = cat_in[i,c_z_peak] - z_pert_l
        z_cat = cat_in[i,c_z_peak]
        z_pert = z_pert_arr[i, i_bt]
        if abs(z_pert - z_cat) > 1.:
            z_pert = z_cat
        Mag_J = FluxToMagnitude(cat_in[i,c_J], cat_in[i,c_ap_corr])
        Mag_K = FluxToMagnitude(cat_in[i,c_Ks], cat_in[i,c_ap_corr])
        short_catalog_arr[i,c_z_peak] = z_pert
        short_catalog_arr[i,c_lmass] = Mass(Mag_K, Mag_J-Mag_K, z_pert)

        if (i+1)%10000==0:
            print "Done perturbing %d sources." % (i+1)

    # Write short catalogs (useful columns only) to files
    #path = 'C:/Users/Jsun1992/Desktop/stacking/EBT_catalogs/sim_catalog_%d.csv' % i_bt
    #path = 'C:/Users/Jsun1992/Desktop/stacking/EBT_catalogs_ASY/sim_catalog_%d.csv'%i_bt
    path = 'C:/Users/Jsun1992/Desktop/stacking/EBT_catalogs_PZ/sim_catalog_%d.csv' % i_bt
    #data_to_store = np.hstack((short_catalog_arr[:,i_bt,:2], short_catalog_arr[:,i_bt,4:]))
    data_to_store = short_catalog_arr
    np.savetxt(path, data_to_store, fmt='%.6f', delimiter=',', 
               header='%s' % str(','.join(header_list)), comments='')
    print 'Wrote %d simulated catalogs to .csv file' % (i_bt+1)

print "Done writing simulated catalogs to files."
