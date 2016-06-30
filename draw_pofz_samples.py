from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq, fsolve

from struct import *


#fileName = '/Users/guochaosun/Desktop/Caltech_OBSCOS/pofz/UVISTA_DR2_master_v2.1.tempfilt'

fileName = '/data/gjsun/catalogs/pofz_calc/UVISTA_DR2_master_v2.1.tempfilt'

with open(fileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
    
NFILT, NTEMP, NZ, NOBJ = unpack("iiii", fileContent[:16])
print NFILT, NTEMP, NZ, NOBJ


tempfilt_flat = unpack("d"*NFILT*NTEMP*NZ, fileContent[16:16+8*NFILT*NTEMP*NZ])
lc_flat = unpack("d"*NFILT, fileContent[16+8*NFILT*NTEMP*NZ: 16+8*NFILT*NTEMP*NZ + 8*NFILT])
zgrid_flat = unpack("d"*NZ, fileContent[16+8*NFILT*NTEMP*NZ + 8*NFILT: 16+8*NFILT*NTEMP*NZ + 8*NFILT + 8*NZ])

zgrid = np.array(list(zgrid_flat))

#fileName = '/Users/guochaosun/Desktop/Caltech_OBSCOS/pofz/UVISTA_DR2_master_v2.1.pz'
fileName = '/data/gjsun/catalogs/pofz_calc/UVISTA_DR2_master_v2.1.pz'

with open(fileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
    
NZ, NOBJ = unpack("ii", fileContent[:8])

chi2fit_flat = unpack("d"*NZ*NOBJ, fileContent[8:8*NZ*NOBJ+8])
chi2fit_arr = np.array(list(chi2fit_flat)).reshape(NOBJ,NZ)
chi2fit_arr = chi2fit_arr.T

pz = chi2fit_arr
pz = np.exp(-0.5*pz)

pz_final = pz/np.trapz(pz[:,0], zgrid)



# Seed random numbers
rand = np.random.RandomState(42)

dim = 100

# Define a z_pert array
# z_pert_arr = np.zeros((NOBJ, dim))

file_to_write = open('/data/gjsun/catalogs/pofz_calc/z_pert_array.dat', 'w+')

for igal in range(NOBJ):
#for igal in range(10):

    xpts = zgrid
    ypts = pz_final[:,igal]

    ncounts = xpts.shape[0]
    pdf_norm = ypts/np.sum(ypts)
    cpdf = np.zeros(ncounts+1)

    for i in range(1,ncounts+1):
        cpdf[i] = cpdf[i-1] + pdf_norm[i-1]
    long_xpts = np.append(xpts, [xpts[-1]*2-xpts[-2]])

    sm_cpdf = InterpolatedUnivariateSpline(long_xpts, cpdf)
    
    
    def sm_cpdf_filt(x):
        return np.minimum(np.maximum(sm_cpdf(x),1.0E-9),1.-1.0E-9)

    
    def get_new_x(prob_in, x_data):
        f_to_solve = lambda x: sm_cpdf_filt(x) - prob_in
        #f_to_solve = lambda x: sm_cpdf(x) - prob_in
        #print 'x_data =', x_data
        try:
            soln = brentq(f_to_solve, x_data[0]-0.001, x_data[-1]+0.1)
            return soln
        except:
            #print prob_in
            #plt.plot(xpts, pdf_norm, 'r-')
            #plt.show()
            #print sm_cpdf_filt(x_data[0]-0.001), sm_cpdf_filt(x_data[-1]+0.1)
            #print f_to_solve(x_data[0]-0.001), f_to_solve(x_data[-1]+0.1)
            print '------------ FAULTY ------------'
            print 'Value Returned: ', xpts[np.argmax(pdf_norm)]
            return xpts[np.argmax(pdf_norm)]
    
    
    prob_in = rand.uniform(1.0E-2,9.9E-1,dim)
    
    for j in range(dim):
        #z_pert_arr[igal, j] = get_new_x(prob_in[j], xpts)
        file_to_write.write( '%.5f\t' % get_new_x(prob_in[j], xpts) )
    file_to_write.write('\n')
    
    #samples = get_new_x(prob_in, xpts)
    #z_pert_arr[igal, :] = samples
    if igal%1000 == 0:
    	print 'DONE %d galaxies!' % igal
    #weights = np.ones_like(samples)/float(len(samples))
    #plt.hist(samples, bins=20, weights=weights)
    #plt.plot(xpts, pdf_norm, 'r-')
    #plt.xlim([0.0,1.0])
    #plt.show()
    
exit(0)