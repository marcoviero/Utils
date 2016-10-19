import pdb
import numpy as np
from astropy.wcs import WCS
from utils import circle_mask
from utils import dist_idl
from utils import gauss_kern
from utils import pad_and_smooth_psf
from utils import shift_twod
from utils import smooth_psf
from lmfit import Parameters, minimize, fit_report

def L_vs_z_m_polynomial_fn(p, zz, mm):
	v = p.valuesdict()
	degreeA = v['degreeA']
	degreeB = v['degreeB']
	nm = len(mm)
	nz = len(zz)

	B = np.zeros([degreeA, nm])
	for j in range(degreeB):  #Offset, slope, (2nd order)
		for i in range(degreeA): #Fit to fluxes
			B[i,:] += v[str('A'+str(i)+str(j))] * mm ** j

	Lout = np.zeros([nz,nm])
	for k in range(nm):
		yk = np.zeros(nz)
		for i in range(degreeA):
			yk += B[i,k] * zz ** i
		Lout[:,k] = yk

	return Lout 

def L_vs_z_m_polynomial_fit(p, zz, mm, L, L_err = None, lam_reg = 0.0):

	v = p.valuesdict()
	degreeA = v['degreeA']
	degreeB = v['degreeB']
	nm = len(mm)
	nz = len(zz)

	B = np.zeros([degreeA, nm])
	for j in range(degreeB):  #Offset, slope, (2nd order)
		for i in range(degreeA): #Fit to fluxes
			B[i,:] += v[str('A'+str(i)+str(j))] * mm ** j

	Lout = L_vs_z_m_polynomial_fn(p, zz, mm)

	return (Lout - L) / L_err + lam_reg / 2.0 / len(L) * np.sum(B ** 2.)


