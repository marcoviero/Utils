import pdb
import six
import numpy as np
from astropy.io import fits
from utils import bin_ndarray as rebin
from utils import gauss_kern
from utils import clean_nans

class Skymaps:

	def __init__(self,file_map,file_noise,psf,color_correction=1.0):
		''' This Class creates Objects for a set of maps/noisemaps/beams/TransferFunctions/etc., 
		at each Wavelength.
		This is a work in progress!
		Issues:  If the beam has a different pixel size from the map, it is not yet able to 
		re-scale it.  Just haven't found a convincing way to make it work.   
		Future Work:
		Will shift some of the work into functions (e.g., read psf, color_correction) 
		and increase flexibility.
		'''
		#READ MAPS
		if file_map == file_noise:
			#SPIRE Maps have Noise maps in the second extension.  
			cmap, hd = fits.getdata(file_map, 1, header = True)
			cnoise, nhd = fits.getdata(file_map, 2, header = True)
		else:
			#This assumes that if Signal and Noise are different maps, they are contained in first extension
			cmap, hd = fits.getdata(file_map, 0, header = True)
			cnoise, nhd = fits.getdata(file_noise, 0, header = True)

		#GET MAP PIXEL SIZE	
		if 'CD2_2' in hd:
			pix = hd['CD2_2'] * 3600.
		else:
			pix = hd['CDELT2'] * 3600.

		#READ BEAMS
		#Check first if beam is a filename (actual beam) or a number (approximate with Gaussian)
		if isinstance(psf, six.string_types):
			beam, phd = fits.getdata(psf, 0, header = True)
			#GET PSF PIXEL SIZE	
			if 'CD2_2' in phd:
				pix_beam = phd['CD2_2'] * 3600.
			elif 'CDELT2' in phd:
				pix_beam = phd['CDELT2'] * 3600.
			else: pix_beam = pix
			#SCALE PSF IF NECESSARY 
			if np.round(10.*pix_beam) != np.round(10.*pix):
				raise ValueError("Beam and Map have different size pixels")
				scale_beam = pix_beam / pix
				pms = np.shape(beam)
				new_shape=(np.round(pms[0]*scale_beam),np.round(pms[1]*scale_beam))
				pdb.set_trace()
				kern = rebin(clean_nans(beam),new_shape=new_shape,operation='ave')
				#kern = rebin(clean_nans(beam),new_shape[0],new_shape[1])
			else: 
				kern = clean_nans(beam)
			self.psf_pixel_size = pix_beam
		else:
			sig = psf / 2.355 / pix
			kern = gauss_kern(psf, np.floor(psf * 8.), pix)

		self.map = clean_nans(cmap) * color_correction
		self.noise = clean_nans(cnoise,replacement_char=1e10) * color_correction
		self.header = hd
		self.pixel_size = pix
		self.psf = kern

	def beam_area_correction(self,beam_area):
		self.map *= beam_area * 1e6
		
	def add_wavelength(self,wavelength):
		self.wavelength = wavelength

	def add_fwhm(self,fwhm):
		self.fwhm = fwhm

	def add_weights(self,file_weights):
		weights, whd = fits.getdata(file_weights, 0, header = True)
		#pdb.set_trace()
		self.noise = clean_nans(1./weights,replacement_char=1e10)


class Field_catalogs:
	def __init__(self, tbl):
		self.table = tbl
		self.nsrc = len(tbl)

	def separate_sf_qt(self):
		sfg = np.ones(self.nsrc)
		#pdb.set_trace()
		for i in range(self.nsrc):
			if (self.table.rf_U_V.values[i] > 1.3) and (self.table.rf_V_J.values[i] < 1.5):
				if (self.table.z_peak.values[i] < 1):
					if (self.table.rf_U_V.values[i] > (self.table.rf_V_J.values[i]*0.88+0.69) ): sfg[i]=0
				if (self.table.z_peak.values[i] > 1):
					if (self.table.rf_U_V.values[i] > (self.table.rf_V_J.values[i]*0.88+0.59) ): sfg[i]=0
		#indsf = np.where(sfg == 1)
		#indqt = np.where(sfg == 0)
		self.table['sfg'] = sfg

	#def separate_nuv_r(self):


	def get_sf_qt_mass_redshift_bins(self, znodes, mnodes):
		self.id_z_ms = {}
		for iz in range(len(znodes[:-1])):
			for jm in range(len(mnodes[:-1])):
				ind_mz_sf =( (self.table.sfg == 1) & (self.table.z_peak >= znodes[iz]) & (self.table.z_peak < znodes[iz+1]) & 
					(10**self.table.LMASS >= 10**mnodes[jm]) & (10**self.table.LMASS < 10**mnodes[jm+1]) ) 
				ind_mz_qt =( (self.table.sfg == 0) & (self.table.z_peak >= znodes[iz]) & (self.table.z_peak < znodes[iz+1]) & 
					(10**self.table.LMASS >= 10**mnodes[jm]) & (10**self.table.LMASS < 10**mnodes[jm+1]) ) 

				#self.id_z_ms['z_'+str(znodes[iz])+'-'+str(znodes[iz+1])+'__m_'+str(mnodes[jm])+'-'+str(mnodes[jm+1])+'_sf'] = self.table.ID[ind_mz_sf].values 
				#self.id_z_ms['z_'+str(znodes[iz])+'-'+str(znodes[iz+1])+'__m_'+str(mnodes[jm])+'-'+str(mnodes[jm+1])+'_qt'] = self.table.ID[ind_mz_qt].values 
				#self.id_z_ms['z_'+str(znodes[iz]).replace('.','p')+'_'+str(znodes[iz+1]).replace('.','p')+'__m_'+str(mnodes[jm]).replace('.','p')+'_'+str(mnodes[jm+1]).replace('.','p')+'_sf'] = self.table.ID[ind_mz_sf].values 
				#self.id_z_ms['z_'+str(znodes[iz]).replace('.','p')+'_'+str(znodes[iz+1]).replace('.','p')+'__m_'+str(mnodes[jm]).replace('.','p')+'_'+str(mnodes[jm+1]).replace('.','p')+'_qt'] = self.table.ID[ind_mz_qt].values 
				self.id_z_ms['z_'+str(round(znodes[iz],3)).replace('.','p')+'_'+str(round(znodes[iz+1],3)).replace('.','p')+'__m_'+str(round(mnodes[jm],3)).replace('.','p')+'_'+str(round(mnodes[jm+1],3)).replace('.','p')+'_sf'] = self.table.ID[ind_mz_sf].values 
				self.id_z_ms['z_'+str(round(znodes[iz],3)).replace('.','p')+'_'+str(round(znodes[iz+1],3)).replace('.','p')+'__m_'+str(round(mnodes[jm],3)).replace('.','p')+'_'+str(round(mnodes[jm+1],3)).replace('.','p')+'_qt'] = self.table.ID[ind_mz_qt].values 

	def get_parent_child_redshift_bins(self,znodes):
		self.id_z_sed = {}
		for ch in self.table.parent.unique():
			for iz in range(len(znodes[:-1])):
				self.id_z_sed['z_'+str(znodes[iz]).replace('.','p')+'_'+str(znodes[iz+1]).replace('.','p')+'__sed'+str(ch)] = self.table.ID[ (self.table.parent == ch) & (self.table.z_peak >= znodes[iz]) & (self.table.z_peak < znodes[iz+1]) ].values 

	def get_parent_child_bins(self):
		self.id_children = {}
		for ch in self.table.parent.unique():
			self.id_children['sed'+str(ch)] = self.table.ID[self.table.parent == ch].values 

	def subset_positions(self,radec_ids):
		''' This positions function is very general.  
			User supplies IDs dictionary, function returns RA/DEC dictionaries with the same keys'''
		#self.ra_dec = {}
		ra_dec = {}
		#ra = {}
		#dec = {}
		#pdb.set_trace()
		for k in radec_ids.keys():
			ra  = self.table.ra[self.table.ID.isin(radec_ids[k])].values
			dec = self.table.dec[self.table.ID.isin(radec_ids[k])].values
			#self.ra_dec[k] = [ra,dec] 
			#ra[k]  = ra
			#dec[k] = dec
			ra_dec[k] = [ra,dec]
		return ra_dec





