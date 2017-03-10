function estimate_mass,K,z,JK,no_scatter=no_scatter
;; Crude estimate for a galaxy as a function of K_tot, J-K color, and
;; redshift.  Should work reasonably well for objects over 0.2<z<3.5.
;; Written by RFQ on 2/18/13.

;; Inputs are the total K magnitude (AB system), z, and the J-K
;; color. By default this randomly perturbs the masses by 0.2 dex.
;; Set the keyword no_scatter to turn this off (in reality the scatter
;; is probably more like .3 dex and is redshift dependent)

;; These are used to calculate the mass from the K and J-K
zmeans = [0.200, 0.400, 0.600, 0.800, 1.050, 1.350, 1.650, 2.000, 2.400, 2.900, 3.500]
intercepts = [18.2842,18.9785,19.2706,19.1569,20.5633,21.5504,19.6128,19.8258,19.8795,23.1529,22.1678]
slopes = [-0.454737,-0.457170,-0.454706,-0.439577,-0.489793,-0.520825,-0.436967,-0.447071,-0.443592,-0.558047,-0.510875]
slopes_col = [0.0661783,-0.0105074,0.00262891,0.140916,0.0321968,0.0601271,0.470524,0.570098,0.455855,0.0234542,0.0162301]


slope = interpol(slopes,zmeans,z)
intercept = interpol(intercepts,zmeans,z)
slope_col = interpol(slopes_col,zmeans,z)

mass = fltarr(n_elements(K))
for i=0,n_elements(K)-1 do mass[i] = slope[i]*K[i]+intercept[i]+slope_col[i]*JK[i]


;; For objects with messed up J-K colors, estimate the masses based on
;; K alone
ind = where((JK lt 0.) or (JK gt 5.),nbad)
if (nbad ge 1) then begin
   intercepts = [18.3347,18.9626,19.2789,19.6839,20.7085,21.8991,22.9160,24.1886,22.6673,23.1514,21.6482]
   slopes = [-0.456550,-0.456620,-0.455029,-0.460626,-0.495505,-0.534706,-0.570496,-0.617651,-0.543646,-0.556633,-0.487324]
   slope = interpol(slopes,zmeans,z)
   intercept = interpol(intercepts,zmeans,z)
   for i=0,nbad-1 do mass[ind[i]] = slope[ind[i]]*K[ind[i]]+intercept[ind[i]]
endif

if not keyword_set(no_scatter) then mass += randomn(seed,n_elements(mass))*.2

return,mass
end

;TO USE
jj=-2.5*alog10(Jf0[ind_flag[j]])+25.
kk=-2.5*alog10(Kf0[ind_flag[j]])+25.
j_k=jj-kk
mass_norm=estimate_mass(kk,redshifts[j],j_k)
mass_diff=masses[j]-mass_norm[0]
massez[j,n0:n0+n1-1]=estimate_mass(kk,$
    zgrid[ind_above[0:n1-1]],j_k)+mass_diff[0]
