import pdb
import numpy as np
import pylab as plt
from utils import find_nearest_index

class Parents_and_Children_SEDs:

    def __init__(self,seds,ids,subsets=4000,u=0.075,leftover_percent = 0.005):
        self.ids = ids
        self.seds = seds
        self.id_parents = []
        
        if subsets > 0:
            indsub = np.random.choice(range(len(ids)),subsets,replace=False)
            sedsub = self.seds[indsub]
            idsub = self.ids[indsub]
        else:
            sedsub = self.seds
            idsub = self.ids

        all_seds =      compare_all_seds(sedsub,u=u)
        pnc_frst =      extract_parent_and_analog_seds(all_seds)
        self.id_parents.append(idsub[pnc_frst[0]])
        parent_sed =    self.seds[pnc_frst[0]]
        analog_seds=    self.seds[pnc_frst[1]]
        remaining_seds= self.seds[pnc_frst[2]]
        remaining_ids = idsub[pnc_frst[2]]
        
        #pdb.set_trace()
        cnt=1
        remaining_fraction = len(remaining_seds)/float(len(sedsub))
        while remaining_fraction > leftover_percent:
            print 'counting' +str(cnt)
            rest_seds = compare_all_seds(remaining_seds,u=u)
            pnc_next = extract_parent_and_analog_seds(rest_seds)
            self.id_parents.append(remaining_ids[pnc_next[0]])
            parent_sed = remaining_seds[pnc_next[0]]
            analog_seds= remaining_seds[pnc_next[1]]
            remaining_seds=remaining_seds[pnc_next[2]] 
            remaining_ids = remaining_ids[pnc_next[2]]
            remaining_fraction = len(remaining_seds)/float(len(sedsub))
            cnt+=1
            
    def get_children(self, ):
        self.id_children = []
        for i in self.id_parents:
            child_seds = compare_seds(self.seds[np.where(self.ids == i)][0],self.seds)
            self.id_children.append(self.ids[child_seds[0]])
            
    def get_childrens_parents(self):
        self.id_childrens_parents = []
        #Need a new function that has seda but more general!
        indp = [np.where(self.ids == self.id_parents[x])[0][0] for x in range(len(self.id_parents))]
        class_all_seds = classify_all_seds(self.seds[indp], self.seds)
        self.id_childrens_parents = np.argmin(class_all_seds,axis=0)
        
    def plot_parents(self,rflam):
        plt.figure()
        plt.xlim(1e3,2.5e4)
        xnorm = find_nearest_index(16000,rflam)
        for i in self.id_parents:
            yplt = self.seds[np.where(self.ids == i)][0]
            plt.loglog(rflam,yplt/yplt[xnorm])
        plt.show()
        
    def plot_histogram(self):
        self.get_childrens_parents()
        plt.figure()
        plt.hist(self.id_childrens_parents,bins=range(len(self.id_parents)))

def compare_seds(seda,sedsb,u=0.05):
    nlam = np.shape(sedsb)[1]
    a12 = (np.sum(seda * sedsb,axis=1) / np.sum(sedsb**2,axis=1) )
    atile = np.tile(a12,(nlam,1))
    #pdb.set_trace()
    b12 = np.sqrt(np.sum(((seda) - (np.multiply(atile.T,sedsb)))**2,axis=1)/np.sum(seda)**2)
    idx = np.where((b12 > 0) & (b12 < u) )
    #pdb.set_trace()
    return [idx[0],a12[idx]]

def compare_all_seds(seds,u=0.05):
    ng = np.shape(seds)[0]
    nlam = np.shape(seds)[1]
    
    ##This is right!  Genius!##################
    sedtile = np.tile(seds,(ng)).reshape(ng,ng,nlam) 
    a12_num = np.sum(np.multiply(sedtile,seds),axis=2) 
    a12_denom = np.sum(seds**2,axis=1) 
    a12 = (a12_num / a12_denom) 
    a12_tile = np.tile(np.reshape(a12,(ng,ng,1)),(1,1,nlam))
    
    b12 = np.sqrt(np.sum( (sedtile - a12_tile*seds)**2,axis=2) / np.sum(sedtile,axis=2)**2)
    #pdb.set_trace()
    ##########################################
    
    idx = (b12 > 0) & (b12 < u)
    
    return idx

def classify_all_seds(parent_seds,all_seds):
    ng = np.shape(all_seds)[0]
    nlam = np.shape(all_seds)[1]
    npts= np.shape(parent_seds)[0]
    
    ##This is right!  Genius!##################
    parent_tile = np.tile(parent_seds,(ng)).reshape(npts,ng,nlam) 
    a12_num = np.sum(np.multiply(parent_tile,all_seds),axis=2) 
    a12_denom = np.sum(all_seds**2,axis=1) 
    a12 = (a12_num / a12_denom) 
    a12_tile = np.tile(np.reshape(a12,(npts,ng,1)),(1,1,nlam))
    b12 = np.sqrt(np.sum( (parent_tile - a12_tile*all_seds)**2,axis=2) / np.sum(parent_tile,axis=2)**2)
    ##########################################
    
    return b12

def extract_parent_and_analog_seds(idx):
    ind_rank= np.argsort(np.sum(idx,axis=1))[::-1]
    ind_max = np.argmax(np.sum(idx,axis=1))
    ind_analogs = np.where(idx[ind_max] == True)
    ind_else    = np.where(idx[ind_max] == False)
    
    out = np.delete(ind_else,np.where(ind_else == ind_max)[1])
    #pdb.set_trace()
    return [ind_max, ind_analogs[0], out] #np.delete(ind_else,np.where(ind_else == ind_max))[0]

def filter_childrens_parents(pids, sedall, ids):
    #id_cps = []
    indp = [np.where(ids == pids[x])[0][0] for x in range(len(pids))]
    class_all_seds = classify_all_seds(sedall[indp], sedall)
    id_cps = np.argmin(class_all_seds,axis=0)
    #pdb.set_trace()
    return [ids[id_cps], id_cps]


