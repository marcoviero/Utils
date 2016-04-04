# ----- Last Updated: 03/30/2016 ----- #

# GJS: This is a demonstration of using penalty functions to perform three-dimension adaptive binning
# GJS: Uncomment the LAST TWO LINES to see how this works

import numpy as np
import matplotlib.pyplot as plt

# Seed random numbers
rand = np.random.RandomState(42)


class ThreeDBinning():
    
    def __init__(self, N_Abins, N_Bbins, N_Cbins, N_thres, N_sample):
        self.N_Abins = N_Abins
        self.N_Bbins = N_Bbins
        self.N_Cbins = N_Cbins
        self.N_Alines = self.N_Abins - 1
        self.N_Blines = self.N_Bbins - 1
        self.N_Clines = self.N_Cbins - 1
        self.N_thres = N_thres
        self.N_sample = N_sample
    
    def make_data(self):
        ###   0 < A < 10, 5 < B < 20, 0 < C < 1   ###
        A_arr = rand.normal(4., 3., self.N_sample)
        B_arr = rand.normal(8., 4., self.N_sample)
        C_arr = rand.normal(0.5, 0.3, self.N_sample)

        data = np.hstack((A_arr[None].T, B_arr[None].T, C_arr[None].T))
        
        # Make uniform bins at the beginning
        A_lp_0 = np.linspace(0,10,self.N_Abins+1)
        B_lp_0 = np.linspace(5,20,self.N_Bbins+1)
        C_lp_0 = np.linspace(0,1,self.N_Cbins+1)

        dic0 = {'A_bins': (self.N_Abins,A_lp_0), 'B_bins': (self.N_Bbins,B_lp_0), 'C_bins': (self.N_Cbins,C_lp_0)}

        dic = dic0
        A_lp_i = np.copy(dic0['A_bins'][1])
        B_lp_i = np.copy(dic0['B_bins'][1])
        C_lp_i = np.copy(dic0['C_bins'][1])

        # Initialize data cube
        data_cube = np.zeros((self.N_Abins, self.N_Bbins, self.N_Cbins))

        # Fill the data cube with the default binning
        for iC in range(self.N_Cbins):
            for iB in range(self.N_Bbins):
                ind = np.where((data[:,1]>=B_lp_0[iB]) & (data[:,1]<B_lp_0[iB+1]) & 
                               (data[:,2]>=C_lp_0[iC]) & (data[:,2]<C_lp_0[iC+1]))[0]
                data_cube[:,iB,iC] = np.histogram(data[ind,0], bins=A_lp_0)[0]
                
        return [data, data_cube, dic0]

    

class AdaptBinning(ThreeDBinning):
    
    def __init__(self, N_Abins=4, N_Bbins=4, N_Cbins=4, N_thres=50, N_sample=10000):
        ThreeDBinning.__init__(self, N_Abins, N_Bbins, N_Cbins, N_thres, N_sample)
        self.N_Abins = N_Abins
        self.N_Bbins = N_Bbins
        self.N_Cbins = N_Cbins
        self.N_Alines = self.N_Abins - 1
        self.N_Blines = self.N_Bbins - 1
        self.N_Clines = self.N_Cbins - 1
        self.N_thres = N_thres
        self.N_sample = N_sample
        
    def Plot_Grid(self, X_Name, Y_Name, Z, dic):
        # Plot the binning in a 2D grid 
        X = dic[X_Name][1]
        Y = dic[Y_Name][1]

        # Get log number counts
        log10_Z = np.log10(Z)
        barvs = np.linspace(0.0,np.log10(self.N_sample/dic[X_Name][0]/dic[Y_Name][0]),5)

        text_xs = (X[0:-1]+X[1::])/2.
        text_ys = (Y[0:-1]+Y[1::])/2.
        plt.figure(figsize=(8,6))
        plt.pcolor(X, Y, log10_Z, cmap='coolwarm', \
                   vmin=0.0, vmax=np.log10(self.N_sample/dic[X_Name][0]/dic[Y_Name][0]))
        for i in range(0,text_xs.shape[0]):
            for j in range(0,text_ys.shape[0]):
                plt.text(text_xs[i],text_ys[j],str(int(Z[j,i])),ha='center',fontsize=12)
        plt.xticks(X, np.round(X,2), fontsize=12)
        plt.yticks(Y, np.round(Y,2), fontsize=12)
        plt.xlabel(X_Name, fontsize=15)
        plt.ylabel(Y_Name, fontsize=15)
        plt.xlim([min(X),max(X)])
        plt.ylim([min(Y),max(Y)])
        cb = plt.colorbar(ticks=barvs)
        cb.ax.tick_params(labelsize=12)
        cb.set_label('Log Number Counts', fontsize=15)
        plt.grid(True)
        plt.show() 

        return 0
    
    
    def P_func_L(self, l_name, l_index, data_cube):
        # ----- Penalty function on the left ----- #
        
        if l_name == 'A_bins':
            temp = data_cube[l_index-1,:,:]
        elif l_name == 'B_bins':
            temp = data_cube[:,l_index-1,:]
        elif l_name == 'C_bins':
            temp = data_cube[:,:,l_index-1]

        if np.sum(temp.flatten())>3.*np.mean([self.N_Abins+self.N_Bbins, self.N_Bbins+self.N_Cbins, \
                                              self.N_Abins+self.N_Cbins])*self.N_thres:
            temp = -np.sum(temp.flatten())**2
        else:
            temp = np.maximum((self.N_thres - temp),0.0)**10
            temp = np.sum(temp.flatten())

        return temp
    
    def P_func_R(self, l_name, l_index, data_cube):
        # ----- Penalty function on the right ----- #
        
        if l_name == 'A_bins':
            temp = data_cube[l_index,:,:]
        elif l_name == 'B_bins':
            temp = data_cube[:,l_index,:]
        elif l_name == 'C_bins':
            temp = data_cube[:,:,l_index]

        if np.sum(temp.flatten())>3.*np.mean([self.N_Abins+self.N_Bbins, self.N_Bbins+self.N_Cbins, \
                                              self.N_Abins+self.N_Cbins])*self.N_thres:
            temp = -np.sum(temp.flatten())**2
        else:
            temp = np.maximum((self.N_thres - temp),0.0)**10
            temp = np.sum(temp.flatten())

        return temp
    
    def P_func_diff(self, l_name, l_index, data_cube):
        return self.P_func_L(l_name, l_index, data_cube) - self.P_func_R(l_name, l_index, data_cube)
    
    
    
    def Run_Adapt(self, subplots=True):
        
        [data, data_cube, dic] = ThreeDBinning(self.N_Abins, self.N_Bbins, self.N_Cbins, \
                                               self.N_thres, self.N_sample).make_data()
        
        print 'Initial Gird =', data_cube
        
        N_bins_tot = self.N_Alines+self.N_Blines+self.N_Clines
        bins_list = []
        for iA in range(self.N_Abins-1):
            bins_list.append('A_bins')
        for iB in range(self.N_Bbins-1):
            bins_list.append('B_bins')
        for iC in range(self.N_Cbins-1):
            bins_list.append('C_bins')
        
        print bins_list
        
        while min(data_cube.flatten())<self.N_thres:
            
            A_lp_i = dic['A_bins'][1]
            B_lp_i = dic['B_bins'][1]
            C_lp_i = dic['C_bins'][1]
            
            # Save values of penalties to a list
            P_func_list = np.empty(N_bins_tot)
            for i in range(N_bins_tot):
                if i<self.N_Alines:
                    P_func_list[i] = self.P_func_diff(bins_list[i], i+1, data_cube)
                elif i<self.N_Alines+self.N_Blines:
                    P_func_list[i] = self.P_func_diff(bins_list[i], i%self.N_Alines+1, data_cube)
                elif i<self.N_Alines+self.N_Blines+self.N_Clines:
                    P_func_list[i] = self.P_func_diff(bins_list[i], i%(self.N_Alines+self.N_Blines)+1, data_cube)
            
            #print P_func_list
            
            # Move the boarder with the largest penalty
            i_move = np.argmax(abs(P_func_list))
            
            
            if i_move<self.N_Abins-1:
                if P_func_list[i_move]>0.0:
                    A_lp_i[i_move+1] += (A_lp_i[i_move+2] - A_lp_i[i_move+1])*0.1
                elif P_func_list[i_move]<0.0:
                    A_lp_i[i_move+1] += (A_lp_i[i_move] - A_lp_i[i_move+1])*0.1
            
            elif i_move<self.N_Abins-1+self.N_Bbins-1:
                if P_func_list[i_move]>0.0:
                    B_lp_i[i_move-self.N_Alines+1] += (B_lp_i[i_move-self.N_Alines+2] - \
                                                       B_lp_i[i_move-self.N_Alines+1])*0.1
                elif P_func_list[i_move]<0.0:
                    B_lp_i[i_move-self.N_Alines+1] += (B_lp_i[i_move-self.N_Alines] - \
                                                       B_lp_i[i_move-self.N_Alines+1])*0.1
            
            elif i_move<self.N_Abins-1+self.N_Bbins-1+self.N_Cbins-1:
                if P_func_list[i_move]>0.0:
                    C_lp_i[i_move-self.N_Alines-self.N_Blines+1] += (C_lp_i[i_move-self.N_Alines-self.N_Blines+2] - \
                                                                     C_lp_i[i_move-self.N_Alines-self.N_Blines+1])*0.1
                elif P_func_list[i_move]<0.0:
                    C_lp_i[i_move-self.N_Alines-self.N_Blines+1] += (C_lp_i[i_move-self.N_Alines-self.N_Blines] -\
                                                                     C_lp_i[i_move-self.N_Alines-self.N_Blines+1])*0.1
            
            
            dic = {'A_bins': (self.N_Abins,A_lp_i), 'B_bins': (self.N_Bbins,B_lp_i), 'C_bins': (self.N_Cbins,C_lp_i)}
            
            
            # Initialize data cube
            data_cube = np.zeros((self.N_Abins, self.N_Bbins, self.N_Cbins))

            # Fill the data cube with the default binning
            for iC in range(self.N_Cbins):
                for iB in range(self.N_Bbins):
                    ind = np.where((data[:,1]>=B_lp_i[iB]) & (data[:,1]<B_lp_i[iB+1]) & 
                                   (data[:,2]>=C_lp_i[iC]) & (data[:,2]<C_lp_i[iC+1]))[0]
                    data_cube[:,iB,iC] = np.histogram(data[ind,0], bins=A_lp_i)[0]
    
            print 'NSources_min = %d' % min(data_cube.flatten())
        
        print 'Final Gird =', data_cube
        
        if subplots == True:
            self.Plot_Grid('B_bins', 'A_bins', np.sum(data_cube,axis=2), dic)
            self.Plot_Grid('C_bins', 'A_bins', np.sum(data_cube,axis=1), dic)
            self.Plot_Grid('C_bins', 'B_bins', np.sum(data_cube,axis=0), dic)
        
        return 0
    
#sample = AdaptBinning()
#sample.Run_Adapt()