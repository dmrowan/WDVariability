#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
from WDranker_2 import badflag_bool, catalog_match
from matplotlib.patheffects import withStroke
import _pickle as pickle
from scipy import optimize
from scipy.stats import norm, chisquare
from progressbar import ProgressBar
import _pickle as pickle
from math import log10, floor

desc="""
WD_SED: Define a class for SEDs
"""

#Things to happen in this class:
#Find SED VOSA information
#Find Teff from gentile fits
#Plot that blackbody
#Try fitting shit

class WDSED:
    def __init__(self, source):
        assert(type(source) == str)
        self.source = source
        
        assert(os.path.isfile("IS.csv"))
        df_IS = pd.read_csv('IS.csv')
        idx_IS = np.where(df_IS['MainID'] == self.source)[0]
        assert(len(idx_IS) == 1)
        idx_IS = idx_IS[0]

        #originally from gentile fusillo fits
        self.TeffH = df_IS['Teff_H'][idx_IS]
        self.loggH = df_IS['log_g_H'][idx_IS]
        self.chi2H = df_IS['Chi2_H'][idx_IS]
        
        self.TeffHe = df_IS['Teff_He'][idx_IS]
        self.loggHe = df_IS['log_g_He'][idx_IS]
        self.chi2He = df_IS['Chi2_He'][idx_IS]

        if self.chi2H < self.chi2He:
            self.atmfit = "H"
            self.Teff = self.TeffH
            self.logg = self.loggH
            self.chi2 = self.chi2H
        else:
            self.atmfit = "He"
            self.Teff = self.TeffHe
            self.logg = self.loggHe
            self.chi2 = self.chi2He

        
        self.objecttype = df_IS['type'][idx_IS]
        self.labelnum = df_IS['labelnum'][idx_IS]
        #Read in vosa sed info
        fpath = self.objecttype+"/"+source+"/sed.dat"
        if not os.path.isfile(fpath):
            return("No sed file found for {}".format(self.source))
        else:
            with open(fpath) as f:
                lines = f.readlines()

        #Initialize dictionary of all filters
        dic_all = {
                'filter':[], 
                'wavelength':[],
                'flux':[],
                'flux_err':[],
                'uplim':[]
            }

        #Iterate through lines
        for line in lines:
            if line.split()[0] != '#' and len(line.split()) != 1:
                if float(line.split()[4]) < 0:
                    continue
                else:
                    dic_all['filter'].append(line.split()[0])
                    dic_all['wavelength'].append(float(line.split()[1]))
                    dic_all['flux'].append(float(line.split()[4]))
                    dic_all['flux_err'].append(float(line.split()[5]))
                    if line.split()[-1] == '-1':
                        dic_all['uplim'].append(1)
                    else:
                        dic_all['uplim'].append(0)
        
        df_all = pd.DataFrame(dic_all)
        
        idx_wise = []
        for idx in range(len(df_all['filter'])):
            if 'WISE' in df_all['filter'][idx]:
                idx_wise.append(idx)

        df_wise = df_all.loc[idx_wise]
        df_wise = df_wise.reset_index(drop=True)
        df_nonIR = df_all.drop(index=idx_wise)
        df_nonIR = df_nonIR.reset_index(drop=True)

        #Assign to object attributes
        self.df_all = df_all
        self.df_wise = df_wise
        self.df_nonIR = df_nonIR


    def addWISE(self, W1=None, W1err=None, W2=None, W2err=None, 
                W3=None, W3err=None, W4=None, W4err=None):

        #Found at http://wise2.ipac.caltech.edu/docs/
        #release/allsky/expsup/sec4_4h.html
        F01 = 309.540
        F02 = 171.787
        F03= 31.674
        F04 = 8.363
        F0list = [F01, F02, F03, F04]

        l1 = 3.4*1e4
        l2 = 4.6*1e4
        l3 = 12*1e4
        l4 = 22*1e4
        llist = [l1, l2, l3, l4]

        Wlist = [W1, W2, W3, W4]
        Werr_list = [W1err, W2err, W3err, W4err]
        flux_densities = []
        errors = []

        for i in range(len(Wlist)):
            W = Wlist[i]
            Werr = Werr_list[i]
            F0 = F0list[i]
            l = llist[i]
            if W is not None:
                f = (3e-5)*F0*10**(-W/2.5)
                f = f / (l**2)
                flux_densities.append(f)
                if Werr is not None:
                    ferr = (3e-5)*F0*10**(-(W+Werr)/2.5)
                    ferr = ferr / (l**2)
                    errors.append(abs(ferr - f))
                else:
                    errors.append("")
            else:
                flux_densities.append("")
                errors.append("")


        df_add = pd.DataFrame({
                'filter':['WISE/WISE.W1', 'WISE/WISE.W2', 
                          'WISE/WISE.W3', 'WISE/WISE.W4'], 
                'wavelength':[33526.0, 46028.0, 115608.0, 220883.0], 
                'flux':flux_densities, 
                'flux_err':errors})
       
        idx_add_drop = []
        for ii in range(len(df_add['flux'])):
            if str(df_add['flux']) == 'nan':
                idx_add_drop.append(ii)

        self.df_add = df_add.drop(index=idx_add_drop)
        self.df_all = self.df_all.append(self.df_add, ignore_index=True)
        self.df_wise = self.df_wise.append(self.df_add, ignore_index=True)

    #Construct plot of SED
    def plotSED(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(self.df_all['wavelength'], self.df_all['flux'], marker='o',
                ls='-',  label='VOSA data')

        ax.minorticks_on()
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=15)
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')

        ax.set_yscale('log')
        minf = min(self.df_all['flux'])
        maxf = max(self.df_all['flux'])
        ax.set_ylim(ymin=10**(round(np.log10(minf), 2)-.5),
                    ymax=10**(round(np.log10(maxf), 2)+.5))

        ax.set_ylabel('Flux ('+r'erg/s/cm$^2$/Å)', fontsize=30)
        ax.set_xlabel('Wavelength (Å)', fontsize=30)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--source", help="source name", type=str)
    args = parser.parse_args()

    sedobject = WDSED('Gaia-DR2-3962760680985110016')
    print(sedobject.df_all)
    sedobject.plotSED()

    sed2 = WDSED('US-1639')
    print(sed2.df_all)
    sed2.addWISE(W1=16.019, W1err=.052)
    print(sed2.df_all)

