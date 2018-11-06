#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
import astropy.convolution
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import WDutils

#Dom Rowan REU 2018

desc="""
WD_Spectra: Class for WD spectra collected with UH88inch
"""

class Spectra:
    def __init__(self, fname):
        assert(os.path.isfile(fname))
        self.fname = fname
        with open(fname) as f:
            lines = f.readlines()

        self.wavelength=[]
        self.flux=[]
        self.err=[]
        self.meta = {}
        for line in lines:
            if '#' in line: 
                if len(line.split())==4:
                    self.meta[line.split()[1]] = line.split()[3]
            else:
                self.wavelength.append(float(line.split()[0]))
                self.flux.append(float(line.split()[1]))
                self.err.append(float(line.split()[2]))

    def set_xmin(self, xmin):
        self.xmin=xmin
    def set_xmax(self, xmax):
        self.xmax=xmax
    def set_ID(self, ID):
        self.ID = ID
    def set_type(self, classification):
        self.type = classification
    def cutoff(self):
        try:
            self.xmin
        except:
            self.xmin = input("Enter wavelength of lower cutoff -- ")
        try:
            self.xmax
        except:
            self.xmax = input("Enter wavelength of upper cutoff -- ")

        return self.xmin, self.xmax


    def meta_string(self,key):
        if key == 'DATE-OBS':
            s = key.title() + ": "+self.meta[key][:10]
        elif key == 'EXPTIME':
            s = "Exposure: "+str(int(float(self.meta[key])))+" s"
        else:
            s = key.title() + ": " + self.meta[key]
        return s

    def smoothflux(self, m):
        smoothflux = astropy.convolution.convolve(
                self.flux, astropy.convolution.Box1DKernel(m))
        return smoothflux

    def plot(self, use_cutoff=False, ax=None):
        if ax==None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        myeffectw = withStroke(foreground='black', linewidth=2)
        txtkwargsw = dict(path_effects=[myeffectw])
        afont = {'fontname':'Keraleeyam'}


        self.flux = [ s*(10**16) for s in self.flux ]
        self.err = [ s*(10**16) for s in self.err ]
        smoothflux = self.smoothflux(4)
        ax.errorbar(self.wavelength, smoothflux, yerr=self.err,
                    color='xkcd:red', ecolor='gray')
        if use_cutoff:
            lower, upper = self.cutoff()
        print(lower, upper)
        ax.set_xlim(xmin=lower, xmax=upper)
        #ax.set_yscale('log')
        try:
            annotation = f"{self.type}\nID #{self.ID}"
            ax.annotate(annotation, (.925,.925), 
                        xycoords='axes fraction', color='xkcd:red',
                        fontsize=20, ha='center', va='top',
                        **afont, **txtkwargsw, bbox=dict(boxstyle='square', fc='.9', ec='.2', lw=2))
        except:
            print("No label num / classification given")

        ax = WDutils.plotparams(ax)
        if ax==None:
            ax.set_ylabel('Flux ('+r'erg/s/cm$^2$/Å)$\times10^{16}$', 
                          fontsize=30)
            ax.set_xlabel('Wavelength (Å)', fontsize=30)
            fig.savefig(self.fname.replace('.dat', '.pdf'))
        return ax
    
def main():
    
   
    s2 = Spectra('GAIA-14766-58321.3017-SNIFS.dat')
    s2.set_xmin(3600)
    s2.set_xmax(7200)
    s2.set_ID(35)
    s2.set_type('DBV')
    print(s2.meta)
    s = Spectra('GaiaDR2-39098-58301.585-SNIFS.dat')
    s.set_xmin(3600)
    s.set_xmax(7200)
    s.set_ID(2)
    s.set_type('DAV')
    print(s.meta)
    

    """
    s = Spectra('SDSS-J220823.66-011534.1-SNIFS1.dat')
    s.set_ID(54)
    s.set_xmin(3600)
    s.set_xmax(7000)
    s.set_type('DAV')

    s2 = Spectra('SDSS-J234829.09-092500.9-SNIFS2.dat')
    s2.set_ID(61)
    s2.set_xmin(3600)
    s2.set_xmax(7000)
    s2.set_type('DAV')
    """

    figT, (axT1, axT2) = plt.subplots(2, 1, figsize=(10, 8))
    axT1 = s.plot(use_cutoff=True, ax=axT1)
    axT2 = s2.plot(use_cutoff=True, ax=axT2)
    axT1.xaxis.set_ticklabels([])
    
    for a in [3970, 4102, 4340, 4861, 6567]:
        myp = 'xkcd:violet'
        axT1.axvspan(a-10, a+10, color=myp, alpha=.5, 
                     label='H lines', zorder=5)
        axT2.axvspan(a-10, a+10, color=myp, alpha=.5, zorder=5)
        #axT1.axvline(a, color='xkcd:violet', alpha=.25, label='H lines')
        #axT2.axvline(a, color='xkcd:violet', alpha=.25)

    afont = {'fontname':'Keraleeyam'}
    for a in [ 3819,3888, 4471, 4713,  5876, 6678]:
        axT1.axvspan(a-10, a+10, color='xkcd:azure', alpha=.35, 
                     label='He I lines', zorder=5)
        axT2.axvspan(a-10, a+10, color='xkcd:azure', alpha=.35, zorder=5)
        #axT1.axvline(a, color='xkcd:azure', alpha=.25, label='He I lines')
        #axT2.axvline(a, color='xkcd:azure', alpha=.25)
    #bbox_circle=dict(boxstyle='circle, pad=.1', fc='w', ec='w', alpha=1)
    #axT1.annotate('HeI', xy=(5500,0),xycoords='data', fontsize=20, 
    #              color='black', ha='center', va='center', bbox=bbox_circle)


    handles, labels = axT1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axT1.legend(by_label.values(), by_label.keys(), fontsize=20,
                loc='upper right', bbox_to_anchor=(1, .4),
                edgecolor='black', framealpha=.9,
                markerscale=.2)


    plt.subplots_adjust(hspace=0, top=.98, right=.98)
    figT.text(.5, .05, 'Wavelength (Å)', fontsize=30, ha='center', 
              va='center')
    figT.text(.03, .5, 'Flux ('+r'erg/s/cm$^2$/Å)$\times10^{16}$', 
              fontsize=30,
              ha='center', va='center', rotation='vertical')
    figT.savefig("SpectraPlot.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    args = parser.parse_args()
    
    main()


