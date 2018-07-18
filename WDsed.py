#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from math import sqrt
from WDviewer import selectidx
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager

#Dom Rowan REU 2018

desc="""
WDsed.py: Create SED from vosa information
"""
#Give x in angstroms
def blackbody(x, a, T):
    #Input x in cm
    #Constants
    h=6.626e-27
    c=2.99793e10
    kb = 1.381e-16

    print(a, T)
    firstterm = (2*h*c**2) / (x**5)
    secondterm = (1) / (np.exp((h*c)/(x*kb*T))-1)
    return a*firstterm*secondterm

"""
#Use scipy curve fit 
popt, pcov = curve_fit(blackbody, df['wavelength'], df['flux'], bounds=([1e-25, 1000], [1e-22, 20000]), p0=[1e-25, 10000])

#xvals to plot 
xvals = np.arange(min(df['wavelength']), max(df['wavelength'])*8, 1e-10)
#fitvals = blackbody(xvals, 8e-1, 1000)
fitvals = blackbody(xvals, popt[0], popt[1])
ax.plot(xvals, fitvals)
"""

def main(group):
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv"))

    if group:
        assert(os.getcwd() == "/home/dmrowan/WhiteDwarfs/InterestingSources/Pulsator")
        assert(os.path.isdir("../VOSA_seds"))
        sourcenames = os.listdir()
    else:
        assert(os.path.isfile("bbody_sed.dat"))
        assert(os.path.isfile("koesterparams.dat"))
        pwd = os.getcwd()
        sourcenames = [ pwd.split("/")[-1] ]

    if os.path.isfile("/home/dmrowan/WhiteDwarfs/InterestingSources/IRSA_wise.csv"):
        df_irsa_wise = pd.read_csv("/home/dmrowan/WhiteDwarfs/InterestingSources/IRSA_wise.csv")
    else:
        df_irsa_wise = None

    #Plot params used later
    myeffect = withStroke(foreground="k", linewidth=1.5)
    txtkwargs = dict(path_effects=[myeffect])
    myeffectw = withStroke(foreground="black", linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}
    arrow = u'$\u2193$'

    
    #Iterate through sources
    for name in sourcenames:
        #Pull ID num
        df_IS = pd.read_csv("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv")
        idx_IS = np.where(df_IS["MainID"] == name)[0]
        assert(len(idx_IS) == 1)
        idx_IS = idx_IS[0]
        labelnum = df_IS['labelnum'][idx_IS]

        if group:
            prefix = name+"/"
        else:
            prefix = ""
        
        if not os.path.isfile(prefix+"bbody_sed.dat"):
            print("No sed found for ", name)
            continue

        with open(prefix+"bbody_sed.dat") as f:
            lines = f.readlines()

        #Make sure we have right model fit
        assert("Black Body" in lines[6])

        #Initialize lists
        filtername = []
        wavelength = []
        flux = []
        flux_err = []
        model = []
        uplim = []
        for line in lines:
            if line.split()[0] != '#' and len(line.split()) != 1:
                if float(line.split()[4]) < 0:
                    continue
                else:
                    filtername.append(line.split()[0])
                    wavelength.append(float(line.split()[1]))
                    flux.append(float(line.split()[4]))
                    flux_err.append(float(line.split()[5]))
                    model.append(float(line.split()[6]))
                    if line.split()[-1] == '1':
                        uplim.append(1)
                    else:
                        uplim.append(0)

        #Seperate WISE
        df = pd.DataFrame({'filter':filtername, 'wavelength':wavelength, 'flux':flux, 'flux_err':flux_err, "model":model, "uplim":uplim})

        idx_wise = []
        for idx in range(len(df['filter'])):
            if 'WISE' in df['filter'][idx]:
                idx_wise.append(idx)

        df_wise = df.loc[idx_wise]
        df_wise = df_wise.reset_index(drop=True)
        df = df.drop(index=idx_wise)
        df = df.reset_index(drop=True)

        
        dic_irsa = {'filter':[], 'wavelength':[], 'flux':[], 'flux_err':[]}
        if df_irsa_wise is not None:
            idx_irsa = np.where(df_irsa_wise['MainID'] == name)[0]
            if len(idx_irsa) != 0:
                dic_irsa['wavelength'] = [33526.0, 46028.0, 115608.0, 220883.0]
                idx_irsa = idx_irsa[0]
                for bandkey in ['F1', 'F2', 'F3', 'F4']:
                    val = df_irsa_wise[bandkey][idx_irsa]
                    valerr = df_irsa_wise[bandkey+'err'][idx_irsa]
                    wisefilter = 'WISE/WISE.W'+bandkey[1]
                    dic_irsa['filter'].append(wisefilter)
                    dic_irsa['flux'].append(val)
                    dic_irsa['flux_err'].append(valerr)

        print(dic_irsa)
        df_irsa = pd.DataFrame(dic_irsa)

        #Generate plot
        fig, ax = plt.subplots(1,1, figsize=(16,12))
        ax.plot(df['wavelength'], df['flux'],  marker='.',markersize=0, 
                ls='-', label='data')
        for i in range(len(df['wavelength'])):
            if df['uplim'][i] == 0:
                ax.errorbar(df['wavelength'][i], df['flux'][i], 
                            yerr=df['flux_err'][i], marker='o', 
                            ls='', ecolor='gray', label='_nolegend_', 
                            markersize=8)
            else:
                ax.errorbar(df['wavelength'][i], df['flux'][i], 
                            yerr=df['flux_err'][i], marker=arrow, ls='', 
                            ecolor='gray', markersize=20)

        ax.plot(df_wise['wavelength'], df_wise['flux'], marker='.', 
                markersize=0, ls='-', label='wise data', color='green')
        for i in range(len(df_wise['wavelength'])):
            if df_wise['uplim'][i] == 0:
                ax.errorbar(df_wise['wavelength'][i], df_wise['flux'][i], 
                            yerr=df_wise['flux_err'][i], marker='o', 
                            color='green', ls='', ecolor='gray', 
                            label='_nolegend_', markersize=8)
            else:
                ax.errorbar(df_wise['wavelength'][i], df_wise['flux'][i], 
                            yerr=df_wise['flux_err'][i], marker=arrow, 
                            color='green', ls='', ecolor='gray', 
                            label='_nolegend_', markersize=20)


        for i in range(len(df_irsa['flux'])):
            ax.errorbar(df_irsa['wavelength'][i], df_irsa['flux'][i], 
                        yerr=df_irsa['flux_err'][i], marker='o', 
                        color='green', ls='', ecolor='gray', 
                        label='_nolegend_', markersize=8)
        #Plot model 
        model_wavelength = list(df['wavelength']) + list(df_wise['wavelength'])
        model_flux = list(df['model']) + list(df_wise['model'])
        ax.plot(model_wavelength, model_flux, ls='--', color='purple', 
                label='bbody fit')
        
        #Plot Params
        ax.set_ylim(ymin=10**(round(np.log10(min(flux)), 2)-.5), 
                    ymax=10**(round(np.log10(max(flux)), 2)+.5))
        ax.set_yscale('log')
        ax.set_ylabel('Flux (erg/s/cm2/A)', fontsize=30)
        ax.set_xlabel('Wavelength (A)', fontsize=30)
        ax.legend(loc=1, fontsize=25)
        #ax.set_title('{} SED'.format(name))
        ax.minorticks_on()
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=15)
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        ax.annotate(str(labelnum), xy=(.05, .95), xycoords='axes fraction', 
                    color='xkcd:red', fontsize=30, ha='center', va='center', 
                    **afont, **txtkwargsw)


        if group:
            fig.savefig("../VOSA_seds/"+name+"_bbody.png")
        else:
            plt.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--use_fit", help="Run scipy curve fit of blackbody", 
                        default=False, action='store_true')
    parser.add_argument("--group", help="Produce pngs of all pulsators", 
                        default=False, action='store_true')
    args= parser.parse_args()

    main(group=args.group)
