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


#Main plotting and fitting function
def main(use_fit):
    #Path assertions
    assert(os.path.isfile('sed.dat'))
    with open('sed.dat') as f:
        lines = f.readlines()

    allfiles = (os.listdir(os.getcwd()))
    for fname in allfiles:
        if fname.endswith('.csv'):
            csvname = fname
            break
    
    i_name = np.where( (np.array(list(csvname)) == 'N') | ( np.array(list(csvname)) == 'F'))[0][0]
    sourcename = csvname[:i_name-1]
    filtername = []
    wavelength = []
    flux = []
    flux_err = []

    for i in range(10, len(lines)):
        filtername.append(lines[i].split()[0])
        wavelength.append(float(lines[i].split()[1]))
        flux.append(float(lines[i].split()[4]))
        flux_err.append(float(lines[i].split()[5]))


    flux = [ f* (1/1e-8) for f in flux ]
    flux_err = [ err * (1/1e-8) for err in flux_err ]
    wavelength = [w * 1e-8 for w in wavelength ]



    df = pd.DataFrame({'filter':filtername, 'wavelength':wavelength, 'flux':flux, 'flux_err':flux_err})

    idx_wise = []
    for idx in range(len(df['filter'])):
        if 'WISE' in df['filter'][idx]:
            idx_wise.append(idx)

    df_wise = df.loc[idx_wise]
    df_wise.reset_index(drop=True)
    df = df.drop(index=idx_wise)
    df.reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(16,12))
    print("-----Non IR Data-----")
    print(df)
    print("-----WISE Data-----")
    print(df_wise)
    print("-"*50)
    if use_fit:

        #Use scipy curve fit 
        popt, pcov = curve_fit(blackbody, df['wavelength'], df['flux'], bounds=([1e-25, 1000], [1e-22, 20000]), p0=[1e-25, 10000])

        #xvals to plot 
        xvals = np.arange(min(df['wavelength']), max(df['wavelength'])*8, 1e-10)
        #fitvals = blackbody(xvals, 8e-1, 1000)
        fitvals = blackbody(xvals, popt[0], popt[1])
        ax.plot(xvals, fitvals)

    ax.errorbar(df['wavelength'], df['flux'], yerr=df['flux_err'], marker='.',markersize=12, ls='--', ecolor='gray')
    ax.errorbar(df_wise['wavelength'], df_wise['flux'], yerr=df_wise['flux_err'], marker='o', color='green', ls='--', ecolor='gray', markersize=8)
    ax.set_ylim(ymin=10**(round(np.log10(min(flux)), 2)-.5), ymax=10**(round(np.log10(max(flux)), 2)+.5))
    ax.set_yscale('log')
    ax.set_ylabel('Flux (erg/cm3/s)')
    ax.set_xlabel('Wavelength (cm)')
    ax.set_title('{} SED'.format(sourcename))
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--use_fit", help="Run scipy curve fit of blackbody", default=False, action='store_true')
    args= parser.parse_args()

    main(use_fit=args.use_fit)
