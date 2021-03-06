#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from scipy.signal import correlate
from matplotlib.patheffects import withStroke
from scipy.optimize import curve_fit

#Dom Rowan REU 2018

desc="""
WDrv.py: RV analysis for eclipsing sources from SNIFS
"""

def expdecay(x, A, gamma, C):
    output = A*np.exp(x*gamma) + C
    print(A, gamma, C)
    return output

#Main plotting and fitting function
def main(datfile, save, l1, l2):
    #Get sourcename for coordinate query
    sourcename = datfile[:-11]
    #Get other filename
    if 'SNIFS1' in datfile:
        datfile1 = datfile
        datfile2 = datfile.replace('SNIFS1', 'SNIFS2')
    else:
        assert('SNIFS2' in datfile)
        datfile1 = datfile.replace('SNIFS2', 'SNIFS1')
        datfile2 = datfile

    #Path assertions
    assert(os.path.isfile(datfile1))
    assert(os.path.isfile(datfile2))

    #Read in both files
    with open(datfile1) as f:
        lines1 = f.readlines()

    with open(datfile2) as f:
        lines2 = f.readlines()


    #Find observation date/time
    ot1 = Time(lines1[1].split()[-1])
    ot2 = Time(lines2[1].split()[-1])

    wavelength1, flux1, flux_err1 = [],[],[]
    for i in range(len(lines1)):
        if lines1[i][0] == '#':
            continue
        else:
            wavelength1.append(float(lines1[i].split()[0]))
            flux1.append(float(lines1[i].split()[1]))
            flux_err1.append(float(lines1[i].split()[2]))
    wavelength2, flux2, flux_err2 = [],[],[]
    for i in range(len(lines2)):
        if lines2[i][0] == '#':
            continue
        else:
            wavelength2.append(float(lines2[i].split()[0]))
            flux2.append(float(lines2[i].split()[1]))
            flux_err2.append(float(lines2[i].split()[2]))
    
    fig, (ax1,ax2, ax3) = plt.subplots(3, 1, figsize=(18,18))
    if l1 is not None and l2 is not None:
        assert(l1 < l2)
        if (l1 > min(wavelength1)) and (l2 < max(wavelength1)):
            idx_1_l1 = np.where( np.array(abs(np.array(wavelength1) - l1)) == np.min(abs(np.array(wavelength1) - l1)) )[0][0]
            idx_1_l2 = np.where( np.array(abs(np.array(wavelength1) - l2)) == np.min(abs(np.array(wavelength1) - l2)) )[0][0]
            idx_2_l1 = np.where( np.array(abs(np.array(wavelength2) - l1)) == np.min(abs(np.array(wavelength2) - l1)) )[0][0]
            idx_2_l2 = np.where( np.array(abs(np.array(wavelength2) - l2)) == np.min(abs(np.array(wavelength2) - l2)) )[0][0]

            outputcorr = np.correlate(flux1[idx_1_l1:idx_1_l2], 
                                      flux2[idx_2_l1:idx_2_l2], 
                                      'full'
                                )
            outputcorr_valid = np.correlate(flux1[idx_1_l1:idx_1_l2], 
                                      flux2[idx_2_l1:idx_2_l2], 
                                      'valid'
                                )
            ax1.axvline(x=wavelength1[idx_1_l1], ls='--')
            ax1.axvline(x=wavelength1[idx_1_l2], ls='--')
            ax2.axvline(x=wavelength2[idx_2_l1], ls='--')
            ax2.axvline(x=wavelength2[idx_2_l2], ls='--')

            popt, pcov = curve_fit(expdecay, flux1[idx_1_l1:idx_1_l2], wavelength1[idx_1_l1:idx_1_l2], p0=[4e-17, -.0002, -(1e-18)], bounds=([1e-20, -.001, -(1e-16)], [1e-15, -.00001, 1e-16]))
            xvals = np.arange(wavelength1[idx_1_l1-200], wavelength1[idx_1_l2+200], 4)
            fitvals = expdecay(xvals, popt[0], popt[1], popt[2]) - 5e-17
            #fitvals = expdecay(xvals, 4e-17, -.0002, 0)
            print(min(fitvals), max(fitvals))
            ax1.plot(xvals, fitvals, label='fit', c='green', lw=4, zorder=2)

        else:
            print("Provide valid l1 and l2 bounds")
            outputcorr = np.correlate(flux1, flux2, 'full')
            outputcorr_valid = np.correlate(flux1, flux2, 'valid')
    else:
        outputcorr = np.correlate(flux1, flux2, 'full')
        outputcorr_valid = np.correlate(flux1, flux2, 'valid')

    print(outputcorr_valid)

    ax1.errorbar(wavelength1, flux1, yerr=flux_err1, marker='.',
                 markersize=12, ls='-', ecolor='gray'
            )
    ax2.errorbar(wavelength2, flux2, yerr=flux_err2, marker='.', 
                 markersize=12, ls='-', ecolor='gray'
            )

    
    ax3.plot(outputcorr)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    fig.text(.02, .66, 'Flux (ergs/s/cm2/A)', va='center', 
             rotation='vertical', fontsize=30
        )
    ax2.set_xlabel('Wavelength (A)', fontsize=20)

    for ax in [ax1, ax2]:
        ax.legend(loc=3)
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', labelsize=15)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)


    #Query IS.csv to find ra, dec
    df_IS = pd.read_csv("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv")
    idx_source = np.where(df_IS['MainID'] == sourcename)[0]
    assert(len(idx_source)==1)
    idx_source = idx_source[0]
    ra = df_IS['ra'][idx_source]
    dec = df_IS['dec'][idx_source]

    hc1 = heliocorrection(ra, dec, ot1)
    hc2 = heliocorrection(ra, dec, ot2)
    hc = np.mean([hc1.value, hc2.value])

    afont = {'fontname':'Keraleeyam'}
    myeffectw = withStroke(foreground="black", linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])

    hc_annotation = "Heliocentric correction: {} km/s".format(str(round(hc,3)))

    ax3.annotate(hc_annotation, xy=(.75, .95), xycoords='axes fraction', 
                 color='xkcd:violet', fontsize=30, 
                 ha='center', va='center', **afont, **txtkwargsw
            )

    if save:
        fig.savefig(sourcename+"_rvplot.png")

    plt.show()


def heliocorrection(ra, dec, ot):
    location = 'gemini_north'
    uh88  = EarthLocation.of_site(location)
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    heliocorr = sc.radial_velocity_correction('heliocentric', 
                                              obstime=ot, location=uh88
                                        )
    return(heliocorr.to(u.km/u.s))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--dat", help="Dat file input (give either one)",
                        type=str, required=True)
    parser.add_argument("--save", help="Save with filename", 
                        default=False, action='store_true')
    parser.add_argument("--l1", help="lower bound angstrom range", default=None, type=float)
    parser.add_argument("--l2", help="upper bound angstrom range", default=None, type=float)
    args= parser.parse_args()

    main(datfile=args.dat, save=args.save, l1=args.l1, l2=args.l2)
