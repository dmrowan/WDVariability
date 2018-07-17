#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
#Dom Rowan REU 2018

desc="""
WDWiseConvert: Convert WISE data into SED usable stuff
"""

#Values to use for testing: US-1639
#WISE 1 mag from IRSA: 16.019, err=.052
#Vosa finds flux 3.2e-18 +- 1.532e-19 (flux in units erg/cm2/s/A)

def main(W1=None, W1err=None, W2=None, W2err=None, W3=None, W3err=None, W4=None, W4err=None):
    #Found at http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    F01 = 309.540
    F02 = 171.787
    F03= 31.674
    F04 = 8.363

    #Values initially in micro meteres, convert to A
    l1 = 3.4*1e4
    l2 = 4.6*1e4
    l3 = 12*1e4
    l4 = 22*1e4
    
    flux_densities = []
    errors = []
    if W1 is not None:
        flux1= (3e-5)*F01*10**(-W1/2.5)
        flux1 = flux1 / (l1**2)
        flux_densities.append(flux1)
        if W1err is not None:
            flux1err = (3e-5)*F01*10**(-(W1+W1err)/2.5)
            flux1err = flux1err/ (l1**2)
            errors.append(abs(flux1err-flux1))

    if W2 is not None:
        flux2= (3e-5)*F02*10**(-W2/2.5)
        flux2 = flux2 / (l2**2)
        flux_densities.append(flux2)
        if W2err is not None:
            flux2err = (3e-5)*F02*10**(-(W2+W2err)/2.5)
            flux2err = flux2err/ (l2**2)
            errors.append(abs(flux2err-flux2))

    if W3 is not None:
        flux3= (3e-5)*F03*10**(-W3/2.5)
        flux3 = flux3 / (l3**2)
        flux_densities.append(flux3)
        if W3err is not None:
            flux3err = (3e-5)*F03*10**(-(W3+W3err)/2.5)
            flux3err = flux3err/ (l3**2)
            errors.append(abs(flux3err-flux3))

    if W4 is not None:
        flux4= (3e-5)*F04*10**(-W4/2.5)
        flux4 = flux4 / (l4**2)
        flux_densities.append(flux4)
        if W4err is not None:
            flux4err = (3e-5)*F04*10**(-(W4+W4err)/2.5)
            flux4err = flux4err/ (l4**2)
            errors.append(abs(flux4err-flux4))
    print(flux_densities)
    print(errors)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--W1", default=None, type=float)
    parser.add_argument("--W2", default=None, type=float)
    parser.add_argument("--W3", default=None, type=float)
    parser.add_argument("--W4", default=None, type=float)
    parser.add_argument("--W1err", default=None, type=float)
    parser.add_argument("--W2err", default=None, type=float)
    parser.add_argument("--W3err", default=None, type=float)
    parser.add_argument("--W4err", default=None, type=float)
    args= parser.parse_args()

    main(W1=args.W1, W1err=args.W1err, W2=args.W2, W2err=args.W2err, W3=args.W3, W3err=args.W3err, W4=args.W4, W4err=args.W4err)

