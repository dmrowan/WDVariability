#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import numpy as np
import subprocess
#Dom Rowan REU 2018

desc="""
WDVosaSort: Sort vosa downloads for new pulsators into directory. Pull sed.
            This version pulls the fit information (bbody and koester)
"""

def main():
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources/puls_vosa')
    assert(os.path.isdir("../Pulsator"))

    filenames = []
    allfilenames = os.listdir(os.getcwd())
    for f in allfilenames:
        if f.endswith('.tgz'):
            filenames.append(f)

    for f in filenames:
        subprocess.run(['tar', '-zxvf', f])

    dirs = [ name[:-4] for name in filenames ]
    
    for dirname in dirs:
        sourcename = os.listdir(dirname+"/objects/")[0]
        if "%2B" in sourcename:
            actualsourcename = sourcename.replace("%2B", "+")
        elif "%" in sourcename:
            actualsourcename = sourcename.replace("%", "-")
        else:
            actualsourcename = sourcename

        #subprocess.run(['cp', dirname+"/objects/"+sourcename+"/sed/"+sourcename+".sed.dat", "../Pulsator/"+actualsourcename+"/sed.dat"])
        subprocess.run(['cp', dirname+"/objects/"+sourcename+"/bestfitp/"+sourcename+".bfit.phot.dat", "../Pulsator/"+actualsourcename+"/bbody_sed.dat"])
        subprocess.run(['cp', dirname+"/objects/"+sourcename+"/fitp/"+sourcename+".koester2.fit.dat", "../Pulsator/"+actualsourcename+"/koesterparams.dat"])

    print("------------------------------------------------------------------")
    for newpulsator in os.listdir("../Pulsator"):
        if ((not os.path.isfile("../Pulsator/"+newpulsator+"/bbody_sed.dat"))
                or (not os.path.isfile("../Pulsator/"+newpulsator+"/koesterparams.dat"))):
            print("No sed found for {}".format(newpulsator))
            print("------------------------------------------------------------------")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
