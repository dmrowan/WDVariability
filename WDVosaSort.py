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
"""

def main():
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources/np_vosadownloads')
    assert(os.path.isdir("../Pulsator"))

    filenames = []
    allfilenames = os.listdir(os.getcwd())
    for f in allfilenames:
        if f.endswith('.tgz'):
            filenames.append(f)

    for f in filenames:
        subprocess.run(['tar', '-zxvf', f])

    dirs = [ name[:-4] for name in filenames ]
    
    sourcelist = []
    for dirname in dirs:
        sourcename = os.listdir(dirname+"/objects/")[0]
        if "%2B" in sourcename:
            actualsourcename = sourcename.replace("%2B", "+")
        elif "%" in sourcename:
            actualsourcename = sourcename.replace("%", "-")
        else:
            actualsourcename = sourcename
        sourcelist.append(sourcename)
        subprocess.run(['cp', dirname+"/objects/"+sourcename+"/sed/"+sourcename+".sed.dat", "../Pulsator/"+actualsourcename+"/sed.dat"])

    print("------------------------------------------------------------------")
    for newpulsator in os.listdir("../Pulsator"):
        if not os.path.isfile("../Pulsator/"+newpulsator+"/sed.dat"):
            print("No sed found for {}".format(newpulsator))
            print("------------------------------------------------------------------")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
