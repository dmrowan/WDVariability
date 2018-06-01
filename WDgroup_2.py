#!/usr/bin/env python
from __future__ import print_function, division
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy import log
from os import path
from astropy.table import Table
import astropy
import pandas as pd
from astropy.stats import LombScargle
import heapq
import matplotlib.image as mpimg
import subprocess
import multiprocessing as mp
import WDranker_2
#Dom Rowan REU 2018

def main(fap, prange, w_pgram, w_expt, w_ac, w_mag, comment, noreplace):
    pool=mp.Pool(processes=4)
    jobs=[]
    
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.csv'):
            if noreplace and os.path.isfile("Output/"+filename[:-4]+"-output.csv"):
                print("Output for "+filename[:-4]+" already exists, skipping")
                continue
            else:
                    #job = pool.apply(WDranker_2.main, args=(filename, fap, prange, w_pgram, w_expt, w_ac, w_mag, comment,))
                    #jobs.append(job)
                    WDranker_2.main(filename, fap, prange, w_pgram, w_expt, w_ac, w_mag, comment)


if __name__ == '__main__':

    desc = """
    This will call the WDranker for all CSVs in a directory. This doesn't show the highest ranking (see WDcompare for that).
    """

    parser = argparse.ArgumentParser(description=desc)
    #Have to include the arguments from WDranker
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    parser.add_argument("--w_pgram", help = "Weight for periodogram", default = 1, type=float)
    parser.add_argument("--w_expt", help= "Weight for exposure time", default = .25, type=float)
    parser.add_argument("--w_ac", help="Weight for autocorrelation", default = 0, type=float)
    parser.add_argument("--w_mag", help= "Weight for magnitude", default=.5, type=float)
    parser.add_argument("--comment", help="Add comments/interactive mode", default=False, action='store_true')
    parser.add_argument("--noreplace", help="Continue for new sources rather than overwriting", default=False, action='store_true')

    args = parser.parse_args()

    main(fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_ac=args.w_ac, w_mag=args.w_mag, comment=args.comment, noreplace=args.noreplace)
