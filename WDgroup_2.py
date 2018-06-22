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
import time
import warnings
import datetime
#Dom Rowan REU 2018

warnings.simplefilter("once")

def main(fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, w_flag, w_magfit,comment, cof, noreplace):
    firsttime = datetime.datetime.now()
    #Create the pool object
    pool=mp.Pool(processes=mp.cpu_count()+2)
    #Queue jobs into this list
    jobs=[]
    
    #Loop through all csv's, adding source csv's to job queue
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.csv'):
            if noreplace and os.path.isfile("Output/"+filename[:-4]+"-output.csv"):
                print("Output for "+filename[:-4]+" already exists, skipping")
                print('-'*100)
                continue
            else:
                    job = pool.apply_async(WDranker_2.main, args=(filename, fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, w_flag, w_magfit,comment,cof, ))
                    jobs.append(job)
    
    #Iterate through all the jobs
    for job in jobs:
        job.get()

    #Print how many minutes, seconds the jobs took
    td = datetime.datetime.now() - firsttime
    print( divmod(td.days*86400 + td.seconds,60) )
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
    parser.add_argument("--w_known", help="Weight for if known (subtracted)", default=.75, type=float)
    parser.add_argument("--w_flag", help="Weight for if more than 25% flagged (subtracted)", default=.5, type=float)
    parser.add_argument("--w_magfit", help="Weight for magfit ratio", default=.25, type=float)
    parser.add_argument("--comment", help="Add comments/interactive mode", default=False, action='store_true')
    parser.add_argument("--cof", help="Use cps or flux", default='cps', type=str)
    parser.add_argument("--noreplace", help="Continue for new sources rather than overwriting", default=False, action='store_true')

    args = parser.parse_args()

    if args.cof != 'cps':
        assert(args.cof == 'flux')
    main(fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_ac=args.w_ac, w_mag=args.w_mag, w_known=args.w_known, w_flag=args.w_flag, w_magfit=args.w_magfit, comment=args.comment, cof=args.cof, noreplace=args.noreplace)
