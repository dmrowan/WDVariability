#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import multiprocessing as mp
import WDranker_2
#Dom Rowan REU 2018

desc = """
This will call the WDranker for all CSVs in a directory. This doesn't show the highest ranking (see WDcompare for that).
"""

def main(fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, w_flag, w_magfit,comment, noreplace):
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
                    job = pool.apply_async(WDranker_2.main, args=(filename, fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, w_flag, w_magfit,comment, ))
                    jobs.append(job)
    
    #Iterate through all the jobs
    for job in jobs:
        job.get()

if __name__ == '__main__':

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
    parser.add_argument("--noreplace", help="Continue for new sources rather than overwriting", default=False, action='store_true')

    args = parser.parse_args()

    main(fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_ac=args.w_ac, w_mag=args.w_mag, w_known=args.w_known, w_flag=args.w_flag, w_magfit=args.w_magfit, comment=args.comment,noreplace=args.noreplace)
