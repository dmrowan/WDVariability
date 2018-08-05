#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import multiprocessing as mp
import WDranker_2
#Dom Rowan REU 2018

desc = """
This will call the WDranker for all CSVs in a directory. 
This doesn't show the highest ranking (see WDcompare for that).
"""

def main(noreplace, noplot):
    #Create the pool object
    pool=mp.Pool(processes=mp.cpu_count()+2)
    #Queue jobs into this list
    jobs=[]
    
    #Loop through all csv's, adding source csv's to job queue
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.csv'):
            if noreplace and os.path.isfile(
                    "Output/"+filename[:-4]+"-output.csv"):
                print("Output for "+filename[:-4]+" already exists, skipping")
                print('-'*100)
                continue
            else:
                    if noplot:
                        job = pool.apply_async(
                                WDranker_2.main, 
                                args=(filename, False))
                    else:
                        job = pool.apply_async(
                                WDranker_2.main, 
                                args=(filename))
                    jobs.append(job)
    
    #Iterate through all the jobs
    for job in jobs:
        job.get()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    #Have to include the arguments from WDranker
    parser.add_argument("--noreplace", 
                        help="Continue rather than overwriting", 
                        default=False, action='store_true')
    parser.add_argument("--noplot", 
                        help="Don't make plots during ranking",
                        default=False, action='store_true')

    args = parser.parse_args()

    main(noreplace=args.noreplace, noplot=args.noplot)
