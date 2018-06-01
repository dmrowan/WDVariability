#!/usr/bin/env python
from __future__ import print_function, division
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy import log
from glob import glob
from subprocess import check_call
import astropy
import pandas as pd
from astropy.stats import LombScargle
import heapq
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import subprocess
#Dom Rowan REU 2018

def main(sortmag, sortcomment, sortperiod, flaghist):
    #Get rid of current AllData.csv if it exists
    if os.path.isfile("Output/AllData.csv"):
        print("Removing csv")
        subprocess.call(['rm', 'Output/AllData.csv'])

    if flaghist:
        ratios = []

    #Make the csv for all information
    #Make dataframe that will have all the information
    bigdf = pd.DataFrame()
    for outputcsv in os.listdir(os.getcwd()+"/Output/"):
        if outputcsv.endswith('.csv'):
            #Import the csv into a pandas df
            df_temp = pd.read_csv(os.getcwd()+"/Output/"+outputcsv)
            bigdf = bigdf.append(df_temp, ignore_index=True)
            if flaghist:
                ratios.append(df_temp['FlaggedRatio'][0])

    if sortmag or sortcomment:
        if sortmag:
            #Sort by magnitude (higher preference then comment, period)
            bigdf = bigdf.sort_values(by=['ABMag'])
        elif sortperiod:
            #Sort by period (lowest first)
            bigdf = bigdf.sort_values(by=['StrongestPeriod'])
        elif sortcomment:
            #Sort by comment
            bigdf = bigdf.sort_values(by=['Comment'])
    else:
        #Sort by rank (default)
        bigdf = bigdf.sort_values('BestRank', ascending=False)

    #Write to CSV        
    bigdf.to_csv("Output/AllData.csv")

    #Generate histogram
    if flaghist:
        plt.hist(ratios)
        plt.title('Ratio of flagged points')
        plt.xlabel('flagged# / total#')
        plt.ylabel('Number')
        plt.show()

if __name__ == "__main__":
    desc = """
    This is used after running WDgroup. It searches through all of the output files and picks the N most interesting sources. It can also be used to make some other interesting plots comparing our results
    """

    parser = argparse.ArgumentParser(description=desc)
    #Have to include the arguments from WDranker
    parser.add_argument("--sortmag", help="Sort by magnitude instead of rank", default=False, action='store_true')
    parser.add_argument("--sortcomment", help="Sort by comment instead of rank", default=False, action='store_true')
    parser.add_argument("--sortperiod", help="Sort by shortest period of strongest peak", default=False, action='store_true')
    parser.add_argument("--flaghist", help="Generate histogram of flagged ratio", default=False, action='store_true')
    args= parser.parse_args()

    main(sortmag = args.sortmag, sortcomment=args.sortcomment, sortperiod=args.sortperiod, flaghist=args.flaghist)
