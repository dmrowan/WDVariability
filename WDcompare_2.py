#!/usr/bin/env python
from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import subprocess
from progressbar import ProgressBar
import math
#Dom Rowan REU 2018

desc = """
WDCompare: Combine all output csv's after running WDranker/WDgroup, 
sort by rank. Options to sort by additional metrics. 
Other options to add to existing AllData.csv and make histograms of flag/rank
"""

#Generate AllData.csv from WDranker output
def main(sortmag, sortperiod):
    #Get rid of current AllData.csv if it exists
    if os.path.isfile("Output/AllData.csv"):
        confirm = input("Hit y to remove current AllData.csv and write new  ")
        if confirm != 'y':
            return
        else:
            subprocess.call(['rm', 'Output/AllData.csv'])

    #Make the csv for all information
    #Make dataframe that will have all the information
    bigdf = pd.DataFrame()
    pbar = ProgressBar()
    print("Iterating through Output files")
    print(len(os.listdir(os.getcwd()+"/Output/")))
    for outputcsv in pbar(os.listdir(os.getcwd()+"/Output/")):
        if outputcsv.endswith('.csv'):
            #Import the csv into a pandas df
            df_temp = pd.read_csv(os.getcwd()+"/Output/"+outputcsv)
            bigdf = bigdf.append(df_temp, ignore_index=True)

    if sortmag or sortperiod: 
        if sortmag:
            #Sort by magnitude 
            bigdf = bigdf.sort_values(by=['ABMag'])
        elif sortperiod:
            #Sort by period (lowest first)
            bigdf = bigdf.sort_values(by=['StrongestPeriod'])
    else:
        #Sort by rank (default)
        bigdf = bigdf.sort_values('BestRank', ascending=False)

    #Write to CSV        
    bigdf.to_csv("Output/AllData.csv", index=False)

#Update current AllData, used to not overwrite comments
def update(sortmag, sortperiod):
    assert(os.path.isfile("Output/AllData.csv"))
    input_df = pd.read_csv("Output/AllData.csv")
    new_df = pd.DataFrame()
    print("Iterating through Output files not yet in table")
    for outputcsv in os.listdir(os.getcwd()+"/Output/"):
        if outputcsv.endswith('.csv'):
            if outputcsv != 'AllData.csv':
                #If source/band not in AllData.csv, add row
                input_idx = np.where((input_df['SourceName'] 
                                      == outputcsv[:-15]) 
                                      & (input_df['Band'] 
                                      == outputcsv[-14:-11]))[0]
                if len(input_idx) == 0:
                    print("Adding {}".format(outputcsv))
                    df_temp = pd.read_csv(os.getcwd()+"/Output/"+outputcsv)
                    new_df = new_df.append(df_temp, ignore_index=True)

    output_df = input_df.append(new_df, ignore_index=True)
    #Sorting options
    if sortmag or sortperiod:
        if sortmag:
            #Sort by magnitude (higher preference then comment, period)
            output_df = output_df.sort_values(by=['ABMag'])
        elif sortperiod:
            #Sort by period (lowest first)
            output_df = output_df.sort_values(by=['StrongestPeriod'])
    else:
        #Sort by rank (default)
        output_df = output_df.sort_values('BestRank', ascending=False)
    
    #Write to CSV
    output_df.to_csv("Output/AllData.csv", index=False)

#generate historgam of flagged ratios
def flaghist():
    assert(os.path.isfile("Output/AllData.csv"))
    input_df = pd.read_csv("Output/AllData.csv")
    #generate histogram
    flaglist = input_df['FlaggedRatio']
    print(len(flaglist))
    flaglist_reduced = []
    for x in flaglist:
        if not math.isnan(x):
            flaglist_reduced.append(x)
    #Additional plot params
    plt.hist(flaglist_reduced)
    plt.title('Ratio of flagged points')
    plt.xlabel('flagged# / total#')
    plt.ylabel('Number')
    plt.show()

def rankhist():
    assert(os.path.isfile("Output/AllData.csv"))
    input_df = pd.read_csv("Output/AllData.csv")
    #Drop rows with < .5 in rank
    idx_lowrank = np.where(input_df['BestRank'] < .5)[0]
    input_df_reduced = input_df.drop(index = idx_lowrank)
    #generate histogram and plot information
    plt.hist(input_df_reduced['BestRank'])
    plt.title("Best Rank")
    plt.xlabel("Rank")
    plt.ylabel("N")
    plt.show()

def commentmatch(oldfname, newfname):
    assert(os.path.isfile(oldfname))
    assert(os.path.isfile(newfname))

    df_old = pd.read_csv(oldfname)
    df_new = pd.read_csv(newfname)

    pbar = ProgressBar()
    for i in pbar(range(len(df_new['SourceName']))):
        if not str(df_new['Comment'][i]) == 'nan':
            continue
        else:
            idx_old = np.where( (df_old['SourceName'] 
                                 == df_new['SourceName'][i]) 
                                 & (df_old['Band'] 
                                 == df_new['Band'][i]) )[0]
            if len(idx_old) == 0:
                continue
            else:
                assert(len(idx_old)==1)
                idx_old = idx_old[0]
                oldcomment = df_old['Comment'][idx_old]
                if str(oldcomment) == 'nan':
                    continue
                else:
                    df_new.loc[i, 'Comment'] = oldcomment

    df_new.to_csv("Output/AllData.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--sortmag", 
                        help="Sort by magnitude instead of rank", 
                        default=False, action='store_true')
    parser.add_argument("--sortperiod", 
                        help="Sort by shortest period of strongest peak", 
                        default=False, action='store_true')
    parser.add_argument("--flaghist", 
                        help="Generate histogram of flagged ratio", 
                        default=False, action='store_true')
    parser.add_argument("--rankhist", 
                        help="Generate histogram of best ranks", 
                        default=False, action='store_true')
    parser.add_argument("--update", 
                        help="Update current csv and re-sort", 
                        default=False, action='store_true')
    parser.add_arugment("--match", 
                        help="Match comments from a previous runthrough", 
                        default=False, action='store_true')
    parser.add_arugment("--old", 
                        help="old alldata.csv for matching", 
                        type=str, default=None)
    parser.add_argument("--new", 
                        help="new alldata.csv for matching", 
                        type=str, default=None)
    args= parser.parse_args()

    if args.flaghist:
        flaghist()
    elif args.rankhist:
        rankhist()
    elif args.update:
        update(sortmag = args.sortmag, sortperiod=args.sortperiod)
    elif args.match:
        commentmatch(args.old, args.new)
    else:
        main(sortmag = args.sortmag,sortperiod=args.sortperiod)

