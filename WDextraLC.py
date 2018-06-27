#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from gPhoton import gphoton_utils
from WDranker_2 import badflag_bool
#Dom Rowan REU 2018

desc="""
Quick light curve to plot with exposure, detrad, flux. Used to check for instrumental causes to variability. 
"""
#Main plotting function
def main(csvname, fap, prange):
    assert(os.path.isfile(csvname))
    #Find source name from csvpath
    csvpath = csvname
    for i in range(len(csvpath)):
        character = csvpath[i]
        if character == 'c':
            endidx=i-5
            break

    source = csvpath[0:endidx]

    #Grab the band (also checks we have source csv)
    if csvpath[-7] == 'N':
        band = 'NUV'
    elif csvpath[-7] == 'F':
        band = 'FUV'
    else:
        print("Not source csv, skipping")
        return

    assert(band is not None)
    bandcolors = {'NUV':'red', 'FUV':'blue'}
    alldata = pd.read_csv(csvpath)

    ###Alldata table corrections###
    #Drop rows with > 10e10 in cps, cps_err, cps < .5
    idx_high_cps = np.where( (abs(alldata['cps_bgsub']) > 10e10) | (alldata['cps_bgsub_err'] > 10e10) | (alldata['counts'] < 1) | (alldata['counts'] > 100000) )[0]
    if len(idx_high_cps) != 0:
        alldata = alldata.drop(index = idx_high_cps)
        alldata = alldata.reset_index(drop=True)

    #Fix rows with weird t_means by averaging t0 and t1
    idx_tmean_fix = np.where( (alldata['t_mean'] < 1) | (alldata['t_mean'] > alldata['t1']) | (np.isnan(alldata['t_mean'])) )[0]
    for idx in idx_tmean_fix:
        t0 = alldata['t0'][idx]
        t1 = alldata['t1'][idx]
        mean = (t1 + t0) / 2.0
        alldata['t_mean'][idx] = mean

    ###Break the alldata table into exposure groups### 
    breaks = []
    for i in range(len(alldata['t0'])):
        if i != 0:
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 100:
                breaks.append(i)

    data = np.split(alldata, breaks)

    df_number = 1

    ###Loop through each exposure group###
    for df in data:
        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find indicies of data above 5 sigma of mean (counts per second column), flagged points, and points with
        # less than 10 seconds of exposure time
        stdev = np.std(df['cps_bgsub'])
        bluepoints = np.where( (df['cps_bgsub'] - np.nanmean(df['cps_bgsub'])) > 5*stdev )[0]
        flag_bool_vals = [ badflag_bool(x) for x in df['flags'] ]
        redpoints1 = np.where(np.array(flag_bool_vals) == True)[0]
        redpoints2 = np.where(df['exptime'] < 10)[0]
        redpoints = np.unique(np.concatenate([redpoints1, redpoints2]))
        redpoints = redpoints + df.index[0]
        bluepoints = bluepoints + df.index[0]

        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))
        df_reduced = df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)
        
        #Remove points where cps_bgsub is nan
        idx_cps_nan = np.where( np.isnan(df_reduced['cps_bgsub']) )[0]
        if len(idx_cps_nan) != 0:
            df_reduced = df_reduced.drop(index=idx_cps_nan)
            df_reduced = df_reduced.reset_index(drop=True)

        if df_reduced.shape[0] < 10:
            df_number +=1
            continue

        #If first point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[0]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[-1]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        
        #Get the cps_bgsub, error and time columns and make correction for relative scales
        cps_bgsub = df_reduced['cps_bgsub']
        cps_bgsub_median = np.median(cps_bgsub)
        cps_bgsub = ( cps_bgsub / cps_bgsub_median ) - 1.0
        cps_bgsub_err = df_reduced['cps_bgsub_err'] / cps_bgsub_median
        #Do the same for flux
        flux_bgsub = df_reduced['flux_bgsub']
        flux_bgsub_median = np.median(flux_bgsub)
        flux_bgsub = ((flux_bgsub / flux_bgsub_median) - 1.0) 
        flux_bgsub_err = (df_reduced['flux_bgsub_err'] / flux_bgsub_median)
        t_mean = df_reduced['t_mean']
        #Exposure time and detrad columns
        detrad = df_reduced['detrad']
        exposuretime = df_reduced['exptime'] - np.median(df_reduced['exptime'])

        #Make the correction for relative scales for redpoints and bluepoints
        if len(redpoints) != 0:
            cps_bgsub_red = df['cps_bgsub'][redpoints]
            cps_bgsub_red = (cps_bgsub_red / cps_bgsub_median) - 1.0
            cps_bgsub_err_red = df['cps_bgsub_err'][redpoints] / cps_bgsub_median
            t_mean_red = df['t_mean'][redpoints]
        if len(bluepoints) != 0:
            cps_bgsub_blue = df['cps_bgsub'][bluepoints]
            cps_bgsub_blue = (cps_bgsub_blue / cps_bgsub_median) - 1.0
            cps_bgsub_err_blue = df['cps_bgsub_err'][bluepoints] / cps_bgsub_median
            t_mean_blue = df['t_mean'][bluepoints]
        

        #Convert to JD here as well
        jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean ]
        plt.errorbar(jd_t_mean, cps_bgsub,  yerr= cps_bgsub_err, color=bandcolors[band], marker='.', ls='', zorder=4, label=band)
        plt.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: #points aren't even red now...
            jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
            plt.errorbar(jd_t_mean_red, cps_bgsub_red, yerr=cps_bgsub_err_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
        if len(bluepoints) != 0: #these points aren't blue either...
            jd_t_mean_blue = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
            plt.errorbar(jd_t_mean_blue, cps_bgsub_blue, yerr=cps_bgsub_err_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')

        plt.scatter(jd_t_mean, detrad, color='green', label='detrad')
        plt.scatter(jd_t_mean, exposuretime, color='blue', label='exposure')
        plt.errorbar(jd_t_mean, flux_bgsub, yerr=flux_bgsub_err, color='orange', label='flux')

        plt.xlabel('Time JD')
        plt.legend(loc=1)
        
        plt.title("LC for {0} {1} exposure group {2}".format(source, band, str(df_number)))
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--csvname", help= "Input full csv file", required=True, type=str)
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    args= parser.parse_args()

    main(csvname=args.csvname, fap=args.fap, prange=args.prange) 
