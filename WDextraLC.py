#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy import log
from os import path
from glob import glob
import astropy
import ipdb
import pandas as pd
from astropy.stats import LombScargle
import heapq
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
import subprocess
import warnings
import importlib
gu = importlib.import_module('gPhoton.gphoton_utils')
import math
#Dom Rowan REU 2018

warnings.simplefilter("once")
np.warnings.simplefilter("once")

#Convert the flag value into a binary string and see if we have a bad flag
def badflag_bool(x):
    bvals = [512,256,128,64,32,16,8,4,2,1]
    val = x
    output_string = ''
    for i in range(len(bvals)):
        if val >= bvals[i]:
            output_string += '1'
            val = val - bvals[i]
        else:
            output_string += '0'
    
    badflag_vals = output_string[0] + output_string[4] + output_string[7] + output_string[8]
    for char in badflag_vals:
        if char == '1':
            return True
            break
        else:
            continue
    return False

#Main ranking function
def main(csvname, fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, w_flag, w_magfit, comment):
    ###Path assertions###
    catalogpath = "/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    sigmamag_path = "Catalog/SigmaMag_reduced.csv"
    sigmamag_percentile_path = "Catalog/magpercentiles.csv"
    assert(os.path.isfile(csvname))
    assert(os.path.isfile(catalogpath))
    assert(os.path.isfile(sigmamag_path))
    assert(os.path.isfile(sigmamag_percentile_path))
    assert(os.path.isdir('PDFs'))
    assert(os.path.isdir('PNGs'))
    assert(os.path.isdir('Output'))

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
        band_other = 'FUV'
    elif csvpath[-7] == 'F':
        band = 'FUV'
        band_other = 'NUV'
    else:
        print("Not source csv, skipping")
        return

    assert(band is not None)
    print(source, band)
    badflags = [2,4,32,512]
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

    ###Query Catalogs###
    bigcatalog = pd.read_csv(catalogpath)
    #Replace hyphens with spaces
    #Have to deal with replacing hyphens in gaia / other sources differently
    nhyphens = len(np.where(np.array(list(source)) == '-')[0])
    if source[0:4] == 'Gaia':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
    elif source[0:5] == 'ATLAS':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
    elif source[0:2] == 'GJ':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
    elif source[0:2] == 'CL':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
    elif source[0:2] == 'LP':
        if nhyphens == 2:
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ', 1))[0]
        else:
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
    elif source[0:2] == 'V*':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
    elif source[0:3] == '2QZ':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ', 1))[0]
    else:
        if nhyphens == 1:
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ' ))[0]
        else:
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ',nhyphens-1))[0]

    #Just doing this for now until I figure out how to deal with ^^^ better
    if len(bigcatalog_idx) == 0:
        print(source, "Not in catalog")
        with open("../brokensources.txt", 'a') as f:
            f.write(source + "\n")
        return
    else:
        bigcatalog_idx = bigcatalog_idx[0]

    simbad_name = bigcatalog['SimbadName'][bigcatalog_idx]
    simbad_types = bigcatalog['SimbadTypes'][bigcatalog_idx]

    ###Break the alldata table into exposure groups### 
    breaks = []
    for i in range(len(alldata['t0'])):
        if i != 0:
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 100:
                breaks.append(i)

    data = np.split(alldata, breaks)

    ###Create lists to fill### - these will be the primary output of main()
    df_number = 1
    c_vals = []
    df_numbers_run = []
    biglc_time = []
    biglc_counts = []
    biglc_err = []
    strongest_periods_list = []

    ###Loop through each exposure group###
    for df in data:
        #Find exposure time
        #Hopefully won't need this when data is fixed
        if len(df['t1']) == 0:
            df_number += 1
            continue
        lasttime = list(df['t1'])[-1]
        firsttime = list(df['t0'])[0]
        exposure = lasttime - firsttime
        c_exposure = (exposure) / 1000

        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find indicies of data above 5 sigma of mean (counts per second column), flagged points, and points with
        # less than 10 seconds of exposure time
        #These colors are not at all accurate redpoints --> grey, bluepoints --> green
        #I just didn't want to change all the variable names. I'm not that good at vim.
        stdev = np.std(df['cps_bgsub'])
        bluepoints = np.where( (df['cps_bgsub'] - np.nanmean(df['cps_bgsub'])) > 5*stdev )[0]
        flag_bool_vals = [ badflag_bool(x) for x in df['flags'] ]
        redpoints1 = np.where(np.array(flag_bool_vals) == True)[0]
        redpoints2 = np.where(df['exptime'] < 10)[0]
        redpoints = np.unique(np.concatenate([redpoints1, redpoints2]))
        redpoints = redpoints + df.index[0]
        bluepoints = bluepoints + df.index[0]

        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))
        flagged_ratio = len(droppoints) / df.shape[0]
        if flagged_ratio > .25:
            c_flagged = 1
        else:
            c_flagged = 0
        df_reduced = df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)
        
        #Remove points where cps_bgsub is nan
        idx_cps_nan = np.where( np.isnan(df_reduced['cps_bgsub']) )[0]
        if len(idx_cps_nan) != 0:
            df_reduced = df_reduced.drop(index=idx_cps_nan)
            df_reduced = df_reduced.reset_index(drop=True)

        if df_reduced.shape[0] < 10:
            #print("Not enough points for this exposure group, skipping. Removed " +  str(len(redpoints)) + " bad points")
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
        flux = df_reduced['flux_bgsub'] * 10
        t_mean = df_reduced['t_mean']
        detrad = df_reduced['detrad']
        exposuretime = df_reduced['exptime']

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
        jd_t_mean = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean ]
        plt.errorbar(jd_t_mean, cps_bgsub,  color=bandcolors[band], marker='.', ls='', zorder=4, label=band)
        plt.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: #points aren't even red now...
            jd_t_mean_red = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
            plt.errorbar(jd_t_mean_red, cps_bgsub_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
        if len(bluepoints) != 0: #these points aren't blue either...
            jd_t_mean_blue = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
            plt.errorbar(jd_t_mean_blue, cps_bgsub_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')

        plt.scatter(jd_t_mean, detrad, color='green', label='detrad')
        plt.scatter(jd_t_mean, exposuretime, color='blue', label='exposure')
        plt.scatter(jd_t_mean, flux, color='orange', label='flux')

        plt.xlabel('Time JD')
        plt.ylabel('Variation in CPS')
        plt.legend(loc=1)
        
        plt.title("LC for {0} {1} exposure group {2}".format(source, band, str(df_number)))
        plt.show()


if __name__ == '__main__':
    
    desc="""
    This produces the ranked value C for a single source dependent on exposure, periodicity, autocorrelation, and other statistical measures
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--csvname", help= "Input full csv file", required=True, type=str)
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    parser.add_argument("--w_pgram", help = "Weight for periodogram", default = 1, type=float)
    parser.add_argument("--w_expt", help= "Weight for exposure time", default = .25, type=float)
    parser.add_argument("--w_ac", help="Weight for autocorrelation", default = 0, type=float)
    parser.add_argument("--w_mag", help= "Weight for magnitude", default=.5, type=float)
    parser.add_argument("--w_known", help="Weight for if known binarity, variability, disk, Z spec type", default=2, type=float)
    parser.add_argument("--w_flag", help="Weight for if more than 25% flagged (subtracted)", default=.5, type=float)
    parser.add_argument("--w_magfit", help="Weight for magfit ratio", default=.25, type=float)
    parser.add_argument("--comment", help="Add comments/interactive mode", default=False, action='store_true')
    args= parser.parse_args()

    main(csvname=args.csvname, fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_ac=args.w_ac, w_mag=args.w_mag, w_known=args.w_known, w_flag=args.w_flag, w_magfit=args.w_magfit, comment=args.comment)
