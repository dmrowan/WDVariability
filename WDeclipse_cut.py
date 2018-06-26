#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os, sys
import warnings
warnings.simplefilter("ignore")
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
import importlib
gu = importlib.import_module('gPhoton.gphoton_utils')
import math
from scipy.stats import chi2
#Dom Rowan REU 2018

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
def main(csvname, fap, prange, cof):
    ###Path assertions###
    assert(os.path.isfile(csvname))
    assert(os.path.isdir("PDFs"))
    #Find source name from csvpath
    csvpath = csvname
    for i in range(len(csvpath)):
        character = csvpath[i]
        if character == 'F':
            endidx=i-1
            band='FUV'
            band_other='NUV'
            break
        elif character == 'N':
            endidx=i-1
            band='NUV'
            band_other='FUV'
            break

    source = csvpath[0:endidx]
    assert(band is not None)
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

    #Get information on flags
    n_rows = alldata.shape[0]
    n_flagged = 0
    for i in range(len(alldata['flags'])):
        if badflag_bool(alldata['flags'][i]):
            n_flagged += 1

    if float(n_rows) == 0:
        allflaggedratio = float('NaN')
    else:
        allflaggedratio = float(n_flagged) / float(n_rows)


    ###See if we have any data in the other band###
    if band=='NUV':
        csvpath_other = csvpath.replace('N', 'F')
    else:
        csvpath_other = csvpath.replace('F', 'N')
    #Look for file in GALEXphot/LCs
    if os.path.isfile(csvpath_other):
        other_band_exists = True
        alldata_other = pd.read_csv(csvpath_other)
    else:
        other_band_exists = False

    if other_band_exists:
        print("Generating additional LC data for " + band_other + " band")
        alldata_other = pd.read_csv(csvpath_other)
        #Drop bad rows, flagged rows
        idx_high_cps_other = np.where( (abs(alldata_other['cps_bgsub']) > 10e10) | (alldata_other['cps_bgsub_err'] > 10e10) | (alldata_other['counts'] < 1) | (alldata_other['counts'] > 100000) )[0]
        #Not interested in looking at red/blue points for other band
            #drop flagged, expt < 10
        idx_other_flagged_bool = [ badflag_bool(x) for x in alldata_other['flags'] ]
        idx_other_flagged = np.where(np.array(idx_other_flagged_bool) == True)[0]
        idx_other_expt = np.where(alldata_other['exptime'] < 10)[0]
        idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other, idx_other_flagged, idx_other_expt]))
        alldata_other = alldata_other.drop(index=idx_other_todrop)
        alldata_other = alldata_other.reset_index(drop=True)

        #Fix rows with weird t_mean time
        #   (some rows have almost zero t_mean, just average t0 and t1 in those rows)
        idx_tmean_fix_other = np.where( (alldata_other['t_mean'] < 1) | (alldata_other['t_mean'] > alldata_other['t1']) | (np.isnan(alldata_other['t_mean'])) )[0]
        for idx in idx_tmean_fix_other:
            t0 = alldata_other['t0'][idx]
            t1 = alldata_other['t1'][idx]
            mean = (t1 + t0) / 2.0
            alldata_other['t_mean'][idx] = mean

        #Make correction for relative scales
        alldata_tmean_other = alldata_other['t_mean']
        alldata_cps_bgsub_other = alldata_other['cps_bgsub']
        alldata_mediancps_other = np.median(alldata_cps_bgsub_other)
        alldata_cps_bgsub_other = ( alldata_cps_bgsub_other / alldata_mediancps_other ) - 1.0
        alldata_cps_bgsub_err_other = alldata_other['cps_bgsub_err'] / alldata_mediancps_other



    ###Break the alldata table into exposure groups### 
    breaks = []
    for i in range(len(alldata['t0'])):
        if i != 0:
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 150:
                breaks.append(i)

    data = np.split(alldata, breaks)
    print("Dividing " + band + " data for source " + source+ " into "+str(len(data))+" exposure groups")

    ###Create lists to fill### - these will be the primary output of main()
    df_number = 1
    biglc_time = []
    biglc_counts = []
    biglc_err = []
    strongest_periods_list = []
    skipnum = 0
    ###Loop through each exposure group###
    for df in data:
        #Find exposure time
        #Hopefully won't need this when data is fixed
        if len(df['t1']) == 0:
            df_number += 1
            skipnum += 1
            continue
        lasttime = list(df['t1'])[-1]
        firsttime = list(df['t0'])[0]
        exposure = lasttime - firsttime

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
        redpoints2 = np.where(df['exptime'] < 5)[0]
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
            #print("Not enough points for this exposure group, skipping. Removed " +  str(len(redpoints)) + " bad points")
            df_number +=1
            skipnum += 1
            continue

        #If first point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[0]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[-1]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        
        #Get the cps_bgsub, error and time columns and make correction for relative scales
        #Standard deviation divided by the median of the error as an interesting thing. Should be significantly above 1. Might be a good way to quickly see whats varying 
        cps_bgsub = df_reduced['cps_bgsub']
        cps_bgsub_median = np.median(cps_bgsub)
        cps_bgsub = ( cps_bgsub / cps_bgsub_median ) - 1.0
        cps_bgsub_err = df_reduced['cps_bgsub_err'] / cps_bgsub_median
        t_mean = df_reduced['t_mean']

        #If we have data in the other band, find points corresponding to this exposure group
        #First get the indicies corresponding to this group in the other band
        if other_band_exists:
            idx_exposuregroup_other = np.where( (alldata_tmean_other > firsttime) & (alldata_tmean_other < lasttime))[0]
            t_mean_other = alldata_tmean_other[idx_exposuregroup_other] - firsttime_mean
            cps_bgsub_other = alldata_cps_bgsub_other[idx_exposuregroup_other]
            cps_bgsub_err_other = alldata_cps_bgsub_err_other[idx_exposuregroup_other]

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

        ####Check if we have an eclipse#####
        lowvals = np.where(np.array(cps_bgsub) < -.9)[0]
        if len(lowvals) != 0:
            for i in range(len(cps_bgsub)):
                if (cps_bgsub[i] < -.25) and (cps_bgsub[i+1] < cps_bgsub[i]):
                    e_start_idx = i-4
                    t_eclipse_start = t_mean[e_start_idx]
                    e_start_found =True
                    break
            for i in range(e_start_idx+4, len(cps_bgsub)):
                if (cps_bgsub[i] > -.25) and (cps_bgsub[i+1] > cps_bgsub[i]):
                    if i+2 < (len(cps_bgsub)-1):
                        e_end_idx = i+2
                        t_eclipse_end = t_mean[e_end_idx]
                    else:
                        e_end_idx = len(cps_bgsub)-1
                        t_eclipse_end = t_mean[e_end_idx]
                    e_end_found = True
                    break
            if e_start_found:
                t_eclipse_start = gu.calculate_jd(t_eclipse_start+firsttime_mean)
                #print("Start of eclipse detected at", t_eclipse_start)

            if e_end_found:
                t_eclipse_end = gu.calculate_jd(t_eclipse_end+firsttime_mean)
                #print("End of eclipse detected at", t_eclipse_end)

            ###Generate plot/subplot information###
            fig = plt.figure(df_number, figsize=(16,12))
            gs.GridSpec(2,2)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle("Exposure group {0} with {1}s".format(str(df_number), str(exposure)))

            #Subplot for LC
            plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=1)
            #Convert to JD here as well
            jd_t_mean = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean ]
            plt.errorbar(jd_t_mean, cps_bgsub, yerr=cps_bgsub_err, color=bandcolors[band], marker='.', ls='', zorder=4, label=band)
            #plt.errorbar(jd_t_mean, cps_bgsub,  color=bandcolors[band], marker='.', ls='-', zorder=4, label=band)
            plt.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
            if len(redpoints) != 0: #points aren't even red now...
                jd_t_mean_red = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
                plt.errorbar(jd_t_mean_red, cps_bgsub_red, yerr=cps_bgsub_err_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
                #plt.errorbar(jd_t_mean_red, cps_bgsub_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
            if len(bluepoints) != 0: #these points aren't blue either...
                jd_t_mean_blue = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
                plt.errorbar(jd_t_mean_blue, cps_bgsub_blue, yerr=cps_bgsub_err_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
                #plt.errorbar(jd_t_mean_blue, cps_bgsub_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
            if other_band_exists:
                #introduce offset here
                jd_t_mean_other = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_other ]
                plt.errorbar(jd_t_mean_other, cps_bgsub_other+2*max(cps_bgsub), yerr=cps_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, label=band_other, alpha=.25)
                plt.axhline(y=2*max(cps_bgsub), alpha=.15, ls='dotted', color=bandcolors[band_other])

            eclipsepoints = plt.ginput(2, show_clicks=True)
            for point in eclipsepoints:
                point_jd = point[0]
                plt.axvline(x=point_jd, ls='--', alpha=.25)

            plt.title(band+' light curve')
            plt.xlabel('Time JD')
            plt.ylabel('Variation in CPS')
            plt.legend(loc=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])


            #Figure out where to cutoff periodogram#
            point_jd_start = eclipsepoints[0][0]
            idx_jd_start = np.where(np.array(abs(jd_t_mean - point_jd_start)) == np.min(abs(jd_t_mean - point_jd_start)))[0][0]
            plt.axvline(x=jd_t_mean[idx_jd_start])
            point_jd_end = eclipsepoints[1][0]
            idx_jd_end = np.where(np.array(abs(jd_t_mean - point_jd_end)) == np.min(abs(jd_t_mean - point_jd_end)))[0][0]
            plt.axvline(x=jd_t_mean[idx_jd_end])

            #Figure out if we have more data before or after eclipse
            if len(t_mean[:idx_jd_start]) > len(t_mean[idx_jd_end:]):
                print("More data before eclipse")
                t_mean = t_mean[:idx_jd_start]
                cps_bgsub = cps_bgsub[:idx_jd_start]
                cps_bgsub_err = cps_bgsub_err[:idx_jd_start]
                detrad = df_reduced['detrad'][:idx_jd_start]
                exptime = df_reduced['exptime'][:idx_jd_start]
            else:
                print("More data after eclipse")
                t_mean = t_mean[idx_jd_end:]
                cps_bgsub = cps_bgsub[idx_jd_end:]
                cps_bgsub_err = cps_bgsub_err[idx_jd_end:]
                detrad = df_reduced['detrad'][idx_jd_end:]
                exptime = df_reduced['exptime'][idx_jd_end:]

            plt.title(band+' light curve')
            plt.xlabel('Time JD')
            plt.ylabel('Variation in CPS')
            plt.legend(loc=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            chi2 = 0
            for val in cps_bgsub:
                chi2 += ( (val)**2 

            #Fist do the periodogram of the data
            ls = LombScargle(t_mean, cps_bgsub)
            freq, amp = ls.autopower(nyquist_factor=1)
            
            #Periodogram for dither information
            ls_detrad = LombScargle(t_mean, detrad)
            freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

            #Periodogram for expt information
            ls_expt = LombScargle(t_mean, exptime)
            freq_expt, amp_expt = ls_expt.autopower(nyquist_factor=1)

            #Calculate false alarm thresholds
            probabilities = [fap]
            faplevels = ls.false_alarm_level(probabilities)
            #Subplot for periodogram
            plt.subplot2grid((2,2), (1,0), colspan=2, rowspan=1)
            plt.plot(freq, amp, 'g-', label='Data')
            plt.plot(freq_detrad, amp_detrad, 'r-', label="Detrad", alpha=.25)
            plt.plot(freq_expt, amp_expt, 'b-', label="Exposure", alpha=.25)
            plt.title(band+' Periodogram')
            plt.xlabel('Freq [Hz]')
            plt.ylabel('Amplitude')
            plt.xlim(0, np.max(freq))
            plt.ylim(0, np.max(amp)*2)
            for level in faplevels:
                idx = np.where(level == faplevels)[0][0]
                fap = probabilities[idx]
                plt.axhline(level, color='black', alpha = .5, ls = '--', label = 'FAP: '+str(fap))

            plt.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            saveimagepath = str("PDFs/"+source+"-"+band+"-"+"eclipse.pdf")
            fig.savefig(saveimagepath)
            df_number += 1

            #Close figure
            fig.clf()
            plt.close('all')
        else:
            print("No eclipse found for exposure group " + str(df_number))
            df_number +=1
        

if __name__ == '__main__':
    
    desc="""
    This produces the ranked value C for a single source dependent on exposure, periodicity, autocorrelation, and other statistical measures
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--csvname", help= "Input full csv file", required=True, type=str)
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    parser.add_argument("--cof", help="Use flux or cps", default='cps', type=str)
    args= parser.parse_args()

    if args.cof != 'cps':
        assert(args.cof == 'flux')

    main(csvname=args.csvname, fap=args.fap, prange=args.prange, cof=args.cof)
