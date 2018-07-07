#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from astropy.stats import LombScargle
import heapq
from WDranker_2 import badflag_bool
#Dom Rowan REU 2018


desc="""
WDeclipse_periodogram: generate plot of periodogram 
"""

#Main plotting function
def main(csvname, fap, prange):
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
    print(source, band)
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
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 100000:
                breaks.append(i)

    data = np.split(alldata, breaks)
    print("Dividing " + band + " data for source " + source+ " into "+str(len(data))+" exposure groups")

    ###Loop through each exposure group###
    for df in data:
        #Find exposure time
        #Hopefully won't need this when data is fixed
        if len(df['t1']) == 0:
            continue

        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find indicies of data above 5 sigma of mean (counts per second column), flagged points, and points with
        # less than 10 seconds of exposure time
        #These colors are not at all accurate redpoints --> grey, bluepoints --> green
        #I just didn't want to change all the variable names. I'm not that good at vim.
        stdev = np.std(df['flux_bgsub'])
        bluepoints = np.where( (df['flux_bgsub'] - np.nanmean(df['flux_bgsub'])) > 5*stdev )[0]
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
            #print("Not enough points for this exposure group, skipping. Removed " +  str(len(redpoints)) + " bad points")
            continue

        #If first point is not within 3 sigma, remove
        if (df_reduced['flux_bgsub'][df_reduced.index[0]] - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['flux_bgsub'][df_reduced.index[-1]] - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])


        #Get the flux_bgsub, error and time columns and make correction for relative scales
        #Standard deviation divided by the median of the error as an interesting thing. Should be significantly above 1. Might be a good way to quickly see whats varying 
        flux_bgsub = df_reduced['flux_bgsub']
        flux_bgsub_median = np.median(flux_bgsub)
        flux_bgsub = ( flux_bgsub / flux_bgsub_median ) - 1.0
        t_mean = df_reduced['t_mean']

        ###Periodogram Creation###
        #Fist do the periodogram of the data
        ls = LombScargle(t_mean, flux_bgsub)
        freq, amp = ls.autopower(nyquist_factor=1)
        
        #Periodogram for dither information
        ls_detrad = LombScargle(t_mean, df_reduced['detrad'])
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

        #Periodogram for expt information
        ls_expt = LombScargle(t_mean, df_reduced['exptime'])
        freq_expt, amp_expt = ls_expt.autopower(nyquist_factor=1)

        #Identify statistically significant peaks
        top5amp = heapq.nlargest(5, amp)
        #top5amp_expt = heapq.nlargest(5, amp_expt)
        top5amp_detrad = heapq.nlargest(5, amp_detrad)
        #Find bad peaks
        bad_detrad = []
        for a in top5amp_detrad:
            if a == float('inf'):
                continue
            idx = np.where(amp_detrad == a)[0]
            f = freq[idx]
            lowerbound = f - prange
            upperbound = f + prange
            bad_detrad.append( (lowerbound, upperbound) )
            
        #Calculate false alarm thresholds
        probabilities = [fap]
        faplevels = ls.false_alarm_level(probabilities)

        sspeaks = [] #freq,amp,fap tuples
        for a in top5amp:
            #False alarm probability threshold
            fapval = ls.false_alarm_probability(a)
            if fapval <= fap:
                ratio = a / ls.false_alarm_level(fap)
                idx = np.where(amp==a)[0]
                f = freq[idx]
                #Now check if it is in any of the bad ranges
                hits = 0
                for tup in bad_detrad:
                    if ( f > tup[0] ) and ( f < tup[1] ):
                        hits+=1

                #If hits is still 0, the peak isnt in any of the bad ranges
                if hits == 0:
                    sspeaks.append( (f, a, fapval, ratio) ) 


        ###Generate plot/subplot information###
        #Subplot for periodogram
        plt.plot(freq, amp, 'g-', label='Data')
        plt.plot(freq_detrad, amp_detrad, 'r-', label="Detrad", alpha=.25)
        plt.plot(freq_expt, amp_expt, 'b-', label="Exposure", alpha=.25)
        plt.title(band+' Periodogram')
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amplitude')
        plt.xlim(0, np.max(freq))
        plt.ylim(0, np.max(amp)*2)
        if any(np.isnan(x) for x in top5amp_detrad):
            print("No detrad peaks")
        else:
            for tup in bad_detrad:
                plt.axvspan(tup[0], tup[1], alpha=.1, color='black')
        
        #ax[0][1].axvline(x=nyquistfreq, color='r', ls='--')
        for level in faplevels:
            idx = np.where(level == faplevels)[0][0]
            fap = probabilities[idx]
            plt.axhline(level, color='black', alpha = .5, ls = '--', label = 'FAP: '+str(fap))

        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--csvname", help= "Input full csv file", required=True, type=str)
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    args=parser.parse_args()
    main(csvname=args.csvname, fap=args.fap, prange=args.prange)
