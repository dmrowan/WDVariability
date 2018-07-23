#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from astropy.stats import LombScargle
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
from scipy.stats import chisquare
from WDranker_2 import badflag_bool
import subprocess
#Dom Rowan REU 2018

desc="""
WDknownevlipse.py: Timing for known eclipse
"""
#Main plotting function
def main():
    csvname = '2MASS-J03030835+0054438-NUV.csv'
    assert(os.path.isfile(csvname))
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

    source = '2MASS-J03030835+0054438'
    bandcolors = {'NUV':'red', 'FUV':'blue'}

    alldata = pd.read_csv(csvpath)
    ###Alldata table corrections###
    #Drop rows with > 10e10 in cps, cps_err, cps < .5
    idx_high_cps = np.where( (abs(alldata['cps_bgsub']) > 10e10) 
                             | (alldata['cps_bgsub_err'] > 10e10) 
                             | (alldata['counts'] < 1) 
                             | (alldata['counts'] > 100000) )[0]

    if len(idx_high_cps) != 0:
        alldata = alldata.drop(index = idx_high_cps)
        alldata = alldata.reset_index(drop=True)

    #Fix rows with weird t_means by averaging t0 and t1
    idx_tmean_fix = np.where( (alldata['t_mean'] < 1) 
                              | (alldata['t_mean'] > alldata['t1']) 
                              | (np.isnan(alldata['t_mean'])) )[0]

    for idx in idx_tmean_fix:
        t0 = alldata['t0'][idx]
        t1 = alldata['t1'][idx]
        mean = (t1 + t0) / 2.0
        alldata['t_mean'][idx] = mean

    ###Break the alldata table into exposure groups### 

    breaks = []
    for i in range(len(alldata['t0'])):
        if i != 0:
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 200:
                breaks.append(i)

    data = np.split(alldata, breaks)
    #eclipse visits are df 3 and 5
    data = [data[2], data[4]]

    #Generate figure
    fig = plt.figure(figsize=(12,6))
    gs1=gs.GridSpec(2,1)
    gs1.update(hspace=0)

    plt.subplots_adjust(top=.98, right=.98, left=.08)
    fig.text(.02, .5, "Relative CPS", va='center', 
             rotation='vertical', fontsize=25)
    plt_idx = 0
    ###Create lists to fill### - these will be the primary output of main()
    ###Loop through each exposure group###
    egress_time = []
    for df in data:
        #Find exposure time
        lasttime = list(df['t1'])[-1]
        firsttime = list(df['t0'])[0]

        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find indicies of data above 5 sigma of mean (counts per second column), flagged points, and points with
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

        #If first point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[0]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[-1]] - np.nanmean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        
        #Get the cps_bgsub, error and time columns and make correction for relative scales
        cps_bgsub = df_reduced['cps_bgsub']
        cps_bgsub_median = np.median(cps_bgsub)
        cps_bgsub = ( cps_bgsub / cps_bgsub_median ) 
        cps_bgsub_err = df_reduced['cps_bgsub_err'] / cps_bgsub_median
        t_mean = df_reduced['t_mean']


        #Make the correction for relative scales for redpoints and bluepoints
        if len(redpoints) != 0:
            cps_bgsub_red = df['cps_bgsub'][redpoints]
            cps_bgsub_red = (cps_bgsub_red / cps_bgsub_median) 
            cps_bgsub_err_red = df['cps_bgsub_err'][redpoints] / cps_bgsub_median
            t_mean_red = df['t_mean'][redpoints]
        if len(bluepoints) != 0:
            cps_bgsub_blue = df['cps_bgsub'][bluepoints]
            cps_bgsub_blue = (cps_bgsub_blue / cps_bgsub_median) 
            cps_bgsub_err_blue = df['cps_bgsub_err'][bluepoints] / cps_bgsub_median
            t_mean_blue = df['t_mean'][bluepoints]

        ###Generate plot/subplot information###
        ax = plt.subplot(gs1[plt_idx])
        #Convert to JD here as well
        jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                      for t in t_mean ]
        t0 = min(jd_t_mean)
        jd_t_mean = [ t - t0 for t in jd_t_mean ]
        ax.errorbar(jd_t_mean, cps_bgsub, yerr=cps_bgsub_err, 
                     color=bandcolors[band], marker='.', ls='', 
                     zorder=4, label=band)
        ax.axhline(y=0, alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: 
            jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                              for t in t_mean_red ]
            jd_t_mean_red = [ t - t0 for t in jd_t_mean_red ]
            plt.errorbar(jd_t_mean_red, cps_bgsub_red, yerr=cps_bgsub_err_red,
                         color='#808080', marker='.', ls='', zorder=2, 
                         alpha=.5, label='Flagged')
        if len(bluepoints) != 0: #these points aren't blue either...
            jd_t_mean_blue = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                               for t in t_mean_blue ]
            jd_t_mean_blue = [ t - t0 for t in t_mean_blue ]
            ax.errorbar(jd_t_mean_blue, cps_bgsub_blue, yerr=cps_bgsub_err_blue, 
                         color='green', marker='.', ls='', zorder=3, 
                         alpha=.5, label='SigmaClip')

        ax.set_xlim(xmin = -.00125, xmax=.02025)
        ax.set_ylim(ymin=-.4, ymax=3.2)
        customticks = [(1/1440)*x for x in np.arange(0, 29, 6)]
        customticklabels = [ 6*x for x in range(0,5) ]
        ax.set_xticks(customticks)
        ax.set_xticklabels(customticklabels)
        ax.minorticks_on()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_ticks_position('both')
        ax.tick_params(direction='in', which='both', labelsize=15)
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')

        ax.annotate(str(round(t0, 4)), xy=(.75, .85), 
                xycoords='axes fraction',                    
                color='black', fontsize=20, zorder=4)

        if plt_idx == 0:
            ax.set_xticklabels([])
            ax.legend(fontsize=25)
        else:
            ax.set_xlabel("Elapsed time (min)", fontsize=25)

        plt_idx += 1

    saveimagepath = str("PDFs/"+source+"-"+band+"-"+"eclipse_compare.pdf")
    fig.savefig(saveimagepath)

    subprocess.run(['cp', saveimagepath, '../../knowneclipse.pdf'])
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
