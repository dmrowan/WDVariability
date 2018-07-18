#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
from scipy.stats import chisquare
from WDranker_2 import badflag_bool
import subprocess
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.font_manager

#Dom Rowan REU 2018

desc="""
WDPlot_Eclipse: Plot tower of eclipses
"""
#Main plotting function
def main(generate):
    assert(os.path.isfile("IS.csv"))
    df_IS = pd.read_csv("IS.csv")
    if generate:
        eclipsenames = []
        for i in range(len(df_IS['MainID'])):
            if df_IS['type'][i] == 'Eclipse':
                eclipsenames.append(df_IS['MainID'][i])
        band_list = []
        df_list = []
        for name in eclipsenames:
            print(name)
            if os.path.isfile('Eclipse/'+name+'/PDFs/'+name+'_NUV_combined.pdf'):
                subprocess.run(['xdg-open', 'Eclipse/'+name+'/PDFs/'+name+'_NUV_combined.pdf'])
            if os.path.isfile('Eclipse/'+name+'/PDFs/'+name+'_FUV_combined.pdf'):
                subprocess.run(['xdg-open', 'Eclipse/'+name+'/PDFs/'+name+'_FUV_combined.pdf'])
            
            bandselection=input("Choose band of best df --- ")
            dfselection = input("In that band, choose best df --- ")
            assert( (bandselection == 'NUV') or (bandselection == 'FUV') )
            dfselection = int(dfselection)
            band_list.append(bandselection)
            df_list.append(dfselection)
        df_eclipse_df = pd.DataFrame({"MainID":eclipsenames, "band":band_list, "df_number":df_list})
        df_eclipse_df.to_csv("eclipse_group.csv")

    assert(os.path.isfile("eclipse_group.csv"))
    input_df = pd.read_csv("eclipse_group.csv")
    eclipsenames = input_df['MainID']
    band_list = input_df['band']
    df_list = input_df['df_number']

    labelnum_list = []
    for name in eclipsenames:
        idx_dfIS = np.where(df_IS['MainID'] == name)[0]
        labelnum_list.append(df_IS['labelnum'][idx_dfIS])


    ###Generate plot/subplot information###
    fig = plt.figure(figsize=(12,8))
    gs1 = gs.GridSpec(5,2) 
    gs1.update(hspace=0, wspace=0.15)
    fig.text(.02, .5, 'Relative CPS', va='center', rotation='vertical', fontsize=30)
    fig.text(.5, .025, 'Minutes from Start', va='center', fontsize=30, ha='center')
    afont = {'fontname':'Keraleeyam'}

    plt.subplots_adjust(top=.98, right=.98)
    myeffect = withStroke(foreground="k", linewidth=.5)
    txtkwargs = dict(path_effects=[myeffect])
    myeffectw = withStroke(foreground="black", linewidth=2)
    txtkwargsw = dict(path_effects=[myeffectw])

    #fig.tight_layout(rect=[0, 0, 1, 1])
    starttimes = []
    for idx_plot in range(len(eclipsenames)):
        name = eclipsenames[idx_plot]
        band = band_list[idx_plot]
        dfbest = df_list[idx_plot]
        labelnum = int(labelnum_list[idx_plot])
        csvpath = "Eclipse/"+name+"/"+name +"-" +band+".csv"
        if band == 'NUV':
            band_other = 'FUV'
        else:
            band_other = 'NUV'

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
            alldata_other = pd.read_csv(csvpath_other)
            #Drop bad rows, flagged rows
            idx_high_cps_other = np.where( (abs(alldata_other['cps_bgsub']) > 10e10) | (alldata_other['cps_bgsub_err'] > 10e10) | (alldata_other['counts'] < 1) | (alldata_other['counts'] > 100000) )[0]
            idx_other_flagged_bool = [ badflag_bool(x) for x in alldata_other['flags'] ]
            idx_other_flagged = np.where(np.array(idx_other_flagged_bool) == True)[0]
            idx_other_expt = np.where(alldata_other['exptime'] < 10)[0]
            idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other, idx_other_flagged, idx_other_expt]))
            alldata_other = alldata_other.drop(index=idx_other_todrop)
            alldata_other = alldata_other.reset_index(drop=True)

            #Fix rows with weird t_mean time
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
                if (alldata['t0'][i] - alldata['t0'][i-1]) >= 200:
                    breaks.append(i)

        data = np.split(alldata, breaks)
        df = data[dfbest-1]

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

        #If less than 10 points, skip
        if df_reduced.shape[0] < 10:
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
            cps_bgsub_blue = ((cps_bgsub_blue / cps_bgsub_median) - 1.0) * 1000
            cps_bgsub_err_blue = df['cps_bgsub_err'][bluepoints] / cps_bgsub_median
            t_mean_blue = df['t_mean'][bluepoints]

        #Subplot for LC
        coords = [(0,0),(1,0),(2,0),(3,0),(4,0), (1,1), (2,1), (3,1), (4,1)]
        plot_coords = coords[idx_plot]
        plt.subplot2grid((5, 2), plot_coords, colspan=1, rowspan=1)
        #plt.subplot2grid((len(eclipsenames), len(eclipsenames)), (idx_plot, 0), colspan=len(eclipsenames), rowspan=1)
        #Convert to JD here as well
        jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean ]

        
        mintimes = [min(jd_t_mean)]
        if len(redpoints) != 0:
            jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
            mintimes.append(min(jd_t_mean_red))
        if len(bluepoints) != 0:
            jd_t_mean_blue = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
            mintimes.append(min(jd_t_mean_blue))
        if other_band_exists:
            if len(cps_bgsub_other) != 0:
                jd_t_mean_other = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_other ]
                mintimes.append(min(jd_t_mean_other))
        """
        jd_min_time = int(min(mintimes))
        jd_t_mean = [ t - jd_min_time for t in jd_t_mean ] 
        """
        t0 = min(jd_t_mean)
        jd_t_mean = [ t - t0 for t in jd_t_mean ]

        ax = plt.subplot(gs1[plot_coords])
        customticks = [(1/1440)*x for x in np.arange(0,29,6)]
        customticklabels = [ 6*x for x in range(0,5) ]
        ax.set_xticks(customticks)
        ax.set_xticklabels(customticklabels)
        ax.errorbar(jd_t_mean, cps_bgsub, yerr=cps_bgsub_err, color=bandcolors[band], marker='.', ls='', zorder=4, label=band)
        ax.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: 
            #jd_t_mean_red = [ t - jd_min_time for t in jd_t_mean_red ]
            jd_t_mean_red = [ t - t0 for t in jd_t_mean_red ]
            ax.errorbar(jd_t_mean_red, cps_bgsub_red, yerr=cps_bgsub_err_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
        if len(bluepoints) != 0: 
            #jd_t_mean_blue = [ t - jd_min_time for t in jd_t_mean_blue ] 
            jd_t_mean_blue = [ t - t0 for t in jd_t_mean_blue ] 
            #ax.errorbar(jd_t_mean_blue, cps_bgsub_blue, yerr=cps_bgsub_err_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
        if other_band_exists:
            if len(cps_bgsub_other) != 0:
                #jd_t_mean_other = [ t - jd_min_time for t in jd_t_mean_other ]
                jd_t_mean_other = [ t - t0 for t in jd_t_mean_other ]
                #introduce offset here
                ax.errorbar(jd_t_mean_other, cps_bgsub_other+2*max(cps_bgsub), yerr=cps_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, label=band_other, alpha=.25)
                ax.axhline(y=2*max(cps_bgsub), alpha=.15, ls='dotted', color=bandcolors[band_other])

        #Plot params
        #if idx_plot==len(eclipsenames)-1:
            #plt.xlabel('Time JD - '+str(jd_min_time), fontsize=20)
            #ax.set_xlabel("JD from start", fontsize=20)
            
        
        if other_band_exists and (len(cps_bgsub_other) != 0):
            ax.set_ylim(ymax = max(cps_bgsub_other+ 2*max(cps_bgsub))*1.2)
        else:
            ax.set_ylim(ymax = max(cps_bgsub)*1.2)

        ax.annotate(str(labelnum), xy=(.95, .85), xycoords='axes fraction', color='xkcd:azure', fontsize=20, horizontalalignment='center', verticalalignment='center', **afont, **txtkwargsw)
        #ax.annotate('{:.2f}'.format(round(t0, 2)), xy=(.05, .12), xycoords='axes fraction', color='black', fontsize=14, horizontalalignment='left', verticalalignment='center', **txtkwargs)
        starttimes.append( '{:.2f}'.format(round(t0, 2)))
        ax.set_xlim(xmin = -.00125, xmax=.02025)

        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', labelsize=12)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params('both', length=8, width=1.8, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        #ax.set_xticks([0, .005, .01, .015, .02])
        if not plot_coords in [(4,0), (4,1)]:
            ax.set_xticklabels([])
        #plt.tight_layout(rect=[.03,0,1,1])

   
    plt.subplot2grid((5, 2), (0,1), colspan=1, rowspan=1)
    plt.xlim(xmin=10, xmax=12)
    plt.ylim(ymin=10, ymax=12)
    plt.errorbar(0,0, yerr=10, color='red', label='NUV', marker='.', ls='')
    plt.errorbar(0,0, yerr=10, color='blue', label='FUV', marker='.', ls='', alpha=.25)
    plt.errorbar(0,0,yerr=10, label='Flagged', color='#808080', marker='.', ls='', alpha=.5)
    plt.legend(fontsize=18, loc=(0, .05), framealpha=.5)
    plt.axis('off')
    
    #plt.show()
    #fig.tight_layout(rect=[.03,0,1,1])
    fig.savefig("EclipseTower.pdf")


    #Generate output table
    df_output = pd.DataFrame({"Name":eclipsenames, "IDnum":[ int(x) for x in labelnum_list ], "Obs Start (JD)": starttimes})
    print(df_output)
    df_output.to_csv("EclipseTableOutput.csv", index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--generate", help="Generate csv which information on best visit", default=False, action='store_true')
    args= parser.parse_args()

    main(generate=args.generate)
