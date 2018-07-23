#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import matplotlib.gridspec as gs
from gPhoton import gphoton_utils
from WDranker_2 import badflag_bool
import subprocess
from matplotlib.patheffects import withStroke

#Dom Rowan REU 2018

desc="""
WDPlot_pulastor: Make LC plots for pulsators
"""
#Main plotting function
def main(generate):
    assert(os.path.isfile("IS.csv"))
    df_IS = pd.read_csv("IS.csv")
    if generate:
        pulsatornames = []
        for i in range(len(df_IS['MainID'])):
            if df_IS['type'][i] == 'Pulsator':
                pulsatornames.append(df_IS['MainID'][i])
        band_list = []
        df_list = []
        for name in pulsatornames:
            print(name)
            if os.path.isfile('Pulsator/'+name+'/PDFs/'
                              +name+'_NUV_combined.pdf'):
                subprocess.run(['xdg-open', 
                                'Pulsator/'+name+'/PDFs/'
                                +name+'_NUV_combined.pdf'])
            if os.path.isfile('Pulsator/'+name+'/PDFs/'
                               +name+'_FUV_combined.pdf'):
                subprocess.run(['xdg-open', 
                                'Pulsator/'+name+'/PDFs/'
                                +name+'_FUV_combined.pdf'])
            
            bandselection=input("Choose band of best df --- ")
            dfselection = input("In that band, choose best df --- ")
            dfselection = int(dfselection)
            band_list.append(bandselection)
            df_list.append(dfselection)
        df_pulsator_groups = pd.DataFrame({"MainID":pulsatornames, 
                                           "band":band_list, 
                                           "df_number":df_list})
        df_pulsator_groups.to_csv("pulsator_group.csv")

    #Plotting function starts here
    assert(os.path.isfile("pulsator_group.csv"))
    #Read in group information
    input_df = pd.read_csv("pulsator_group.csv")
    pulsatornames = input_df['MainID']
    band_list = input_df['band']
    df_list = input_df['df_number']
    types = input_df['type']

    #Find ID numbers
    labelnum_list = []
    for name in pulsatornames:
        idx_dfIS = np.where(df_IS['MainID'] == name)[0]
        labelnum_list.append(int(df_IS['labelnum'][idx_dfIS]))

    #Create new df to sort
    df_sort = pd.DataFrame({'names':pulsatornames, 
                            'band':band_list, 
                            'df_list':df_list, 
                            'types':types,
                            'labelnum':labelnum_list,
                    })
    df_sort = df_sort.sort_values(by='labelnum')
    df_sort = df_sort.reset_index(drop=True)

    pulsatornames_sorted = df_sort['names']
    band_list_sorted = df_sort['band']
    df_list_sorted = df_sort['df_list']
    type_list_sorted = df_sort['types']
    labelnum_list_sorted = df_sort['labelnum']
    ###Generate plot/subplot information###
    nfigs = 8
    nfigs_0 = nfigs
    fig = plt.figure(figsize=(16,nfigs*2.5))
    gs1 = gs.GridSpec(nfigs,2) 
    gs1.update(hspace=0, wspace=0.2)
    fig.text(.02, .5, 'Flux (MMI)', va='center', 
             rotation='vertical', fontsize=30)
    fig.text(.5, .05, 'Minutes from Start', 
             va='center', fontsize=30, ha='center')
    afont = {'fontname':'Keraleeyam'}

    plt.subplots_adjust(top=.98, right=.9)
    myeffectw = withStroke(foreground="black", linewidth=2) 
    txtkwargsw = dict(path_effects=[myeffectw]) 
    figfull = False
    lastfig = False
    fignumber = 1

    #fig.tight_layout(rect=[0, 0, 1, 1])
    starttimes = []
    for idx_plot in range(len(pulsatornames_sorted)):
        nleft = len(pulsatornames_sorted) - idx_plot
        if figfull:
            if nleft < 2*nfigs:
                if nleft % 2 != 0:
                    nfigs = nleft//2 + 1
                else:
                    nfigs = int(nleft / 2)
                lastfig = True
            plt.gcf().clear()
            fig = plt.figure(figsize=(16,nfigs*2.5))
            gs1 = gs.GridSpec(nfigs,2) 
            gs1.update(hspace=0, wspace=0.2)
            fig.text(.02, .5, 'Flux MMI', va='center', 
                     rotation='vertical', fontsize=30)
            fig.text(.5, .05, 'Minutes from Start', 
                     va='center', fontsize=30, ha='center')
            afont = {'fontname':'Keraleeyam'}
            plt.subplots_adjust(top=.98, right=.9)
            figfull=False

        name = pulsatornames_sorted[idx_plot]
        #band = band_list[idx_plot]
        band='NUV'
        dfbest = df_list_sorted[idx_plot]
        labelnum = labelnum_list_sorted[idx_plot]
        objecttype = type_list_sorted[idx_plot]
        if objecttype == 'Pulsator':
            csvpath = "Pulsator/"+name+"/"+name +"-" +band+".csv"
        else:
            assert(objecttype == 'KnownPulsator')
            csvpath = "KnownPulsator/"+name+"/"+name+"-"+band+".csv"
        if not os.path.isfile(csvpath):
            #Only FUV data exists
            band = 'FUV'
            csvpath = csvpath.replace('NUV', band)
        if band == 'NUV':
            band_other = 'FUV'
        else:
            band_other = 'NUV'

        bandcolors = {'NUV':'red', 'FUV':'blue'}

        alldata = pd.read_csv(csvpath)

        ###Alldata table corrections###
        #Drop rows with > 10e10 in cps, cps_err, cps < .5
        idx_high_cps = np.where( (abs(alldata['cps_bgsub']) > 10e10) | 
                                 (alldata['cps_bgsub_err'] > 10e10) | 
                                 (alldata['counts'] < 1) | 
                                 (alldata['counts'] > 100000) )[0]
        if len(idx_high_cps) != 0:
            alldata = alldata.drop(index = idx_high_cps)
            alldata = alldata.reset_index(drop=True)

        #Fix rows with weird t_means by averaging t0 and t1
        idx_tmean_fix = np.where( (alldata['t_mean'] < 1) | 
                                  (alldata['t_mean'] > alldata['t1']) | 
                                  (np.isnan(alldata['t_mean'])) )[0]
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
            idx_high_cps_other = np.where( 
                    (abs(alldata_other['cps_bgsub']) > 10e10) 
                    | (alldata_other['cps_bgsub_err'] > 10e10) 
                    | (alldata_other['counts'] < 1) 
                    | (alldata_other['counts'] > 100000) )[0]
            idx_other_flagged_bool = [ badflag_bool(x) 
                                       for x in alldata_other['flags'] ]
            idx_other_flagged = np.where(
                    np.array(idx_other_flagged_bool) == True)[0]
            idx_other_expt = np.where(alldata_other['exptime'] < 10)[0]
            idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other, 
                                                         idx_other_flagged, 
                                                         idx_other_expt]))
            alldata_other = alldata_other.drop(index=idx_other_todrop)
            alldata_other = alldata_other.reset_index(drop=True)

            #Fix rows with weird t_mean time
            idx_tmean_fix_other = np.where( 
                    (alldata_other['t_mean'] < 1) 
                    | (alldata_other['t_mean'] > alldata_other['t1']) 
                    | (np.isnan(alldata_other['t_mean'])) )[0]
            for idx in idx_tmean_fix_other:
                t0 = alldata_other['t0'][idx]
                t1 = alldata_other['t1'][idx]
                mean = (t1 + t0) / 2.0
                alldata_other['t_mean'][idx] = mean


            #Make correction for relative scales
            alldata_tmean_other = alldata_other['t_mean']
            alldata_flux_bgsub_other = alldata_other['flux_bgsub']
            alldata_medianflux_other = np.median(alldata_flux_bgsub_other)
            alldata_flux_bgsub_other = (( 
                    alldata_flux_bgsub_other / alldata_medianflux_other ) 
                    - 1.0) * 1000
            alldata_flux_bgsub_err_other = (
                    alldata_other['flux_bgsub_err'] 
                    / alldata_medianflux_other) * 1000


        ###Break the alldata table into exposure groups### 
        breaks = []
        for i in range(len(alldata['t0'])):
            if i != 0:
                if (alldata['t0'][i] - alldata['t0'][i-1]) >= 200:
                    breaks.append(i)

        data = np.split(alldata, breaks)
        df = data[dfbest-1]

        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        lasttime = list(df['t1'])[-1]
        firsttime = list(df['t0'])[0]

        #Find indicies of data above 5 sigma of mean 
        #(counts per second column), flagged points, and points with
        # less than 10 seconds of exposure time
        stdev = np.std(df['flux_bgsub'])
        bluepoints = np.where( (df['flux_bgsub'] 
                                - np.nanmean(df['flux_bgsub'])) > 5*stdev )[0]
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
        if (df_reduced['flux_bgsub'][df_reduced.index[0]] 
                - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['flux_bgsub'][df_reduced.index[-1]] 
                - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        
        #Convert units to flux MMI
        flux_bgsub = df_reduced['flux_bgsub']
        flux_bgsub_median = np.median(flux_bgsub)
        flux_bgsub = (( flux_bgsub / flux_bgsub_median ) - 1.0) * 1000
        flux_bgsub_err = (df_reduced['flux_bgsub_err'] 
                         / flux_bgsub_median) * 1000
        t_mean = df_reduced['t_mean']

        #Match with other band
        if other_band_exists:
            idx_eg_other = np.where( 
                    (alldata_tmean_other > firsttime) 
                    & (alldata_tmean_other < lasttime))[0]
            t_mean_other = (alldata_tmean_other[idx_eg_other] 
                    - firsttime_mean)
            flux_bgsub_other = alldata_flux_bgsub_other[idx_eg_other]
            flux_bgsub_err_other = alldata_flux_bgsub_err_other[idx_eg_other]

        #Flux MMI correction
        if len(redpoints) != 0:
            flux_bgsub_red = df['flux_bgsub'][redpoints]
            flux_bgsub_red = ((flux_bgsub_red / flux_bgsub_median) 
                    - 1.0) * 1000
            flux_bgsub_err_red = (df['flux_bgsub_err'][redpoints] 
                                 / flux_bgsub_median) * 1000
            t_mean_red = df['t_mean'][redpoints]
        if len(bluepoints) != 0:
            flux_bgsub_blue = df['flux_bgsub'][bluepoints]
            flux_bgsub_blue = ((flux_bgsub_blue / flux_bgsub_median) 
                    - 1.0) * 1000
            flux_bgsub_err_blue = (df['flux_bgsub_err'][bluepoints] 
                    / flux_bgsub_median) * 1000
            t_mean_blue = df['t_mean'][bluepoints]

        #Subplot for LC
        coords = []
        #for x in range(0,2):
        #    for y in range(0, 5):
        #        coords.append( (y, x) )
        for y in range(0,nfigs):
            coords.append( (y,0) )
            coords.append( (y,1) )
        if idx_plot > (2*nfigs)-1:
            while idx_plot > (2*nfigs)-1:
                idx_plot = idx_plot - ((2*nfigs))
            if lastfig:
                idx_plot = idx_plot #used to have minus two here
        plot_coords = coords[idx_plot]
        plt.subplot2grid((nfigs, 2), plot_coords, colspan=1, rowspan=1)
        #Convert to JD here as well
        jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                for t in t_mean ]

        mintimes = [min(jd_t_mean)]
        if len(redpoints) != 0:
            jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                    for t in t_mean_red ]
            mintimes.append(min(jd_t_mean_red))
        if len(bluepoints) != 0:
            jd_t_mean_blue = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                    for t in t_mean_blue ]
            mintimes.append(min(jd_t_mean_blue))
        if other_band_exists:
            if len(flux_bgsub_other) != 0:
                jd_t_mean_other = [ gphoton_utils.calculate_jd(t
                        +firsttime_mean) for t in t_mean_other ]
                mintimes.append(min(jd_t_mean_other))
        """
        jd_min_time = int(min(mintimes))
        jd_t_mean = [ t - jd_min_time for t in jd_t_mean ] 
        """
        t0 = min(jd_t_mean)
        jd_t_mean = [ t - t0 for t in jd_t_mean ]

        ax1 = plt.subplot(gs1[plot_coords])
        ax1.set_xlim(xmin = -.00125, xmax=.02025)
        ax1.minorticks_on()
        ax1.tick_params(direction='in', which='both', labelsize=12)
        customticks = [(1/1440)*x for x in np.arange(0,29,6)]
        customticklabels = [ 6*x for x in range(0,5) ]
        ax1.set_xticks(customticks)
        ax1.set_xticklabels(customticklabels)
        ax1.tick_params(axis='y', colors=bandcolors[band], which='both')
        if band=='FUV' and (not other_band_exists):
            ax1.yaxis.tick_right()
        ax1.xaxis.set_ticks_position('both')
        ax1.errorbar(jd_t_mean, flux_bgsub, yerr=flux_bgsub_err, 
                    color=bandcolors[band], marker='.', ls='', 
                    zorder=4, label=band, alpha=.5)
        ax1.axhline(alpha=.3, ls='dotted', color='black')
        if len(redpoints) != 0: 
            jd_t_mean_red = [ t - t0 for t in jd_t_mean_red ]
            ax1.errorbar(jd_t_mean_red, flux_bgsub_red, 
                        yerr=flux_bgsub_err_red, color='#808080', 
                        marker='.', ls='', zorder=2, alpha=.5, 
                        label='Flagged')
        if len(bluepoints) != 0: 
            jd_t_mean_blue = [ t - t0 for t in jd_t_mean_blue ] 
            ax1.errorbar(jd_t_mean_blue, flux_bgsub_blue, 
                        yerr=flux_bgsub_err_blue, color='green', 
                        marker='.', ls='', zorder=3, alpha=.5, 
                        label='SigmaClip')
        if other_band_exists:
            ax2 = ax1.twinx()
            ax2.minorticks_on()
            ax2.tick_params(direction='in', which='both', labelsize=12)
            if len(flux_bgsub_other) != 0:
                jd_t_mean_other = [ t - t0 for t in jd_t_mean_other ]
                ax2.errorbar(jd_t_mean_other, 
                            flux_bgsub_other, 
                            yerr=flux_bgsub_err_other, 
                            color=bandcolors[band_other], marker='.', 
                            ls='', zorder=1, label=band_other, alpha=.4)

            ax2.tick_params(axis='y', colors=bandcolors[band_other], which='both')
        #Plot params
        #if idx_plot==len(pulsatornames)-1:
            #plt.xlabel('Time JD - '+str(jd_min_time), fontsize=20)
            #ax.set_xlabel("JD from start", fontsize=20)
            
        if other_band_exists and (len(flux_bgsub_other) != 0):
            ax1.set_ylim(ymax = max(flux_bgsub) * 1.2)
            ax2.set_ylim(ymax = max(flux_bgsub_other) * 1.2)
        else:
            ax1.set_ylim(ymax = max(flux_bgsub)*1.2)

        if objecttype == 'Pulsator':
            typecolor='xkcd:red'
        else:
            typecolor='xkcd:violet'
        ax1.annotate(str(labelnum), xy=(.95, .85), xycoords='axes fraction', 
                    color=typecolor, fontsize=25, ha='center', va='center', 
                    **afont, **txtkwargsw, zorder=10)
        #ax.annotate('{:.2f}'.format(round(t0, 2)), xy=(.05, .12), xycoords='axes fraction', color='black', fontsize=14, horizontalalignment='left', verticalalignment='center', **txtkwargs)
        starttimes.append( '{:.2f}'.format(round(t0, 2)))

        #ax.yaxis.set_ticks_position('both')
        for axis in [ax1, ax2]:
            axis.tick_params('both', length=8, width=1.8, which='major')
            axis.tick_params('both', length=4, width=1, which='minor')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)
            if other_band_exists:
                ax2.spines[axis].set_linewidth(1.5)

        #ax.set_xticks([0, .005, .01, .015, .02])
        if lastfig:
            if ((not nleft == 2) and 
                    (not plot_coords in [(nfigs-1,0), (nfigs-1,1)])):
                ax1.set_xticklabels([])
            
        else:
            if not plot_coords in [(nfigs-1,0), (nfigs-1,1)]:
                ax1.set_xticklabels([])
        #plt.tight_layout(rect=[.03,0,1,1])

   
        if plot_coords == coords[-1]: 
            figfull=True
            fig.tight_layout(rect=[.03,0,1,1])
            fig.savefig("LCappendix"+str(fignumber)+".pdf")
            fignumber += 1


    fig.savefig("LCappendix"+str(fignumber)+".pdf")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--generate", 
                        help="Generate csv which information on best visit", 
                        default=False, action='store_true')
    args= parser.parse_args()
    main(generate=args.generate)
