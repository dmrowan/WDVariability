#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
from astropy.time import Time
import collections
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import random
random.seed()
from statsmodels.stats.proportion import proportion_confint
import time
import WDranker_2
import WDutils
import WDVisit
import yt

desc="""
WD_Recovery: Procedure for injecting and recovering synthetic optical 
data from WD-1145.
"""

def FindCutoff(path_array, percentile):
    assert(os.path.isfile('AllData.csv'))
    df = pd.read_csv('AllData.csv')
    source_names = [ p[:-8] for p in path_array ]
    idx_drop = []
    for i in range(len(df['Band'])):
        if ((df['Band'][i] == 'FUV') or 
                (df['SourceName'][i] not in source_names)):
            idx_drop.append(i)
    print(len(idx_drop))
    df = df.drop(index=idx_drop)
    cutoff = np.percentile(df['BestRank'], percentile)
    return cutoff


#Generate list of sources used for planet injection
def genMagLists(plot=False):
    print("---Generating Source Lists---")
    #Initialize lists
    mag_list = []
    path_list = []
    bigcatalog = pd.read_csv("MainCatalog.csv")
    k2catalog = pd.read_csv("K2.csv")
    k2ra = []
    k2dec = []
    for i in range(len(k2catalog['RA'])):
        ra_c = k2catalog['RA'][i]
        dec_c = k2catalog['Dec'][i]
        s= ra_c + " " + dec_c
        c = SkyCoord(s, unit=(u.hourangle, u.degree))
        k2ra.append(c.ra.deg)
        k2dec.append(c.dec.deg)
    kepler_catalog = SkyCoord(ra=k2ra*u.degree,dec=k2dec*u.degree)
    #Iterate through NUV files in cwd
    pbar = ProgressBar()
    for filename in pbar(os.listdir(os.getcwd())):
        if filename.endswith('.csv') and 'NUV' in filename:
            source = filename[:-8]
            our_idx = WDutils.catalog_match(source, bigcatalog)
            if len(our_idx) == 0:
                print("No catalog match for ", filename)
                continue
            else:
                our_idx = our_idx[0]
                our_ra = bigcatalog['ra'][our_idx]
                our_dec = bigcatalog['dec'][our_idx]
                our_c = SkyCoord(ra=our_ra*u.deg, dec=our_dec*u.deg)
                kidx, d2d, d3d = our_c.match_to_catalog_sky(kepler_catalog)
                if d2d < 5*u.arcsec:
                    kidx = int(kidx)
                    #print("Our RA", our_ra)
                    #print("Our DEC", our_dec)
                    #print("Kepler RA",k2catalog['RA'][kidx])
                    #print("Kepler DEC",k2catalog['Dec'][kidx])
                    continue
                else:
                    #Do a full data reduction
                    alldata = pd.read_csv(filename)
                    alldata = WDutils.df_fullreduce(alldata)
                    alldata = WDutils.tmean_correction(alldata)
                    mag = np.nanmedian(alldata['mag_bgsub'])
                    data = WDutils.dfsplit(alldata, 100)
                    #See if we have any good visits
                    for df in data:
                        visit = WDVisit.Visit(df, filename, mag=mag)
                        if visit.good_df() == True:
                            if visit.existingperiods() == False:
                                if visit.high_rank() == False:
                                    mag_list.append(mag)
                                    path_list.append(filename)
                                    break

                        """
                        if len(df['t1']) == 0:
                            continue
                        else:
                            exposuretup = WDranker_2.find_cEXP(df)
                            exposure = exposuretup.exposure
                            tflagged = exposuretup.t_flagged
                            if ((tflagged > (exposure / 4)) 
                                    or (exposure < 500)):
                                continue
                            else:
                                mag_list.append(mag)
                                path_list.append(filename)
                                break
                        """
                    continue
    median = np.median(mag_list)
    print("Median magnitude --- ", median)

    #Magnitude Histogram
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(12, 12))
        bins = np.arange(min(mag_list), max(mag_list)+.1, .1)
        ax.hist(mag_list, bins=bins, color='xkcd:red', linewidth=1.2, 
                edgecolor='black')
        ax = WDutils.plotparams(ax)

        fig.savefig("/home/dmrowan/WhiteDwarfs/NUVmaghist.pdf")

    #Save as pickle
    
    print("Saving pickle")
    print(len(mag_list), len(path_list))
    mag_array = np.array(mag_list)
    path_array = np.array(path_list)
    tup = (mag_array,path_array)
    with open('MagList3.pickle', 'wb') as handle:
        pickle.dump(tup, handle)


#Choose a source and visit
def selectLC(binvalue, binsize, mag_array, path_array, visit_array):
    #First need to find indicies for bin
    binupper = binvalue + binsize 
    idx_bin = np.where( (mag_array >= binvalue)
                       &(mag_array < binupper) )[0]
    idx_choice = np.random.choice(idx_bin)
    filename = path_array[idx_choice]
    
    data_indicies = visit_array[idx_choice]
    #If there were no good visits, pick a new source
    if len(data_indicies) == 0:
        print(filename, "no sources in visit list")
        visit, source_mag, filename, visit_i = selectLC(
                binvalue, binsize, mag_array, path_array, visit_array)
        return visit, source_mag, filename, visit_i
    
    else:
        #Standard data reduction procedure
        assert(os.path.isfile(filename))
        usecols = ['t0', 't1', 't_mean',
                   'mag_bgsub',
                   'cps_bgsub', 'cps_bgsub_err', 'counts',
                   'flux_bgsub', 'flux_bgsub_err',
                   'detrad', 'flags', 'exptime']
        alldata = pd.read_csv(filename, usecols=usecols)
        #Data reduction and time correction
        alldata = WDutils.df_reduce(alldata)
        alldata = WDutils.tmean_correction(alldata)
        #Split into visits 
        data = WDutils.dfsplit(alldata, 100)
        source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)

        idx_df = np.random.choice(data_indicies)

        df = data[idx_df]
        visit = WDVisit.Visit(df, filename, source_mag)
        visit_i = idx_df

        return visit, source_mag, filename, visit_i


#Select random observation and time chunk
def selectOptical(opticalLC, plot=False, exposure=30, center=None):
    exphalf = exposure/2
    maxtime = max(opticalLC['time'])
    #Find idx range corresponding to exposure
    idx_low = np.where(opticalLC['time'] < exphalf)[0][-1]
    idx_high = np.where(opticalLC['time'] > maxtime-exphalf)[0][0]
    idx_center = np.arange(idx_low, idx_high, 1)
    if center is None:
        time_center = np.random.choice(opticalLC['time'][idx_center])
    else:
        time_center = center
    df_optical = opticalLC[(opticalLC['time'] > time_center-exphalf) 
                          &(opticalLC['time'] < time_center+exphalf)]
    df_optical = df_optical.reset_index(drop=True)
    #Reset time to start from zero
    tmin = df_optical['time'][0]
    for i in range(len(df_optical['time'])):
        df_optical.loc[i, 'time'] = df_optical['time'][i] - tmin
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(20, 6))
        ax.scatter(df_optical['time'], df_optical['flux'])
        ax = WDutils.plotparams(ax)
        #plt.show()
        return df_optical, ax
    else:
        return df_optical, None

#Full injection procedure for single source
def main(mag, bs, mag_array, path_array, visit_array, 
         opticalLC, fname, verbose):
    #Select a visit
    visit, source_mag, filename, visit_i = selectLC(
            mag, bs, mag_array, path_array, visit_array)

    #Select a multiplicative factor
    mf = random.random() * 2
    mf = round(mf, 3) 
    visit.inject(opticalLC, mf)
    result = visit.assessrecovery()
    tup = (mag, source_mag, mf, result)

    #Generate output string
    outputstr = f"{source_mag},{mf},{result},{filename},{visit_i}"

    #Append to outputfile
    if fname is not None:
        os.system("echo {0} >> {1}".format(outputstr, fname))

    if verbose:
        print(outputstr)
    return tup

def wrapper(mag_array, path_array, opticalLC, 
            iterations, ml, mu, bs, p, fname, verbose):
    #Set up magnitude bins
    time_start = time.time()
    minbin=ml
    maxbin=mu
    bins = np.arange(minbin, maxbin+bs, bs)
    bins = np.array([ round(b, 1) for b in bins ])

    #Create mp pool and iterate
    pool = mp.Pool(processes = p)
    jobs=[]
    for i in range(iterations):
        for mag in bins:
            job = pool.apply_async(main, args=(mag, bs, mag_array, 
                                               path_array, visit_array, opticalLC, 
                                               fname, verbose,))
            jobs.append(job)
    for job in jobs:
        job.get()

    time_end = time.time()
    time_str = f"{time_end - time_start}"
    if fname is not None:
        os.system("echo {0} >> {1}".format(time_str, fname))

#Iterate through all visits of a source
def testfunction(filename, output):
    usecols = ['t0', 't1', 't_mean',
               'mag_bgsub',
               'cps_bgsub', 'cps_bgsub_err', 'counts',
               'flux_bgsub', 'flux_bgsub_err',
               'detrad', 'flags', 'exptime']
    assert(os.path.isfile(filename))
    alldata = pd.read_csv(filename, usecols=usecols)
    #Data reduction and time correction
    alldata = WDutils.df_reduce(alldata)
    alldata = WDutils.tmean_correction(alldata)
    #Split into visits 
    data = WDutils.dfsplit(alldata, 100)
    source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
    visit_list = []
    for df in data:
        visit = WDVisit.Visit(df, filename, source_mag)
        if visit.good_df() == True: 
            if visit.existingperiods() == False:
                visit_list.append(visit)

    #If there were no good visits, pick a new source
    if len(visit_list) == 0:
        outputstr = filename + " no sources in visit list"
        if output is not None:
            fname = output
            os.system("echo {0} >> {1}".format(outputstr, fname))
    else:
        for visit in visit_list:
            mf = random.random() * 2
            mf = round(mf, 3) 
            try:
                visit.inject(opticalLC, mf)
                result = visit.assessrecovery()
            except:
                result = "Error with source "+filename

            #Generate output string
            outputstr = (str(source_mag)+","
                        +str(mf)+","
                        +str(result))

            print(outputstr)
            if output is not None:
                fname = output
                os.system("echo {0} >> {1}".format(outputstr, fname))

#Multiprocessing of testfunction
def testfunction_wrapper(path_array, output, p):
    #Create mp pool and iterate
    pool = mp.Pool(processes = p)
    jobs=[]
    for filename in path_array:
        job = pool.apply_async(testfunction, args=(filename, output,))
        jobs.append(job)
    for job in jobs:
        job.get()

def gen_resultarray(fname):
    assert(os.path.isfile(fname))
    #Parse results
    with open(fname) as f:
        lines = f.readlines()
    #mag = [[float(x.strip()) for x in line.split(',')][0] for line in lines]
    #mf = [[float(x.strip()) for x in line.split(',')][1] for line in lines]
    #out = [[float(x.strip()) for x in line.split(',')][2] for line in lines]
    mag = [float(line.split(',')[0]) for line in lines]
    mf = [float(line.split(',')[1]) for line in lines]
    out = [float(line.split(',')[2]) for line in lines]

    #Define our axes
    minbin=14
    maxbin=22
    bs = .2
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Fill 2d numpy arrays 
    array2d = np.array([magbins, mfbins])
    totalarray = np.zeros((len(magbins), len(mfbins)))
    recoveryarray = np.zeros((len(magbins), len(mfbins)))
    resultarray = np.zeros((len(magbins), len(mfbins)))
    print("---Iterating Through Mags---")
    pbar = ProgressBar()
    for i in pbar(range(len(mag))):
        magval = mag[i]
        mfval = mf[i]
        result = out[i]
        magidx = np.where(array2d[0] <= magval)[0].max()
        mfidx = np.where(array2d[1] <= mfval)[0].max()
        totalarray[magidx, mfidx] += 1
        recoveryarray[magidx, mfidx] += result
    #Use recovery/total to calculate result
    for x in range(totalarray.shape[0]):
        for y in range(totalarray.shape[1]):
            if totalarray[x, y] == 0:
                resultarray[x, y] = 0
            else:
                resultarray[x,y] = (recoveryarray[x,y] / 
                                    totalarray[x,y]) * 100

    with open('resultarray.pickle', 'wb') as p:
        pickle.dump(resultarray, p)

    return resultarray

#Recovery plot with mf cuts in additional subplot
def ProjectionPlot(resultarray):
    #Define our axes
    minbin=14
    maxbin=22
    bs = .2
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Gridspec plot
    fig, outer_ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.subplots_adjust(bottom=.15)
    #gs_outer = gs.GridSpec(1, 1)
    #outer_ax = plt.subplot(gs_outer[0])
    #Subplot for imshow
    gs_inner = gs.GridSpecFromSubplotSpec(2,1, subplot_spec=outer_ax,
                                          hspace=0, height_ratios=[1,1.25])
    ax0 = plt.subplot(gs_inner[1])

    #Use np.flip and T to set up axes correctly
    im = ax0.imshow(np.flip(resultarray,1).T, cmap='viridis', 
                    extent=(14,22,0,2), aspect='auto')
    ax0.set_xlabel(r'$GALEX$'+' NUV (mag)', fontsize=25)
    ax0.set_ylabel('Scale Factor', fontsize=25)
    ax0.set_xticks([x for x in np.arange(14, 22, 1)])
    ax0 = WDutils.plotparams(ax0)
    ax0.xaxis.get_major_ticks()[0].set_visible(False)

    #Subplot for magnitude bar
    axM = plt.subplot(gs_inner[0])
    mf_values = [.5, 1, 1.5]
    current_z = 3
    colors = ['xkcd:azure', 'xkcd:blue', 'xkcd:darkblue']
    for i in range(len(mf_values)):
        idx = np.where(mfbins == mf_values[i])[0]
        slicevalues = resultarray.T[idx]
        
        axM.bar(magbins+.05, slicevalues[0], 
               label=r'$s='+str(mf_values[i])+r'$',
               width=.1, edgecolor='black', zorder=current_z, 
               color=colors[i])

        current_z -= 1

    axM.legend(loc=1)
    axM.set_xlim(xmin=14, xmax=22)
    axM.xaxis.set_ticklabels([])
    axM = WDutils.plotparams(axM)
    #axM.yaxis.get_major_ticks()[0].set_visible(False)
    axM.spines['bottom'].set_linewidth(2)
    axM.set_ylim(ymax=axM.get_ylim()[1]+5)
    axM.set_ylabel("Percent\nRecovered", fontsize=20)
    
    axcb = fig.colorbar(im, ax=outer_ax.ravel().tolist(), pad=.04, aspect=30)

    fig.savefig('ProjectionPlot.pdf')

#Basic recovery plot with single subplot
def RecoveryPlot(resultarray):

    #Define our bin arrays
    minbin=14
    maxbin=22
    bs = .2
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Gridspec plot
    fig, ax0 = plt.subplots(1, 1, figsize=(12,5))

    #Subplot for array
    ax0 = plt.gca()

    #Define custom colormap


    mycm = yt.make_colormap([(np.array([88,88,88])/255, 5),
                           (np.array([51, 8, 94])/255, 5),
                           ('dpurple', 5),
                           ('purple', 5),
                           ('dblue', 5),
                           ('blue', 5),
                           (np.array([0, 100, 20])/255, 5),
                           ('dgreen', 5),
                           ('green', 5),
                           (np.array([173, 255, 47])/255, 5),
                           ('yellow', 5),
                           (np.array([255, 200, 50])/255, 5),
                           ('orange', 5),
                           (np.array([255, 69, 0])/255, 5),
                           (np.array([168, 0, 0])/255, 5),
                           (np.array([84, 0, 0])/255, 5)],
                           name='mycm', interpolate=False)

    #Use np.flip and T to set up axes correctly
    im = ax0.imshow(np.flip(resultarray,1).T, cmap='mycm', 
                     extent=(14,22,0,2), aspect='auto')
    ax0.set_xlabel(r'$GALEX$'+' NUV (mag)', fontsize=25)
    ax0.set_ylabel('Scale Factor', fontsize=25)
    ax0.set_xticks([x for x in np.arange(14, 22, 1)])
    ax0 = WDutils.plotparams(ax0)
    ax0.xaxis.get_major_ticks()[0].set_visible(False)
    ax0.set_xlim(xmin=15, xmax=21)
    ax0.set_yticks([.5, 1, 1.5, 2])
    ax0.yaxis.set_minor_locator(MultipleLocator(.1))


    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_ylabel('Recovery Rate (%)', fontsize=25)
    plt.subplots_adjust(bottom=.175, top=.98)
    fig.savefig('RecoveryPlot.svg')

def ComparisonPlot(fname, opticalLC):
    d_sources = {
            'bright':{
                '1':{ '1':[], '0':[]}, 
                '0.5':{ '1':[], '0':[]}},
            'medium':{
                '1':{ '1':[], '0':[]}, 
                '0.5':{ '1':[], '0':[]}},
            'faint':{
                '1':{ '1':[], '0':[]}, 
                '0.5':{ '1':[], '0':[]}},
            }
    #Parse source list
    assert(os.path.isfile(fname))
    with open(fname) as f:
        lines = f.readlines()
    for l in lines:
        if (l[0]=='#') or (len(l.split())==0):
            continue
        else:
            ll = l.split()
            mag = ll[0].replace(',', '')
            mf = ll[1].replace(',', '')
            rresult = ll[2].replace(',', '')
            filepath = ll[3].replace(',','')
            visit_i = int(ll[4])
            if float(mag) < 17:
                d_sources['bright'][mf][rresult].extend([filepath, visit_i])
            elif float(mag) > 19.9:
                d_sources['faint'][mf][rresult].extend([filepath, visit_i])
            else:
                d_sources['medium'][mf][rresult].extend([filepath, visit_i])

    
    myeffectw = withStroke(foreground='black', linewidth=1)
    txtkwargsw = dict(path_effects=[myeffectw])
    afont = {'fontname':'Keraleeyam'}

    d_plotting = {'bright':d_sources['bright']['1']['1'],
                  'medium':d_sources['medium']['1']['1'],
                  'faint':d_sources['faint']['1']['1']}

    mypurple = (100/255, 0/255, 200/255)
    myblue = (0, 0, 160/255)
    myred = (168/255, 0, 0)

    fig = plt.figure(figsize=(32, 12))
    fs = 35

    mygs = gs.GridSpec(2, 3)
    plt.subplots_adjust(top=.98, right=.94, left=.06, 
                        bottom=.12, wspace=0.1)

    fig.text(.5, .05, 'Time Elapsed (min)', fontsize=fs, ha='center',
             va='center')
    fig.text(0.02, .5, 'Normalized Flux', fontsize=fs, 
             ha='left', va='center', rotation=90)
    axOptical = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

    axOptical.scatter(opticalLC['time'], opticalLC['flux'], 
                      color='xkcd:violet', marker='o', s=5)
    axOptical = WDutils.plotparams(axOptical)
    axOptical.set_xlim(xmin=min(opticalLC['time']), 
                       xmax=max(opticalLC['time']))
    axOptical.tick_params(which='both', labelsize=20)

    axbright = plt.subplot2grid((2,3), (1, 0), colspan=1, rowspan=1)
    axmedium = plt.subplot2grid((2,3), (1, 1), colspan=1, rowspan=1)
    axfaint = plt.subplot2grid((2,3), (1, 2), colspan=1, rowspan=1)


    axes_list = [axbright, axmedium, axfaint]
    axis_i = 0
    #center_list = [105, 185]
    center_list = [55, 105, 185]
    exp_list = []
    for magkey in d_plotting.keys():
        sourcefile = d_plotting[magkey][0]
        visit_i = d_plotting[magkey][1]
        assert(os.path.isfile(sourcefile))
        usecols = ['t0', 't1', 't_mean',
                   'mag_bgsub',
                   'cps_bgsub', 'cps_bgsub_err', 'counts',
                   'flux_bgsub', 'flux_bgsub_err',
                   'detrad', 'flags', 'exptime']
        alldata = pd.read_csv(sourcefile, usecols=usecols)
        #Data reduction and time correction
        alldata = WDutils.df_reduce(alldata)
        alldata = WDutils.tmean_correction(alldata)
        #Split into visits 
        data = WDutils.dfsplit(alldata, 100)
        source_mag = round(np.nanmedian(alldata['mag_bgsub']),2)

        good_list = []
        for j in range(len(data)):
            df = data[j]
            vtemp = WDVisit.Visit(df, sourcefile, source_mag)
            if vtemp.good_df() == True: 
                if vtemp.existingperiods() == False:
                    good_list.append(j)

        df = data[good_list[visit_i]]
        visit = WDVisit.Visit(df, sourcefile, source_mag) 
        visit_2 = WDVisit.Visit(df, sourcefile, source_mag)
        visit.inject(opticalLC, 1, center=center_list[axis_i])
        visit_2.inject(opticalLC, .5, center=center_list[axis_i])
        
        result1 = visit.assessrecovery()
        result2 = visit_2.assessrecovery()

        label_list = []
        for r in [result1, result2]:
            if r == 1:
                label_list.append(", recovered")
            else:
                label_list.append(", unrecovered")
        ax_c = axes_list[axis_i]
        markers1, caps1, bars1 = ax_c.errorbar(
                visit.t_mean, visit.flux_injected, 
                yerr=visit.flux_err, color='xkcd:red',
                marker='.', ls='', alpha=.75, ms=12, zorder=3, 
                label=r'$s=1.0$'+label_list[0])
        markser2, caps2, bars2 = ax_c.errorbar(
                visit_2.t_mean, visit_2.flux_injected, 
                yerr=visit_2.flux_err, color='xkcd:indigo',
                marker='.', ls='', alpha=.5, ms=12, zorder=2,
                label=r'$s=0.5$'+label_list[1])
        [bar.set_alpha(.35) for bar in bars1]
        [bar.set_alpha(.35) for bar in bars2]
        """
        if visit.FUVexists():
            ax_c.errorbar(visit.t_mean_FUV, visit.flux_injected_FUV,
                         yerr=visit.flux_err_FUV, color='blue',
                         marker='.', ls='', alpha=.5, ms=10)
        """
        mag_text = r'mag$=$' + str(source_mag) 
        ax_c.text(.05, .15, mag_text, zorder=5, 
                  transform=ax_c.transAxes,
                  color='black', fontsize=25, 
                  ha='left', va='top', 
                  **afont, #**txtkwargsw,
                  bbox=dict(boxstyle='square', fc='w', ec='none',
                      alpha=.5))

        ax_c.legend(loc=1, edgecolor='black', framealpha=0.9, 
                    markerscale=1, fontsize=20)

        axis_i += 1
        exp_list.append(visit.exposure)

    for a in [axbright, axmedium, axfaint]:
        a = WDutils.plotparams(a)
        a.set_xlim(0, 30)
        a.xaxis.get_major_ticks()[0].set_visible(False)
        a.xaxis.get_major_ticks()[-1].set_visible(False)
        a.tick_params(which='both', labelsize=20)
        a.set_ylim(ymin=a.get_ylim()[0]-.05)

    for i in range(len(center_list)):
        center = center_list[i]
        exposure = (exp_list[i]) / 60
        lower_time  = center - (exposure/2)
        upper_time = center + (exposure/2)
        condition1 = (opticalLC['time'] >= lower_time) 
        condition2 = (opticalLC['time'] <= upper_time)
        condition_idx = condition1 & condition2
        optical_idx = np.where(condition_idx)[0]
        flux_values = [ opticalLC['flux'][ii] for ii in optical_idx ]
        lower_flux = min(flux_values) - .075
        upper_flux = max(flux_values) + .075
        width = upper_time - lower_time
        height = upper_flux - lower_flux
        p = mpl.patches.Rectangle((lower_time, lower_flux), width=width, 
                                  height=height, edgecolor='black', lw=2,
                                  facecolor='#E6E6E6', alpha=.4)
        axOptical.add_patch(p)
        
        con1 = mpl.patches.ConnectionPatch(
                xyA=(lower_time, lower_flux), 
                xyB=(axes_list[i].get_xlim()[0], axes_list[i].get_ylim()[1]),
                coordsA="data", coordsB="data", axesA=axOptical, 
                axesB=axes_list[i], color='black', alpha=.4, lw=2)
        con2 = mpl.patches.ConnectionPatch(
                xyA=(upper_time, lower_flux), 
                xyB=(axes_list[i].get_xlim()[1], axes_list[i].get_ylim()[1]),
                coordsA="data", coordsB="data", axesA=axOptical, 
                axesB=axes_list[i], color='black', alpha=.4, lw=2)
        axOptical.add_artist(con1)
        axOptical.add_artist(con2)
       

    fig.savefig("RecoveryCompare.jpeg", dpi=500)

def SlicePlot(resultarray, mag_array, visit_array):
    #Define our axes
    minbin=14
    maxbin=22
    bs = .2
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])

    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Subplot for magnitude bar
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    plt.subplots_adjust(hspace=0, left=.17, right=.88, top=.97, bottom=.12)
    #mf_values = [.5, 1, 1.5]
    mf_values = [.5, 1, 1.5] 
    #z_values = [3, 2, 1]
    z_values = [1, .5, 0]
    colors = ['#4400F4', '#0000FF', '#7E00D8', '#A000BC', '#CF007E', '#FF0000']
    slicecolors = ["#8229b8", "#33bb00", "#0069ef", "#ff4319"][::-1]
    #Create three separate bar graphs
    for i in range(len(mf_values)):
        idx = np.where(mfbins == mf_values[i])[0]
        slicevalues = resultarray.T[idx]
        
        ax.bar(magbins+(bs/2), slicevalues[0], 
               #label=r'$s='+str(mf_values[i])+r'$',
               label=f"{r'$s='}{mf_values[i]}{r'$'}",
               width=bs, edgecolor='.15', zorder=z_values[i],
               color=slicecolors[i])

    
    visit_hist_mag = []
    for i in range(len(visit_array)):
        for ii in range(len(visit_array[i])):
            visit_hist_mag.append(mag_array[i])

    #ax2, ax2_1 = WDutils.DoubleY(ax2, colors=(slicecolors[3], slicecolors[0]))
    n1, bins1, patches1 = ax2.hist(mag_array, bins=magbins, color=slicecolors[3],
             linewidth=1.2, edgecolor='black', zorder=1)
    #n2, bins2, pathces2 = ax2_1.hist(visit_hist_mag, bins=magbins, color=slicecolors[0],
    #         linewidth=1.2, edgecolor='black', zorder=1)
    #visits_per_source = n2/n1
    #ax2.clear()
    #ax2.bar(magbins[:-1], n2/n1, width=.2, align='edge')

    fig.text(.5, .05, r'$GALEX$ NUV (mag)', fontsize=20,
             ha='center', va='center')
    ax.set_ylabel('Percent Recovered', fontsize = 20)
    #ax2.set_yticks([50, 100, 150, 200, 250, 300])
    ax2.set_ylabel("N WD", fontsize=20)

    ax.legend(fontsize=15, loc=1,
              edgecolor='black', framealpha=.9, markerscale=.2)

    ax.set_yticks([20, 40, 60, 80, 100])

    ax.set_xlim(xmin=15, xmax=21)
    ax2.set_xlim(xmin=15, xmax=21)
    ax = WDutils.plotparams(ax)
    ax2 = WDutils.plotparams(ax2)
    ax.set_xticklabels([])

    for a in [ax, ax2]:
        a.tick_params('both', length=4, width=1.8, which='minor')
        a.tick_params('both', length=8, width=2.6, which='major')
        a.xaxis.set_minor_locator(MultipleLocator(.2))
   

    fig.savefig('SlicePlot.pdf')


#Histogram of magnitudes and bar graph of n visits
def maglist_hist(mag_array, path_array, visit_array):
    #Generate Figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.set_xlabel(r'$GALEX$ NUV (mag)', fontsize=20)


    colors = ['#4400F4', '#0000FF', '#7E00D8', '#A000BC', '#CF007E', '#FF000']

    #Top axes is a histogram of sources binned by magnitude
    bins = np.array([ round(b, 1) for b  in np.arange(14, 21.1, .1) ])
    ax.hist(mag_array, bins=bins, color=colors[0], linewidth=1.2, 
            edgecolor='black')
    ax.set_xlim(14, 21)
    ax = WDutils.plotparams(ax)

    #Pickle load - so we don't have to re-run to create plots
    if os.path.isfile('visitarray.pickle'):
        with open('visitarray.pickle', 'rb') as p:
            n_visits = pickle.load(p)
    else:
        #Filling array over same range as histogram
        n_visits = []
        pbar = ProgressBar()
        #Same procedure used for selecting sources for maglist
        for filename in pbar(path_array):
            print(filename)
            usecols = ['t0', 't1', 't_mean',
                       'mag_bgsub',
                       'cps_bgsub', 'cps_bgsub_err', 'counts',
                       'flux_bgsub', 'flux_bgsub_err',
                       'detrad', 'flags', 'exptime']
            assert(os.path.isfile(filename))
            alldata = pd.read_csv(filename, usecols=usecols)
            #Data reduction and time correction
            alldata = WDutils.df_reduce(alldata)
            alldata = WDutils.tmean_correction(alldata)
            #Split into visits 
            data = WDutils.dfsplit(alldata, 100)
            source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
            visit_list = []
            for df in data:
                visit = WDVisit.Visit(df, filename, source_mag)
                if visit.good_df() == True: 
                    if visit.existingperiods() == False:
                        visit_list.append(visit)

            n_visits.append(len(visit_list))

        #Save as a pickle for future use
        with open('visitarray.pickle', 'wb') as p:
            pickle.dump(n_visits, p)

    visit_mags = []
    for i in range(len(mag_array)):
        for ii in range(n_visits[i]):
            visit_mags.append(mag_array[i])

    #Bottom axes: bar graph over same xrange as hist
    ax, ax1 = WDutils.DoubleY(ax, colors=(colors[0], colors[4]))
    ax1.hist(visit_mags, bins=bins, color=colors[3], lw=1.2, 
             edgecolor='black')
    ax1.set_xlim(14, 21)   
    ax1 = WDutils.plotparams(ax1)
    ax.set_ylabel('N WD Sources', fontsize=20, color=colors[0])
    ax1.set_ylabel('N GALEX Visits', fontsize=20, color=colors[3])

    fig.savefig('MagnitudeHistograms.pdf')


#Find what fraction of of the optical light curve has detectable transits
def DetectableTransits(opticalLC, threshold):
    iterations = 10000
    if not isinstance(threshold, list):
        threshold = [ threshold ]
    for i in range(len(threshold)):
        if threshold[i] > 1:
            threshold[i] = threshold[i] / 100
    results = []
    for t in threshold:
        print("Threshold: ", t)
        counter = 0
        pbar = ProgressBar()
        for i in pbar(range(iterations)):
            df_optical, ax = selectOptical(opticalLC, plot=False, exposure=30)
            idx_below = np.where(df_optical['flux'] <= t)[0]
            #There must be at least five points below threshold
            if len(idx_below) >= 5:
                counter += 1
            else:
                continue
        fraction = counter / iterations
        results.append(fraction)
    d = dict(zip(threshold, results))
    return d

#Find sources for comparision plot easily
def SourceSearch(desiredmag, mf, desired_result, mag_array, path_array, 
                 visit_array,
                 opticalLC, center=55, mrange=1):
    lowermag = desiredmag - mrange
    uppermag = desiredmag + mrange
    mag_range = [ round(b, 1) for b in np.arange(
        desiredmag-0.5, desiredmag+0.5, 0.1) ]
    desiredbin = np.random.choice(mag_range)

    #Iterate through sources in mag range until five are found 
    sources_found = []
    i = 1
    while len(sources_found) < 5:
        if i%25 == 0:
            print(i, len(sources_found))
        i += 1
        visit, source_mag, filename, visit_i = selectLC(
                desiredbin, 0.1, mag_array, path_array, visit_array)
        if visit.exposure < 1700:
            continue
        else:
            if type(mf) == list:
                visit_2 = copy.deepcopy(visit)
                visit.inject(opticalLC, mf[0], center=center)
                result = visit.assessrecovery()
                if result == desired_result[0]:
                    visit_2.inject(opticalLC, mf[1], center=center)
                    result_2 = visit_2.assessrecovery()
                    if result_2 == desired_result[1]:
                        sources_found.append([filename, visit_i[0]])
            else:
                visit.inject(opticalLC, mf, center=185)
                result = visit.assessrecovery()
                if result == desired_result:
                    if [filename, visit_i[0]] not in sources_found:
                        sources_found.append([filename, visit_i[0]])

    #Should probably format output to a txt file
    print(sources_found)

#Compute max occurrence rate
def OccurrenceRate(resultarray, mag_array, plot=False):

    #Default resultarray params
    minbin=14
    maxbin=22
    bs = .2
    magbins = np.arange(minbin, maxbin+bs, bs)
    magbins = np.array([ round(b, 1) for b in magbins ])
    mfbins = np.arange(0, 2, .1)
    mfbins = np.array([ round(b, 1) for b in mfbins ])

    #Initialize our output arrays
    N_excluded = np.zeros(len(mfbins))
    N_excluded_inclination = np.zeros(len(mfbins))
    orate = np.zeros(len(mfbins))
    orate_inclination = np.zeros(len(mfbins))

    #Iterate through each scale bin
    for i in range(resultarray.shape[1]):
        total = 0
        total_inclination = 0
        #Iterate through magnitude bins
        for ii in range(resultarray.shape[0]):

            #Find number of sources in magbin
            binval = magbins[ii]
            condition1 = mag_array >= binval
            condition2 = mag_array < binval + .1
            conditionmatch = condition1 & condition2
            idx = np.where(conditionmatch)[0]
            n_sources = len(idx)

            #Access resultarray value
            p_detection = resultarray[ii][i]
            n_excluded_ii = n_sources * (p_detection/100)
            #Computed on 85-90 degree range
            inclination_scale = .0038
            n_excluded_inclination = n_excluded_ii * .0038

            #Add values to totals
            total += n_excluded_ii
            total_inclination += n_excluded_inclination
        #Use jeffreys method to compute upper bounds
        lowBound, highBound = proportion_confint(0, total, 
                alpha=1-.9973, method='jeffreys')
        lowBound2, highBound2 = proportion_confint(0, total_inclination,
                alpha=1-.9973, method='jeffreys')
        orate[i] = highBound * 100
        orate_inclination[i] = highBound2 * 100
        #Store result in output arrays
        N_excluded[i] = total
        N_excluded_inclination[i] = total_inclination

    #Add extra value to end to make plots look nicer
    mfbins_adjusted = np.append(mfbins, 2.0)
    orate = np.append(orate, orate[-1])
    N_excluded = np.append(N_excluded, N_excluded[-1])
    orate_inclination = np.append(orate_inclination, 
            orate_inclination[-1])
    N_excluded_inclination = np.append(N_excluded_inclination,
            N_excluded_inclination[-1])

    #Values at 0.8 are quoted occurrence rates
    idx_rectangle = np.where(mfbins_adjusted == 1.0)[0][0]
    r_height = orate[idx_rectangle]
    r_height_2 = orate_inclination[idx_rectangle]
    n_height = N_excluded[idx_rectangle]
    print(f"Maximum Occurrence rate: {r_height}%")
    print(f"Max Occurrence rate w/ Inclination: {r_height_2}%")
    print(f"N Excluded: {n_height}")
    #Store output in named tuple
    OutputTup = collections.namedtuple('OccurrenceTup', ['rate', 'rate_w_i'])
    tup = OutputTup(r_height, r_height_2)
    #Generate occurrence plot
    if plot:
        #fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.2, 4), sharex=True)
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 2.7))
        #plt.subplots_adjust(hspace=0, left=.15, right=.85, top=.98)
        plt.subplots_adjust(left=.17, right=.85, top=.96, bottom=.21)
       
        #Define colors in rgb
        #mypurple = (100/255, 0/255, 200/255)
        #myblue = (0, 0, 160/255)
        #myred = (168/255, 0, 0)

        #colors = [np.array([82, 0, 239])/255, 
        #          np.array([145, 0, 202])/255, 
        #          np.array([202, 0, 133])/255]
        colors = ["#721cbd", "#ff652b"]
        ds = 'steps-post'
        #Double axes plots
        ax0, ax0_2 = WDutils.DoubleY(ax0, colors=(colors[0], colors[1]))
        ax0.plot(mfbins_adjusted, orate, drawstyle=ds, 
                 color=colors[0], ls='-', lw=2)
        ax0_2.plot(mfbins_adjusted, N_excluded, drawstyle=ds, 
                   color=colors[1], 
                   ls='-', lw=2)

        #Draw rectangle for 0.8 scale factor 
        p = mpl.patches.Rectangle((1.0, 0), width=.1, height=r_height,
                                  edgecolor='black', lw=1, facecolor='black',
                                  alpha=.6)
        ax0.add_patch(p)
        
        #Set ticklocs
        #ax0.yaxis.set_major_locator(MultipleLocator(.2))
        #ax0.yaxis.set_minor_locator(MultipleLocator(.1))
        ax0.xaxis.set_major_locator(MultipleLocator(.4))
        ax0.xaxis.set_minor_locator(MultipleLocator(.1))

        ax0.set_xlabel("Scale Factor", fontsize=20)
        ax0.set_ylabel("Max Occurrence \nRate (%)", fontsize=20)
        ax0_2.set_ylabel("N Excluded \nDetections", fontsize=20)
        ax0.set_xlim(xmin=0, xmax=2.0)
        ax0.set_ylim(ymin=0)
        ax0_2.set_ylim(ymin=0)

        #Default save in current directory
        fig.savefig("OccurrencePlot.svg")

    #Generate table output
    dic = {"ScaleFactor":mfbins_adjusted, 
            "Nexcluded":[round(x,3) for x in N_excluded],
           "OccurrenceRate":[round(x,3) for x in orate]}
    df_output = pd.DataFrame(dic)
    df_output.to_csv("OccurrenceRate.dat", index=False)

    return tup

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--iter", help="Number of iterations to run", 
                        type=int, default=100)
    parser.add_argument("--ml", help="Lower mag range", 
                        type=float, default=14)
    parser.add_argument("--mu", help="Upper mag range", 
                        type=float, default=22)
    parser.add_argument("--bs", help="Magnitude bin size", 
                        type=float, default=.1)
    parser.add_argument("--p", help="Number of processes to spawn", 
                        type=int, default=4)
    parser.add_argument("--output", help="Output filename to save results",
                        type=str, default=None)
    parser.add_argument("--v", help='Verbose', 
                        default=False, action='store_true')
    parser.add_argument("--test", help="Use test function wrapper",
                        default=False, action='store_true')
    parser.add_argument("--plot", help="3 plotting options",
                        default=False, action='store_true')
    parser.add_argument("--dt", help="Find detectable transit percentage",
                        default=False, action='store_true')
    parser.add_argument("--ss", help="Search for sources for comparison plot",
                        default=False, action='store_true')
    parser.add_argument("--gen_array", help="Generate result array pickle",
                        default=None, type=str)
    parser.add_argument("--gen_maglist", help="Generate MagList",
                        default=False, action='store_true')
    parser.add_argument("--gen_visitarray", help="Generate Visit array", 
                        default=False, action='store_true')
    args=parser.parse_args()

    #Argument assertions
    assert(args.iter > 0)
    assert( (args.ml > 10) and (args.ml < 30) and (args.ml < args.mu) )
    assert( (args.mu > 10) and (args.mu < 30) )
    assert( (args.bs >= .1) and (args.bs <=2 ) )
    assert(args.p >= 1)

    if args.gen_maglist:
        genMagLists(plot=False)

    #Pickle loads
    with open('1145LC.pickle', 'rb') as p:
        opticalLC = pickle.load(p)
    with open('MagList4.pickle', 'rb') as p2:
        MagList = pickle.load(p2)
    if os.path.isfile('resultarray.pickle'):
        with open('resultarray.pickle', 'rb') as p3:
            resultarray = pickle.load(p3)
    
    mag_array = MagList[0]
    path_array = MagList[1]
    visit_array = MagList[2]

    if args.test:
        testfunction_wrapper(path_array, args.output, args.p)
    elif args.dt:
        result = DetectableTransits(opticalLC, [.9, .75, .5])
        print(result)
    elif args.ss:
        desiredmag = float(input("Enter desired magnitude --- "))
        assert(13 < desiredmag < 22)
        desiredmf = float(input("Scale factor --- "))
        assert(0 <= desiredmf <= 2)
        desiredresult = int(input("Result (1 or 0) --- "))
        assert(desiredresult in [0,1])
        secondcriteria = input("Second criteria set? --- ")
        if len(secondcriteria) == 0:
            SourceSearch(desiredmag, desiredmf, desiredresult, 
                         mag_array, path_array, visit_array, opticalLC)
        else:
            desiredmf_2 = float(input("Scale factor --- "))
            assert(0 <= desiredmf_2 <= 2)
            desiredresult_2 = int(input("Result (1 or 0) --- "))
            assert(desiredresult_2 in [0,1])
            SourceSearch(desiredmag, [desiredmf, desiredmf_2],
                         [desiredresult, desiredresult_2], 
                         mag_array, path_array, visit_array, opticalLC)

        
    elif args.plot:
        selection = input("(1) Projection Plot \n"
                          +"(2) Recovery Plot \n"
                          +"(3) Comparison Plot \n"
                          +"(4) Slice Plot \n"
                          +"(5) Magnitude Hist \n"
                          +"(6) Occurence Plot \n"
                          +"---Select Plot Type---")
        if selection == '1':
            ProjectionPlot(resultarray)
        elif selection == '2':
            RecoveryPlot(resultarray)
        elif selection == '3':
            assert(os.path.isfile('comparisons.txt'))
            ComparisonPlot('comparisons.txt', opticalLC)
        elif selection == '4':
            SlicePlot(resultarray, mag_array, visit_array)
        elif selection == '5':
            maglist_hist(mag_array, path_array)
        elif selection == '6':
            print(OccurrenceRate(resultarray, mag_array, plot=True))
        else:
            print("Invalid selection")
    elif args.gen_array is not None:
        gen_resultarray(args.gen_array)
    elif args.gen_maglist:
        print("MagList generated")
    elif args.gen_visitarray:
        gen_visitarray(mag_array, path_array)
    else:
        wrapper(mag_array, path_array, opticalLC, args.iter, args.ml,
                args.mu, args.bs, args.p, args.output, args.v)
   
