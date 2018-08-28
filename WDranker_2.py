#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
import collections
from gPhoton import gphoton_utils
import heapq
import math
import matplotlib.gridspec as gs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import WDutils 
#Dom Rowan REU 2018


desc="""
WDranker_2.py: produces the ranked value c for a single source dependent on 
exposure, periodicity, autocorrelation, and other statistical measures
"""

#Metric calculation magnitude RMS scatter
def find_cRMS(m_ab, sigma_mag, band):
    #Path assertions
    sigmamag_percentile_path_NUV = "Catalog/magpercentiles_NUV.csv"
    sigmamag_percentile_path_FUV = "Catalog/magpercentiles_FUV.csv"
    assert(os.path.isfile(sigmamag_percentile_path_NUV))
    assert(os.path.isfile(sigmamag_percentile_path_FUV))
    #Read in mag percentile information
    if band == 'NUV':
        percentile_df = pd.read_csv(sigmamag_percentile_path_NUV)
    else:
        assert(band == 'FUV')
        percentile_df = pd.read_csv(sigmamag_percentile_path_FUV)

    magbins = percentile_df['magbin']
    magbins = np.array(magbins)
    percentile50 = percentile_df['median']
    lowerbound = percentile_df['lower']
    upperbound = percentile_df['upper']
    #Take metric as ratio to median in bin
    if m_ab < 21.5:
        sigmamag_idx = np.where(magbins <= m_ab)[0]
        if len(sigmamag_idx) > 0:
            sigmamag_idx = max(sigmamag_idx)
        else:
            sigmamag_idx = len(magbins)-1

        #sigmamag_idx = np.where(
        #        abs(m_ab-magbins) == min(abs(m_ab-magbins)))[0]
        sigmafit_val = float(percentile50[sigmamag_idx])
        sigmafit_val_upper = float(upperbound[sigmamag_idx])
        if ((sigma_mag > sigmafit_val) 
                and (sigma_mag < sigmafit_val_upper)):
            c_magfit = sigma_mag/ sigmafit_val
        elif sigma_mag>= sigmafit_val_upper:
            c_magfit = sigmafit_val_upper / sigmafit_val
        else:
            c_magfit = 0
    else:
        c_magfit = 0

    return c_magfit

#Metric calculation for exposure & quality flag
def find_cEXP(df):
    #Get raw exposure value
    lasttime = list(df['t1'])[-1]
    firsttime = list(df['t0'])[0]
    exposure = lasttime - firsttime

    coloredtup = WDutils.ColoredPoints(df)
    redpoints = coloredtup.redpoints
    bluepoints = coloredtup.bluepoints
    droppoints = np.unique(np.concatenate([redpoints, bluepoints]))

    #Exposure metric in ks
    c_exposure = (exposure) / 1000
    #Correction for flagged time
    t_flagged = (len(droppoints) * 15) / 1000 
    c_exposure = c_exposure - t_flagged
    
    OutputTup = collections.namedtuple('OutputTup', ['firsttime',
                                                     'lasttime',
                                                     'exposure',
                                                     'c_exposure',
                                                     't_flagged'])
    tup = OutputTup(firsttime, lasttime, exposure, c_exposure, t_flagged)
    return tup

def find_cPGRAM(ls, amp_detrad, exposure=1800):
    freq, amp = ls.autopower(nyquist_factor=1)
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
        lowerbound = f - .0005
        upperbound = f + .0005
        bad_detrad.append( (lowerbound, upperbound) )
        
    #Calculate false alarm thresholds
    fap = .05
    probabilities = [fap]
    faplevels = ls.false_alarm_level(probabilities)

    ditherfapval = .25
    ditherfaplevel = ls.false_alarm_level(ditherfapval)
    ditherperiod_exists=False
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
                    ditherperiod_exists = True
            #If hits is still 0, the peak isnt in any of the bad ranges
            if hits == 0:
                sspeaks.append( (f, a, fapval, ratio) ) 
        elif (fapval > fap) and (fapval <= ditherfapval):
            idx = np.where(amp==a)[0]
            f = freq[idx]
            for tup in bad_detrad:
                if (f > tup[0]) and ( f < tup[1] ):
                    ditherperiod_exists=True

    #This is a crude way to ensure we don't get any dither harmonics
    if ditherperiod_exists:
        sspeaks = []
    #Grab the info to show the strongest peak for the source
    if len(sspeaks) != 0:
        sspeaks_amp = [ peak[1] for peak in sspeaks ] 
        sspeaks_freq = [ peak[0] for peak in sspeaks ]
        sspeaks_fap = [ peak[2] for peak in sspeaks ] 
        sspeaks_ratio = [ peak[3] for peak in sspeaks ]
        strongest_freq = (
                sspeaks_freq[np.where(np.asarray(sspeaks_amp) 
                == max(sspeaks_amp))[0][0]])
        strongest_period_ratio = (
                sspeaks_ratio[np.where(np.asarray(sspeaks_amp)
                ==max(sspeaks_amp))[0][0]])
        strongest_period_fap = (
                sspeaks_fap[np.where(np.asarray(sspeaks_amp)
                ==max(sspeaks_amp))[0][0]])
        strongest_period = 1.0 / strongest_freq
        strongest_period_tup = (strongest_period[0], 
                                strongest_period_ratio, 
                                strongest_period_fap)
    else:
        strongest_period_tup = (-1,-1)

    c_periodogram = 0
    for peak in sspeaks:
        if (peak[0] < (1/ (exposure))) or (peak[0] > (1/25)):
                c_periodogram += peak[3] * .125
        else:
            c_periodogram += peak[3]

    OutputTup = collections.namedtuple('OutputTup', ['c', 
                                                     'ditherperiod_exists',
                                                     'strongest_period_tup',
                                                     'sspeaks',
                                                     'bad_detrad',
                                                     ])
    tup = OutputTup(c_periodogram, ditherperiod_exists, 
                    strongest_period_tup, sspeaks, bad_detrad)
    return tup

#Calculate Welch Stetson Variability Metric.
def find_cWS(t_mean, t_mean_other, 
            flux_bgsub, flux_bgsub_other,
            flux_bgsub_err, flux_bgsub_err_other,
            ditherperiod_exists,
            other_band_exists):

    ws_times = [] 
    ws_flux = [] 
    ws_flux_err = [] 
    ii_previous = 0     #Index to reduce number of iterations 
    for i in range(len(t_mean)):
        t = t_mean[i]
        f = flux_bgsub[i]
        ferr = flux_bgsub_err[i]
        matchfound = False
        if other_band_exists:
            for ii in range(ii_previous, len(t_mean_other)): 
                tother = t_mean_other[ii]
                fother = flux_bgsub_other[ii]
                ferrother = flux_bgsub_err_other[ii]
                if abs(t-tother) < 1.5:
                    matchfound = True
                    ws_times.append( (t, tother) )
                    ws_flux.append( (f, fother) )
                    ws_flux_err.append( (ferr, ferrother) )
                    ii_previous = ii
                    break
        if matchfound == False:
            ws_times.append( (t, None) )
            ws_flux.append( (f, None) )
            ws_flux_err.append( (ferr, None) )

    assert(len(ws_times) 
           == len(ws_flux) 
           == len(ws_flux_err) 
           == len(t_mean))
    ws_sum = 0
    for i in range(len(ws_flux)):
        fluxtup = ws_flux[i]
        errtup = ws_flux_err[i]
        if math.isnan(errtup[0]):
            deltaband=0
        else:
            deltaband = ((fluxtup[0] - np.nanmean(flux_bgsub)) 
                          / errtup[0])
        if fluxtup[1] is None:
            deltaother = 1
        else:
            assert(other_band_exists)
            if len(flux_bgsub_other) == 0:
                deltaother = 1
            else:
                if str(errtup[1]) == str(float('NaN')):
                    deltaother = 1
                else:
                    deltaother = ((fluxtup[1] 
                                   - np.nanmean(flux_bgsub_other)) 
                                   / errtup[1])

        ws_sum += deltaband*deltaother
    
    c_ws = np.sqrt(1/(len(ws_times)*(len(ws_times)-1))) * ws_sum

    #Crude way of making dither not count
    if ditherperiod_exists:
        c_ws = 0

    return c_ws

def selfcorrelation(flux_bgsub):
    autocorr_result = np.correlate(flux_bgsub, flux_bgsub, mode='full')
    autocorr_result = autocorr_result[int(autocorr_result.size/2):]

    if any(np.isinf(x) for x in autocorr_result):
        print("Infinite Values in Autocorr for group ")
        #Reassign Autocorr_result to be a bunch of zeros
        numberofzeros = len(autocorr_result)
        autocorr_result = np.zeros(numberofzeros)

    return autocorr_result
#Main ranking function
def main(csvname,
         makeplot,
         w_pgram=1,
         w_expt=.2, 
         w_WS=.30, 
         w_magfit=.25,
         comment=False):

    ###Path assertions###
    catalogpath = ("/home/dmrowan/WhiteDwarfs/"
                  +"Catalogs/MainCatalog_reduced_simbad_asassn.csv")
    sigmamag_path_NUV = "Catalog/SigmaMag_NUV.csv"
    sigmamag_path_FUV = "Catalog/SigmaMag_FUV.csv"
    assert(os.path.isfile(csvname))
    assert(os.path.isfile(catalogpath))
    assert(os.path.isfile(sigmamag_path_NUV))
    assert(os.path.isfile(sigmamag_path_FUV))
    assert(os.path.isdir('PDFs'))
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

    bandcolors = {'NUV':'red', 'FUV':'blue'}
    alldata = pd.read_csv(csvpath)
    ###Alldata table corrections###
    alldata = WDutils.df_reduce(alldata)

    #Fix rows with incorrecct t_means by averaging t0 and t1
    alldata = WDutils.tmean_correction(alldata)

    ###Apparent Magnitude### 
    m_ab = np.nanmedian(alldata['mag_bgsub'])
    sigma_mag_all = median_absolute_deviation(alldata['mag_bgsub'])
    magdic = {"mag":[m_ab], "sigma":[sigma_mag_all], "weight":[1]}
    #c_magfit_all = find_cRMS(m_ab, sigma_mag_all, band)


    ###See if we have any data in the other band###
    if band == 'NUV':
        csvpath_other = csvpath.replace('NUV', 'FUV')
    else:
        csvpath_other = csvpath.replace('FUV', 'NUV')
    #Look for file in GALEXphot/LCs
    csvpath_other = '/home/dmrowan/WhiteDwarfs/GALEXphot/LCs/'+csvpath_other
    if os.path.isfile(csvpath_other):
        other_band_exists = True
        alldata_other = pd.read_csv(csvpath_other)
    else:
        other_band_exists = False

    if other_band_exists:
        #print("Generating additional LC data for " + band_other + " band")
        alldata_other = pd.read_csv(csvpath_other)
        alldata_other = WDutils.df_fullreduce(alldata_other)
        #Fix rows with weird t_mean time
        alldata_other = WDutils.tmean_correction(alldata_other)

        #Make correction for relative scales
        relativetup_other = WDutils.relativescales(alldata_other)
        alldata_tmean_other = relativetup_other.t_mean
        alldata_flux_bgsub_other = relativetup_other.flux
        alldata_flux_bgsub_err_other = relativetup_other.err

    ###Query Catalogs###
    bigcatalog = pd.read_csv(catalogpath)
    bigcatalog_idx = WDutils.catalog_match(source, bigcatalog)
    if len(bigcatalog_idx) == 0:
        print(source, "Not in catalog")
        with open("../brokensources.txt", 'a') as f:
            f.write(source + "\n")
        return
    else:
        bigcatalog_idx = bigcatalog_idx[0]

    spectype = bigcatalog['spectype'][bigcatalog_idx]
    variability = bigcatalog['variability'][bigcatalog_idx]
    binarity = bigcatalog['binarity'][bigcatalog_idx]
    hasdisk = bigcatalog['hasdisk'][bigcatalog_idx]
    simbad_name = bigcatalog['SimbadName'][bigcatalog_idx]
    simbad_types = bigcatalog['SimbadTypes'][bigcatalog_idx]
    gmag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]


    ###Break the alldata table into exposure groups### 
    data = WDutils.dfsplit(alldata, 100)
    print("Dividing {0} data for {1} into {2} exposure groups".format(
        band, source, str(len(data))))

    #Initialize Lists
    df_number = 1
    c_vals = []
    c_ws_vals = []
    c_magfit_vals = []
    c_exp_vals = []
    c_pgram_vals = []
    df_numbers_run = []
    biglc_time = []
    biglc_counts = []
    biglc_err = []
    strongest_periods_list = []
    fap_list = []
    ditherperiod_exists = False

    ###Loop through each exposure group###
    for df in data:
        if len(df['t1']) == 0:
            df_number += 1
            continue
        ###Dataframe corrections###
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find exposure, c_exposure
        exposuretup = find_cEXP(df)
        firsttime = exposuretup.firsttime
        lasttime = exposuretup.lasttime
        exposure = exposuretup.exposure
        c_exposure = exposuretup.c_exposure
        c_exp_vals.append(c_exposure)

        #Filter for red and blue points
        coloredtup = WDutils.ColoredPoints(df)
        redpoints = coloredtup.redpoints
        bluepoints = coloredtup.bluepoints
        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))
        
        #Corrections for relative scales
        relativetup = WDutils.relativescales(df)
        t_mean = relativetup.t_mean
        flux_bgsub = relativetup.flux
        flux_bgsub_err = relativetup.err

        if len(redpoints) != 0:
            t_mean_red = [ t_mean[ii] for ii in redpoints]
            flux_bgsub_red = [ flux_bgsub[ii] for ii in redpoints]
            flux_bgsub_err_red = [ flux_bgsub_err[ii] for ii in redpoints]
        if len(bluepoints) != 0:
            t_mean_blue = [ t_mean[ii] for ii in bluepoints ] 
            flux_bgsub_blue = [ flux_bgsub[ii] for ii in bluepoints ]
            flux_bgsub_err_blue = [ flux_bgsub_err[ii] for ii in bluepoints ]

        #Drop red and blue points
        df_reduced = df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)
        

        if df_reduced.shape[0] < 10:
            df_number +=1
            continue

        #Drop bad first and last points
        df_reduced = WDutils.df_firstlast(df_reduced)

        #Have to do this again to get the reduced indicies
        relativetup_reduced = WDutils.relativescales(df_reduced)
        t_mean = relativetup_reduced.t_mean
        flux_bgsub = relativetup_reduced.flux
        flux_bgsub_err = relativetup_reduced.err

        
        #Math points in other band
        if other_band_exists:
            idx_exposuregroup_other = np.where( 
                    (alldata_tmean_other > firsttime) 
                    & (alldata_tmean_other < lasttime))[0]

            t_mean_other = np.array(
                    alldata_tmean_other[idx_exposuregroup_other] 
                    - firsttime_mean)

            flux_bgsub_other = np.array(
                    alldata_flux_bgsub_other[idx_exposuregroup_other])

            flux_bgsub_err_other = np.array(
                    alldata_flux_bgsub_err_other[idx_exposuregroup_other])


        ###Periodogram Creation###
        #Fist do the periodogram of the data
        ls = LombScargle(t_mean, flux_bgsub)
        freq, amp = ls.autopower(nyquist_factor=1)
        
        #Periodogram for dither information
        detrad = df_reduced['detrad']
        ls_detrad = LombScargle(t_mean, detrad)
        freq_detrad, amp_detrad = ls_detrad.autopower(nyquist_factor=1)

        #Periodogram for expt information
        exptime = df_reduced['exptime']
        ls_expt = LombScargle(t_mean, exptime)
        freq_expt, amp_expt = ls_expt.autopower(nyquist_factor=1)

        #Periodogram metric
        pgram_tup = find_cPGRAM(ls, amp_detrad, exposure)
        c_periodogram = pgram_tup.c
        ditherperiod_exists = pgram_tup.ditherperiod_exists
        strongest_period_tup = pgram_tup.strongest_period_tup 
        if strongest_period_tup[0] != -1:
            strongest_periods_list.append(strongest_period_tup)
            fap_list.append(strongest_period_tup[2])

        c_pgram_vals.append(c_periodogram)
        sspeaks = pgram_tup.sspeaks

        #Welch Stetson Variability Metric.
        if other_band_exists:
            c_ws = find_cWS(t_mean, t_mean_other, 
                            flux_bgsub, flux_bgsub_other,
                            flux_bgsub_err, flux_bgsub_err_other,
                            ditherperiod_exists,
                            other_band_exists)
        else:
            c_ws = find_cWS(t_mean, None, 
                            flux_bgsub, None,
                            flux_bgsub_err, None,
                            ditherperiod_exists,
                            other_band_exists)

        c_ws_vals.append(c_ws)

        #Sigma Mag Metric
        ###Grab magnitude information###
        df_sigma_mag = median_absolute_deviation(df_reduced['mag_bgsub'])
        magdic["mag"].append(m_ab)
        magdic["sigma"].append(df_sigma_mag)
        magdic["weight"].append(.25)
        c_magfit = find_cRMS(m_ab, df_sigma_mag, band)
        c_magfit_vals.append(c_magfit)

        ###Autocorrelation results###
        autocorr_result = selfcorrelation(flux_bgsub)

        #####GENERATE RATING#####
        C = ((w_pgram * c_periodogram) 
            + (w_expt * c_exposure) 
            + (w_magfit * c_magfit) 
            + (w_WS * c_ws))
        print("Exposure group "+str(df_number)+" ranking: "+ str(C))
        c_vals.append(C)


        if makeplot:
            ###Generate plot/subplot information###
            fig = plt.figure(df_number, figsize=(16,12))
            gs.GridSpec(4,4)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(
                    "Exposure group {0} with {1}s \nRanking: "
                    +"{2} {3} significant peaks".format(
                        str(df_number), str(exposure), 
                        str(C), str(len(sspeaks))))

            #Subplot for LC
            plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=2)
            #Convert to JD here as well
            jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                          for t in t_mean ]
            plt.errorbar(jd_t_mean, flux_bgsub, yerr=flux_bgsub_err, 
                         color=bandcolors[band], marker='.', ls='', 
                         zorder=4, label=band)
            plt.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
            if len(redpoints) != 0: 
                jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) 
                                  for t in t_mean_red ]
                plt.errorbar(jd_t_mean_red, flux_bgsub_red, 
                             yerr=flux_bgsub_err_red, color='#808080', 
                             marker='.', ls='', zorder=2, alpha=.5, 
                             label='Flagged')
            if len(bluepoints) != 0: 
                jd_t_mean_blue = [ gphoton_utils.calculate_jd(
                                   t+firsttime_mean) 
                                   for t in t_mean_blue ]
                plt.errorbar(jd_t_mean_blue, flux_bgsub_blue, 
                             yerr=flux_bgsub_err_blue, 
                             color='green', marker='.', ls='', 
                             zorder=3, alpha=.5, label='SigmaClip')
            if other_band_exists:
                jd_t_mean_other = [ gphoton_utils.calculate_jd(
                                    t+firsttime_mean) 
                                    for t in t_mean_other ]
                plt.errorbar(jd_t_mean_other, 
                             flux_bgsub_other, 
                             yerr=flux_bgsub_err_other, 
                             color=bandcolors[band_other], marker='.', 
                             ls='', zorder=1, label=band_other, alpha=.25)

            ax = plt.gca()
            ax = WDutils.plotparams(ax)
            plt.title(band+' light curve')
            plt.xlabel('Time JD')
            plt.ylabel('Flux mmi')
            plt.legend(loc=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            #Subplot for autocorr
            plt.subplot2grid((4,4), (2,2), colspan=1, rowspan=2)
            plt.plot(autocorr_result, 'b-', label='data')
            plt.title('Autocorrelation')
            plt.xlabel('Delay')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            ax = plt.gca()
            ax = WDutils.plotparams(ax)

            #Subplot for periodogram
            plt.subplot2grid((4,4), (2,0), colspan=2, rowspan=2)
            ax = plt.gca()
            ax = WDutils.plotparams(ax)
            ax.plot(freq, amp, 'g-', label='Data')
            ax.plot(freq_detrad, amp_detrad, 'r-', label="Detrad", alpha=.25)
            ax.plot(freq_expt, amp_expt, 'b-', label="Exposure", alpha=.25)
            ax.set_title(band+' Periodogram')
            ax.set_xlabel('Freq [Hz]')
            ax.set_ylabel('Amplitude')
            ax.set_xlim(0, np.max(freq))
            try:
                ax.set_ylim(0, np.max(amp)*2)
            except:
                print("Issue with periodogram axes")

            top5amp_detrad = heapq.nlargest(5, amp_detrad)
            bad_detrad = pgram_tup.bad_detrad
            if any(np.isnan(x) for x in top5amp_detrad):
                print("No detrad peaks for exposure group " + str(df_number))
            else:
                for tup in bad_detrad:
                    ax.axvspan(tup[0], tup[1], alpha=.1, color='black')
            
            #ax[0][1].axvline(x=nyquistfreq, color='r', ls='--')
            for level in [.05]:
                ax.axhline(level, color='black', alpha = .5, 
                            ls = '--', label = 'FAP: '+str(level))
            ax.axhline(.25, color='black', alpha=.5, 
                        ls=':', label = 'FAP: '+str(.25))

            ax.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            #Subplot for png image
            plt.subplot2grid((4,4), (2,3), colspan=1, rowspan=2)
            pngfile = ("/home/dmrowan/WhiteDwarfs/GALEXphot/pngs/"
                      +source+".png")
            img1 = mpimg.imread(pngfile)
            plt.imshow(img1)
            #Turn of axes 
            #ax[1][1].axis('off')
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            saveimagepath = str(
                    "PDFs/"+source+"-"+band+"qlp"+str(df_number)+".pdf")
            fig.savefig(saveimagepath)

            #Close figure
            fig.clf()
            plt.close('all')
        
        #Information for big light curve
        biglc_time.append(np.nanmean(t_mean + firsttime_mean))
        biglc_counts.append(np.nanmean(flux_bgsub))
        biglc_err.append(
                np.std(flux_bgsub_err) / np.sqrt(df_reduced.shape[0]))

        df_numbers_run.append(df_number)
        df_number += 1



    ###Find the total rank, best rank, and best group###
    totalrank = np.sum(c_vals)
    if len(c_vals) !=0:
        bestrank = max(c_vals)
        idx_best = np.where(np.array(c_vals) == bestrank)[0][0]
        best_expt_group = df_numbers_run[idx_best]
        c_ws_best = c_ws_vals[idx_best]
        c_magfit_best = c_magfit_vals[idx_best]
        c_ws_max = max(c_ws_vals)
        c_exp_max = max(c_exp_vals)
        c_pgram_max = max(c_pgram_vals)
    else:
        bestrank = 0
        idx_best = 0
        best_expt_group=0
        c_ws_best = 0
        c_magfit_best = 0
        c_ws_max = 0
        c_exp_max = 0
        c_pgram_max = 0
    print(source, "Total rank: " + str(totalrank), 
          "Best rank: " + str(bestrank), 
          "Best group: " + str(best_expt_group))

    ###Get most prevalent period from strongest_periods_list###
    all_periods = [ tup[0] for tup in strongest_periods_list ]
    all_ratios = [ tup[1] for tup in strongest_periods_list ] 
    if len(all_periods) > 1:
        period_to_save = all_periods[np.where(np.asarray(all_ratios) 
                                              == max(all_ratios))[0][0]]
        best_fap = min(fap_list)
    elif len(all_periods) == 1:
        period_to_save = all_periods[0]
        period_to_save = round(period_to_save,3)
        best_fap = min(fap_list)
    else:
        period_to_save = ''
        best_fap = ''

    #Generate output csv with pandas
    outputdic = {
            "SourceName":[source], 
            "Band":[band], 
            "TotalRank":[round(totalrank, 3)], 
            "BestRank":[round(bestrank, 3)], 
            "Comment":[""],
            "ABmag":[round(m_ab, 2)], 
            "StrongestPeriod":[period_to_save], 
            "False Alarm Prob.":[best_fap],
            "WS metric":[c_ws_best],
            "c_magfit":[c_magfit_best],
            "SimbadName":[simbad_name],
            "SimbadTypes":[simbad_types],
            "Spectype":[spectype],
            "KnownVariable":[variability], 
            "Binarity":[binarity],
            "Hasdisk":[hasdisk],
            "c_ws_max":[c_ws_max],
            "c_exp_max":[c_exp_max],
            "c_pgram_max":[c_pgram_max],
            }
    dfoutput = pd.DataFrame(outputdic)
    dfoutput.to_csv("Output/"+source+"-"+band+"-output.csv", index=False)


    if makeplot:
        #####Generate multiplage pdf#####

        ###Page 1###
        #Drop flagged rows from alldata
        alldata_flag_bool_vals = [ WDutils.badflag_bool(x) 
                                   for x in alldata['flags'] ]
        alldata_flag_idx = np.where(
                np.array(alldata_flag_bool_vals) == True)[0]
        alldata = alldata.drop(index = alldata_flag_idx)
        alldata = alldata.reset_index(drop=True)
        #Make the correction for relative scales
        alldata_tmean = alldata['t_mean']
        alldata_flux_bgsub = alldata['flux_bgsub']
        alldata_medianflux = np.nanmedian(alldata_flux_bgsub)
        alldata_flux_bgsub = ( alldata_flux_bgsub / alldata_medianflux ) - 1.0
        alldata_flux_bgsub_err = (alldata['flux_bgsub_err'] 
                                  / alldata_medianflux)

        #Convert to JD
        alldata_jd_tmean = [ gphoton_utils.calculate_jd(t) 
                             for t in alldata_tmean ] 
        biglc_jd_time = [ gphoton_utils.calculate_jd(t) for t in biglc_time ]
        if other_band_exists:
            alldata_jd_tmean_other = [ gphoton_utils.calculate_jd(t) 
                                       for t in alldata_tmean_other ]

        #See if ASASSN data exists:
        if type(bigcatalog['ASASSNname'][bigcatalog_idx]) != str:
            asassn_exists = False
        else:
            asassn_exists = True
            asassn_name = bigcatalog['ASASSNname'][bigcatalog_idx]

        #Plot ASASSN data
        if asassn_exists:
            print("ASASSN data exists")
            figall = plt.figure(figsize=(16,12))
            gs.GridSpec(2, 2)
            figall.tight_layout(rect=[0, .03, 1, .95])
            #Plot total light curve
            plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=1)
            plt.errorbar(biglc_jd_time, biglc_counts, yerr=biglc_err, 
                         color=bandcolors[band], marker='.', ls='-',  
                         zorder=3, ms=15, label=band)
            plt.errorbar(alldata_jd_tmean, alldata_flux_bgsub, 
                         yerr=alldata_flux_bgsub_err, color='black', 
                         marker='.', zorder=2, ls='', alpha=.125)
            plt.xlabel('Time [s]')
            plt.ylabel('Relative Counts per Second')
            #Plot data in other band
            if other_band_exists:
                print("Plotting additional LC data for " + 
                      band_other + " band")
                plt.errorbar(alldata_jd_tmean_other, alldata_flux_bgsub_other, 
                             yerr=alldata_flux_bgsub_err_other, 
                             color=bandcolors[band_other], marker='.', 
                             ls='', zorder=1, alpha=.25, label=band_other)
            plt.xlabel('Time [s]')
            plt.ylabel('Flux MMI')
            plt.legend()

            #Plot ASASSN data
            plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
            axASASSN_LC = plt.gca()
            axASASSN_LC = WDutils.plotASASSN_LC(axASASSN_LC, asassn_name)
            axASASSN_LC = WDutils.plotparams(axASASSN_LC)

            plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
            axASASSN_pgram = plt.gca()
            axASASSN_pgram = WDutils.plotASASSN_pgram(axASASSN_pgram, 
                                                      asassn_name)
            axASASSN_pgram = WDutils.plotparams(axASASSN_pgram)
            
        else:
            figall, axall = plt.subplots(1,1,figsize=(16,12))
            figall.tight_layout(rect=[0, 0.03, 1, 0.95])
            #Plot total light curve
            axall.errorbar(biglc_jd_time, biglc_counts, 
                           yerr=biglc_err, color=bandcolors[band], 
                           marker='.', ls='-',  zorder=3, ms=15, label=band)
            axall.errorbar(alldata_jd_tmean, alldata_flux_bgsub, 
                           yerr=alldata_flux_bgsub_err, color='black', 
                           marker='.', zorder=2, ls='', alpha=.125)
            #Plot data in other band
            if other_band_exists:
                print("Plotting additional LC data for " 
                      + band_other + " band")
                axall.errorbar(alldata_jd_tmean_other, 
                               alldata_flux_bgsub_other, 
                               yerr=alldata_flux_bgsub_err_other, 
                               color=bandcolors[band_other], marker='.', 
                               ls='', zorder=1, alpha=.25, label=band_other)
            axall.set_xlabel('Time [s]')
            axall.set_ylabel('Flux MMI')
            axall.legend()

        #Supertitle
        figall.suptitle("Combined Light Curve for {0} in {1} "
                        +"\nBest rank {2} in exposure group {3} "
                        +"\nTotal rank {4} in {5} exposure groups".format(
                            source, 
                            band, 
                            str(round(bestrank,2)), 
                            str(best_expt_group), 
                            str(round(totalrank, 2)), 
                            str(len(data)))
                        )

        all1saveimagepath = str("PDFs/"+source+"-"+band+"all1"+".pdf")
        figall.savefig(all1saveimagepath)
        #Clear figure
        figall.clf()
        plt.close('all')

        ###Page 2### Magnitude sigma plot and Source information
        #Get info from sigmamag csv file (from WDsigmamag)
        figall2, axall2 = plt.subplots(2,1,figsize=(16,12))
        figall2.tight_layout(rect=[0, 0.03, 1, 0.95])
        if band == 'NUV':
            df_sigmamag = pd.read_csv(sigmamag_path_NUV)
        else:
            assert(band == 'FUV')
            df_sigmamag = pd.read_csv(sigmamag_path_FUV)
        #Pull values, weights
        allmags = df_sigmamag['m_ab']
        allsigma = df_sigmamag['sigma_m']
        df_alphas = df_sigmamag['weight']
        rgb_1 = np.zeros((len(df_alphas),4))
        rgb_1[:,3] = df_alphas
        #Create magnitude bins using np.digitize
        axall2[0].scatter(allmags,allsigma,color=rgb_1, zorder=1, s=5)

        #Get information from magdic
        sourcemags = np.array(magdic['mag'])
        sourcesigmas =np.array(magdic['sigma'])
        sourcealphas = np.array(magdic['weight'])
        #Make lists for arrow points (above .3 sigma)
        arrow_mag = []
        arrow_sigma = []
        arrow_alpha = []
        idx_arrow = np.where(sourcesigmas > .3)[0]
        for idx in idx_arrow:
            arrow_mag.append(sourcemags[idx])
            arrow_sigma.append(.29)
            arrow_alpha.append(sourcealphas[idx])

        #Drop these indicies from the source arrays
        sourcemags = np.delete(sourcemags, idx_arrow)
        sourcesigmas = np.delete(sourcesigmas, idx_arrow)
        sourcealphas = np.delete(sourcealphas, idx_arrow)

        #Make color code information
        rgb_2 = np.zeros((len(sourcealphas), 4))
        rgb_2[:,0] = 1.0
        rgb_2[:,3] = sourcealphas
        
        #Make color code information for arrow
        rgb_arrow = np.zeros((len(arrow_alpha),4))
        rgb_arrow[:,0] = .3
        rgb_arrow[:,1] = .7
        rgb_arrow[:,2] = 1.0
        rgb_arrow[:,3] = arrow_alpha


        axall2[0].scatter(sourcemags, sourcesigmas, color=rgb_2, zorder=2)
        axall2[0].scatter(arrow_mag, arrow_sigma, 
                          color=rgb_arrow, marker="^", zorder=3)
        axall2[0].set_title("Sigma as a function of AB mag")
        axall2[0].set_xlabel("AB mag")
        axall2[0].set_ylabel("Sigma")
        axall2[0].set_ylim(ymin=0)
        axall2[0].set_ylim(ymax=.3)
        axall2[0].set_xlim(xmin=13)
        axall2[0].set_xlim(xmax=23)

        ###Information for text subplot
        axall2[1].set_ylim(ymin=0, ymax=1)
        information1 = """
        Source name:         \n
        Band:                \n
        ABMagnitude:         \n
        g Magitude:          \n
        Spectral Type:       \n
        SIMBAD Designation:  \n
        SIMBAD Type list:    \n
        Known Variability:   \n
        Known Binarity:      \n
        Has Disk:            \n
        Strongest Period:    \n
        """
        information2 = """
        {0} \n
        {1} \n
        {2} \n
        {3} \n
        {4} \n
        {5} \n
        {6} 
        {7} \n
        {8} \n
        {9} \n
        {10} \n
        """.format(source, band, str(round(m_ab,4)), str(round(gmag, 4)), 
                   spectype, simbad_name, simbad_types, variability, 
                   binarity, hasdisk, period_to_save
            )
        axall2[1].text(.2, 1, information1, size=15, ha='left', va='top')
        axall2[1].text(.7, 1, information2, size=15, ha='right', va='top')
        axall2[1].axis('off')

        all2saveimagepath = str("PDFs/"+source+"-"+band+"all2"+".pdf")
        figall2.savefig(all2saveimagepath)

        #Clear figure
        figall.clf()
        plt.close('all')

        #Generate PDF
        subprocess.run(['PDFcreator', '-s', source, '-b', band])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            "--csvname", 
            help= "Input full csv file", 
            required=True, type=str)
    parser.add_argument(
            "--comment", 
            help="Add comments/interactive mode", 
            default=False, action='store_true')
    args= parser.parse_args()

    main(csvname=args.csvname,
         comment=args.comment,
         makeplot=True)
