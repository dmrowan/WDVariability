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
import subprocess
import warnings
import importlib
gu = importlib.import_module('gPhoton.gphoton_utils')
#Dom Rowan REU 2018

warnings.simplefilter("once")

#Function to read in ASASSN data - even weird tables 
def readASASSN(path):
    jd_list = []
    mag_list = []
    mag_err_list = []
    with open(path) as f:
        for line in f:
            if line[0].isdigit():
                datlist = line.rstrip().split()
                jd_list.append(datlist[0])
                mag_list.append(datlist[7])
                mag_err_list.append(datlist[8])

    i=0
    while i < len(mag_err_list):
        if float(mag_err_list[i]) > 10:
            del jd_list[i]
            del mag_list[i]
            del mag_err_list[i]
        else:
            i += 1

    jd_list = [ float(element) for element in jd_list ] 
    mag_list = [ float(element) for element in mag_list ]
    mag_err_list = [ float(element) for element in mag_err_list ]

    return [jd_list, mag_list, mag_err_list]



def main(csvname, fap, prange, w_pgram, w_expt, w_ac, w_mag, w_known, comment):
    #Path assertions
    assert(os.path.isfile(csvname))
    catalogpath = "/home/dmrowan/WhiteDwarfs/Catalogs/BigCatalog_Final.csv"
    assert(os.path.isfile(catalogpath))
    sigmamag_path = "Catalog/SigmaMag.csv"
    assert(os.path.isfile(sigmamag_path))
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

    bandcolors = {'NUV':'red', 'FUV':'blue'}

    alldata = pd.read_csv(csvpath)
    
    #Drop rows with > 10e10 in cps, cps_err, cps < .5
    idx_high_cps = np.where( (abs(alldata['cps_bgsub']) > 10e10) | (alldata['cps_bgsub_err'] > 10e10) | (alldata['counts'] < 1) )[0]
    if len(idx_high_cps) != 0:
        alldata = alldata.drop(index = idx_high_cps)
        alldata = alldata.reset_index(drop=True)

    #Fix rows with weird t_means
    #    (some rows have almost zero t_mean, just average t0 and t1 in those rows)
    idx_tmean_fix = np.where( (alldata['t_mean'] < 1) | (alldata['t_mean'] > alldata['t1']) | (np.isnan(alldata['t_mean'])) )[0]
    for idx in idx_tmean_fix:
        t0 = alldata['t0'][idx]
        t1 = alldata['t1'][idx]
        mean = (t1 + t0) / 2.0
        alldata['t_mean'][idx] = mean

    #Get information on flags
    n_rows = alldata.shape[0]
    n_flagged = len( np.where(alldata['flags'] != 0)[0])
    if float(n_rows) ==0:
        flaggedratio = float('NaN')
    else:
        flaggedratio = float(n_flagged) / float(n_rows)

    ###Apparent Magnitude### - could also be done using conversion from flux 
    m_ab = np.nanmean(alldata['mag_bgsub'])
    sigma_mag_all = np.nanstd( (alldata['mag_bgsub_err_1'] + alldata['mag_bgsub_err_2'])/2.0 )
    #Calculate c_mag based on ranges:
    if m_ab < 16:
        c_mag = 1
    elif m_ab < 17:
        c_mag = .5
    elif m_ab < 18:
        c_mag = .25
    else:
        c_mag = 0

    magdic = {"mag":[m_ab], "sigma":[sigma_mag_all], "weight":[1]}

    ###Check if in knownvariable###
    c_known = 0
    df_known_variables_DAV = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/DAVInstabilityStrip.csv")
    dav_ra_dec = [ "".join(radec.split()) for radec in df_known_variables_DAV['RA,DEC (J2000)'] ]
    if source in dav_ra_dec:
        print("Known DAV variable")
        c_known = 1

    df_known_variables_DBV = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/DBVInstabilityStrip.csv")
    dbv_ra_dec = [ "".join(radec.split()) for radec in df_known_variables_DBV['RA_DEC'] ]
    if source in dbv_ra_dec:
        print("Known DBV variable")
        c_known = 1

    ###See if we have any data in the other band###
    csvpath_other = list(csvpath)
    csvpath_other[-7] = band_other[0]
    csvpath_other = "".join(csvpath_other)
    if os.path.isfile(csvpath_other):
        other_band_exists = True
        alldata_other = pd.read_csv(csvpath_other)
    else:
        other_band_exists = False

    if other_band_exists:
        print("Generating additional LC data for " + band_other + " band")
        alldata_other = pd.read_csv(csvpath_other)
        #Drop bad rows, flagged rows
        idx_high_cps_other = np.where( (abs(alldata_other['cps_bgsub']) > 10e10) | (alldata_other['cps_bgsub_err'] > 10e10) | (alldata_other['counts'] < 1) )[0]
        #Not interested in looking at red/blue points for other band
            #drop flagged, expt < 10
        idx_other_flagged = np.where( alldata_other['flags'] != 0 | (alldata_other['exptime'] < 10) )[0]
        idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other, idx_other_flagged]))
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
            if (alldata['t0'][i] - alldata['t0'][i-1]) >= 100:
                breaks.append(i)

    data = np.split(alldata, breaks)
    print("Dividing " + band + " data for source " + source+ " into "+str(len(data))+" exposure groups")

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
        c_exposure = (lasttime - firsttime) / 1000

        ###Dataframe corrections###
        
        #Reset first time in t_mean to be 0
        firsttime_mean = df['t_mean'][df.index[0]]
        df['t_mean'] = df['t_mean'] - firsttime_mean

        #Find indicies of data above 5 sigma of mean (counts per second column), flagged points, and points with
        # less than 10 seconds of exposure time
        stdev = np.std(df['cps_bgsub'])
        bluepoints = np.where( (df['cps_bgsub'] - np.mean(df['cps_bgsub'])) > 5*stdev )[0]
        redpoints = np.where( (df['flags']!=0) | (df['exptime'] < 10) )[0]
        redpoints = redpoints + df.index[0]
        bluepoints = bluepoints + df.index[0]

        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))

        #print("Removing " +  str(len(redpoints)) + " bad points")
        df_reduced = df.drop(index=droppoints)
        df_reduced = df_reduced.reset_index(drop=True)

        #Remove points where cps_bgsub is nan
        idx_cps_nan = np.where( np.isnan(df_reduced['cps_bgsub']) )[0]
        if len(idx_cps_nan) != 0:
            df_reduced = df_reduced.drop(index=idx_cps_nan)
            df_reduced = df_reduced.reset_index(drop=True)

        if df_reduced.shape[0] < 7:
            print("Not enough points for this exposure group, skipping. Removed " +  str(len(redpoints)) + " bad points")
            df_number +=1
            continue

        #If first point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[0]] - np.mean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])

        #If last point is not within 3 sigma, remove
        if (df_reduced['cps_bgsub'][df_reduced.index[-1]] - np.mean(df['cps_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])

        #Grab magnitude information
        df_m_ab = np.nanmean(df_reduced['mag_bgsub'])
        df_sigma_mag = np.nanstd( (df_reduced['mag_bgsub_err_1'] + df_reduced['mag_bgsub_err_2'])/2.0 )

        magdic["mag"].append(df_m_ab)
        magdic["sigma"].append(df_sigma_mag)
        magdic["weight"].append(.25)

        #Get the cps_bgsub, error and time columns and make correction for relative scales
        cps_bgsub = df_reduced['cps_bgsub']
        cps_bgsub_median = np.median(cps_bgsub)
        cps_bgsub = ( cps_bgsub / cps_bgsub_median ) - 1.0
        cps_bgsub_err = df_reduced['cps_bgsub_err'] / cps_bgsub_median
        t_mean = df_reduced['t_mean']
        #If we have data in the other band, find points corresponding to this exposure group
        #We've already done the relative scale correction (on the entire alldata_other table)
        #First get the indicies corresponding to this group in the other band
        if other_band_exists:
            idx_exposuregroup_other = np.where( (alldata_tmean_other > firsttime) & (alldata_tmean_other < lasttime))[0]
            t_mean_other = alldata_tmean_other[idx_exposuregroup_other] - firsttime_mean
            cps_bgsub_other = alldata_cps_bgsub_other[idx_exposuregroup_other]
            cps_bgsub_err_other = alldata_cps_bgsub_err_other[idx_exposuregroup_other]

        #Convert times to JD
        #t_mean_2 = [ gu.calculate_jd(time) for time in t_mean ] 

        #Make the correction for relative scales for redpoints and purplepoints
        if len(redpoints) != 0:
            cps_bgsub_red = df['cps_bgsub'][redpoints]
            cps_bgsub_median_red = np.median(cps_bgsub_red)
            cps_bgsub_red = (cps_bgsub_red / cps_bgsub_median_red) - 1.0
            cps_bgsub_err_red = df['cps_bgsub_err'][redpoints] / cps_bgsub_median_red
            t_mean_red = df['t_mean'][redpoints]
        if len(bluepoints) != 0:
            cps_bgsub_blue = df['cps_bgsub'][bluepoints]
            cps_bgsub_median_blue = np.median(cps_bgsub_blue)
            cps_bgsub_blue = (cps_bgsub_blue / cps_bgsub_median_blue) - 1.0
            cps_bgsub_err_blue = df['cps_bgsub_err'][bluepoints] / cps_bgsub_median_blue
            t_mean_blue = df['t_mean'][bluepoints]
        ###Periodogram Creation###
        #Fist do the periodogram of the data
        #if df_number == 7:
        #    print(cps_bgsub)
        ls = LombScargle(t_mean, cps_bgsub)
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
        
        c_periodogram = 0
        for peak in sspeaks:
            if len(sspeaks) < 3:
                c_periodogram += peak[3]
            if len(sspeaks) >= 3:
                c_periodogram += peak[3] * .25

        #Grab the info to show the strongest peak for the source
        if len(sspeaks) != 0:
            sspeaks_amp = [ peak[1] for peak in sspeaks ] 
            sspeaks_freq = [ peak[0] for peak in sspeaks ] 
            sspeaks_ratio = [ peak[3] for peak in sspeaks ]
            strongest_freq = (sspeaks_freq[np.where(np.asarray(sspeaks_amp) == max(sspeaks_amp))[0][0]])
            strongest_period_ratio = (sspeaks_ratio[np.where(np.asarray(sspeaks_amp)==max(sspeaks_amp))[0][0]])
            strongest_period = 1.0 / strongest_freq
            strongest_periods_list.append((strongest_period[0], strongest_period_ratio))

        ###Autocorrelation results###
        autocorr_result = np.correlate(cps_bgsub, cps_bgsub, mode='full')
        autocorr_result = autocorr_result[int(autocorr_result.size/2):]
            
        if any(np.isinf(x) for x in autocorr_result):
            print("Infinite Values in Autocorr for group "+str(df_number))
            #Reassign Autocorr_result to be a bunch of zeros
            numberofzeros = len(autocorr_result)
            autocorr_result = np.zeros(numberofzeros)

        '''
        #I removed this all for now because the autocorr wasn't that useful, and because it brings up alot of errors in test cases
        ac_x = range(len(autocorr_result))
        
        popt, pcov = curve_fit(fitfunc, ac_x, autocorr_result)
        
        params = np.polyfit(ac_x, autocorr_result, 1)
        residuals = autocorr_result - (ac_x*params[0]+params[1])
        #residuals = autocorr_result - fitfunc(ac_x, *popt)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((autocorr_result-np.mean(autocorr_result))**2)
        if ss_tot == 0:
            r_squared = 0
            c_autocorr = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            c_autocorr = 1 - r_squared
        '''
        c_autocorr = 0

        ###Generate rating###
        C = (w_pgram * c_periodogram) + (w_expt * c_exposure) + (w_ac * c_autocorr) + (w_mag * c_mag) - (w_known * c_known)
        print("Exposure group "+str(df_number)+" ranking: "+ str(C))
        c_vals.append(C)


        ###Generate plot/subplot information###
        fig, ax = plt.subplots(2,2,figsize=(16,12))
        fig.suptitle(source+" "+band+ "\n" + 
                "AB Magnitude: " + str(round(m_ab, 2)) + ", Exposure: "+str(exposure)+ " seconds \n" +
                "Exposure group: " + str(df_number) +"\n"+
                #"SSPs: "+str(len(sspeaks)) + "\n" +
                "Periodogram ratio: "+str(c_periodogram) + "\n" +
                "Ranking: " + str(C)
                )

        #Subplot for LC
        #Convert to JD here as well
        jd_t_mean = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean ]
        ax[0][0].errorbar(jd_t_mean, cps_bgsub, yerr=cps_bgsub_err, color=bandcolors[band], marker='.', ls='-', zorder=4, label=band)
        ax[0][0].axhline(alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: #points aren't even red now...
            jd_t_mean_red = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
            ax[0][0].errorbar(jd_t_mean_red, cps_bgsub_red, yerr=cps_bgsub_err_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
        if len(bluepoints) != 0: #these points aren't blue either...
            jd_t_mean_blue = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
            ax[0][0].errorbar(jd_t_mean_blue, cps_bgsub_blue, yerr=cps_bgsub_err_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
        if other_band_exists:
            #introduce offset here
            jd_t_mean_other = [ gu.calculate_jd(t+firsttime_mean) for t in t_mean_other ]
            ax[0][0].errorbar(jd_t_mean_other, cps_bgsub_other+2*max(cps_bgsub), yerr=cps_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, label=band_other, alpha=.25)
            ax[0][0].axhline(y=2*max(cps_bgsub), alpha=.15, ls='dotted', color=bandcolors[band_other])

        ax[0][0].set_title(band+' light curve')
        ax[0][0].set_xlabel('Time JD')
        ax[0][0].set_ylabel('Variation in CPS')
        ax[0][0].legend()

        #Subplot for autocorr
        ax[1][0].plot(autocorr_result, 'b-', label='data')
        #ax[1][0].plot(ac_x, fitfunc(ac_x, *popt), 'g-', label='fit')
        #ax[1][0].plot(residuals, 'r--', alpha=.25, label='residuals')
        #ax[1][0].plot(ac_x, ac_x*params[0]+params[1], 'r--', alpha=.5, label='linear fit')
        ax[1][0].set_title('Autocorrelation')
        ax[1][0].set_xlabel('Delay')
        #ax[1][0].legend()

        #Subplot for periodogram
        ax[0][1].plot(freq, amp, 'g-', label='Data')
        ax[0][1].plot(freq_detrad, amp_detrad, 'r-', label="Detrad", alpha=.25)
        ax[0][1].plot(freq_expt, amp_expt, 'b-', label="Exposure", alpha=.25)
        ax[0][1].set_title(band+' Periodogram')
        ax[0][1].set_xlabel('Freq [Hz]')
        ax[0][1].set_ylabel('Amplitude')
        ax[0][1].set_xlim(0, np.max(freq))
        if any(np.isnan(x) for x in top5amp_detrad):
            print("No detrad peaks for exposure group " + str(df_number))
        else:
            for tup in bad_detrad:
                ax[0][1].axvspan(tup[0], tup[1], alpha=.1, color='black')
        
        #ax[0][1].axvline(x=nyquistfreq, color='r', ls='--')
        for level in faplevels:
            idx = np.where(level == faplevels)[0][0]
            fap = probabilities[idx]
            ax[0][1].axhline(level, color='black', alpha = .5, ls = '--', label = 'FAP: '+str(fap))

        ax[0][1].legend()
        #Subplot for png image
        pngfile = "/home/dmrowan/WhiteDwarfs/GALEXphot/pngs/"+source+".png"
        img1 = mpimg.imread(pngfile)
        ax[1][1].imshow(img1)
        #Turn of axes 
        ax[1][1].axis('off')

        saveimagepath = str("PNGs/"+source+"-"+band+"qlp"+str(df_number)+".png")
        fig.savefig(saveimagepath)
        df_numbers_run.append(df_number)
        df_number += 1

        #Close figure
        fig.clf()
        plt.close('all')
        
        #Information for big light curve
        biglc_time.append(np.mean(t_mean + firsttime_mean))
        biglc_counts.append(np.mean(cps_bgsub))
        biglc_err.append(np.std(cps_bgsub_err) / np.sqrt(df_reduced.shape[0]))

    #Find the total rank, best rank, and best group
    totalrank = np.sum(c_vals)
    if len(c_vals) !=0:
        bestrank = max(c_vals)
        idx_best = np.where(np.array(c_vals) == bestrank)[0][0]
        best_expt_group = df_numbers_run[idx_best]
    else:
        bestrank = 0
        idx_best = 0
        best_expt_group=0
    print(source, "Total rank: " + str(totalrank), "Best rank: " + str(bestrank), "Best group: " + str(best_expt_group))

    ###Commenting/Interactive Mode###
    if comment:
        if bestrank >= 0:
            bestimagepath = "PNGs/"+source+"-"+band+"qlp"+str(best_expt_group)+".png"
            subprocess.call(['display', bestimagepath])
            comment_value = input("Message code: ")
    else:
        comment_value=""

    ###Get most prevalent period from strongest_periods_list###
    all_periods = [ tup[0] for tup in strongest_periods_list ]
    all_ratios = [ tup[1] for tup in strongest_periods_list ] 
    if len(all_periods) > 1:
        period_to_save = all_periods[np.where(np.asarray(all_ratios) == max(all_ratios))[0][0]]
    elif len(all_periods) == 1:
        period_to_save = all_periods[0]
        period_to_save = round(period_to_save,3)
    else:
        period_to_save = ''

    ###Query Catalogs###
    bigcatalog = pd.read_csv(catalogpath)
    #Get alternate designation if it exists
    if source[0].isdigit():
        sdss_source=source
        idx_sdss = np.where(bigcatalog['SDSSDesignation'] == source)[0]
        gaia_source = bigcatalog['GaiaDesignation'][idx_sdss]
        variability = bigcatalog['variability'][idx_sdss]
        binarity = bigcatalog['binarity'][idx_sdss]
        hasdisk = bigcatalog['hasdisk'][idx_sdss]
        mwdd_type = bigcatalog['mwdd_type'[idx_sdss]
    elif source[0] == "G"
        gaia_source=source
        idx_gaia = np.where(bigcatalog['GaiaDesignation'] == source)[0]
        sdss_source = bigcatalog['SDSSDesignation'][idx_gaia]
        variability = bigcatalog['variability'][idx_gaia]
        binarity = bigcatalog['binarity'][idx_gaia]
        hasdisk = bigcatalog['hasdisk'][idx_gaia]
        mwdd_type = bigcatalog['mwdd_type'][idx_gaia]

    #Query Simbad Catalog
    if not os.path.isfile("Catalog/AllCatalog_Simbad.csv"):
        print("Creating catalog")
        subprocess.run(["WDsearch"])

    #Read in Simbad catalog
    df_simbad = pd.read_csv("Catalog/AllCatalog_Simbad.csv")
    idx_simbad = np.where(df_simbad["SourceName"] == source)[0]
    if len(idx_simbad) > 0:
        simbad_name = df_simbad["SimbadName"][idx_simbad]
        simbad_types = df_simbad["SimbadType2"][idx_simbad]
    else:
        simbad_name = None
        simbad_types = None
    #Read in SDSS info
    df_sdss = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/SDSSCatalog.csv")
    idx_sdss = np.where(df_sdss['SDSS-J'] == source)[0]
    if len(idx_sdss) != 0:
        idx_sdss = idx_sdss[0]
        g_mag = df_sdss['g'][idx_sdss]
        sdss_dtype = df_sdss['dtype'][idx_sdss]
    else:
        g_mag = ""
        sdss_dtype = ""

    #Generate output csv with pandas
    dfoutput = pd.DataFrame()
    dfoutput = dfoutput.append({"SourceName":source, "band":band, "TotalRank":round(totalrank,3), "BestRank":round(bestrank,3), "Comment":comment_value, "ABMag":round(m_ab, 2), "StrongestPeriod":period_to_save, "SimbadName":simbad_name, "gmag":g_mag, "dType":sdss_dtype, "FlaggedRatio":flaggedratio, "KnownVariable":c_known}, ignore_index=True)
    dfoutput.to_csv("Output/"+source+"-"+band+"-output.csv", index=False)


    #####Generate multiplage pdf#####

    ###Page 1###
    #Drop flagged rows from alldata
    alldata_flag_idx = np.where( alldata['flags'] !=0)[0]
    alldata = alldata.drop(index = alldata_flag_idx)
    alldata = alldata.reset_index(drop=True)
    #Make the correction for relative scales
    alldata_tmean = alldata['t_mean']
    alldata_cps_bgsub = alldata['cps_bgsub']
    alldata_mediancps = np.nanmedian(alldata_cps_bgsub)
    alldata_cps_bgsub = ( alldata_cps_bgsub / alldata_mediancps ) - 1.0
    alldata_cps_bgsub_err = alldata['cps_bgsub_err'] / alldata_mediancps

    #Convert to JD
    alldata_jd_tmean = [ gu.calculate_jd(t) for t in alldata_tmean ] 
    biglc_jd_time = [ gu.calculate_jd(t) for t in biglc_time ]
    if other_band_exists:
        alldata_jd_tmean_other = [ gu.calculate_jd(t) for t in alldata_tmean_other ]

    #Try to find ASASSN data
    if os.path.isfile('../ASASSNphot/ap_phot/'+source+'_V.dat'):
        print("ASASSN data exists")
        figall, axall = plt.subplots(2,1, figsize=(16,12))
        #Plot total light curve
        axall[0].errorbar(biglc_jd_time, biglc_counts, yerr=biglc_err, color=bandcolors[band], marker='.', ls='-',  zorder=3, ms=15, label=band)
        axall[0].errorbar(alldata_jd_tmean, alldata_cps_bgsub, yerr=alldata_cps_bgsub_err, color='black', marker='.', zorder=2, ls='', alpha=.125)
        axall[0].set_xlabel('Time [s]')
        axall[0].set_ylabel('Relative Counts per Second')
        #Plot data in other band
        if other_band_exists:
            print("Plotting additional LC data for " + band_other + " band")
            axall[0].errorbar(alldata_jd_tmean_other, alldata_cps_bgsub_other, yerr=alldata_cps_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, alpha=.25, label=band_other)
        axall[0].set_xlabel('Time [s]')
        axall[0].set_ylabel('Relative Counts per Second')
        axall[0].legend()

        #Plot ASASSN data
        ASASSN_output_V = readASASSN('../ASASSNphot/ap_phot/'+source+'_V.dat')
        ASASSN_JD_V = ASASSN_output_V[0]
        ASASSN_mag_V = ASASSN_output_V[1]
        ASASSN_mag_err_V = ASASSN_output_V[2]

        ASASSN_output_g = readASASSN('../ASASSNphot/ap_phot/'+source+'_V.dat')
        ASASSN_JD_g = ASASSN_output_g[0]
        ASASSN_mag_g = ASASSN_output_g[1]
        ASASSN_mag_err_g = ASASSN_output_g[2]

        axall[1].errorbar(ASASSN_JD_V, ASASSN_mag_V, yerr=ASASSN_mag_err_V, color='blue', ls='-', label='V band')
        axall[1].errorbar(ASASSN_JD_g, ASASSN_mag_g, yerr=ASASSN_mag_err_g, color='green', ls='-', label='g band')
        axall[1].set_xlabel('JD')
        axall[1].set_ylabel("V Magnitude")
        axall[1].set_title('ASASSN LC')
        axall[1].legend()
    else:
        print("No ASASSN data")
        figall, axall = plt.subplots(1,1,figsize=(16,12))
        #Plot total light curve
        axall.errorbar(biglc_jd_time, biglc_counts, yerr=biglc_err, color=bandcolors[band], marker='.', ls='-',  zorder=3, ms=15, label=band)
        axall.errorbar(alldata_jd_tmean, alldata_cps_bgsub, yerr=alldata_cps_bgsub_err, color='black', marker='.', zorder=2, ls='', alpha=.125)
        #Plot data in other band
        if other_band_exists:
            print("Plotting additional LC data for " + band_other + " band")
            axall.errorbar(alldata_jd_tmean_other, alldata_cps_bgsub_other, yerr=alldata_cps_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, alpha=.25, label=band_other)
        axall.set_xlabel('Time [s]')
        axall.set_ylabel('Relative Counts per Second')
        axall.legend()

    #If its a known variable, add that to the title
    if c_known == 0:
        figall.suptitle("Combined Light Curve for " + source +"-"+ band+ " \n"+
                "AB magnitude " + str(round(m_ab, 2)) + "\n" +
                "Total rank: " + str(round(totalrank,2)) + " in "+str(len(data))+ " exposure groups \n"+
                "Best rank: " +str(round(bestrank,2))+ " in exposure group " + str(best_expt_group) +"\n"+
                "SDSS Type: " + str(sdss_dtype) + " g mag: " + str(g_mag) + "\n"
                )
    else:
        figall.suptitle("Combined Light Curve for " + source +"-"+ band+ " Known Variable \n"+
                "AB magnitude " + str(round(m_ab, 2)) + "\n" +
                "Total rank: " + str(round(totalrank,2)) + " in "+str(len(data))+ " exposure groups \n"+
                "Best rank: " +str(round(bestrank,2))+ " in exposure group " + str(best_expt_group) +"\n"+
                "SDSS Type: " + str(sdss_dtype) + " g mag: " + str(g_mag) + "\n"
                )
    
    all1saveimagepath = str("PNGs/"+source+"-"+band+"all1"+".png")
    figall.savefig(all1saveimagepath)
    #Clear figure
    figall.clf()
    plt.close('all')

    ###Page 2### Magnitude sigma plot and Source information
    #Get info from sigmamag csv file (from WDsigmamag)
    figall2, axall2 = plt.subplots(2,1,figsize=(16,12))
    df_sigmamag = pd.read_csv(sigmamag_path)
    df_alphas = df_sigmamag['weight']
    rgb_1 = np.zeros((len(df_alphas),4))
    rgb_1[:,3] = df_alphas
    axall2[0].scatter(df_sigmamag['m_ab'], df_sigmamag['sigma_m'], color=rgb_1, zorder=2)

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

    axall2[0].scatter(sourcemags, sourcesigmas, color=rgb_2, zorder=3)
    axall2[0].scatter(arrow_mag, arrow_sigma, color=rgb_arrow, marker="^", zorder = 1)
    axall2[0].set_title("Sigma as a function of AB mag")
    axall2[0].set_xlabel("AB mag")
    axall2[0].set_ylabel("Sigma")
    axall2[0].set_ylim(ymin=0)
    axall2[0].set_ylim(ymax=.3)

    ###Information for text subplot
    axall2[1].set_ylim(ymin=0, ymax=1)
    if c_known == 0:
        formatknown = ''
    else:
        formatknown = 'y'
    information = """
    SDSS source name: {0}
    Gaia source name: {1}
    Other Designations: {2} \n
    Known Variable: {3} \n
    ABMagnitude: {4} \n
    SIMBAD Designation: {5}
    SIMBAD Type list: {6} \n
    MWDD Type: {7} \n
    Known Variability (MWDD): {8} \n
    Known Binarity: {9} \n
    Known Disk: {10} \n
    """.format(sdss_source, gaia_source, band, formatknown, str(round(m_ab,2))simbad_name, simbad_types, mwdd_type, variability, binarity, hasdisk)
    axall2[1].axis('off')
    axall2[1].text(0, .5, information, size=20)

    all2saveimagepath = str("PNGs/"+source+"-"+band+"all2"+".png")
    figall2.savefig(all2saveimagepath)

    #Clear figure
    figall.clf()
    plt.close('all')

    #Generate PDF
    subprocess.run(['PDFcreator', '-s', source, '-b', band])
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
    parser.add_argument("--w_known", help="Weight for if known (subtracted)", default=.75, type=float)
    parser.add_argument("--comment", help="Add comments/interactive mode", default=False, action='store_true')
    args= parser.parse_args()

    main(csvname=args.csvname, fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_ac=args.w_ac, w_mag=args.w_mag, w_known=args.w_known, comment=args.comment)
