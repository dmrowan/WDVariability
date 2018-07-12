#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from astropy.stats import LombScargle
import heapq
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
import subprocess
from gPhoton import gphoton_utils
#Dom Rowan REU 2018


desc="""
WDranker_2.py: produces the ranked value c for a single source dependent on exposure, periodicity, autocorrelation, and other statistical measures
"""

#function to read in asassn data - even weird tables 
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
def main(csvname, fap, prange, w_pgram, w_expt, w_WS, w_mag, w_known, w_flag, w_magfit, comment):
    ###Path assertions###
    catalogpath = "/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    sigmamag_path_NUV = "Catalog/SigmaMag_NUV.csv"
    sigmamag_percentile_path_NUV = "Catalog/magpercentiles_NUV.csv"
    sigmamag_path_FUV = "Catalog/SigmaMag_FUV.csv"
    sigmamag_percentile_path_FUV = "Catalog/magpercentiles_FUV.csv"
    assert(os.path.isfile(csvname))
    assert(os.path.isfile(catalogpath))
    assert(os.path.isfile(sigmamag_path_NUV))
    assert(os.path.isfile(sigmamag_percentile_path_NUV))
    assert(os.path.isfile(sigmamag_path_FUV))
    assert(os.path.isfile(sigmamag_percentile_path_FUV))
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
    print(source, band)
    bandcolors = {'NUV':'red', 'FUV':'blue'}
    alldata = pd.read_csv(csvpath)
    ###Alldata table corrections###
    #Drop rows with > 10e10 in cps, cps_err, cps < .5
    idx_high_cps = np.where( (alldata['cps_bgsub'] > 10e10) | (alldata['cps_bgsub_err'] > 10e10) | (alldata['counts'] < 1) | (alldata['counts'] > 100000)  | (alldata['cps_bgsub'] < -10000) )[0]
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

    ###Apparent Magnitude### - could also be done using conversion from flux 
    m_ab = np.nanmedian(alldata['mag_bgsub'])
    sigma_mag_all = np.nanstd( (alldata['mag_bgsub_err_1'] + alldata['mag_bgsub_err_2'])/2.0 )
    #Calculate c_mag based on ranges:
    if m_ab > 13 and m_ab < 25:
        c_mag = m_ab**(-1) * 10
    else:
        c_mag = 0

    magdic = {"mag":[m_ab], "sigma":[sigma_mag_all], "weight":[1]}

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
    if m_ab < 20.75:
        sigmamag_idx = np.where(abs(m_ab-magbins) == min(abs(m_ab-magbins)))[0]
        sigmafit_val = float(lowerbound[sigmamag_idx])
        sigmafit_val_upper = float(upperbound[sigmamag_idx])
        if sigma_mag_all > sigmafit_val and sigma_mag_all < sigmafit_val_upper:
            c_magfit = sigma_mag_all / sigmafit_val
        elif sigma_mag_all >= sigmafit_val_upper:
            c_magfit = sigmafit_val_upper / sigmafit_val
        else:
            c_magfit = 0
    else:
        c_magfit = 0


    ###See if we have any data in the other band###
    csvpath_other = list(csvpath)
    csvpath_other[-7] = band_other[0]
    csvpath_other = "".join(csvpath_other)
    #Look for file in GALEXphot/LCs
    csvpath_other = '/home/dmrowan/WhiteDwarfs/GALEXphot/LCs/'+csvpath_other
    if os.path.isfile(csvpath_other):
        other_band_exists = True
        alldata_other = pd.read_csv(csvpath_other)
    else:
        other_band_exists = False

    if other_band_exists:
        print("Generating additional LC data for " + band_other + " band")
        alldata_other = pd.read_csv(csvpath_other)
        #Drop bad rows, flagged rows
        idx_high_cps_other = np.where( (alldata_other['cps_bgsub'] > 10e10) | (alldata_other['cps_bgsub_err'] > 10e10) | (alldata_other['counts'] < 1) | (alldata_other['counts'] > 100000)  | (alldata_other['cps_bgsub'] < -10000) )[0]
        #Not interested in looking at red/blue points for other band
            #drop flagged, expt < 10
        idx_other_flagged_bool = [ badflag_bool(x) for x in alldata_other['flags'] ]
        idx_other_flagged = np.where(np.array(idx_other_flagged_bool) == True)[0]
        idx_other_expt = np.where(alldata_other['exptime'] < 10)[0]
        
        idx_other_todrop = np.unique(np.concatenate([idx_high_cps_other, idx_other_flagged, idx_other_expt]))
        alldata_other = alldata_other.drop(index=idx_other_todrop)
        alldata_other = alldata_other.reset_index(drop=True)

        stdev_other = np.std(alldata_other['flux_bgsub'])
        idx_fivesigma_other = np.where( (alldata_other['flux_bgsub'] - np.nanmean(alldata_other['flux_bgsub'])) > 5*stdev_other )[0]
        alldata_other = alldata_other.drop(index=idx_fivesigma_other)
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
        alldata_flux_bgsub_other = alldata_other['flux_bgsub']
        alldata_medianflux_other = np.median(alldata_flux_bgsub_other)
        alldata_flux_bgsub_other = (( alldata_flux_bgsub_other / alldata_medianflux_other ) - 1.0) * 1000
        alldata_flux_bgsub_err_other = (alldata_other['flux_bgsub_err'] / alldata_medianflux_other) * 1000

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
        #with open("../brokensources.txt", 'a') as f:
        #    f.write(source + "\n")
        return
    else:
        bigcatalog_idx = bigcatalog_idx[0]

    spectype = bigcatalog['spectype'][bigcatalog_idx]
    variability = bigcatalog['variability'][bigcatalog_idx]
    binarity = bigcatalog['binarity'][bigcatalog_idx]
    hasdisk = bigcatalog['hasdisk'][bigcatalog_idx]
    simbad_name = bigcatalog['SimbadName'][bigcatalog_idx]
    simbad_types = bigcatalog['SimbadTypes'][bigcatalog_idx]
    gmag = bigcatalog['sdss_g'][bigcatalog_idx]

    #Known information changes rank:
    if str(binarity) != 'nan' or str(variability) != 'nan' or str(hasdisk) != 'nan': 
        c_known = 1
    else:
        c_known = 0

    if str(spectype) != 'nan':
        if any(spectype) == 'Z':
            c_known += 1


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
    c_ws_vals = []
    df_numbers_run = []
    biglc_time = []
    biglc_counts = []
    biglc_err = []
    strongest_periods_list = []
    ditherperiod_exists = False

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

        #Find indicies of data above 5 sigma of mean , flagged points, and points with
        # less than 10 seconds of exposure time
        stdev = np.std(df['flux_bgsub'])
        bluepoints = np.where( (df['flux_bgsub'] - np.nanmean(df['flux_bgsub'])) > 5*stdev )[0]
        flag_bool_vals = [ badflag_bool(x) for x in df['flags'] ]
        redpoints1 = np.where(np.array(flag_bool_vals) == True)[0]
        redpoints2 = np.where(df['exptime'] < 10)[0]
        redpoints = np.unique(np.concatenate([redpoints1, redpoints2]))
        redpoints = redpoints + df.index[0]
        bluepoints = bluepoints + df.index[0]

        droppoints = np.unique(np.concatenate([redpoints, bluepoints]))
        flagged_ratio = len(droppoints) / df.shape[0]
        c_flagged = -1*flagged_ratio
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
        if (df_reduced['flux_bgsub'][df_reduced.index[0]] - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[0])
            df_reduced = df_reduced.reset_index(drop=True)

        #If last point is not within 3 sigma, remove
        if (df_reduced['flux_bgsub'][df_reduced.index[-1]] - np.nanmean(df['flux_bgsub'])) > 3*stdev:
            df_reduced = df_reduced.drop(index=df_reduced.index[-1])
            df_reduced = df_reduced.reset_index(drop=True)

        ###Grab magnitude information###
        df_m_ab = np.nanmean(df_reduced['mag_bgsub'])
        df_sigma_mag = np.nanstd( (df_reduced['mag_bgsub_err_1'] + df_reduced['mag_bgsub_err_2'])/2.0 )

        magdic["mag"].append(df_m_ab)
        magdic["sigma"].append(df_sigma_mag)
        magdic["weight"].append(.25)
        
        #Get the flux_bgsub, error and time columns and make correction for relative scales
        flux_bgsub = df_reduced['flux_bgsub']
        flux_bgsub_median = np.median(flux_bgsub)
        flux_bgsub = (( flux_bgsub / flux_bgsub_median ) - 1.0) * 1000
        flux_bgsub_err = (df_reduced['flux_bgsub_err'] / flux_bgsub_median) * 1000
        t_mean = df_reduced['t_mean']
        
        #If we have data in the other band, find points corresponding to this exposure group
        #First get the indicies corresponding to this group in the other band
        if other_band_exists:
            idx_exposuregroup_other = np.where( (alldata_tmean_other > firsttime) & (alldata_tmean_other < lasttime))[0]
            t_mean_other = np.array(alldata_tmean_other[idx_exposuregroup_other] - firsttime_mean)
            flux_bgsub_other = np.array(alldata_flux_bgsub_other[idx_exposuregroup_other])
            flux_bgsub_err_other = np.array(alldata_flux_bgsub_err_other[idx_exposuregroup_other])
            idx_flux_nan_other = np.where(np.isnan(flux_bgsub_other))[0]
            if len(idx_flux_nan_other) != 0:
                t_mean_other = np.delete(t_mean_other, idx_flux_nan_other)
                flux_bgsub_other = np.delete(flux_bgsub_other, idx_flux_nan_other)
                flux_bgsub_err_other = np.delete(flux_bgsub_err_other, idx_flux_nan_other)

        if df_number == 6:
            print(flux_bgsub_other)
        #Make the correction for relative scales for redpoints and bluepoints
        if len(redpoints) != 0:
            flux_bgsub_red = df['flux_bgsub'][redpoints]
            flux_bgsub_red = ((flux_bgsub_red / flux_bgsub_median) - 1.0) * 1000
            flux_bgsub_err_red = (df['flux_bgsub_err'][redpoints] / flux_bgsub_median) * 1000
            t_mean_red = df['t_mean'][redpoints]
        if len(bluepoints) != 0:
            flux_bgsub_blue = df['flux_bgsub'][bluepoints]
            flux_bgsub_blue = ((flux_bgsub_blue / flux_bgsub_median) - 1.0) * 1000
            flux_bgsub_err_blue = (df['flux_bgsub_err'][bluepoints] / flux_bgsub_median) * 1000
            t_mean_blue = df['t_mean'][bluepoints]

        ###Additional metric### - Ratio of std / sigma
        #flux_std = np.std(df_reduced['flux_bgsub'])
        #median_error = np.median(df_reduced['flux_bgsub_err'])
        #c_uncertainty = flux_std / median_error


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
                        ditherperiod_exists = True
                #If hits is still 0, the peak isnt in any of the bad ranges
                if hits == 0:
                    sspeaks.append( (f, a, fapval, ratio) ) 

        #This is a crude way to ensure we don't get any dither harmonics
        if ditherperiod_exists:
            sspeaks = []
        #Grab the info to show the strongest peak for the source
        if len(sspeaks) != 0:
            sspeaks_amp = [ peak[1] for peak in sspeaks ] 
            sspeaks_freq = [ peak[0] for peak in sspeaks ]
            sspeaks_fap = [ peak[2] for peak in sspeaks ] 
            sspeaks_ratio = [ peak[3] for peak in sspeaks ]
            strongest_freq = (sspeaks_freq[np.where(np.asarray(sspeaks_amp) == max(sspeaks_amp))[0][0]])
            strongest_period_ratio = (sspeaks_ratio[np.where(np.asarray(sspeaks_amp)==max(sspeaks_amp))[0][0]])
            strongest_period_fap = (sspeaks_fap[np.where(np.asarray(sspeaks_amp)==max(sspeaks_amp))[0][0]])
            strongest_period = 1.0 / strongest_freq
            strongest_periods_list.append((strongest_period[0], strongest_period_ratio))


        c_periodogram = 0
        for peak in sspeaks:
            if (peak[0] < (1/ (exposure))) or (peak[0] > (1/25)):
                    c_periodogram += peak[3] * .125
            else:
                c_periodogram += peak[3]

        #Welch Stetson Variability Metric
        #Iterate through each time to find if there is a matching time in other band
        ws_times = [] #Array of tuples t_mean, t_mean_other matching (if exists)
        ws_flux = [] #Array of flux_bgsub, flux_bgsub_other matching (if exists)
        ws_flux_err = [] #Array of tuples flux_bgsub_err, flux_bgsub_err_other matching (if exists)
        ii_previous = 0 #Index to reduce number of iterations 
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

        assert(len(ws_times) == len(ws_flux) == len(ws_flux_err) == len(t_mean))
        ws_sum = 0
        for i in range(len(ws_flux)):
            fluxtup = ws_flux[i]
            errtup = ws_flux_err[i]
            deltaband = (fluxtup[0] - np.nanmean(flux_bgsub)) / errtup[0]
            if fluxtup[1] is None:
                deltaother = 1
            else:
                assert(other_band_exists)
                if len(flux_bgsub_other) == 0:
                    deltaother = 1
                else:
                    deltaother = (fluxtup[1] - np.nanmean(flux_bgsub_other)) / errtup[1]

            ws_sum += deltaband*deltaother
        
        c_ws = np.sqrt(1/(len(ws_times)*(len(ws_times)-1))) * ws_sum

        #Crude way of making dither not count
        if ditherperiod_exists:
            c_ws = 0

        c_ws_vals.append(c_ws)

        ###Autocorrelation results###
        autocorr_result = np.correlate(flux_bgsub, flux_bgsub, mode='full')
        autocorr_result = autocorr_result[int(autocorr_result.size/2):]

        if any(np.isinf(x) for x in autocorr_result):
            print("Infinite Values in Autocorr for group "+str(df_number))
            #Reassign Autocorr_result to be a bunch of zeros
            numberofzeros = len(autocorr_result)
            autocorr_result = np.zeros(numberofzeros)

        #####GENERATE RATING#####
        C = (w_pgram * c_periodogram) + (w_expt * c_exposure) + (w_mag * c_mag) + (w_flag * c_flagged) + (w_known * c_known) + (w_magfit * c_magfit) + (w_WS * c_ws)
        if C < 0:
            C = 0
        print("Exposure group "+str(df_number)+" ranking: "+ str(C))
        c_vals.append(C)


        ###Generate plot/subplot information###
        fig = plt.figure(df_number, figsize=(16,12))
        gs.GridSpec(4,4)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Exposure group {0} with {1}s \nRanking: {2} {3} significant peaks".format(str(df_number), str(exposure), str(C), str(len(sspeaks))))

        #Subplot for LC
        plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=2)
        #Convert to JD here as well
        jd_t_mean = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean ]
        plt.errorbar(jd_t_mean, flux_bgsub, yerr=flux_bgsub_err, color=bandcolors[band], marker='.', ls='', zorder=4, label=band)
        #plt.errorbar(jd_t_mean, flux_bgsub,  color=bandcolors[band], marker='.', ls='-', zorder=4, label=band)
        plt.axhline(alpha=.3, ls='dotted', color=bandcolors[band])
        if len(redpoints) != 0: #points aren't even red now...
            jd_t_mean_red = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_red ]
            plt.errorbar(jd_t_mean_red, flux_bgsub_red, yerr=flux_bgsub_err_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
            #plt.errorbar(jd_t_mean_red, flux_bgsub_red, color='#808080', marker='.', ls='', zorder=2, alpha=.5, label='Flagged')
        if len(bluepoints) != 0: #these points aren't blue either...
            jd_t_mean_blue = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_blue ]
            plt.errorbar(jd_t_mean_blue, flux_bgsub_blue, yerr=flux_bgsub_err_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
            #plt.errorbar(jd_t_mean_blue, flux_bgsub_blue, color='green', marker='.', ls='', zorder=3, alpha=.5, label='SigmaClip')
        if other_band_exists:
            #introduce offset here
            jd_t_mean_other = [ gphoton_utils.calculate_jd(t+firsttime_mean) for t in t_mean_other ]
            plt.errorbar(jd_t_mean_other, flux_bgsub_other+2*max(flux_bgsub), yerr=flux_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, label=band_other, alpha=.25)
            plt.axhline(y=2*max(flux_bgsub), alpha=.15, ls='dotted', color=bandcolors[band_other])

        plt.title(band+' light curve')
        plt.xlabel('Time JD')
        plt.ylabel('Flux mmi')
        plt.legend(loc=1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        #Subplot for autocorr
        plt.subplot2grid((4,4), (2,2), colspan=1, rowspan=2)
        plt.plot(autocorr_result, 'b-', label='data')
        #ax[1][0].plot(ac_x, fitfunc(ac_x, *popt), 'g-', label='fit')
        #ax[1][0].plot(residuals, 'r--', alpha=.25, label='residuals')
        #ax[1][0].plot(ac_x, ac_x*params[0]+params[1], 'r--', alpha=.5, label='linear fit')
        plt.title('Autocorrelation')
        plt.xlabel('Delay')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #ax[1][0].legend()

        #Subplot for periodogram
        plt.subplot2grid((4,4), (2,0), colspan=2, rowspan=2)
        plt.plot(freq, amp, 'g-', label='Data')
        plt.plot(freq_detrad, amp_detrad, 'r-', label="Detrad", alpha=.25)
        plt.plot(freq_expt, amp_expt, 'b-', label="Exposure", alpha=.25)
        plt.title(band+' Periodogram')
        plt.xlabel('Freq [Hz]')
        plt.ylabel('Amplitude')
        plt.xlim(0, np.max(freq))
        plt.ylim(0, np.max(amp)*2)
        if any(np.isnan(x) for x in top5amp_detrad):
            print("No detrad peaks for exposure group " + str(df_number))
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

        #Subplot for png image
        plt.subplot2grid((4,4), (2,3), colspan=1, rowspan=2)
        pngfile = "/home/dmrowan/WhiteDwarfs/GALEXphot/pngs/"+source+".png"
        img1 = mpimg.imread(pngfile)
        plt.imshow(img1)
        #Turn of axes 
        #ax[1][1].axis('off')
        plt.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        saveimagepath = str("PDFs/"+source+"-"+band+"qlp"+str(df_number)+".pdf")
        fig.savefig(saveimagepath)
        df_numbers_run.append(df_number)
        df_number += 1

        #Close figure
        fig.clf()
        plt.close('all')
        
        #Information for big light curve
        biglc_time.append(np.nanmean(t_mean + firsttime_mean))
        biglc_counts.append(np.nanmean(flux_bgsub))
        biglc_err.append(np.std(flux_bgsub_err) / np.sqrt(df_reduced.shape[0]))



    ###Find the total rank, best rank, and best group###
    totalrank = np.sum(c_vals)
    if len(c_vals) !=0:
        bestrank = max(c_vals)
        idx_best = np.where(np.array(c_vals) == bestrank)[0][0]
        best_expt_group = df_numbers_run[idx_best]
        c_ws_best = c_ws_vals[idx_best]
    else:
        bestrank = 0
        idx_best = 0
        best_expt_group=0
        c_ws_best = 0
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

    #Generate output csv with pandas
    outputdic = {
            "SourceName":[source], 
            "Band":[band], 
            "TotalRank":[round(totalrank, 3)], 
            "BestRank":[round(bestrank, 3)], 
            "Comment":[comment_value], 
            "ABmag":[round(m_ab, 2)], 
            "MagUncertainty":[round(c_mag,3)],
            "StrongestPeriod":[period_to_save], 
            "WS metric":[c_ws_best],
            "SimbadName":[simbad_name],
            "SimbadTypes":[simbad_types],
            "FlaggedRatio":[allflaggedratio],
            "Spectype":[spectype],
            "KnownVariable":[variability], 
            "Binarity":[binarity],
            "Hasdisk":[hasdisk],
            }
    dfoutput = pd.DataFrame(outputdic)
    dfoutput.to_csv("Output/"+source+"-"+band+"-output.csv", index=False)


    #####Generate multiplage pdf#####

    ###Page 1###
    #Drop flagged rows from alldata
    alldata_flag_bool_vals = [ badflag_bool(x) for x in alldata['flags'] ]
    alldata_flag_idx = np.where(np.array(alldata_flag_bool_vals) == True)[0]
    alldata = alldata.drop(index = alldata_flag_idx)
    alldata = alldata.reset_index(drop=True)
    #Make the correction for relative scales
    alldata_tmean = alldata['t_mean']
    alldata_flux_bgsub = alldata['flux_bgsub']
    alldata_medianflux = np.nanmedian(alldata_flux_bgsub)
    alldata_flux_bgsub = ( alldata_flux_bgsub / alldata_medianflux ) - 1.0
    alldata_flux_bgsub_err = alldata['flux_bgsub_err'] / alldata_medianflux

    #Convert to JD
    alldata_jd_tmean = [ gphoton_utils.calculate_jd(t) for t in alldata_tmean ] 
    biglc_jd_time = [ gphoton_utils.calculate_jd(t) for t in biglc_time ]
    if other_band_exists:
        alldata_jd_tmean_other = [ gphoton_utils.calculate_jd(t) for t in alldata_tmean_other ]

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
        plt.errorbar(biglc_jd_time, biglc_counts, yerr=biglc_err, color=bandcolors[band], marker='.', ls='-',  zorder=3, ms=15, label=band)
        plt.errorbar(alldata_jd_tmean, alldata_flux_bgsub, yerr=alldata_flux_bgsub_err, color='black', marker='.', zorder=2, ls='', alpha=.125)
        plt.xlabel('Time [s]')
        plt.ylabel('Relative Counts per Second')
        #Plot data in other band
        if other_band_exists:
            print("Plotting additional LC data for " + band_other + " band")
            plt.errorbar(alldata_jd_tmean_other, alldata_flux_bgsub_other, yerr=alldata_flux_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, alpha=.25, label=band_other)
        plt.xlabel('Time [s]')
        plt.ylabel('Flux MMI')
        plt.legend()

        #Plot ASASSN data
        plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
        ASASSN_output_V = readASASSN('../ASASSNphot_2/'+asassn_name+'_V.dat')
        ASASSN_JD_V = ASASSN_output_V[0]
        ASASSN_mag_V = ASASSN_output_V[1]
        ASASSN_mag_err_V = ASASSN_output_V[2]

        ASASSN_output_g = readASASSN('../ASASSNphot_2/'+asassn_name+'_g.dat')
        ASASSN_JD_g = ASASSN_output_g[0]
        ASASSN_mag_g = ASASSN_output_g[1]
        ASASSN_mag_err_g = ASASSN_output_g[2]

        plt.errorbar(ASASSN_JD_V, ASASSN_mag_V, yerr=ASASSN_mag_err_V, color='blue', ls='-', label='V band', ecolor='gray')
        plt.errorbar(ASASSN_JD_g, ASASSN_mag_g, yerr=ASASSN_mag_err_g, color='green', ls='-', label='g band', ecolor='gray')
        try:
            maxmag_g = max(ASASSN_mag_g)
        except:
            maxmag_g = 20
        try:
            minmag_g = min(ASASSN_mag_g)
        except:
            minmag_g = 10
        minmag_V = min(ASASSN_mag_V)
        maxmag_V = max(ASASSN_mag_V)
        maxmag = max(maxmag_V, maxmag_g)
        minmag = min(minmag_V, minmag_g)
        plt.ylim(maxmag, minmag)
        plt.xlabel('JD')
        plt.ylabel("V Magnitude")
        plt.title('ASASSN LC')
        plt.legend()

        plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
        if len(ASASSN_JD_V) > 5:
            lsV = LombScargle(ASASSN_JD_V, ASASSN_mag_V, dy=ASASSN_mag_err_V)
            freqV, ampV = lsV.autopower(nyquist_factor=1)
            plt.plot(freqV, ampV, color='blue', label='V mag', zorder=2)
            plt.xlim(xmax=(1/30))
            plt.axhline(y=lsV.false_alarm_level(.1), color='blue', alpha=.5, ls='-')
        if len(ASASSN_JD_g) > 5:
            lsg = LombScargle(ASASSN_JD_g, ASASSN_mag_g, dy=ASASSN_mag_err_g)
            freqg, ampg = lsg.autopower(nyquist_factor=1)
            plt.plot(freqg, ampg, color='green', label='g mag', zorder=1)
            plt.xlim(xmax=(1/30))
            plt.axhline(y=lsg.false_alarm_level(.1), color='green', alpha=.5, ls='-')
    
        #Print frequencies
        if False:
            idx_asassn_max_v = np.where(np.array(ampV)==max(ampV))[0]
            print("Frequency V band: ", freqV[idx_asassn_max_v])
            idx_asassn_max_g = np.where(np.array(ampg)==max(ampg))[0]
            print("Frequency g band: ", freqg[idx_asassn_max_g])

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Periodogram for ASASSN Data')
        plt.legend(loc=1)

        
    else:
        print("No ASASSN data")
        figall, axall = plt.subplots(1,1,figsize=(16,12))
        figall.tight_layout(rect=[0, 0.03, 1, 0.95])
        #Plot total light curve
        axall.errorbar(biglc_jd_time, biglc_counts, yerr=biglc_err, color=bandcolors[band], marker='.', ls='-',  zorder=3, ms=15, label=band)
        axall.errorbar(alldata_jd_tmean, alldata_flux_bgsub, yerr=alldata_flux_bgsub_err, color='black', marker='.', zorder=2, ls='', alpha=.125)
        #Plot data in other band
        if other_band_exists:
            print("Plotting additional LC data for " + band_other + " band")
            axall.errorbar(alldata_jd_tmean_other, alldata_flux_bgsub_other, yerr=alldata_flux_bgsub_err_other, color=bandcolors[band_other], marker='.', ls='', zorder=1, alpha=.25, label=band_other)
        axall.set_xlabel('Time [s]')
        axall.set_ylabel('Flux MMI')
        axall.legend()

    #Supertitle
    figall.suptitle("Combined Light Curve for {0} in {1} \nBest rank {2} in exposure group {3} \nTotal rank {4} in {5} exposure groups".format(source, band, str(round(bestrank,2)), str(best_expt_group), str(round(totalrank, 2)), str(len(data)))
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
    axall2[0].scatter(arrow_mag, arrow_sigma, color=rgb_arrow, marker="^", zorder = 3)
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
    """.format(source, band, str(round(m_ab,4)), str(round(gmag, 4)), spectype, simbad_name, simbad_types, variability, binarity, hasdisk, period_to_save)
    #axall2[1].text(.2, .4, information, size=15, horizontalalignment='left', verticalalignment='center')
    axall2[1].text(.2, 1, information1, size=15, horizontalalignment='left', verticalalignment='top')
    axall2[1].text(.7, 1, information2, size=15, horizontalalignment='right', verticalalignment='top')
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
    parser.add_argument("--csvname", help= "Input full csv file", required=True, type=str)
    parser.add_argument("--fap", help = "False alarm probability theshold for periodogram", default=.05, type=float)
    parser.add_argument("--prange", help = "Frequency range for identifying regions in periodogram due to expt and detrad", default=.0005, type=float)
    parser.add_argument("--w_pgram", help = "Weight for periodogram", default = 1, type=float)
    parser.add_argument("--w_expt", help= "Weight for exposure time", default = .20, type=float)
    parser.add_argument("--w_WS", help="Weight for Welch Stetson variability metric", default = .25, type=float)
    parser.add_argument("--w_mag", help= "Weight for magnitude", default=.5, type=float)
    parser.add_argument("--w_known", help="Weight for if known binarity, variability, disk, Z spec type", default=.5, type=float)
    parser.add_argument("--w_flag", help="Weight for flagged ratio", default=-.5, type=float)
    parser.add_argument("--w_magfit", help="Weight for magfit ratio", default=.30, type=float)
    parser.add_argument("--comment", help="Add comments/interactive mode", default=False, action='store_true')
    args= parser.parse_args()

    main(csvname=args.csvname, fap=args.fap, prange=args.prange, w_pgram=args.w_pgram, w_expt=args.w_expt, w_WS=args.w_WS, w_mag=args.w_mag, w_known=args.w_known, w_flag=args.w_flag, w_magfit=args.w_magfit, comment=args.comment)
