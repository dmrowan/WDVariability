#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
from astropy.coordinates import Angle
from astropy import units as u
import collections
import numpy as np
import pandas as pd
#Dom Rowan REU 2018

desc="""
WDutils: Utiliy functions for WD project
"""

#Read in ASASSN tables
def readASASSN(path):
    #Initialize lists
    jd_list = []
    mag_list = []
    mag_err_list = []
    #Open ASASSN file and parse
    with open(path) as f:
        for line in f:
            if line[0].isdigit():
                datlist = line.rstrip().split()
                jd_list.append(datlist[0])
                mag_list.append(datlist[7])
                mag_err_list.append(datlist[8])

    #Iterate through to delete bad rows
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


#Return True if bad flag exists in flagval
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

    badflag_vals = (output_string[0]
            + output_string[4]
            + output_string[7]
            + output_string[8])

    for char in badflag_vals:
        if char == '1':
            return True
            break
        else:
            continue
    return False

#Return idx of catalog match (name based query)
def catalog_match(source, bigcatalog):
    #Based of nhyphens and leading characters of source
    nhyphens = len(np.where(np.array(list(source)) == '-')[0])
    if ( (source[0:4] == 'Gaia') or
            (source[0:2] in ['GJ', 'CL', 'V*', 'PN']) or
            (source[0:3] in ['Ton', 'CL*', 'Cl*', 'RRS', 'MAS']) or
            (source[0:5] in ['LAWDS']) or
            ('GMS97' in source) or
            ('FOCAP' in source) or
            ('KAB' in source) or
            ('NCA' in source)):
            bigcatalog_idx = np.where(bigcatalog['MainID']
                                      == source.replace('-', ' '))[0]
    elif source[0:5] == 'ATLAS':
        bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
    elif source[0:2] == 'LP':
        if nhyphens == 2:
            bigcatalog_idx = np.where(bigcatalog['MainID']
                                      == source.replace('-', ' ', 1))[0]
        elif nhyphens == 3:
            source_r = source.replace('-', ' ', 1)[::-1]
            source_r = source_r.replace('-', ' ',1)[::-1]
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source_r)[0]
        else:
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
    elif source[0:3] == '2QZ':
        bigcatalog_idx = np.where(bigcatalog['MainID']
                                  == source.replace('-', ' ', 1))[0]
    elif source[0:2] == 'BD':
        if nhyphens == 2:
            bigcatalog_idx = np.where(bigcatalog['MainID']
                    == source[::-1].replace('-', ' ', 1)[::-1])[0]
        else:
            bigcatalog_idx = np.where(
                    bigcatalog['MainID'] == source.replace('-', ' '))[0]
    else:   #SDSS sources go in here
        if nhyphens == 1:
            bigcatalog_idx = np.where(
                    bigcatalog['MainID'] == source.replace('-', ' ' ))[0]
        else:
            bigcatalog_idx = np.where(
                    bigcatalog['MainID'] ==
                    source.replace('-', ' ',nhyphens-1))[0]

    return(bigcatalog_idx)

#Basic data filtering for gPhoton output
def df_reduce(df):
    idx_reduce = np.where( (df['cps_bgsub'] > 10e10)
        | (df['cps_bgsub_err'] > 10e10)
        | (df['counts'] < 1)
        | (df['counts'] > 100000)
        | (np.isnan(df['cps_bgsub']))
        | (df['flux_bgsub'] < 0)
        | (df['cps_bgsub'] < -10000) )[0]

    if len(idx_reduce) != 0:
        df = df.drop(index=idx_reduce)
        df = df.reset_index(drop=True)

    return df 

#Additional reduction of flagged points, expt, sigmaclip
def df_fullreduce(df):
    df = df_reduce(df)

    idx_flagged_bool = [ badflag_bool(x) for x in df['flags'] ]
    idx_flagged = np.where(np.array(idx_flagged_bool) == True)[0]
    idx_expt = np.where(df['exptime'] < 10)[0]

    stdev = np.std(df['flux_bgsub'])
    idx_sigmaclip = []
    if len(df['flux_bgsub']) != 0:
        if not df['flux_bgsub'].isnull().all():
            idx_sigmaclip = np.where(
                    abs(df['flux_bgsub'] - np.nanmean(df['flux_bgsub']))
                        > 5*stdev)[0]

    idx_drop = np.unique(np.concatenate([idx_flagged, 
                                         idx_expt,
                                         idx_sigmaclip]))
    df = df.drop(index=idx_drop)
    df = df.reset_index(drop=True)
    return df

#If the first and last points aren't within 3 sigma, remove
def df_firstlast(df):
    stdev = np.std(df['flux_bgsub'])
    if (df['flux_bgsub'][df.index[0]]
            - np.nanmean(df['flux_bgsub'])) > 3*stdev:
        df = df.drop(index=df.index[0])
        df = df.reset_index(drop=True)

    if (df['flux_bgsub'][df.index[-1]]
            - np.nanmean(df['flux_bgsub'])) > 3*stdev:
        df = df.drop(index=df.index[-1])
        df = df.reset_index(drop=True)

    return df


#Make correction for t_mean by averaging t0 and t1
def tmean_correction(df):
    idx_tmean_fix = np.where( (df['t_mean'] < 1)
                            | (df['t_mean'] > df['t1'])
                            | (np.isnan(df['t_mean'])))[0]
    
    for idx in idx_tmean_fix:
        t0 = df['t0'][idx]
        t1 = df['t1'][idx]
        mean = (t1 + t0) / 2.0
        df['t_mean'][idx] = mean

    return df

#Split df into visits, defined by tbreak
def dfsplit(df, tbreak):
    #tbreak in seconds
    breaks = []
    for i in range(len(df['t0'])):
        if i != 0:
            if (df['t0'][i] - df['t0'][i-1]) >= tbreak:
                breaks.append(i)

    data = np.split(df, breaks)
    return data


#Basic plot params (so I don't drive myself insane)
def plotparams(ax):
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=15)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

#Find indicies of flaggedpoints, sigmaclip, exposure gap
def ColoredPoints(df):
    stdev = np.std(df['flux_bgsub'])
    bluepoints = np.where(
            abs(df['flux_bgsub'] - np.nanmean(df['flux_bgsub']))
            > 5*stdev )[0]
    flag_bool_vals = [ badflag_bool(x) for x in df['flags'] ]
    redpoints1 = np.where(np.array(flag_bool_vals) == True)[0]
    redpoints2 = np.where(df['exptime'] < 10)[0]
    redpoints = np.unique(np.concatenate([redpoints1, redpoints2]))
    redpoints = redpoints + df.index[0]
    bluepoints = bluepoints + df.index[0]

    OutputTup = collections.namedtuple('OutputTup', ['redpoints',
                                                     'bluepoints'])
    tup = OutputTup(redpoints, bluepoints)
    return tup

#Output arrays for percent flux, err flux 
def relativescales(df):
    flux_bgsub = df['flux_bgsub']
    median = np.median(flux_bgsub)
    flux_bgsub = ((flux_bgsub/median)-1.0)*100
    flux_err = (
            df['flux_bgsub_err'] / median)*100
    t_mean = df['t_mean']

    OutputTup = collections.namedtuple('OutputTup', ['t_mean', 
                                                     'flux',
                                                     'err'])
    tup = OutputTup(t_mean, flux_bgsub, flux_err)
    return tup

def raconvert(h, m, s):
    ra = Angle((h,m,s), unit='hourangle')
    return ra.degree

def decconvert(d, m, s):
    dec = Angle((d,m,s), u.deg)
    return dec.degree

def tohms(ra, dec, fancy=False):
    ra1 = Angle(ra, u.deg)
    dec1 = Angle(dec, u.deg)
    #print(ra1.hms)
    #print(dec1.dms)
    if not fancy:
        return ra1.hms, dec1.dms
    else:
        r_hours = str(int(ra1.hms.h))
        if ra1.hms.h < 10:
            r_hours = '0'+r_hours

        r_minutes = str(int(ra1.hms.m))
        if ra1.hms.m < 10:
            r_minutes = '0'+r_minutes

        if ra1.hms.s < 10:
            r_seconds = '0' +str(round(ra1.hms.s,2))
        else:
            r_seconds = str(round(ra1.hms.s,2))

        r_string = "{0}{1}{2}".format(r_hours, r_minutes, r_seconds)
        #Negative dec case
        if dec1.dms.d < 0:
            d_degrees = str(int(dec1.dms.d))
            if abs(dec1.dms.d) < 10:
                d_degrees = '0'+d_degrees

            d_minutes = str(int(-1*dec1.dms.m))
            if abs(dec1.dms.m) < 10:
                d_minutes = '0'+d_minutes

            if abs(dec1.dms.s) < 10:
                d_seconds = '0' + str(round(-1*dec1.dms.s, 2))
            else:
                d_seconds = str(round(-1*dec1.dms.s, 2))

            d_string = "{0}{1}{2}".format(d_degrees, d_minutes, d_seconds)
        else:
            d_degrees = str(int(dec1.dms.d))
            if abs(dec1.dms.d) < 10:
                d_degrees = '0'+d_degrees

            d_minutes = str(int(dec1.dms.m))
            if abs(dec1.dms.m) < 10:
                d_minutes = '0'+d_minutes

            if abs(dec1.dms.s) < 10:
                d_seconds = '0' + str(round(dec1.dms.s, 2))
            else:
                d_seconds = str(round(dec1.dms.s, 2))
            d_string = "+{0}{1}{2}".format(d_degrees, d_minutes, d_seconds)
        return r_string+d_string


            



