from __future__ import print_function, division, absolute_import
# Third party imports
from astropy.coordinates import Angle
from astropy.stats import LombScargle
from astropy import units as u
import collections
import numpy as np
import pandas as pd
#Dom Rowan REU 2018

desc="""
WDutils: Utility functions for WD project
"""

#-----------------------------------------------------------------------------
def readASASSN(path):
    '''
    Read in an ASASSN dat file
    
    :param path: file path to ASASSN dat file

    :type path: str

    :returns: list of time, list of magnitudes, list of err
    '''
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def badflag_bool(x):
    '''
    Determine if a bad flag exists in flag value

    :param x: flag value from gPhoton 'flag' column

    :type x: int

    :returns: Boolean -- existence of bad flag
    '''
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def catalog_match(source, bigcatalog):
    '''
    Return idx of catalog match using name based query

    :param source: name of object to query

    :type source: str

    :param bigcatalog: catalog to query

    :type bigcatalog: pandas df

    :returns: numpy array -- indicies of catalog match
    '''
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def df_reduce(df):
    '''
    Basic data filtering for gPhoton output

    :param df: gPhoton output df

    :type df: pandas DataFrame with columns cps_bgsub, counts, flux_bgsub

    :returns: pandas DataFrame -- reduced from initial with reset indicies
    '''
              
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

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def df_fullreduce(df):
    '''
    Additional reduction of flagged points, expt, sigmaclip

    :parmam df: gPhoton output pandas DataFrame

    :type df: pandas DataFrame with columns cps_bgsub, counts, flux_bgsub, 
              flags, exptime

    :returns: pandas DataFrame -- reduced from initial with reset indicies
    '''
    
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def df_firstlast(df: object) -> object:
    '''
    If the first and last points aren't within 3 sigma, remove

    :parmam df: gPhoton output pandas DataFram

    :type df: pandas DataFrame with columns flux_bgsub

    :returns: pandas DataFrame -- reduced from initial with reset indicies
    '''

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
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
def tmean_correction(df):
    '''
    Make correction for t_mean by averaging t0 and t1

    :parmam df: gPhoton output pandas DataFram

    :type df: pandas DataFrame with columns flux_bgsub

    :returns: pandas DataFrame -- corrected t_mean column
    '''
    idx_tmean_fix = np.where( (df['t_mean'] < 1)
                            | (df['t_mean'] > df['t1'])
                            | (np.isnan(df['t_mean'])))[0]
    
    for idx in idx_tmean_fix:
        t0 = df['t0'][idx]
        t1 = df['t1'][idx]
        mean = (t1 + t0) / 2.0
        df['t_mean'][idx] = mean

    return df
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def dfsplit(df, tbreak):
    '''
    Split df into visits, defined by tbreak

    :parmam df: gPhoton output pandas DataFram

    :type df: pandas DataFrame with columns flux_bgsub

    :param tbreak: time gap in seconds by which to divide visits

    :type tbreak: int, float

    :returns: pandas DataFrame -- numpy array of pandas DataFrames
    '''
    breaks = []
    for i in range(len(df['t0'])):
        if i != 0:
            if (df['t0'][i] - df['t0'][i-1]) >= tbreak:
                breaks.append(i)

    data = np.split(df, breaks)
    return data
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plotparams(ax):
    '''
    Basic plot params 

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=15)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def DoubleY(ax, colors=('black', 'black')):
    '''
    Create a double y axis with two seperate colors

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :param colors: 2-tuple of axes colors

    :type colors: tuple length 2

    :returns: two axes, modified original and new y scale
    '''
    if (type(colors) != tuple) or (len(colors) != 2):
        raise TypeError("colors must be 2-tuple")
    ax2 = ax.twinx()
    ax.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    for a in [ax, ax2]:
        a.minorticks_on()
        a.tick_params(direction='in', which='both', labelsize=15)
        a.tick_params('both', length=8, width=1.8, which='major')
        a.tick_params('both', length=4, width=1, which='minor')
    ax.tick_params('y', colors=colors[0], which='both')
    ax2.tick_params('y', colors=colors[1], which='both')
    return ax, ax2
        
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def ColoredPoints(df):
    '''
    Find indicies of flaggedpoints, sigmaclip, exposure gap

    :param df: gPhoton output pandas DataFrame

    :type df: pandas DataFrame with columns flux_bgsub, flags, exptime

    :returns: named tuple with attributes:
                   tup.redpoints -- indicies of flagged points, low exptime
                   tup.bluepoints -- indicies of sigma clip
    '''
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def relativescales(df):
    '''
    Output arrays for percent flux, err flux 

    :param df: gPhoton output pandas DataFrame

    :type df: pandas DataFrame with columns flux_bgsub, t_mean, flux_bgsub_err

    :returns: named tuple with attributes:
                   tup.t_mean -- time column
                   tup.flux -- relative flux %
                   tup.err -- relative flux err
    '''
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
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def relativescales_1(df):
    '''
    Output arrays for percent flux, err flux in 0-1 relative scale

    :param df: gPhoton output pandas DataFrame

    :type df: pandas DataFrame with columns flux_bgsub, t_mean, flux_bgsub_err

    :returns: named tuple with attributes:
                   tup.t_mean -- time column
                   tup.flux -- relative flux 
                   tup.err -- relative flux err
    '''
    flux_bgsub = df['flux_bgsub']
    median = np.median(flux_bgsub)
    flux_bgsub = ((flux_bgsub/median))
    flux_err = df['flux_bgsub_err'] / median
    t_mean = df['t_mean']

    OutputTup = collections.namedtuple('OutputTup', ['t_mean', 
                                                     'flux',
                                                     'err'])
    tup = OutputTup(t_mean, flux_bgsub, flux_err)
    return tup
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def raconvert(h, m, s):
    '''
    Convert hour minute second RA to decimal

    :param h: RA hour

    :type h: int, float

    :param m: RA minute

    :type m: int, float

    :param s: RA second

    :type s: int, float

    :returns: float -- right ascension in decimal format
    '''
    ra = Angle((h,m,s), unit='hourangle')
    return ra.degree
#----------------------------------------------------------------------------

#Convert degree minute second to decimal
#----------------------------------------------------------------------------
def decconvert(d, m, s):
    '''
    Convert degree minute second DEC to decimal

    :param d: DEC hour

    :type d: int, float

    :param m: DEC minute

    :type m: int, float

    :param s: DEC second

    :type s: int, float

    :returns: float -- declination in decimal format
    '''
    dec = Angle((d,m,s), u.deg)
    return dec.degree
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def tohms(ra=0, dec=0, fancy=False):
    '''
    Convert decimals ra and dec to hms and dms

    :param ra: right ascension

    :type ra: int, float

    :param dec: declination

    :type dec: int, float

    :param fancy: return a fancy string output (default=False)

    :type fancy: Boolean

    :returns: { if not Fancy -- right ascension, declination in hms dms
              { else: string formated hhmmss.ss+ddmmss.ss
    '''
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
            d_degrees = str(int(-1*dec1.dms.d))
            if abs(dec1.dms.d) < 10:
                d_degrees = '0'+d_degrees

            d_minutes = str(int(-1*dec1.dms.m))
            if abs(dec1.dms.m) < 10:
                d_minutes = '0'+d_minutes

            if abs(dec1.dms.s) < 10:
                d_seconds = '0' + str(round(-1*dec1.dms.s, 2))
            else:
                d_seconds = str(round(-1*dec1.dms.s, 2))

            d_string = r'$-{0}{1}{2}$'.format(d_degrees, d_minutes, d_seconds)
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
            d_string = r'$+{0}{1}{2}$'.format(d_degrees, d_minutes, d_seconds)
        return r_string+d_string
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def coordrange(r, d, sep=2):
    '''
    Give ra and dec range around input coords

    :param r: source ra in degrees
    
    :type ra: int or float

    :param d: source declination in degrees

    :type dec: int or float

    :returns: 4-tuple of lower ra, upper ra, lower dec, upper dec in str

    '''
    ra = Angle(r, u.deg)
    dec = Angle(d, u.deg)
    lowerra = ra - (2*u.arcsecond)
    upperra = ra + (2*u.arcsecond)
    lowerdec = dec - (2*u.arcsecond)
    upperdec = dec + (2*u.arcsecond)
    sp = dict(sep=(' '), precision=2, pad=True)
    print(f"{lowerra.to_string(unit=u.hour, **sp)} \n"
          f"{upperra.to_string(unit=u.hour, **sp)} \n"
          f"{lowerdec.to_string(unit=u.deg, **sp)} \n"
          f"{upperdec.to_string(unit=u.deg, **sp)}")
    return((lowerra.to_string(unit=u.hour, **sp), 
            upperra.to_string(unit=u.hour, **sp),
            lowerdec.to_string(unit=u.deg, **sp),
            upperdec.to_string(unit=u.deg, **sp)))
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plotASASSN_LC(ax, asassn_name):
    '''
    Plot ASASSN light curve

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :param asassn_name: base ASASSN file path

    :type asassn_name: str

    :returns: matplotlib axes object -- light curve of ASASSN data
    '''
    ASASSN_output_V = readASASSN(
            '../ASASSNphot_2/'+asassn_name+'_V.dat')
    ASASSN_JD_V = ASASSN_output_V[0]
    ASASSN_mag_V = ASASSN_output_V[1]
    ASASSN_mag_err_V = ASASSN_output_V[2]

    ASASSN_output_g = readASASSN(
            '../ASASSNphot_2/'+asassn_name+'_g.dat')
    ASASSN_JD_g = ASASSN_output_g[0]
    ASASSN_mag_g = ASASSN_output_g[1]
    ASASSN_mag_err_g = ASASSN_output_g[2]

    ax.errorbar(ASASSN_JD_V, ASASSN_mag_V,
                 yerr=ASASSN_mag_err_V, color='blue',
                 ls='-', label='V band', ecolor='gray')
    ax.errorbar(ASASSN_JD_g, ASASSN_mag_g,
                 yerr=ASASSN_mag_err_g, color='green',
                 ls='-', label='g band', ecolor='gray')
    #Having some issues here, set default ranges if there is a problem
    try:
        maxmag_g = max(ASASSN_mag_g)
        minmag_g = min(ASASSN_mag_g)
        minmag_V = min(ASASSN_mag_V)
        maxmag_V = max(ASASSN_mag_V)
    except:
        maxmag_g = 20
        minmag_g = 10
        minmag_V = 10
        maxmag_V = 20
    maxmag = max(maxmag_V, maxmag_g)
    minmag = min(minmag_V, minmag_g)
    try:
        ax.set_ylim(maxmag, minmag)
    except:
        ax.set_ylim(20, 10)

    ax.set_xlabel('JD')
    ax.set_ylabel("V Magnitude")
    ax.set_title('ASASSN LC')
    ax.legend()

    return ax
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def plotASASSN_pgram(ax, asassn_name):
    '''
    Plot ASASSN periodogram

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :param asassn_name: base ASASSN file path

    :type asassn_name: str

    :returns: matplotlib axes object -- periodogram of ASASSN data
    '''
    ASASSN_output_V = readASASSN(
            '../ASASSNphot_2/'+asassn_name+'_V.dat')
    ASASSN_JD_V = ASASSN_output_V[0]
    ASASSN_mag_V = ASASSN_output_V[1]
    ASASSN_mag_err_V = ASASSN_output_V[2]

    ASASSN_output_g = readASASSN(
            '../ASASSNphot_2/'+asassn_name+'_g.dat')
    ASASSN_JD_g = ASASSN_output_g[0]
    ASASSN_mag_g = ASASSN_output_g[1]
    ASASSN_mag_err_g = ASASSN_output_g[2]
    if len(ASASSN_JD_V) > 5:
        #Select the largest time group
        breaksASN_V = []
        for i in range(len(ASASSN_JD_V)):
            if i != 0:
                if (ASASSN_JD_V[i] - ASASSN_JD_V[i-1]) >= 100:
                    breaksASN_V.append(i)

        Vgroups_JD = []
        Vgroups_mag = []
        Vgroups_mag_err = []
        for i in range(len(breaksASN_V)):
            if i == 0:
                Vgroups_JD.append(ASASSN_JD_V[:breaksASN_V[i]])
                Vgroups_mag.append(ASASSN_mag_V[:breaksASN_V[i]])
                Vgroups_mag_err.append(
                        ASASSN_mag_err_V[:breaksASN_V[i]])
            elif i == len(breaksASN_V) -1:
                Vgroups_JD.append(ASASSN_JD_V[breaksASN_V[i]:])
                Vgroups_mag.append(ASASSN_mag_V[breaksASN_V[i]:])
                Vgroups_mag_err.append(
                        ASASSN_mag_err_V[breaksASN_V[i]:])
            else:
                Vgroups_JD.append(
                        ASASSN_JD_V[breaksASN_V[i-1]:breaksASN_V[i]])
                Vgroups_mag.append(
                        ASASSN_mag_V[breaksASN_V[i-1]:breaksASN_V[i]])
                Vgroups_mag_err.append(
                        ASASSN_mag_err_V[breaksASN_V[i-1]:breaksASN_V[i]])

        length_V_list = [ len(l) for l in Vgroups_JD ]
        if len(length_V_list) > 0:

            idx_Vlongest = np.where(np.array(length_V_list)
                                    == max(length_V_list))[0][0]
            ASASSN_pgramV_JD = Vgroups_JD[idx_Vlongest]
            ASASSN_pgramV_mag = Vgroups_mag[idx_Vlongest]
            ASASSN_pgramV_err = Vgroups_mag_err[idx_Vlongest]

            #Generate LS periodogram
            lsV = LombScargle(ASASSN_pgramV_JD,
                              ASASSN_pgramV_mag,
                              dy=ASASSN_pgramV_err)
            freqV, ampV = lsV.autopower(nyquist_factor=1)
            ax.plot(freqV, ampV, color='blue', label='V mag', zorder=2)
            ax.set_xlim(xmax=(1/30))
            ax.axhline(y=lsV.false_alarm_level(.1),
                        color='blue', alpha=.5,
                        ls='--', label='.1 fal')
    if len(ASASSN_JD_g) > 5:
        lsg = LombScargle(ASASSN_JD_g,
                          ASASSN_mag_g, dy=ASASSN_mag_err_g)
        freqg, ampg = lsg.autopower(nyquist_factor=1)
        ax.plot(freqg, ampg, color='green', label='g mag', zorder=1)
        ax.set_xlim(xmax=(1/30))
        ax.axhline(y=lsg.false_alarm_level(.1),
                    color='green', alpha=.5, ls='--', label='.1 fal')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.set_title('Periodogram for ASASSN Data')
    ax.legend(loc=1)

    return ax
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def flux_to_mag(band, flux):
    assert(band in ['NUV', 'FUV'])
    if band == 'NUV':
        pivot_wavelength = 2297 #angstroms
    else:
        pivot_wavelength = 1524 #angstroms
    #Convert to flux per frequency in Jy
    fv = 3.34e4*(pivot_wavelength)**2 * flux
    #Calculate ab magnitude
    m_ab = -2.5*np.log10(fv) + 8.90
    return m_ab
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def align_yaxis(ax1, v1, ax2, v2):
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)



