#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import argparse
import pandas as pd
import subprocess
from astropy import units as u
from astropy.coordinates import SkyCoord
import _pickle as pickle
from progressbar import ProgressBar

#Dom Rowan REU 2018

desc="""
MakeCatalog: Generate Catalog of sources. Additional simbad/asassn functions
"""
#Read in pickle information
def main(sep, fname, noexpt_cut):
    picklefname = '/home/dmrowan/WhiteDwarfs/Catalogs/checked.pickle'
    with open(picklefname, 'rb') as p:
        picklelist = pickle.load(p)

    picklelist_names = [ pic[0] for pic in picklelist ]

    #Initialize dictionary
    output_dic = {
            "GaiaDesignation":[],
            "SDSSDesignation":[],
            "MWDDDesignation":[], 
            "ATLASDesignation":[],
            "ra":[],
            "dec":[],
            "gaia_parallax":[],
            "gaia_parallax_error":[],
            "gaia_g_mean_flux":[],
            "gaia_g_mean_flux_error":[],
            "gaia_g_mean_mag":[],
            "gaia_bp_mean_flux":[],
            "gaia_bp_mean_flux_error":[],
            "gaia_bp_mean_mag":[],
            "gaia_rp_mean_flux":[],
            "gaia_rp_mean_flux_error":[],
            "gaia_rp_mean_mag":[],

            "sdss_u":[],
            "sdss_su":[],
            "sdss_g":[],
            "sdss_sg":[],
            "sdss_r":[],
            "sdss_sr":[],
            "sdss_i":[],
            "sdss_si":[],
            "sdss_z":[],
            "sdss_sz":[],
            "sdss_dtype":[],

            "gaia_nuv_time":[],
            "gaia_fuv_time":[],
            "sdss_nuv_time":[],
            "sdss_fuv_time":[],

            "mwdd_type":[],
            "variability":[],
            "binarity":[], 
            "hasdisk":[],
            "mwdd_nuv_time":[],
            "mwdd_fuv_time":[],

            "atlas_nuv_time":[],
            "atlas_fuv_time":[],
    }

    df_SDSS = pd.read_csv("SDSSCatalog.csv")
    df_SDSS['ra'] = df_SDSS['SDSS-J'].str[0:2]+":"+ df_SDSS['SDSS-J'].str[2:4]+":"+ df_SDSS['SDSS-J'].str[4:9]
    df_SDSS['dec'] = df_SDSS['SDSS-J'].str[9:12] + ":" + df_SDSS['SDSS-J'].str[12:14] + ":" + df_SDSS['SDSS-J'].str[14:]

    df_GAIA = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/GaiaWDcatalog.csv")
    gaia_catalog = SkyCoord(ra=list(df_GAIA['ra'])*u.degree, dec=list(df_GAIA['dec'])*u.degree)
    print("Grabbing SDSS sources, adding GAIA info if catalog match exists")
    pbar1 = ProgressBar()
    for i in pbar1(range(len(df_SDSS['ra']))):
        stringcoord = df_SDSS['ra'][i] + " " + df_SDSS['dec'][i]
        c = SkyCoord(stringcoord, unit=(u.hour, u.deg))
        idx, d2d, d3d = c.match_to_catalog_sky(gaia_catalog)
        if d2d < sep*u.arcsec:
            #Grab information from GAIA catalog at index idx
            idx = int(idx)
            output_dic['GaiaDesignation'].append(df_GAIA['designation'][idx])
            output_dic['ra'].append(df_GAIA['ra'][idx])
            output_dic['dec'].append(df_GAIA['dec'][idx])
            output_dic['gaia_parallax'].append(df_GAIA['parallax'][idx] * 10e3)
            output_dic['gaia_parallax_error'].append(df_GAIA['parallax_error'][idx] * 10e3)
            output_dic['gaia_g_mean_flux'].append(df_GAIA['phot_g_mean_flux'][idx])
            output_dic['gaia_g_mean_flux_error'].append(df_GAIA['phot_g_mean_flux_error'][idx])
            output_dic['gaia_g_mean_mag'].append(df_GAIA['phot_g_mean_mag'][idx])
            output_dic['gaia_bp_mean_flux'].append(df_GAIA['phot_bp_mean_flux'][idx])
            output_dic['gaia_bp_mean_flux_error'].append(df_GAIA['phot_bp_mean_flux_error'][idx])
            output_dic['gaia_bp_mean_mag'].append(df_GAIA['phot_bp_mean_mag'][idx])
            output_dic['gaia_rp_mean_flux'].append(df_GAIA['phot_rp_mean_flux'][idx])
            output_dic['gaia_rp_mean_flux_error'].append(df_GAIA['phot_rp_mean_flux_error'][idx])
            output_dic['gaia_rp_mean_mag'].append(df_GAIA['phot_rp_mean_mag'][idx])
            pickleidxgaia = np.where(np.array(picklelist_names) == 'GaiaDR2-'+"".join(list(df_GAIA['designation'][idx])[9:]))[0]
            if len(pickleidxgaia) > 0:
                output_dic['gaia_nuv_time'].append(picklelist[pickleidxgaia[0]][1])
                output_dic['gaia_fuv_time'].append(picklelist[pickleidxgaia[0]][2])
            else:
                output_dic['gaia_nuv_time'].append(float("NaN"))
                output_dic['gaia_fuv_time'].append(float("NaN"))
        else:
            output_dic['GaiaDesignation'].append("")
            output_dic['gaia_parallax'].append("")
            output_dic['gaia_parallax_error'].append("")
            output_dic['gaia_g_mean_flux'].append("")
            output_dic['gaia_g_mean_flux_error'].append("")
            output_dic['gaia_g_mean_mag'].append("")
            output_dic['gaia_bp_mean_flux'].append("")
            output_dic['gaia_bp_mean_flux_error'].append("")
            output_dic['gaia_bp_mean_mag'].append("")
            output_dic['gaia_rp_mean_flux'].append("")
            output_dic['gaia_rp_mean_flux_error'].append("")
            output_dic['gaia_rp_mean_mag'].append("")
            output_dic['gaia_nuv_time'].append(float('NaN'))
            output_dic['gaia_fuv_time'].append(float('NaN'))
            
            #Turn SDSS coords into decimal degrees
            output_dic['ra'].append(c.ra.degree)
            output_dic['dec'].append(c.dec.degree)

        #SDSS information
        output_dic['SDSSDesignation'].append(df_SDSS['SDSS-J'][i])
        output_dic['sdss_u'].append(df_SDSS['u'][i])
        output_dic['sdss_su'].append(df_SDSS['su'][i])
        output_dic['sdss_g'].append(df_SDSS['g'][i])
        output_dic['sdss_sg'].append(df_SDSS['sg'][i])
        output_dic['sdss_r'].append(df_SDSS['r'][i])
        output_dic['sdss_sr'].append(df_SDSS['sr'][i])
        output_dic['sdss_i'].append(df_SDSS['i'][i])
        output_dic['sdss_si'].append(df_SDSS['si'][i])
        output_dic['sdss_z'].append(df_SDSS['z'][i])
        output_dic['sdss_sz'].append(df_SDSS['sz'][i])
        output_dic['sdss_dtype'].append(df_SDSS['dtype'][i])

        #Get expt info from pickle
        pickleidxsdss = np.where( np.array(picklelist_names) == df_SDSS['SDSS-J'][i] )[0]
        if len(pickleidxsdss) > 0:
            output_dic['sdss_nuv_time'].append(picklelist[pickleidxsdss[0]][1])
            output_dic['sdss_fuv_time'].append(picklelist[pickleidxsdss[0]][2])
        else:
            output_dic['sdss_nuv_time'].append(float('NaN'))
            output_dic['sdss_fuv_time'].append(float('NaN'))

    #Now loop through sources in the gaia catalog
    print("Looping through GAIA sources not yet in catalog")
    pbar2 = ProgressBar()
    for i in pbar2(range(len(df_GAIA['designation']))):
        #Check if they already exist (i.e. also in sdss)
        if not df_GAIA['designation'][i] in output_dic['GaiaDesignation']:
            output_dic['GaiaDesignation'].append(df_GAIA['designation'][i])
            output_dic['ra'].append(df_GAIA['ra'][i])
            output_dic['dec'].append(df_GAIA['dec'][i])
            output_dic['gaia_parallax'].append(df_GAIA['parallax'][i] * 10e3)
            output_dic['gaia_parallax_error'].append(df_GAIA['parallax_error'][i] * 10e3)
            output_dic['gaia_g_mean_flux'].append(df_GAIA['phot_g_mean_flux'][i])
            output_dic['gaia_g_mean_flux_error'].append(df_GAIA['phot_g_mean_flux_error'][i])
            output_dic['gaia_g_mean_mag'].append(df_GAIA['phot_g_mean_mag'][i])
            output_dic['gaia_bp_mean_flux'].append(df_GAIA['phot_bp_mean_flux'][i])
            output_dic['gaia_bp_mean_flux_error'].append(df_GAIA['phot_bp_mean_flux_error'][i])
            output_dic['gaia_bp_mean_mag'].append(df_GAIA['phot_bp_mean_mag'][i])
            output_dic['gaia_rp_mean_flux'].append(df_GAIA['phot_rp_mean_flux'][i])
            output_dic['gaia_rp_mean_flux_error'].append(df_GAIA['phot_rp_mean_flux_error'][i])
            output_dic['gaia_rp_mean_mag'].append(df_GAIA['phot_rp_mean_mag'][i])
            pickleidxgaia = np.where(np.array(picklelist_names) == 'GaiaDR2-'+"".join(list(df_GAIA['designation'][i])[9:]))[0]
            if len(pickleidxgaia) > 0:
                output_dic['gaia_nuv_time'].append(picklelist[pickleidxgaia[0]][1])
                output_dic['gaia_fuv_time'].append(picklelist[pickleidxgaia[0]][2])
            else:
                output_dic['gaia_nuv_time'].append(float("NaN"))
                output_dic['gaia_fuv_time'].append(float("NaN"))

            output_dic['SDSSDesignation'].append("")
            output_dic['sdss_u'].append("")
            output_dic['sdss_su'].append("")
            output_dic['sdss_g'].append("")
            output_dic['sdss_sg'].append("")
            output_dic['sdss_r'].append("")
            output_dic['sdss_sr'].append("")
            output_dic['sdss_i'].append("")
            output_dic['sdss_si'].append("")
            output_dic['sdss_z'].append("")
            output_dic['sdss_sz'].append("")
            output_dic['sdss_dtype'].append("")
            output_dic['sdss_nuv_time'].append(float("NaN"))
            output_dic['sdss_fuv_time'].append(float("NaN"))

    #Montreal WD survey
    df_MWDD = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/MWDD-export.csv")
    #Grab expt from pickle
    mwdd_nuv = []
    mwdd_fuv = []
    for wdid in df_MWDD['wdid']:
        wdid = ''.join(wdid.split())
        pickleidxmwdd = np.where(np.array(picklelist_names) == wdid)[0]
        if len(pickleidxmwdd > 0):
            wdid_nuv = picklelist[pickleidxmwdd[0]][1]
            wdid_fuv = picklelist[pickleidxmwdd[0]][2]
        else:
            wdid_nuv = 0
            wdid_fuv = 0
        
        mwdd_nuv.append(wdid_nuv)
        mwdd_fuv.append(wdid_fuv)

    df_MWDD['nuv_time'] = mwdd_nuv
    df_MWDD['fuv_time'] = mwdd_fuv

    #Drop rows where expt < 1000 in both bands
    #MWDD_expt_drop = np.where( (df_MWDD['nuv_time'] < 1000) & (df_MWDD['fuv_time'] < 1000) )[0]
    #print("Dropping {0} rows in MWDD for low exposure out of a total {1} rows".format(str(len(MWDD_expt_drop)), str(len(df_MWDD['wdid']))))
    #df_MWDD = df_MWDD.drop(index=MWDD_expt_drop)
    #df_MWDD = df_MWDD.reset_index(drop=True)

    dropnan_mwdd = np.where(df_MWDD['icrsra'].isnull())[0]
    if len(dropnan_mwdd) > 0:
        df_MWDD = df_MWDD.drop(index=dropnan_mwdd)
        df_MWDD = df_MWDD.reset_index(drop=True)

    #Make conversion to deg for ra and dec
    ra_converted = []
    dec_converted = []
    for i in range(len(df_MWDD['icrsra'])):
        c1 = SkyCoord(str(df_MWDD['icrsra'][i] + " " + df_MWDD['icrsdec'][i]), unit=(u.hour, u.deg))
        ra_converted.append(c1.ra.deg)
        dec_converted.append(c1.dec.deg)

    df_MWDD['ra'] = ra_converted
    df_MWDD['dec'] = dec_converted
    mwdd_catalog = SkyCoord(ra=list(df_MWDD['ra'])*u.degree, dec=list(df_MWDD['dec'])*u.degree)

    #Go through all sources currently in outputdic
    #If it matches up with source in MWDD, add info 
    print("Adding MWDD info for current sources in catalog where match exists")
    pbar3 = ProgressBar()
    for i in pbar3(range(len(output_dic['SDSSDesignation']))):
        c1 = SkyCoord(output_dic['ra'][i]*u.degree, output_dic['dec'][i]*u.degree)
        idx, d2d, d3d = c1.match_to_catalog_sky(mwdd_catalog)
        #idx refers to place in df_mwdd that matches with c1 from output_dic
        #If seperation < threshold
        if d2d < sep*u.arcsec:
            idx = int(idx)
            output_dic['MWDDDesignation'].append(df_MWDD['wdid'][idx])
            output_dic['mwdd_type'].append(df_MWDD['spectype'][idx])
            output_dic['variability'].append(df_MWDD['variability'][idx])
            output_dic['binarity'].append(df_MWDD['binarity'][idx])
            output_dic['hasdisk'].append(df_MWDD['hasdisk'][idx])
            output_dic['mwdd_nuv_time'].append(df_MWDD['nuv_time'][idx])
            output_dic['mwdd_fuv_time'].append(df_MWDD['fuv_time'][idx])
        else:
            output_dic['MWDDDesignation'].append("")
            output_dic['mwdd_type'].append("")
            output_dic['variability'].append("")
            output_dic['binarity'].append("")
            output_dic['hasdisk'].append("")
            output_dic['mwdd_nuv_time'].append(float("NaN"))
            output_dic['mwdd_fuv_time'].append(float("NaN"))

    #Now need to add sources that are new from MWDD (not in GAIA or SDSS)
    #These are new rows at the bottom of the dic/table
    print("Adding new sources from MWDD")
    pbar4 = ProgressBar()
    for i in pbar4(range(len(df_MWDD['wdid']))):
        #Check if they already exist in output
        if not df_MWDD['wdid'][i] in output_dic['MWDDDesignation']:
            output_dic['MWDDDesignation'].append(df_MWDD['wdid'][i])
            output_dic['ra'].append(df_MWDD['ra'][i])
            output_dic['dec'].append(df_MWDD['dec'][i])
            output_dic['mwdd_type'].append(df_MWDD['spectype'][i])
            output_dic['variability'].append(df_MWDD['variability'][i])
            output_dic['binarity'].append(df_MWDD['binarity'][i])
            output_dic['hasdisk'].append(df_MWDD['hasdisk'][i])
            output_dic['mwdd_nuv_time'].append(df_MWDD['nuv_time'][i])
            output_dic['mwdd_fuv_time'].append(df_MWDD['fuv_time'][i])
            #Need to add info for other columns that will be empty here
                #(ATLAS column not generated yet)
            output_dic['SDSSDesignation'].append("")
            output_dic['sdss_u'].append("")
            output_dic['sdss_su'].append("")
            output_dic['sdss_g'].append("")
            output_dic['sdss_sg'].append("")
            output_dic['sdss_r'].append("")
            output_dic['sdss_sr'].append("")
            output_dic['sdss_i'].append("")
            output_dic['sdss_si'].append("")
            output_dic['sdss_z'].append("")
            output_dic['sdss_sz'].append("")
            output_dic['sdss_dtype'].append("")
            output_dic['sdss_nuv_time'].append(float("NaN"))
            output_dic['sdss_fuv_time'].append(float("NaN"))
            output_dic['GaiaDesignation'].append("")
            output_dic['gaia_parallax'].append("")
            output_dic['gaia_parallax_error'].append("")
            output_dic['gaia_g_mean_flux'].append("")
            output_dic['gaia_g_mean_flux_error'].append("")
            output_dic['gaia_g_mean_mag'].append("")
            output_dic['gaia_bp_mean_flux'].append("")
            output_dic['gaia_bp_mean_flux_error'].append("")
            output_dic['gaia_bp_mean_mag'].append("")
            output_dic['gaia_rp_mean_flux'].append("")
            output_dic['gaia_rp_mean_flux_error'].append("")
            output_dic['gaia_rp_mean_mag'].append("")
            output_dic['gaia_nuv_time'].append(float("NaN"))
            output_dic['gaia_fuv_time'].append(float("NaN"))


    #ATLAS Catalog Crosslist - same procedure as for MWDD
    df_ATLAS = pd.read_csv('/home/dmrowan/WhiteDwarfs/Catalogs/ATLAS_WDcatalogue.csv')
    #Get expt info from pickle
    atlas_nuv = []
    atlas_fuv = []
    for name in df_ATLAS['ATLASname']:
        pickleidxatlas = np.where(np.array(picklelist_names) == name)[0]
        if len(pickleidxatlas > 0):
            atlas_nuv.append(picklelist[pickleidxatlas[0]][1])
            atlas_fuv.append(picklelist[pickleidxatlas[0]][2])
        else:
            atlas_nuv.append(0)
            atlas_fuv.append(0)

    df_ATLAS['nuv_time'] = atlas_nuv
    df_ATLAS['fuv_time'] = atlas_fuv

    #Drop rows with < 1000 s in both expt bands
    #ATLAS_expt_drop = np.where( (df_ATLAS['nuv_time'] < 1000) & (df_ATLAS['fuv_time'] < 1000) )[0]
    #print("Dropping {} rows in ATLAS for low exposure".format(str(len(ATLAS_expt_drop))))
    #df_ATLAS = df_ATLAS.drop(index=ATLAS_expt_drop)
    #df_ATLAS = df_ATLAS.reset_index(drop=True)

    atlas_catalog = SkyCoord(ra=list(df_ATLAS['ra'])*u.degree, dec=list(df_ATLAS['dec'])*u.degree)
    #First add to existing rows (add info if we match, add nan if we dont)
    print("Adding ATLAS info to existing sources in catalog if match exists")
    pbar5 = ProgressBar()
    for i in pbar5(range(len(output_dic['SDSSDesignation']))):
        c2 = SkyCoord(output_dic['ra'][i]*u.degree, output_dic['dec'][i]*u.degree)
        #idx refers to place in df_ATLAS that matches with c2 from output_dic
        idx, d2d, d3d = c2.match_to_catalog_sky(atlas_catalog)
        if d2d < sep*u.arcsec:
            idx = int(idx)
            output_dic['ATLASDesignation'].append(df_ATLAS['ATLASname'][idx])
            output_dic['atlas_nuv_time'].append(df_ATLAS['nuv_time'][idx])
            output_dic['atlas_fuv_time'].append(df_ATLAS['fuv_time'][idx])
        else:
            output_dic['ATLASDesignation'].append("")
            output_dic['atlas_nuv_time'].append(float("NaN"))
            output_dic['atlas_fuv_time'].append(float("NaN"))

    #Now add rows for atlas designations that don't yet exist in the output_dic
    print("Adding new sources from ATLAS")
    pbar6 = ProgressBar()
    for i in pbar6(range(len(df_ATLAS['ATLASname']))):
        if not df_ATLAS['ATLASname'][i] in output_dic['ATLASDesignation']:
            output_dic['ATLASDesignation'].append(df_ATLAS['ATLASname'][i])
            output_dic['ra'].append(df_ATLAS['ra'][i])
            output_dic['dec'].append(df_ATLAS['dec'][i])
            output_dic['atlas_nuv_time'].append(df_ATLAS['nuv_time'][i])
            output_dic['atlas_fuv_time'].append(df_ATLAS['fuv_time'][i])

            output_dic['SDSSDesignation'].append("")
            output_dic['sdss_u'].append("")
            output_dic['sdss_su'].append("")
            output_dic['sdss_g'].append("")
            output_dic['sdss_sg'].append("")
            output_dic['sdss_r'].append("")
            output_dic['sdss_sr'].append("")
            output_dic['sdss_i'].append("")
            output_dic['sdss_si'].append("")
            output_dic['sdss_z'].append("")
            output_dic['sdss_sz'].append("")
            output_dic['sdss_dtype'].append("")
            output_dic['sdss_nuv_time'].append(float("NaN"))
            output_dic['sdss_fuv_time'].append(float("NaN"))
            output_dic['GaiaDesignation'].append("")
            output_dic['gaia_parallax'].append("")
            output_dic['gaia_parallax_error'].append("")
            output_dic['gaia_g_mean_flux'].append("")
            output_dic['gaia_g_mean_flux_error'].append("")
            output_dic['gaia_g_mean_mag'].append("")
            output_dic['gaia_bp_mean_flux'].append("")
            output_dic['gaia_bp_mean_flux_error'].append("")
            output_dic['gaia_bp_mean_mag'].append("")
            output_dic['gaia_rp_mean_flux'].append("")
            output_dic['gaia_rp_mean_flux_error'].append("")
            output_dic['gaia_rp_mean_mag'].append("")
            output_dic['gaia_nuv_time'].append(float("NaN"))
            output_dic['gaia_fuv_time'].append(float("NaN"))
            output_dic['MWDDDesignation'].append("")
            output_dic['mwdd_type'].append("")
            output_dic['variability'].append("")
            output_dic['binarity'].append("")
            output_dic['hasdisk'].append("")
            output_dic['mwdd_nuv_time'].append(float("NaN"))
            output_dic['mwdd_fuv_time'].append(float("NaN"))


    #Construct df
    output_df = pd.DataFrame(output_dic)

    #Drop rows with type other than WD
    non_wd_idx = np.where( (output_df['sdss_dtype'].str[0] != 'D') & (output_df['sdss_dtype'] != '') )[0]
    output_df = output_df.drop(index=non_wd_idx)
    output_df = output_df.reset_index(drop=True)


    #Make column for maximum exposure time
    nuv_time_max = []
    fuv_time_max = []
    for i in range(len(output_df['sdss_nuv_time'])):
        nuv_time_max.append( np.nanmax(np.asarray([output_df['sdss_nuv_time'][i], output_df['gaia_nuv_time'][i], output_df['mwdd_nuv_time'][i], output_df['atlas_nuv_time'][i]])) )
        fuv_time_max.append( np.nanmax(np.asarray([output_df['sdss_fuv_time'][i], output_df['gaia_fuv_time'][i], output_df['mwdd_fuv_time'][i], output_df['atlas_nuv_time'][i]])) )

    output_df['nuv_time_max'] = nuv_time_max
    output_df['fuv_time_max'] = fuv_time_max

    if not noexpt_cut:
        #Drop rows with < 1000s of expt in both bands
        expt_drop = np.where( (output_df['nuv_time_max'] < 1000) & (output_df['fuv_time_max'] < 1000) )[0]
        print("Dropping " + str(len(expt_drop)) + " rows due to exposure time limitation")
        output_df = output_df.drop(index=expt_drop)
        output_df = output_df.reset_index(drop=True)


    #Save to CSV
    output_df.to_csv(fname, index=False)

#Simbad matching
def addsimbad(sep, fname):
    assert(os.path.isfile(fname))
    input_df = pd.read_csv(fname)
    #Generate simbad script
    with open("simbadscript.txt", "w") as f:
        f.write('''format object f1 "%MAIN_ID | "+ "%OTYPELIST"''')
        f.write("\n set radius {}s".format(sep))
        f.write("\n set limit 1")
    
    #Loop through all ra, dec in input_df
    for ra, dec in zip(input_df['ra'], input_df['dec']):
        with open("simbadscript.txt", "a") as f:
            f.write("\n coo {0} {1}".format(ra,dec))
    
    print("Simbad script created. \
            Go to http://simbad.u-strasbg.fr/simbad/sim-fscript \
            \n Upload script, download result")
    move_input = input("Hit enter if script is processed and in Catalogs \
            directory. Enter m if needs to be moved from downloads")
    if move_input == 'm':
        print("Moving file to Catalogs folder")
        subprocess.run(['mv', '/home/dmrowan/Downloads/simbad.txt', '.'])

    assert(os.path.isfile('simbad.txt'))
    with open('simbad.txt', 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if '::error::' in lines[i]:
            idx_error = i
        elif '::data::' in lines[i]:
            idx_data = i
            break

    idx_no_simbad = []
    for i in range(idx_error, idx_data+1):
        if lines[i][0] == '[':
            for ii in range(len(lines[i])):
                if lines[i][ii] == "]":
                    endbracket = ii
                    break
        
            idx_no_simbad.append(int(lines[i][1:endbracket]) - 4)

    returned_sources = []
    for i in range(idx_data+2, len(lines)):
        vertlineidx = np.where(np.array(list(lines[i])) == "|")[0]
        quoteidx = np.where(np.array(list(lines[i]))=='"')[0]
        if len(vertlineidx > 0) and len(quoteidx) > 0:
            txtsourcename = lines[i][:(vertlineidx[0]-1)]
            txtobstype = lines[i][quoteidx[-1]+1:]
            returned_sources.append( [txtsourcename, txtobstype] )

    simbad_names = []
    simbad_types = []
    for i in range(len(input_df['ra'])):
        if i in idx_no_simbad:
            simbad_names.append("")
            simbad_types.append("")
        else:
            simbad_names.append(returned_sources[0][0])
            simbad_types.append(returned_sources[0][1])
            returned_sources.pop(0)

    input_df['SimbadName'] = simbad_names
    input_df['SimbadTypes'] = simbad_types

    updated_df_name = "{}_simbad.csv".format(fname[:-4])
    input_df.to_csv(updated_df_name, index=False)

#Asassn matching
def addasassn(sep, fname):
    #Read in dataframe
    assert(os.path.isfile(fname))
    input_df = pd.read_csv(fname)
    asassn_column = []
    asassn_file_list = os.listdir('/home/dmrowan/WhiteDwarfs/ASASSNphot_2')
    acdat = '/home/dmrowan/WhiteDwarfs/Catalogs/asassncoords.dat'
    assert(os.path.isfile(acdat))
    with open(acdat) as f:
        lines = f.readlines()

    gaia_asassn_name = []
    gaia_asassn_ra = []
    gaia_asassn_dec = []

    for line in lines:
        lsplit = line.split()
        gaia_asassn_name.append(lsplit[0])
        gaia_asassn_ra.append(float(lsplit[1]))
        gaia_asassn_dec.append(float(lsplit[2]))

    numbers_asassn_name = []
    numbers_asassn_ra = []
    numbers_asassn_dec = []
    for filename in asassn_file_list:
        if filename[0] != 'G':
            if filename[:-6] in numbers_asassn_name:
                continue
            else:
                numbers_asassn_name.append(filename[:-6])
                name = filename[:-6]
                stringcoord = name[:2]+':'+name[2:4]+ \
                        ':'+name[4:9]+' '+name[9:12]+ \
                        ':'+name[12:14]+':'+name[14:]
                c = SkyCoord(stringcoord, unit=(u.hour, u.deg))
                numbers_asassn_ra.append(c.ra.deg)
                numbers_asassn_dec.append(c.dec.deg)

    all_names = gaia_asassn_name + numbers_asassn_name
    all_ra = gaia_asassn_ra + numbers_asassn_ra
    all_dec = gaia_asassn_dec + numbers_asassn_dec
    assert( len(all_names) == len(all_ra) == len(all_dec) )
    asassn_catalog = SkyCoord(ra=all_ra*u.degree, dec=all_dec*u.degree)
    pbar = ProgressBar()
    for i in pbar(range(len(input_df['MainID']))):
        if input_df['MainID'][i][0:4] == 'Gaia':
            dirhead= '/home/dmrowan/WhiteDwarfs/ASASSNphot_2/'
            if os.path.isfile(dirhead+input_df['MainID'][i].replace(' ','')+'_V.dat'):
                asassn_column.append(input_df['MainID'][i].replace(' ', ''))
            else:
                asassn_column.append("")
        else:
            c2 = SkyCoord(input_df['ra'][i]*u.degree, input_df['dec'][i]*u.degree)
            idx, d2d, d3d = c2.match_to_catalog_sky(asassn_catalog)
            if d2d < sep*u.arcsec:
                asassn_column.append(all_names[idx])
            else:
                asassn_column.append("")


    input_df["ASASSNname"] = asassn_column
    updated_df_name = "{}_asassn.csv".format(fname[:-4])
    input_df.to_csv(updated_df_name, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--sep", help="Arcsecond seperation for searching catalogs", default=5, type=float)
    parser.add_argument("--simbad", help="Add Simbad information to existing catalog", default=False, action='store_true')
    parser.add_argument("--asassn", help="Add ASASSN information to existing catalog", default=False, action='store_true')
    parser.add_argument("--fname", help="Name to write/read big catalog", required=True, type=str)
    parser.add_argument("--noexpt_cut", help="Dont do full cut, keep all expt", default=False, action='store_true')
    args= parser.parse_args()

    if args.simbad:
        addsimbad(sep = args.sep, fname = args.fname)
    elif args.asassn:
        addasassn(sep = args.sep, fname = args.fname)
    else:
        main(sep = args.sep, fname=args.fname, noexpt_cut=args.noexpt_cut)

