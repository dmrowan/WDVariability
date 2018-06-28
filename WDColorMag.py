#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
#Dom Rowan REU 2018

desc="""
WDColorMag: Produce color mag plot with shading based on best rank. Used to help identify interesting sources/confirm ranking
"""

#Main Plotting function:
def main():
    #Path assertions
    assert(os.path.isfile("Output/AllData.csv"))
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"))

    #Read in dataframes
    df_rank = pd.read_csv("Output/AllData.csv")
    bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv")

    #Only looking at top 3000 sources
    df_rank = df_rank[:3001]
    #Read in ranks, sources
    no_gaia_data_counter = 0

    bp_rp_list = []
    absolute_mag_list = []
    rank_list = []

    for source, i in zip(df_rank['SourceName'], range(len(df_rank['SourceName']))):
        #Find source in BigCatalog
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

        if len(bigcatalog_idx) == 0:
                print(source, "Not in catalog")
                no_gaia_data_counter += 1
        else:
            #Grab Gaia information 
            bigcatalog_idx = bigcatalog_idx[0]
        
            gaia_parallax = bigcatalog['gaia_parallax'][bigcatalog_idx]
            if str(gaia_parallax) == 'nan':
                no_gaia_data_counter += 1
            else:
                #Calcualte bp-rp, absolute mag
                g_mag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]
                bp_mag = bigcatalog['gaia_bp_mean_mag'][bigcatalog_idx]
                rp_mag = bigcatalog['gaia_rp_mean_mag'][bigcatalog_idx]

                gaia_parallax = gaia_parallax / 10e3
                absolute_mag_list.append(g_mag + 5*np.log10(gaia_parallax/100))
     
                bp_rp_list.append(bp_mag - rp_mag)

                rank_list.append(df_rank['BestRank'][i])


    assert(len(bp_rp_list) == len(absolute_mag_list) == len(rank_list))

    #Plotting params
    df_plot = pd.DataFrame({'absolutemag':absolute_mag_list, 'bp_rp':bp_rp_list, 'rank':rank_list})
    df_plot = df_plot.sort_values(by=['rank'])
    plt.figure(figsize=(16,12))
    plt.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c=df_plot['rank'], cmap='autumn_r')
    plt.colorbar().set_label("Rank", fontsize=20)
    plt.ylim(ymin=max(absolute_mag_list)+.5, ymax=min(absolute_mag_list)-.5)
    plt.title("Gaia CMD \n Sources not included: {}".format(no_gaia_data_counter), fontsize=20)
    plt.xlabel("BP-RP", fontsize=20)
    plt.ylabel("AbsoluteMag", fontsize=20)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
