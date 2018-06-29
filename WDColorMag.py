#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
import matplotlib.widgets as wd
import numpy as np
import argparse
import pandas as pd
from math import sqrt
from WDviewer import selectidx
#Dom Rowan REU 2018

desc="""
WDColorMag: Produce color mag plot with shading based on best rank. Used to help identify interesting sources/confirm ranking
"""

#Main Plotting function:
def main(pick, view, region):
    #Path assertions
    assert(os.path.isfile("Output/AllData.csv"))
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"))

    #Read in dataframes
    df_rank = pd.read_csv("Output/AllData.csv")
    bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv")

    #Only looking at top 3000 sources
    df_rank_reduced = df_rank[:3001]
    #Read in ranks, sources
    no_gaia_data_counter = 0

    bp_rp_list = []
    absolute_mag_list = []
    rank_list = []
    source_list = []
    band_list = []
    for source, i in zip(df_rank_reduced['SourceName'], range(len(df_rank_reduced['SourceName']))):
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

                rank_list.append(df_rank_reduced['BestRank'][i])
                source_list.append(source)
                band_list.append(df_rank_reduced['Band'][i])


    assert(len(bp_rp_list) == len(absolute_mag_list) == len(rank_list) == len(source_list))

    #Plotting params
    df_plot = pd.DataFrame({'source':source_list, 'band':band_list,'rank':rank_list, 'absolutemag':absolute_mag_list, 'bp_rp':bp_rp_list})
    df_plot = df_plot.sort_values(by=['rank'])
    plt.figure(figsize=(16,12))
    plt.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c=df_plot['rank'], cmap='autumn_r')
    plt.colorbar().set_label("Rank", fontsize=20)
    plt.ylim(ymin=max(absolute_mag_list)+.5, ymax=min(absolute_mag_list)-.5)
    plt.title("Gaia CMD \n Sources not included: {}".format(no_gaia_data_counter), fontsize=20)
    plt.xlabel("BP-RP", fontsize=20)
    plt.ylabel("AbsoluteMag", fontsize=20)

    #Grab specific point
    if pick:
        pick_point = plt.ginput(1, show_clicks=True)[0]
        coords = []
        for i in range(len(df_plot['absolutemag'])):
            color_coords = df_plot['bp_rp'][i]
            mag_coords = df_plot['absolutemag'][i]
            coords.append( (color_coords, mag_coords) )
        distances = []
        for i in range(len(coords)):
            deltax = pick_point[0] - coords[i][0]
            deltay = pick_point[1] - coords[i][1]
            distances.append(sqrt(deltax**2 + deltay**2))
        assert(len(distances) == len(coords))
        closest_indicies = np.where(np.array(distances) == np.min(distances))[0]
        for idx in closest_indicies:
            print(df_plot.iloc[idx])
        

        if view:
            for idx in closest_indicies:
                idx_alldata = np.where( (df_rank['SourceName'] == df_plot['source'][idx]) & (df_rank['Band'] == df_plot['band'][idx]) )[0]
                idx_alldata = idx_alldata[0] + 2
                
                selectidx(idx_alldata, comment=False)

        plt.show()

    #Grab all points within a specified region
    elif region:
        #Draw box and blox corners/lines
        point_list = plt.ginput(2, show_clicks=True)
        xvals = [ tup[0] for tup in point_list ]
        yvals = [ tup[1] for tup in point_list ] 
        plt.plot([xvals[0], xvals[0]], [yvals[0], yvals[1]], 'b-')
        plt.plot([xvals[1], xvals[1]], [yvals[0], yvals[1]], 'b-')
        plt.plot([xvals[0], xvals[1]], [yvals[0], yvals[0]], 'b-')
        plt.plot([xvals[0], xvals[1]], [yvals[1], yvals[1]], 'b-')
        plt.plot(xvals, yvals, 'bo')
        

        df_region = pd.DataFrame({'source':[], 'band':[], 'rank':[], 'absolutemag':[], 'bp_rp':[]})

        for i in range(len(df_plot['absolutemag'])):
            color = df_plot['bp_rp'][i]
            mag = df_plot['absolutemag'][i]
            if (color > min(xvals)) and (color < max(xvals)):
                if (mag > min(yvals)) and (color < max(yvals)):
                    df_temp = df_plot.ix[i]
                    df_region = df_region.append(df_temp, ignore_index=True)

        df_region = df_region.sort_values(by=['rank'], ascending=False)
        print(df_region)
        plt.show()

    else:
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--pick", help="Use Lasso to select points and output sourcename", default=False, action='store_true')
    parser.add_argument("--view", help="View pdf of selected source", default=False, action='store_true')
    parser.add_argument("--region", help="Draw a box and grab info on all sources in region", default=False, action='store_true')
    args= parser.parse_args()

    main(args.pick, args.view, args.region)
