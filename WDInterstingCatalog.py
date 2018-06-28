#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import numpy as np
import argparse
import pandas as pd
#Dom Rowan REU 2018

desc="""
WDInterestingCatalog: Generate catalog of interesting sources, including ra and dec info on sources
"""

#Main Plotting function:
def main():
    #Path assertions
    assert(os.path.isdir("Eclipse"))
    assert(os.path.isdir("KnownPulsator"))
    assert(os.path.isdir("Pulsator"))
    assert(os.path.isdir("PossiblePulsator"))
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"))

    #Read in dataframes
    bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv")

    #Import types into lists
    eclipse_sources = os.listdir("Eclipse")
    known_pulsators = os.listdir("KnownPulsator")
    new_pulsators = os.listdir("Pulsator")
    possible_pulsators = os.listdir("PossiblePulsator")

    #Initialize Lists
    mainID = []
    objecttype = []
    ra_list = []
    dec_list = []

    #Grab object types
    for source in eclipse_sources:
        mainID.append(source)
        objecttype.append("Eclipse")

    for source in known_pulsators:
        mainID.append(source)
        objecttype.append("KnownPulsator")
    
    for source in new_pulsators:
        mainID.append(source)
        objecttype.append("NewPulsator")

    for source in possible_pulsators:
        mainID.append(source)
        objecttype.append("PossiblePulsator")

    for source in mainID:
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
        else:
            #Find RA and DEC
            bigcatalog_idx = bigcatalog_idx[0]
            ra_list.append(bigcatalog['ra'][bigcatalog_idx])
            dec_list.append(bigcatalog['dec'][bigcatalog_idx])

    #Create and store DF
    output_df = pd.DataFrame({"MainID":mainID, "Type":objecttype, "Ra":ra_list, "Dec":dec_list})
    output_df.to_csv("InterestingSources.csv")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
