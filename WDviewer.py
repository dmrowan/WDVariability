#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import subprocess
#Dom Rowan REU 2018

desc="""
Three functional options: 
    (1) view/comment on functions in range  
    (2) select specific idx --i to view/comment 
    (3) Move interesting sources directory under specific headers
"""

#View and comment on sources in range f, l+1
def main(f,l):
    assert(os.path.isfile("Output/AllData.csv"))
    df_rank = pd.read_csv("Output/AllData.csv")
    #Use the indicies on the csv 
    f = f-2
    l = l-2
    for i in range(f,l+1):
        #Load in n files at a time (used to reduce pdf load time issue)
        numload = 8
        loadimages=False
        #Load images if we are at the first index 
        if (i == f) & (i % numload != 0):
            upper= i+(numload-(i%numload))
            loadimages=True
        #Load images every n times (n defined by numload above)
        elif i % numload == 0:
            loadimages=True
            #If we are reaching the end of input range, go to l+1
            if (l-i)>numload:
                upper = i+numload
            else:
                upper=l+1
        #Use the upperlimit to load indicies in range
        if loadimages:
            for ii in range(i,upper):
                source = df_rank['SourceName'][ii]
                band = df_rank['Band'][ii]
                #Account for special characters
                if '[' in source:
                   source = source.replace('[', "\[").replace(']','\]')
                   filename = "PDFs/{0}_{1}_combined.pdf".format(source, 
                                                                 band)
                else:
                    filename = "PDFs/{0}_{1}_combined.pdf".format(source, 
                                                                  band)
                #Open PDF
                subprocess.run(['xdg-open', filename])

        #Commenting options (stored after all in range run or break called)
        if str(df_rank['Comment'][i]) != 'nan':
            print("Current comment: ", df_rank['Comment'][i])

        comment = input(df_rank['SourceName'][i]+
                        " -- Enter comment to save into csv." +
                        "Hit enter for no comment or to keep" +
                        "existing comment --- ")
        if comment == "break":
            break
        elif len(comment) != 0:
            df_rank.loc[i, 'Comment'] = comment

    df_rank.to_csv("Output/AllData.csv", index=False)

#Display specific index pdf
def selectidx(n, comment=True):
    assert(os.path.isfile("Output/AllData.csv"))
    df_rank = pd.read_csv("Output/AllData.csv")
    #Using index on csv, not innate to df
    i = n-2
    source = df_rank['SourceName'][i]
    band = df_rank['Band'][i]
    #Account for special characters
    if '[' in source:
       source = source.replace('[', "\[").replace(']','\]')
       filename = "PDFs/{0}_{1}_combined.pdf".format(source, band)
    else:
        filename = "PDFs/{0}_{1}_combined.pdf".format(source, band)
    #Open PDf
    subprocess.run(['xdg-open', filename])

    #Commenting Options
    if str(df_rank['Comment'][i]) != 'nan':
        print("Current comment: ", df_rank['Comment'][i])
    if comment:
        comment = input("Enter comment to save into csv. " +
                        "Hit enter for no comment or to keep " +
                        "existing comment --- ")
        if len(comment) != 0:
            df_rank.loc[i, 'Comment'] = comment
            
        df_rank.to_csv("Output/AllData.csv", index=False)

def name(source, comment=True):
    assert(os.path.isfile("Output/Alldata.csv"))
#Add source to InterestingSources 
def move(i):
    assert(os.path.isfile("Output/AllData.csv"))
    df_rank = pd.read_csv("Output/AllData.csv")
    #Using index on csv, not innate to df
    i = i-2
    source = df_rank['SourceName'][i]
    band = df_rank['Band'][i]
    #Account for special characters 
    #Get pdf and csvname 
    if '[' in source:
       source = source.replace('[', "\[").replace(']','\]')
       pdffilename = "PDFs/{0}_{1}_combined.pdf".format(source, band)
       csvfilename = "{0}-{1}.csv".format(source, band)
    else:
        pdffilename = "PDFs/{0}_{1}_combined.pdf".format(source, band)
        csvfilename = "{0}-{1}.csv".format(source, band)
    #User input - select object type for categorization
    objecttype = input("Enter category to move to: " +
                       "Pulsator, KnownPulsator, PossiblePulsator, " +
                       "or Eclipse --- ")
    valid_object_types = ["Pulsator", 
                          "KnownPulsator",
                          "PossiblePulsator", 
                          "Eclipse"]
    if not (objecttype in valid_object_types):
        print("Invalid object type")
        return
    else:
        #Check if directory exists for source ID (FUV and NUV share directory)
        if os.path.isdir("../InterestingSources/"+objecttype+"/"+source):
            print("Directory already exists")
            if not os.path.isdir("../InterestingSources/"+
                                 objecttype+"/"+source+'/PDFs'):
                subprocess.run(['mkdir', '../InterestingSources/'+
                                objecttype+'/'+source+'/PDFs'])
        else:
            print("Making directory for source")
            subprocess.run(['mkdir', '../InterestingSources/'+
                            objecttype+'/'+source])
            subprocess.run(['mkdir', '../InterestingSources/'+
                            objecttype+'/'+source+'/PDFs'])

        #Move pdf file and csvfile
        subprocess.run(['cp', pdffilename, '../InterestingSources/'+
                        objecttype+"/"+source+"/PDFs/"])
        subprocess.run(['cp', csvfilename, '../InterestingSources/'+
                        objecttype+"/"+source+"/"])
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--i", 
            help="Select row of source to view, opening pdf in window", 
            default=None, type=int)
    parser.add_argument("--f", 
            help="If using fromtop, identify first index", 
            default=2, type=int)
    parser.add_argument("--l", 
            help="If using fromtop, identify last index", 
            default=12, type=int)
    parser.add_argument("--move", 
            help="Move file in index to Interesting sources", 
            default=None, type=int)
    parser.add_argument("--name", 
            help="Select source to view by name, opening pdf in window", 
            default=None, type=str)
    args= parser.parse_args()

    if args.move is not None:
        move(args.move)
    elif args.name is not None:
        name(args.name)
    elif args.i is not None:
        selectidx(args.i)
    else:
        main(args.f, args.l)
