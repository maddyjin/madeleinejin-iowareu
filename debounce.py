#!/usr/local/bin/python
# Copyright 2023, The University of Iowa.  All rights reserved.
# Permission is hereby given to use and reproduce this software 
# for non-profit educational purposes only.
#
# Code to debounce HCW visits.
# Oscar Andersen and Alberto Maria Segre 
#
# Default debounce interval is 180 seconds, which we determined
# experimentally.
#
# To use:
#    % debounce.py [seconds=180] < infile.csv > outfile.csv
# will transform infile.csv into outfile.csv by merging subsequent
# visits by the same HCW to the same room into one extended visit,
# provided they are separated by no more than the specified debouncing
# interval.
#
import sys
import csv
import datetime

# Expects input data to contain, in order:
#   vid,rid,fid,hid,jtid,uid,did,shift,itime,otime,duration,nhitime,nritime,idisp,odisp
# see pull.py for details. Produces identical format file as outcome.
def debounce(interval=180):
    lastVisit= {} # key is hid, stores entire visit (a row from the file)
    revisedVisits= [] # records merged visits and excludes debounced visits

    # Creates a dictionary reader object csvin from STDIN
    csvin = csv.DictReader(sys.stdin)

    # For each entry in csvin, where row is a dictionary.
    nvisits = 0
    for row in csvin:
        nvisits = nvisits + 1
        if row['hid'] not in lastVisit: 
            # No previous visit by hcw hid, so by definition a visit
            # to a "new" room. Save this visit in the lastVist
            # dictionary, indexed by the hid.
            #sys.stderr.write("Inserting new visit {}\n".format(row))
            lastVisit[row['hid']] = row
        elif row['rid'] == lastVisit[row['hid']]['rid'] and int((datetime.datetime.fromisoformat(row['itime']) - datetime.datetime.fromisoformat(lastVisit[row['hid']]['otime'])).total_seconds()) <= interval:
            # This is a return visit to the same room rid by hcw hid
            # within maxgap time interval and without any intervening
            # visit to another room.
            #sys.stderr.write("Updating existing visit {} => {}\n".format(lastVisit[row['hid']], int((datetime.datetime.fromisoformat(row['otime']) - datetime.datetime.fromisoformat(lastVisit[row['hid']]['itime'])).total_seconds())))
            lastVisit[row['hid']]['otime'] = row['otime']
            lastVisit[row['hid']]['duration'] = int((datetime.datetime.fromisoformat(lastVisit[row['hid']]['otime']) - datetime.datetime.fromisoformat(lastVisit[row['hid']]['itime'])).total_seconds())
        else: 
            # previously visited a different room, so by definition a
            # visit to a "new" room
            #sys.stderr.write("Closing exisiting visit {}\nInserting new visit {}\n".format(lastVisit[row['hid']],row))
            revisedVisits.append(lastVisit[row['hid']])
            lastVisit[row['hid']] = row

    # Flush any visits left in lastVisit.
    revisedVisits.extend(lastVisit.values())

    # Sort these by itime and then print them out; would have been
    # more efficient to maintain in sorted order all along. 
    revisedVisits.sort(key=lambda x: x['itime'])

    # Write data to STDOUT.
    sys.stdout.write("vid,rid,fid,hid,jtid,uid,did,shift,itime,otime,duration,nhitime,nritime,idisp,odisp\n")
    for row in revisedVisits:
        sys.stdout.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(row['vid'], row['rid'], row['fid'], row['hid'], row['jtid'],
                                                                           row['uid'], row['did'], row['shift'], row['itime'], row['otime'],
                                                                           row['duration'], row['nhitime'], row['nritime'],
                                                                           row['idisp'], row['odisp']))

    # Write status update to stderr
    sys.stderr.write("{} visits read; {} debounced visits written\n".format(nvisits, len(revisedVisits)))

if __name__ == '__main__':
    # sys.argv[0] will be the command name, and there will be either 0
    # or 1 additional argument given (max separating time interval in
    # seconds, which defaults to 180).
    try:
        if len(sys.argv) == 2:
            # Time interval specified
            debounce(int(sys.argv[1]))
        elif len(sys.argv) == 1:
            # Default behavior
            debounce();
    except:
        sys.stderr.write("Usage:\n  {} [seconds=180] < infile.csv > outfile.csv\n".format(sys.argv[0]))
