#!/usr/local/bin/python
# Copyright 2023, The University of Iowa.  All rights reserved.
# Permission is hereby given to use and reproduce this software 
# for non-profit educational purposes only.
#
# Code to detect HCW shifts.
# Oscar Andersen and Alberto Maria Segre 
#
# Uses an approach similar to debouncing, except here we merge
# consecutive visits even if they are not to the same room, and we
# tend to work with large interval separations (default 10h=36000s).
#
# To use:
#    % shifts.py [seconds=36000] < infile.csv > outfile.csv
# will transform infile.csv into outfile.csv by merging subsequent
# visits by the same HCW to any room into one shift, producing a csv
# file of shifts, including a graphical 24h portrayal in column 1.
#
import sys
import csv
import datetime

def shifts(interval=36000):
    lastShift= {} # key is hid, stores entire visit (a row from the file)
    newShifts= [] # records shifts

    # Creates a dictionary reader object csvin from STDIN
    csvin = csv.DictReader(sys.stdin)

    # For each entry in csvin, where row is a dictionary.
    nvisits = 0
    for row in csvin:
        if row['hid'] not in lastShift: 
            # No previous visit by hcw hid, so by definition a new
            # shift. Save this shift in the lastShift dictionary,
            # indexed by the hid.
            lastShift[row['hid']] = row
	    # Initialize visit count and cumulative visit duration,
	    # stored, for now, in the average duration slot.
            lastShift[row['hid']]['vcount']=1
            lastShift[row['hid']]['avgvdur']=int(row['duration'])
        elif int((datetime.datetime.fromisoformat(row['itime']) - datetime.datetime.fromisoformat(lastShift[row['hid']]['otime'])).total_seconds()) <= interval:
            # This is a new shift by hcw hid within maxgap time
            # interval. Update cumulative counters.
            lastShift[row['hid']]['vcount']=lastShift[row['hid']]['vcount'] + 1
            lastShift[row['hid']]['avgvdur']=lastShift[row['hid']]['avgvdur']+int(row['duration'])
            # Update existing data, extending duration.
            lastShift[row['hid']]['otime'] = row['otime']
            lastShift[row['hid']]['duration'] = int((datetime.datetime.fromisoformat(lastShift[row['hid']]['otime']) - datetime.datetime.fromisoformat(lastShift[row['hid']]['itime'])).total_seconds())
        else: 
            # Previous shift was too long ago, so by definition now a
            # new shift. Update and close the previous one, then start
            # the new one. 
            lastShift[row['hid']]['avgvdur']=lastShift[row['hid']]['avgvdur']/(60*lastShift[row['hid']]['vcount'])
            newShifts.append(lastShift[row['hid']])
            # Start the new shift.
            lastShift[row['hid']] = row
            lastShift[row['hid']]['vcount']=1
            lastShift[row['hid']]['avgvdur']=int(row['duration'])
        # Increment 0-indexed visit counter.
        nvisits = nvisits + 1
            
    # Flush any shifts left in lastShift.
    for hid in lastShift.keys():
        lastShift[hid]['avgvdur']=lastShift[hid]['avgvdur']/(60*lastShift[hid]['vcount'])
    newShifts.extend(lastShift.values())

    # Sort these by itime and jtid then print them out; would have
    # been more efficient to maintain in sorted order all along.
    newShifts.sort(key=lambda x: (x['itime'], x['jtid']))

    # Write data to STDOUT. These data are almost exactly like the
    # input data, except vid/rid values are replaced with a shift
    # sketch, individual shift/idisp/odisp are lost, and vduration is
    # scaled to hours.
    sys.stdout.write("plot,hid,jtid,utid,ftid,shift,stime,etime,vcount,vmean(m),duration(h)\n")
    for row in newShifts:
        sys.stdout.write("{},{},{},{},{},{},{},{},{},{:05.2f},{:05.2f}\n".format(plotshift(datetime.datetime.fromisoformat(row['itime']),
                                                                                           datetime.datetime.fromisoformat(row['otime'])),

                                                                                 row['hid'], row['jtid'], row['utid'], row['ftid'], row['shift'], row['itime'], row['otime'],
                                                                                 row['vcount'], row['avgvdur'], int(row['duration'])/(60*60)))
# Old version incorporates uid and fid, which won't be defined for remixed or synthetic data.
#        sys.stdout.write("shift,hid,jtid,uid,utid,fid,ftid,stime,etime,vcount,vmean(m),duration(h)\n")
#        sys.stdout.write("{},{},{},{},{},{},{},{},{},{},{},{:05.2f},{:05.2f}\n".format(plotshift(datetime.datetime.fromisoformat(row['itime']),
#                                                                                                 datetime.datetime.fromisoformat(row['otime'])),
#                                                                                       row['hid'], row['jtid'], row['uid'], row['utid'],
#                                                                                       row['fid'], row['ftid'], row['shift'], row['itime'], row['otime'],
#                                                                                       row['vcount'], row['avgvdur'], int(row['duration'])/(60*60)))

    # Write status update to stderr
    sys.stderr.write("{} visits read; {} shifts written\n".format(nvisits, len(newShifts)))

# Return a graphic display of the shift, where itime and otime are datetime objects.
def plotshift(itime, otime):
    ihead = {0:'|Mon ', 1:'|Tue ', 2:'|Wed ', 3:'|Thu ', 4:'|Fri ', 5:'|Sat ', 6:'|Sun '}
    otail = {0:' Mon|', 1:' Tue|', 2:' Wed|', 3:' Thu|', 4:' Fri|', 5:' Sat|', 6:' Sun|'}
    if itime.day == otime.day:
        # Day shift
        shift = ihead[itime.weekday()] + itime.hour*' ' + (otime.hour+1-itime.hour)*'-' + (25-otime.hour)*' ' + otail[otime.weekday()]
    elif (otime - itime).days < 1:
        # Night shift
        shift = ihead[itime.weekday()] + otime.hour*'-' + (itime.hour+1-otime.hour)*' ' + (25-itime.hour)*'-' + otail[otime.weekday()]
    else: 
        # Multi-day shift
        shift = ihead[itime.weekday()] + 26*'-' + otail[otime.weekday()]
    return(shift)

if __name__ == '__main__':
    # sys.argv[0] will be the command name, and there will be either 0
    # or 1 additional argument given (min separating time interval in
    # seconds, which defaults to 36000).
#    try:
        if len(sys.argv) == 2:
            # Time interval specified
            shifts(int(sys.argv[1]))
        elif len(sys.argv) == 1:
            # Default behavior
            shifts();
#    except:
#        sys.stderr.write("Usage:\n  {} [seconds=36000] < infile.csv > outfile.csv\n".format(sys.argv[0]))
