#!/usr/local/bin/python
# Copyright 2023, The University of Iowa.  All rights reserved.
# Permission is hereby given to use and reproduce this software 
# for non-profit educational purposes only.
#
# Code to produce remixed visit schedules from  SSense data.
# Madeleine Jin and Alberto Maria Segre
#
# To use:
#    % remix.py [cfile=config.cfg] < infile.csv > outfile.csv
# where config.cfg contains the remix parameters:
#    ndays: <int>        number of days in remix
#    first_day: <int>    index of first day (0=Monday)
#    day_classification: random, week, day
#
# Here, remixes may randomly sample days, sample while respecting
# weekday/weekend, or sample while respecting day-of-week constraints.
import sys
import csv
import datetime
from random import random, randint, choice, choices

######################################################################
# Generates a remix of the visit data obtained from stdin according to
# the configuration found in configuration file cfile.
#
def remix(cfile):
    # Parse the configuration file, ignoring any options you don't
    # know about. In this way, we can share pull's configuration file
    # and just add the remix-specific parameters.
    with open(cfile, 'r') as cfile:
        # To parse day of week.
        dow = {'0':0, 'mon':0, 'monday':0, '1':1, 'tue':1, 'tues':1, 'tuesday':1,
               '2':2, 'wed':2, 'weds':2, 'wednesday':2, '3':3, 'thu':3, 'thur':3, 'thurs':3, 'thursday':3,
               '4':4, 'fri':4, 'friday':4, '5':5, 'sat':5, 'saturday':5, '6':6, 'sun':6, 'sunday':6,
               '7':7, 'weekday':7, '8':8, 'weekend':8}
        # Default values.
        ndays = 14   		# 2 weeks
        stday = 0		# start on 0=Mon, 1=Tue, ... 6=Sun, 7=weekday, 8=weekend
        stype = 'random'	# choose randomly (alt: dow, weekday)
        shint = 36000		# shift separting interval (10h default)
        for line in cfile:
            # Skip any blank or comment lines
            if line.strip() == '' or line.lstrip()[0] == '#':
                continue
            line=[ element.strip() for element in line.lower().split(':') ]
            if line[0] == 'ndays':
                ndays=int(line[1].split('#')[0].strip())
            elif line[0] == 'stday':
                stday=dow[line[1].split('#')[0].strip().lower()]
            elif line[0] == 'stype':
                stype=line[1].split('#')[0].strip().lower()
            elif line[0] == 'shint':
                shint=int(line[1].split('#')[0].strip())
        #print("ndays={}, stday={}, stype={}, shint={}".format(ndays, stday, stype, shint))
 
    # Creates a dictionary reader object csvin from STDIN
    csvin = csv.DictReader(sys.stdin)
    ## DEBUG
    ##csvin = csv.DictReader(open('test.csv', 'r'))

    sdate = edate = None # visit start and end dates as datetime objects
    visits = []          # list of dictionaries (visits) in input order (sorted by itime).
    today = '0000-00-00' # current date in timestamp string format.
    shifts = []          # 0-indexed list of dictionaries (shifts).
    openShifts = {}      # key is hid: used to assign shift index.
    days = []	         # 0-indexed lists of dictionaries (days).
    weekly = { 'weekday':set(), 'weekend':set() } # day indexes by itime/stype
    daily = { i:set() for i in range(7) }         # day indexes by itime/stype
    rnew = 0             # maps rid to small integer rid
    rmap = {}            # maps rid to small integer rid

    # For each entry in csvin, where row is a dictionary. A row is
    # represented as:
    #   vid,rid,fid,ftid,hid,jtid,uid,utid,did,shift,itime,otime,duration,nhitime,nritime,idisp,odisp
    # but here we will only retain:
    #   vid,rid,ftid,hid,jtid,utid,shift,itime,otime,duration,nhitime,nritime,idisp,odisp
    # with blank values for nhitime and nritime pending recalculation.
    for row in csvin:
        # Remap the room number.
        if row['rid'] not in rmap:
            rnew = rnew + 1
            rmap[row['rid']] = rnew
        visits.append({'vid':len(visits), 'rid':rmap[row['rid']], 'ftid':row['ftid'], 'hid':row['hid'], 'jtid':row['jtid'],
                       'utid':row['utid'], 'shift':row['shift'], 'itime':row['itime'], 'otime':row['otime'], 'duration':row['duration'],
                       'nritime':'', 'nhitime':'', 'idisp':row['idisp'], 'odisp':row['odisp']})
        # Use the new structure
        row=visits[-1]

        # A shift is represented as:
        #     hid,jtid,utid,ftid,shift,stime,etime,vcount,avgvdur(m),duration(h),[v1,v2...]
        # where otime,vcount,duration, and the visit lists are
        # updated as visits accumulate, and avgvdur(m) is updated when
        # the shift closes out. Note itime,otime will eventually be
        # output as stime,etime since they correspond to shift
        # start/end and not and individual visit's in/out of room.
        idatetime = datetime.datetime.fromisoformat(row['itime'])
        odatetime = datetime.datetime.fromisoformat(row['otime'])
        if sdate is None or sdate > idatetime:
            sdate = idatetime
        if edate is None or edate < odatetime:
            edate = odatetime
        if row['hid'] in openShifts and int((idatetime - datetime.datetime.fromisoformat(openShifts[row['hid']]['otime'])).total_seconds()) <= shint:
            # This is a new visit by hcw hid within an existing shift
            # (i.e., within shint time interval). Update cumulative counters.
            openShifts[row['hid']]['vcount']=openShifts[row['hid']]['vcount'] + 1
            openShifts[row['hid']]['avgvdur']=openShifts[row['hid']]['avgvdur']+int(row['duration'])
        
            # Update existing data, extending duration.
            openShifts[row['hid']]['otime'] = row['otime']
            openShifts[row['hid']]['duration'] = int((idatetime - datetime.datetime.fromisoformat(openShifts[row['hid']]['itime'])).total_seconds())
            openShifts[row['hid']]['visits'].append(row['vid'])
        
            # Check to see if shift's updated otime conflicts with
            # DST; if so, mark appropriate day accordingly.
            if timechange(openShifts[row['hid']]['otime'][:10]):
                if openShifts[row['hid']] in days[-1]['shifts']:
                    days[-1]['dst'] = True
                else:
                    days[-2]['dst'] = True
        else: 
            # This visit is a new shift for hcw hid (beyond maxgap
            # time, or no previous visit on record).
            if row['hid'] in openShifts:
                # Update and close out existing shift. 
                openShifts[row['hid']]['avgvdur']=openShifts[row['hid']]['avgvdur']/(60*openShifts[row['hid']]['vcount'])
                shifts.append(openShifts[row['hid']])

            # Start the new shift. A shifts fields are copied from its
            # initial visit, then updated as visits accumulate. We start
            # with:
            #   hid,jtid,utid,ftid,shift,itime,otime
            # 
            # TODO: maybe trim down useless attributes in shifts....?
            openShifts[row['hid']] = {'hid':row['hid'],'jtid':row['jtid'],'utid':row['utid'],'ftid':row['ftid'],
                                      'shift':row['shift'],'itime':row['itime'],'otime':row['otime'],'duration':row['duration']}
            openShifts[row['hid']]['vcount']=1
            openShifts[row['hid']]['avgvdur']=int(row['duration'])
            openShifts[row['hid']]['visits']=[row['vid']]
            shifts.append(openShifts[row['hid']])

            # Check this shift's itime against today.
            if row['itime'][:10] != today:
                # New shift begins a new day 
                today = row['itime'][:10]
                # Days are marked as affected by DST, but otherwise
                # just contain a list of shifts.
                days.append({ 'dst':timechange(today), 'shifts':[openShifts[row['hid']]] })

                # Note days/shifts already update, hence the -1
                #print("{}: day {}, shift {}".format(today, len(days)-1, len(shifts)-1))
            else:
                # Append new shift to existing day
                days[-1]['shifts'].append(openShifts[row['hid']])

            # Also add new shift to daily (indexed by itime dow) and
            # weekly (indexed by itime weekday/weekend) as
            # appropriate.
            daily[idatetime.weekday()].add(len(days)-1)
            if idatetime.weekday() < 5:
                weekly['weekday'].add(len(days)-1)
            else:
                weekly['weekend'].add(len(days)-1)
    # Flush any shifts left in openShifts.
    for hid in openShifts.keys():
        openShifts[hid]['avgvdur']=openShifts[hid]['avgvdur']/(60*openShifts[hid]['vcount'])

    #print("{} visits in {} shifts over {} days between {} and {}".format(len(visits), len(shifts), len(days), sdate, edate))
    #print("days={}\n\nshifts={}\n\ndaily={}\n\nweekly={}".format(days, shifts, daily, weekly))
    #print("####################################################################")
    #for v in visits:
    #    print(v)
    #print("####################################################################")
    #for s in shifts:
    #    print(s)

    # Now we're ready to sample the requisite days from the list of
    # days (spanning sdate to edate). Select specific days based on
    # stype (random, daily, weekly), stday (0-6,7-8) (both from the
    # config file), as well as sdow derived, from sdate.
    sdow = sdate.weekday()
    # Handle special stday values 7 (weekday) and 8 (weekend).
    if stday == 7:	# Random weekday start
        stday = randint(0, 4)
    elif stday == 8:	# Random weekend start
        stday = randint(5, 6)
    # Make sure you filter out and DST affected days.
    if stype == 'random':
        # Randomly chosen days can be selected all at once.
        newdays = choices([ days[i] for i in range(len(days)) if not days[i]['dst'] ], k=ndays)
    elif stype == 'daily':
        newdays = []
        for i in range(ndays):
            newdays.append(choice([ days[j] for j in daily[(stday+i-sdow)%7] 
                                    if not days[j]['dst'] ]))
    elif stype == 'weekly':
        newdays = []
        for i in range(ndays):
            if (stday+i-sdow)%7 < 5:
                # DST happens only on weekends; no need to check weekdays!
                newdays.append(choice([ days[j] for j in weekly['weekday'] ]))
            else:
                newdays.append(choice([ days[j] for j in weekly['weekend']
                                        if not days[j]['dst'] ]))

    # At this point, newdays is a sequence of (possibly non-unique)
    # shifts, which contain (possibly non-unique) visits. We need to
    # relabel the dates, picking a random start date within the
    # original frame's time span (while honoring daily/weekly
    # constraints, if any), and then reify the hids.

    # Pick the target time interval from the original visits.
    offset = datetime.timedelta(days=randint(0, ((edate - sdate)).days-ndays-1))
    edate = (sdate + offset + datetime.timedelta(days=ndays)).strftime('%Y-%m-%d')
    sdate = (sdate + offset).strftime('%Y-%m-%d')
    #print("Relabel {} day interval between {} and {} [offset {}]".format(ndays, sdate, edate, offset))

    # Next, scan the original shifts from the appropriate time
    # interval and construct a hcw census for the period of time
    # corresponding to the remixed data stream. The keys are jtids,
    # the values are dictionaries of shifts...
    hnew = 0
    hmap = {} 		 # Remap hids to small ints as you go
    staff = {}		 # {'day':n, 'night':n}
    today = '0000-00-00'
    for shift in shifts:
        if shift['itime'] < sdate:
            # Discard early shifts.
            continue
        elif shift['itime'] >= edate:
            # Skip late shifts.
            break
        if shift['itime'][:10] != today:
            # First shift of the day.
            today = shift['itime'][:10]
        if shift['jtid'] not in staff:
            staff[shift['jtid']] = {}
        if shift['hid'] not in hmap.keys():
            # Remap this new hid to a small int; add to staff[shift['jtid']]
            hnew = hnew + 1
            #print('Remapping HCW {} to {}'.format(shift['hid'], hnew))
            hmap[shift['hid']] = str(hnew)
            staff[shift['jtid']][hmap[shift['hid']]] = {'day':0, 'night':0}
        if shift['shift'] == 'day':
            staff[shift['jtid']][hmap[shift['hid']]]['day'] += 1
        else:
            staff[shift['jtid']][hmap[shift['hid']]]['night'] += 1

    # Convert values to represent the probability of hcw working a
    # {day, night} on any given day.
    for jtid in staff.keys():
        for hid in staff[jtid].keys():
            staff[jtid][hid]['day'] = staff[jtid][hid]['day']/ndays
            staff[jtid][hid]['night'] = staff[jtid][hid]['night']/ndays

    # Show HR census.
    #for day in sorted(staff.keys()):
    #    print('{}: {}'.format(day, staff[day]))
    #print("=======>{}", hnew)

    # OK, now we're ready to reify the new visits. Scan the shifts in
    # each day of newdays, assigning each shift to an available HCW
    # from staff based on their observed probability of working on any
    # given day.
    #
    # If you run out of a particular type of HCW, clone a new one.
    #
    # Remember the only HW constraint we explicitly honor is that
    # night+day on consecutive days not allowed, but day + night
    # are. We could probably do better for balance or fairness of work
    # schedules. Also, night/day sequences are not unheard of but
    # probably rare. So a better approach would use probabilities to
    # model these, but for now we'll see how well this naive version
    # works.
    #
    # As we do this, we'll construct newsvisits and newshifts,
    # containing copies of the original visits and shifts so that we
    # can edit them in place.
    newshifts = []
    newvisits = []
    today = datetime.datetime.fromisoformat(sdate)
    nightwork = set()
    for day in newdays:
        #print("Today is {}".format(today))
        hmap = {}       # maps staff[hid] -> shift[hid] (reset daily)
        censor = nightwork
        nightwork = set()
        for shift in day['shifts']:
            # Start by making a copy of the shift from
            # day['shifts']. Because these may be used multiple times,
            # we'll want to update on copy of the original.
            #
            # TODO: shouldn't we be more parsimonious here? We don't
            # need 'visits', for example.
            newshifts.append(shift.copy())
            newshifts[-1]['visits']=None

            # OK, lets work on the hid first.
            #print('hid {} in {}?'.format(shift['hid'], hmap))
            if shift['hid'] not in hmap.keys():
                if shift['jtid'] not in staff:
                    staff[shift['jtid']] = {}

                # Find an as yet unassigned hid in staff with an
                # appropriate shift type open.
                #print('looking to map hid {} jtid {}'.format(shift['hid'], shift['jtid']))
                avail = [ hid for hid in staff[shift['jtid']].keys()
                               if hid not in hmap.keys() and
                                  (shift['shift'] == 'night' or hid not in censor) and
                                  random() <= staff[shift['jtid']][str(hid)][shift['shift']] ]
                #print('choose from {} entries'.format(len(avail)))
                if avail:
                    hmap[shift['hid']] = choice(avail)
                    #print(' => mapping hid {} to {}'.format(shift['hid'], hmap[shift['hid']]))
                else:
                    # Remap this new hid to a small int corresponding
                    # to a new hcw of the appropriate jtid added to
                    # staff[shift['jtid']] by clonining one of the
                    # existing hcws with a non-zero shift value (if
                    # there is one) or just resorting to a made-up hcw
                    # who works half the possible days.
                    hnew = hnew + 1
                    hmap[shift['hid']] = str(hnew)
                    available = [ entry for entry in staff[shift['jtid']].values() if entry[shift['shift']] > 0 ] 
                    if available:
                        staff[shift['jtid']][hmap[shift['hid']]] = choice(available)
                    elif shift['shift'] == 'day':
                        staff[shift['jtid']][hmap[shift['hid']]] = {'day':0.5, 'night':0.0}
                    else:
                        staff[shift['jtid']][hmap[shift['hid']]] = {'day':0.0, 'night':0.5}
                    #print(' => mapping hid {} to NEW {}'.format(shift['hid'], hmap[shift['hid']]))

            # OK, we've got the hid, so update newshifts[-1], and note
            # if this hid is working today's night shift so we don't
            # give him/her a lousy double.
            hid = newshifts[-1]['hid'] = hmap[shift['hid']]
            if shift['shift'] == 'night':
                nightwork.add(hid)

            # Next, update stime and etime.
            idelta = today - datetime.datetime.fromisoformat(shift['itime'][:10])
            newshifts[-1]['itime'] = str(datetime.datetime.fromisoformat(shift['itime']) + idelta)
            #odelta = today - datetime.datetime.fromisoformat(shift['otime'][:10])
            newshifts[-1]['otime'] = str(datetime.datetime.fromisoformat(shift['otime']) + idelta)

            # Finally, create new visits corresponding to the visits
            # in this shift. These will get sorted before being
            # dumped, so for now we construct then in shift order.
            #
            # Recall a visit has form:
            #   vid,rid,fid,ftid,hid,jtid,uid,utid,did,shift,itime,otime,duration,nhitime,nritime,idisp,odisp
            # most of which can just be copied from the visit itself.
            #
            # Like with shifts, we'll then fill in the rid, hid,
            # itime/otime upfront and then rescan to fill in nhitime
            # and nritime.
            for v in shift['visits']:
                newvisits.append(visits[v].copy())
                newvisits[-1]['vid'] = len(newvisits)   # new visits will be 1-indexed
                newvisits[-1]['hid'] = hid
                # Now fix itime and otime
                idelta = today - datetime.datetime.fromisoformat(visits[v]['itime'][:10])
                newvisits[-1]['itime'] = str(datetime.datetime.fromisoformat(visits[v]['itime']) + idelta)
                #odelta = today - datetime.datetime.fromisoformat(visits[v]['otime'][:10])
                newvisits[-1]['otime'] = str(datetime.datetime.fromisoformat(visits[v]['otime']) + idelta)

        # Update notion of today as you go.
        today = today + datetime.timedelta(days=1)

    #print("{} visits in {} shifts".format(len(newvisits), len(newshifts)))
    # Show HR census.
    #for day in sorted(staff.keys()):
    #    print('{}: {}'.format(day, staff[day]))
    #print("=======>{}", hnew)
    #print("####################################################################")
    #newvisits.sort(key=lambda x: (x['itime'],x['otime']))
    #for v in newvisits:
    #    print(v)
    #print("####################################################################")
    #for s in newshifts:
    #    print(s)

    # Now sort the shifts by stime,etime.
    newshifts.sort(key=lambda x: (x['itime'],x['jtid']))
    # Similarly, shift the visits by itime,otime.
    newvisits.sort(key=lambda x: (x['itime']+x['otime']))

    # Almost done. The last thing to do is to recompute nhitime and
    # nritime for the now sorted remixed visit data.  We'll keep a
    # dictionary cache (indexed by hid) of each hcw's last visit
    # (lastv), and a dictionary cache (indexed by rid) of each room's
    # most recent entry event (laste) so it they can be conveniently
    # accessed for modification.
    lastv = {}          # Visit indexes with as yet unknown nhitimes
    laste = {}		# Visit indexes with as yet uknown nritimes
    i = 0		# Index of visit in visits
    # Remember, nhitime is the hcw's next itime to any room, and
    # nritime is the room's next visit itime by any hcw.
    for i in range(len(newvisits)):
        # If hid had a previous visit, insert your nhitime in that record.
        if newvisits[i]['hid'] in lastv:
            newvisits[lastv[newvisits[i]['hid']]]['nhitime']=newvisits[i]['itime']
        # Cache current visit's index pending its nhitime determination
        # from a subsequent visit.
        lastv[newvisits[i]['hid']] = i

        # If rid had a previous visit, insert your nritime in that record.
        if newvisits[i]['rid'] in laste:
            newvisits[laste[newvisits[i]['rid']]]['nritime']=newvisits[i]['itime']
        # Cache current visit's index pending its nhitime determination
        # from a subsequent visit.
        laste[newvisits[i]['rid']] = i

    # Done; return both lists.
    return(newshifts, newvisits)

######################################################################
# Determines if the timestamp given represents a day when the time
# changes. The idea is to avoid using these days in a remix because
# they will have either one too few or one too many hours. Instead our
# remix days will always have exactly 24h. In the US, DST starts on
# the second Sunday of March and ends on the first Sunday of November
# (all of the original data is from the US).
#
def timechange(timestamp):
    timestamp=datetime.datetime.strptime(timestamp, "%Y-%m-%d")
    if timestamp.month == 3 and timestamp.weekday() == 6:
        # Second Sunday in March: 23h day.
        return(7 < timestamp.day <= 14)
    elif timestamp.month == 11 and timestamp.weekday() == 6:
        # First Sunday in November: 25h day.
        return(0 < timestamp.day <= 7)
    return(False)

######################################################################
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

######################################################################
# Now we can take output of pull as input on stdin
# and one optional parameter which is the name of the config file which defaults to config.cfg
# and produces on stdout a new visits file of resampled data.
if __name__ == '__main__':
    # sys.argv[0] will be the command name, and there will be either 0
    # or 1 additional argument given (configuration file name,
    # defaults to config.cfg).
#    try:
        if len(sys.argv) == 2:
            # Configuration file named
            shifts, visits = remix(sys.argv[1]);
        elif len(sys.argv) == 1:
            # Use default configuration file
            shifts, visits = remix("config.cfg");
        else:
            raise InvocationError

        sys.stderr.write("plot,hid,jtid,utid,ftid,shift,stime,etime,vcount,vmean(m),duration(h)\n")
        for s in shifts:
            sys.stderr.write("{},{},{},{},{},{},{},{},{},{:05.2f},{:05.2f}\n".format(plotshift(datetime.datetime.fromisoformat(s['itime']),
                                                                                               datetime.datetime.fromisoformat(s['otime'])),
                                                                                     s['hid'], s['jtid'], s['utid'], s['ftid'], s['shift'],
                                                                                     s['itime'], s['otime'],
                                                                                     s['vcount'], s['avgvdur'], int(s['duration'])/(60*60)))
        sys.stdout.write('vid,rid,ftid,hid,jtid,utid,shift,itime,otime,duration,nhitime,nritime,idisp,odisp\n')
        i = 0
        for v in visits:
            i = i+1
            sys.stdout.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i,v['rid'],v['ftid'],v['hid'],v['jtid'],v['utid'],v['shift'],
                                                                            v['itime'],v['otime'],v['duration'],v['nhitime'],v['nritime'],
                                                                            v['idisp'],v['odisp']))
#    except:
#        sys.stderr.write("Usage:\n  {} [config] < infile.csv > outfile.csv\nWhere:\n  config is the name of the configuration file [default=config.cfg]\n".format(sys.argv[0]))
        
