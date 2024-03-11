#!/usr/local/bin/python
# Copyright 2023, The University of Iowa.  All rights reserved.
# Permission is hereby given to use and reproduce this software 
# for non-profit educational purposes only.
#
# Code to pull visit data from SSense database.
# Jacob Nyberg, Madeleine Jin, Cael Elmore and Alberto Maria Segre
#
# To use:
#    % pull.py [cfile=config.cfg] > outfile.csv
# where config.cfg contains the query configuration:
#   stime: timestamp
#   etime: timestamp
#   shift: day|night
#   unit:  u-u,u... 
#   dept:  d-d,d... 
#   jtype: j-j,j... 
#   facilities: f-f,f...
# where missing or blank lines default to all possible values.
#
# Also allows aliases (alternate forms of keywords):
#   fid = facilities
#   uid = unit
#   jtid = jtype
#   did = dept
# Ignores unrecoginzed configuration file directives.
#
import sys
import csv
import pymysql
from datetime import datetime
from os import access, R_OK
from os.path import isfile

# Takes a range specification on integers, consisting of any
# combination of comma separated individual values and
# hyphen-separated integer ranges and returns a list of SQL constraint
# fragments which, when applied to the variable in question, describe
# the range of integers. So, for example, for range specification:
#   1,2,5-8,12
# sqlRange() will return
#   ['BETWEEN 5 AND 8', 'IN (1, 2, 12)']
# which can be expressed in SQL as:
#   (x BETWEEN 5 AND 8 OR x IN (1, 2, 12))
#
def sqlRange(string):
    expr = []
    enum = []
    for entry in [ entry.strip() for entry in string.strip().split(',') ]:
        if '-' in entry:
            expr.append('BETWEEN {} AND {}'.format(int(entry.split('-')[0]),int(entry.split('-')[1])))
        else:
            enum.append(int(entry))
    if len(enum)>1:
        return(expr + ['IN {}'.format(tuple(enum))])
    elif len(enum)==1:
        return(expr + ['= {}'.format(enum[0])])
    else:
        return(expr)

# Generates and executes a mySQL query for the ssense database that
# returns information from the database in regards to visits.  The
# specific attributes and values within are extracted via text file,
# who's name or path is the only parameter.  The file should be
# formatted where each line denotes a specification of what attribute
# you want and what value you want from the attribute.
#
# Ignores lines starting with # or any trailing comments after #.
#
def pull(cfile):
    # Check if configuration file exists.
    assert isfile(cfile) and access(cfile, R_OK), "File {} doesn't exist or isn't readable".format(cfile)

    # Parse the configuration file. 
    stime=etime=shift=units=depts=jtypes=facilities=None
    with open(cfile, 'r') as cfile:
        for line in cfile:
            # Skip any blank or comment lines
            if line.strip() == '' or line.lstrip()[0] == '#':
                continue
            line=[ element.strip() for element in line.lower().split(':') ]
            if line[0] == 'stime':
                stime=(':'.join(line[1:])).split('#')[0].strip()
            elif line[0] == 'etime':
                etime=(':'.join(line[1:])).split('#')[0].strip()
            elif line[0] == 'shift':
                shift=line[1].split('#')[0].strip()
            elif line[0] == 'units' or line[0] == 'uid':
                units=sqlRange(line[1].split('#')[0].strip())
            elif line[0] == 'depts' or line[0] == 'did':
                depts=sqlRange(line[1].split('#')[0].strip())
            elif line[0] == 'jtypes' or line[0] == 'jtid':
                jtypes=sqlRange(line[1].split('#')[0].strip())
            elif line[0] == 'facilities' or line[0] == 'fid':
                facilities=sqlRange(line[1].split('#')[0].strip())
            #else:
            #    sys.stderr.write("-- Ignoring unrecognized configuration file entry: {}\n".format(line))

    # Now build the query a little bit at a time. Later, we might want
    # to build an alternate, more human readable, version of this
    # query.
    #   v.vid, v.rid, r.fid, v.hid, j.jtid, r.uid, h.did, v.shift, v.itime, v.otime, v.duration, v.idisp, v.odisp 
    query = "SELECT v.vid as vid, v.rid as rid, r.fid as fid, f.ftid as ftid, v.hid as hid, j.jtid as jtid,\n"
    query+= "       r.uid as uid, u.utid as utid, h.did as did, v.shift as shift, v.itime as itime,\n"
    query+= "       v.otime as otime, v.duration as duration, v.idisp as idisp, v.odisp as odisp\n"
    query+= "FROM visits v, rooms r, units u, hcws h, jobs j, facilities f\n"
    query+= "WHERE v.rid=r.rid AND r.uid=u.uid AND v.hid=h.hid AND h.jid=j.jid AND r.fid=f.fid\n{}"
    query+= "ORDER BY fid, itime, otime;\n"
    # Associated csv output header: note nhitime is the hcw's next
    # itime (to any room, for determining end-of-shift), and nritime
    # is the room's next visit itime (by any hcw, for determining
    # time-to-next-visit).
    header= "vid,rid,fid,ftid,hid,jtid,uid,utid,did,shift,itime,otime,duration,nhitime,nritime,idisp,odisp\n"

    # Let's construct the constraint expression from the configuration
    # file that specializes the WHERE clause of the SELECT.
    constraints=''
    if stime is not None and etime is not None:
        # Use between 
        constraints=constraints + '      AND v.itime BETWEEN "{}" AND "{}"\n'.format(datetime.fromisoformat(stime).isoformat(' '),
                                                                                 datetime.fromisoformat(etime).isoformat(' '))
    elif stime is not None:
        # Need to convert stime to a MySQL compatible timestamp.
        constraints=constraints + '      AND v.itime>="{}"\n'.format(datetime.fromisoformat(stime).isoformat(' '))
    elif etime is not None:
        # Need to convert etime to a MySQL compatible timestamp.
        constraints=constraints + '      AND v.itime<="{}"\n'.format(datetime.fromisoformat(etime).isoformat(' '))
    if shift is not None:
        # Specify shift type
        constraints=constraints + '      AND v.shift="{}"\n'.format(shift)
    if facilities is not None:
        constraints=constraints + '      AND ({})\n'.format(' OR '.join([ 'r.fid ' + str(spec) for spec in facilities ]))
    if jtypes is not None:
        constraints=constraints + '      AND ({})\n'.format(' OR '.join([ 'j.jtid ' + str(spec) for spec in jtypes ]))
    if depts is not None:
        constraints = constraints + '      AND ({})\n'.format(' OR '.join([ 'h.did ' + str(spec) for spec in depts ]))
    if units is not None:
        constraints = constraints + '      AND ({})\n'.format(' OR '.join([ 'r.uid ' + str(spec) for spec in units ]))

    # Connect to database, input query, read output as database, save as a tsv
    connection = pymysql.connect(host="localhost", database='ssense', user='uihc', password='uihc')
    cursor = connection.cursor()
    cursor.execute(query.format(constraints))

    # Process each visit; accumulating visit records chronologically
    # in visit.
    #
    # As you add visits, you will need to add this hcw's itime as the
    # nhitime of hcw's previous visit. We keep a dictionary cache
    # (indexed by hid) of each hcw's last visit (lastv) so it can be
    # conveniently accessed for modification.
    #
    # In a similar fashion, you will need to add this room's itime as
    # the nritime of the room's previous visit. We keep a simlar
    # dictionary cache (indexed by rid) of each room's most recent
    # entry event (laste) so it can be conveniently accessed for
    # modification.
    #
    lastv = {}          # Visit indexes with as yet unknown nhitimes
    laste = {}		# Visit indexes with as yet uknown nritimes
    visits = []		# List of all visits
    i = 0		# Index of visit in visits
    # Process a visit at a time.
    visit = cursor.fetchone()
    while visit:
        # Keep visits, as lists (not tuples) in the visits
        # list. You'll need to add initially blank fields for nhitime
        # and nritime.
        visits.append(list(visit[:13] + ('','',) + visit[13:])) # Add to visits list

        # If hid had a previous visit, insert your nhitime in that record.
        if visit[4] in lastv:  
            visits[lastv[visit[4]]][13]=visit[10]
        # Cache current visit's index pending its nhitime determination
        # from a subsequent visit.
        lastv[visit[4]] = i

        # If rid had a previous visit, insert your nritime in that record.
        if visit[1] in laste:
            visits[laste[visit[1]]][14]=visit[10]
        # Cache current visit's index pending its nhitime determination
        # from a subsequent visit.
        laste[visit[1]] = i
            
        # Process next visit.
        visit = cursor.fetchone()
        i = i + 1
    # Done with query results.
    connection.close()

    # Now you can dump the nhitime,nritime-augmented list of visits.
    sys.stdout.write(header)
    writer = csv.writer(sys.stdout);
    writer.writerows(visits)

    # Write query to STDERR in SQL comment format
    sys.stderr.write("-- {} total entries retrieved\n".format(i))
    sys.stderr.write(query.format(constraints))
    # Exit
    return()

if __name__ == '__main__':
    # sys.argv[0] will be the command name, and there will be either 0
    # or 1 additional argument given (configuration file name,
    # defaults to config.cfg).
#    try:
        if len(sys.argv) == 2:
            # Configuration file named
            pull(sys.argv[1])
        elif len(sys.argv) == 1:
            # Default behavior
            pull("config.cfg");
        else:
            raise InvocationError
#    except:
#         sys.stderr.write("Usage:\n  {} [config] > outfile.csv 2> config.sql\nWhere:\n  config is the name of the configuration file [default=config.cfg]\n".format(sys.argv[0]))
