#!/usr/bin/env python
from __future__ import print_function,division
from shamfi_lib import *
from shamfi.git_helper import *
import argparse
from numpy import array,where,arange

parser = argparse.ArgumentParser(description="Converts a srclist between RTS or WODEN formats \
                                 Should automatically detect if base catalouge is RTS or WODEN format. \
                                 Currently assumes only one SOURCE is in the srclist i.e. an \
                                 output from shamfi.py or combine_srclists_shamfi.py")

parser.add_argument('--srclist', default=False,
    help='srclist to be converted')
parser.add_argument('--rts2woden', default=False, action='store_true',
    help='Optional - specify to convert from RTS to WODEN type srclist')
parser.add_argument('--woden2rts', default=False, action='store_true',
    help='Optional - specify to convert RTS type srclist')
parser.add_argument('--outname',default='convert',
    help='Name for output srclist - defaults to "srclist_coverted"+woden or rts')

args = parser.parse_args()

##Check file exists
srclist = check_file_exists(args.srclist,'--srclist')

##Check if RTS or WODEN
type,lines = check_rts_or_woden_get_lines(srclist)

##Override automatic
if args.woden2rts:
    type = 'woden'
elif args.rts2woden:
    type = 'rts'

##Convert to the opposite format
if type == 'rts':
    convert_type = 'woden'
elif type == 'woden':
    convert_type = 'rts'

print('Assuming srclist is a %s type, will convert to %s' %(type,convert_type))

##If specfied, make a name, if not, make a generic one
if args.outname == 'convert':
    outname = 'srclist_converted-%s.txt' %convert_type
##Check specified name is a text file, correct if not
else:
    outname = check_if_txt_extension(args.outname)

##If a WODEN, convert to an RTS
if type == 'woden':
    with open(outname,'w+') as outfile:

        write_git_header(outfile)

        _,name,_,_,_,_,_,_,_ = lines[0].split()
        ##The RTS only allows names up to 16 characters long so clip here
        name = name[:16]

        ##Split into COMPONENTS
        lines = array(lines[1:])
        comp_ends = where(lines == 'ENDCOMPONENT')[0]

        ##Write first component out as the SOURCE part of the RTS srclist
        write_woden_component_as_RTS(lines[:comp_ends[0]], outfile, name = name)

        ##For the rest, write as COMPONENTS
        for ind in arange(len(comp_ends) - 1):
            write_woden_component_as_RTS(lines[comp_ends[ind]:comp_ends[ind+1]], outfile)

        outfile.write('ENDSOURCE')
    # outfile.close()

##If RTS, convert to WODEN
elif type == 'rts':
    ##Parse and format the RTS srclist into the RTS_source class
    all_RTS_sources = get_RTS_sources(args.srclist, [])
    write_woden_from_RTS_sources(all_RTS_sources, outname)
