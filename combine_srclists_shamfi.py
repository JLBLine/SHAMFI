#!/usr/bin/env python
from __future__ import print_function,division
from shamfi_lib import *
import argparse
from sys import exit

gitlabel = get_gitlabel()

parser = argparse.ArgumentParser(description="Combines multiple srclists (RTS or WODEN) into one source. \
                                 Assumes that each srclist only contains one SOURCE currently")
parser.add_argument('--srclist', action='append',
    help='Any number of srclists (of either RTS or WODEN, not a combination) to combine. Repeat \
          the argument as necessary, e.g. \
          --srclist=srclist_one.txt --srclist=srclist_two.txt --srclist=srclist_three.txt')
parser.add_argument('--outname',default='srclist_combined',help='Name for output srclist - defaults to "srclist_combined"+woden or rts')

args = parser.parse_args()

##Check all the files to be combined exist
srclists = [check_file_exists(srclist,'--srclist') for srclist in args.srclist]

def get_woden_comp_nums(lines):
    '''Using the first line in lines, grab information on WODEN components'''
    _,_,_,num_point,_,num_gauss,_,num_shape,num_coeffs = lines[0].split()

    return int(num_point),int(num_gauss),int(num_shape),int(num_coeffs)

srctype,lines_base = check_rts_or_woden_get_lines(srclists[0])
print("First srclist is a %s type. Assuming they are all" %srctype)

if args.outname == 'srclist_combined':
    outname = 'srclist-%s_combined.txt' %srctype
else:
    outname = check_if_txt_extension(args.outname)

if srctype == 'woden':
    tot_num_point,tot_num_gauss,tot_num_shape,tot_num_coeffs = get_woden_comp_nums(lines_base)

    all_lines = [lines_base]
    for srclist in srclists[1:]:
        srctype,lines_extra = check_rts_or_woden_get_lines(srclist)
        num_point,num_gauss,num_shape,num_coeffs = get_woden_comp_nums(lines_extra)

        tot_num_point += num_point
        tot_num_gauss += num_gauss
        tot_num_shape += num_shape
        tot_num_coeffs += num_coeffs

        all_lines.append(lines_extra)

    with open(outname,'w+') as outfile:
        outfile.write('##Combined with SHAMFI commit %s\n' %gitlabel)
        outfile.write('SOURCE combined_woden P %d G %d S %d %d\n' %(tot_num_point,tot_num_gauss,tot_num_shape,tot_num_coeffs))

        for lines in all_lines:
            for line in lines:
                if 'SOURCE' in line or line == '':
                    pass
                else:
                    outfile.write(line + '\n')

        outfile.write('ENDSOURCE')

elif srctype == 'rts':
    all_RTS_sources = []

    for srclist in srclists:
        all_RTS_sources = get_RTS_sources(srclist, all_RTS_sources)

    write_singleRTS_from_RTS_sources(all_RTS_sources,outname,gitlabel)
