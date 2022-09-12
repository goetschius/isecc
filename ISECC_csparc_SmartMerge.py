#!/usr/bin/env python3.5
#
########
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
########
#
# This program generates subparticles from cryosparc v2.X .cs files
#
# Works only on I1 alignment files from cryosparc
# Idealized Fivefold:	0.000, 0.618, 1.000
# Idealized Threefold:	0.382, 0.000, 1.000
# Idealized Twofold:	0.000, 0.000, 1.000
#
#	NOTE: Chimera command "getcrd sel" reports coordinates in Angstroms
#	      Vector will be automatically converted to voxels
#
# Daniel Goetschius, 18 Oct 2019
#
# 
########

import argparse
import sys
import os
import math
import numpy as np
from pyquaternion import Quaternion

print( "\nATTENTION:" )
print( "This script is included purely for historical purposes only. It is unmaintained and I DO NOT recommend using it." )
print( "Last updated on 2019-11-07.\n" )

def remove_field_name(a, name):                 # stackoverflow.com/questions/15575878
	names = list(a.dtype.names)
	if name in names:
		names.remove(name)
	b = a[names]
	return b

def add_field(a, descr):        # stackoverflow.com/questions/1201817
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    """
    if descr in a.dtype.descr:
        print( descr, " is already in file!")
        return a

    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")

    extended_dtype = a.dtype.descr + [descr]

    b = np.empty(a.shape, dtype=extended_dtype)
    for name in a.dtype.names:
        b[name] = a[name]
    print("Added field", descr, "to file")
    return b


## This function merges two csfiles
def smart_merge(cs_1, cs_2):

    ### Removing some stuff as a test
    remove_list = [ 'alignments3D/alpha', 'alignments3D/alpha_min' ]
    print("testing")
    for item in remove_list:
        try:
            cs_1 = remove_field_name(cs_1, item)
        except:
            pass
    print("test")

    ### I know that I want the following fields in the output file
    desired_fields = np.type=[('alignments3D/alpha', '<f4'), ('alignments3D/alpha_min', '<f4'), ('alignments3D/class_ess', '<f4'), ('alignments3D/class_posterior', '<f4'), ('alignments3D/class', '<u4'), ('alignments3D/cross_cor', '<f4'), ('alignments3D/error', '<f4'), ('alignments3D/error_min', '<f4'), ('alignments3D/image_pow', '<f4'), ('alignments3D/pose_ess', '<f4'), ('alignments3D/pose', '<f4', (3,)), ('alignments3D/psize_A', '<f4'), ('alignments3D/resid_pow', '<f4'), ('alignments3D/shift_ess', '<f4'), ('alignments3D/shift', '<f4', (2,)), ('alignments3D/slice_pow', '<f4'), ('alignments3D/split', '<u4'), ('alignments3D/weight', '<f4'), ('blob/idx', '<u4'), ('blob/import_sig', '<u8'), ('blob/path', 'S60'), ('blob/psize_A', '<f4'), ('blob/shape', '<u4', (2,)), ('blob/sign', '<f4'), ('ctf/accel_kv', '<f4'), ('ctf/amp_contrast', '<f4'), ('ctf/anisomag', '<f4', (4,)), ('ctf/bfactor', '<f4'), ('ctf/cs_mm', '<f4'), ('ctf/df1_A', '<f4'), ('ctf/df2_A', '<f4'), ('ctf/df_angle_rad', '<f4'), ('ctf/exp_group_id', '<u4'), ('ctf/phase_shift_rad', '<f4'), ('ctf/scale_const', '<f4'), ('ctf/scale', '<f4'), ('ctf/shift_A', '<f4', (2,)), ('ctf/tetra_A', '<f4', (4,)), ('ctf/tilt_A', '<f4', (2,)), ('ctf/trefoil_A', '<f4', (2,)), ('ctf/type', 'S9'), ('uid', '<u8'), ('location/micrograph_uid', '<u8'), ('location/exp_group_id', '<u4'), ('location/micrograph_path', 'S141'), ('location/micrograph_shape', '<u4', (2,)), ('location/center_x_frac', '<f4'), ('location/center_y_frac', '<f4')]

    ### These are the fields that already exist in the inputs
    input1_fieldnames = list(cs_1.dtype.names)
    input2_fieldnames = list(cs_2.dtype.names)

    ### These fields are unique to cs_1
    cs_1_unique = list(set(input1_fieldnames) - set(input2_fieldnames))
    print("Unique to cs_1", cs_1_unique)
    ### And these fields are unique to cs_2
    cs_2_unique = list(set(input2_fieldnames) - set(input1_fieldnames))
    print("Unique to cs_2", cs_2_unique)
    ### And these are the shared/common fieldnames
    cs_1_2_shared = list(np.intersect1d(input1_fieldnames, input2_fieldnames))
    print("Common to both", cs_1_2_shared)



    ### Sort the input files by uid
    cs_1.sort(order='uid')
    cs_2.sort(order='uid')
#    print((cs_1['uid']==cs_2['uid]').all())
    print( np.array_equal( cs_1['uid'], cs_2['uid'] ) )

    ### Copy the original input 1 as the basis for our output
    cs_new = np.copy(cs_1)

    ### Start adding in the necessary fields
    for item in desired_fields:
        try:
            cs_new = add_field(cs_new, item)
        except:
            print("Couldn't add", item, ". It probably already exists.")

    print(cs_new[0])
    ### Copy over the data entries
    print("Attempting to copy items from input 1")
    for item in desired_fields:
        try:
            cs_new[item[0]] = cs_1[item[0]]
        except:
            pass

    print("Attempting to copy items from input 2")
    for item in desired_fields:
        try:
            cs_new[item[0]] = cs_2[item[0]]
        except:
            pass

    try:
        print(cs_1[0]['alignments3D/pose'], "\n")
    except:
        pass
    try:
        print(cs_2[0]['alignments3D/pose'], "\n")
    except:
        pass
    print("\n", cs_new.dtype.fields, cs_new[0], "\n")

    np.save("merged_csfile.cs", cs_new)

    return

def main(args):

	if args.input_1.endswith(".cs") and args.input_2.endswith(".cs"):
		cs_1 = np.load(args.input_1)
		cs_2 = np.load(args.input_2)
		smart_merge(cs_1, cs_2)
	else:
		print( "Please provide valid input files." )
		sys.exit()

	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_1", "-i1", required=True, help="first .cs file")
	parser.add_argument("--input_2", "-i2", required=True, help="second .cs file")
	sys.exit(main(parser.parse_args()))

