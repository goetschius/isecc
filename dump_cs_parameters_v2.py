#!/usr/bin/env python3.5
#
########
#
# This script dumps the parameters for the first item in a .cs file
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

import argparse
import sys
import os
import numpy as np

def dump_parameters(csfile) : 

	print( "\nDumping parameters from input:\n" )

	print( "Fields are:", csfile.dtype.fields, "\n" )

#	if int( csfile[0]['blob/sign'] ) == int( -1 ):       # relion standard
#		print( "Data sign is:", int( -1 ), ": relion standard!\n" )
#	elif int( csfile[0]['blob/sign'] ) == int( 1 ):       # cryosparc standard
#		print( "Data sign is:", int( 1 ), ": cryosparc standard!\n" )

	print( "### Parameters for particle 0 are below: ###" )
	for item in csfile.dtype.fields :
		print( item, ":", csfile[0][item] )
	print( "\n" )

	print( "### Parameters for particle 1 are below: ###" )
	for item in csfile.dtype.fields :
		print( item, ":", csfile[1][item] )
	print( "\n" )



def main(args):

	if args.input.endswith(".cs"):
		cs = np.load(args.input)
		dump_parameters(cs)
	else:
		print( "Please provide a valid input file." )
		sys.exit()

	return 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="path to cryosparc_P#_J#_###_particles.cs file")
	sys.exit(main(parser.parse_args()))

