#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime
from . import symops
from . import starparse

I1Quaternions = symops.getSymOps()

def defocusAdjust( defocusU, defocusV, deltaDefocus ) :
	## From K. Zhang 2016
	## z_theta = z_u*cos^2(theta - theta_astig) + z_v*sin^2(theta - theta_astig)
	## Relion star file provides rlnDefocusU (Angst), rlnDefocusV (Angst), rlnDefocusAngle (degrees)
		## see src/metadata_label.h
		## rlnDefocusAngle = Angle between X and defocus U direction (in degrees)
		## Zhang: theta_astig "is the fixed angle between axis zu and x-axis of Cartesian coordinate system"
	## if defocusU == defocusV, theta_astig = 0, deltaZAngst = defocusOffsetU = defocusOffsetV

	### Attempt to reflect the astigmatism by applying deltaDefocus unequally to defocusU and defocusV
	defocus_ratio = np.true_divide( defocusU, defocusV )	

	## Average defocusU and defocusV
	a = np.array( [ defocusU, defocusV ] )
	defocus_average = np.average( a )

	### Changed sign of defocus adjustment on 20 Dec 2019
	correction_factor = np.true_divide( ( defocus_average + deltaDefocus ), defocus_average )

	defocusU = defocusU * correction_factor
	defocusV = defocusV * correction_factor

	defocusU = np.around(defocusU,6)	# round to 6 decimals
	defocusV = np.around(defocusV,6)	# round to 6 decimals


	return defocusU, defocusV


def reverseDefocus( input, output, FlippedDefocusU, FlippedDefocusV, FlippedXYZ_string ) :

	filename = input
	header, fullheader = starparse.getStarHeader( filename, 'null' )
	fullheader = fullheader[3:]

	a = starparse.getStarData( filename, len( header ) )

	reverse_defocus_ndarray = np.asarray( a, order='C' )

	### Get the necessary indices
	UID_index = starparse.getUID( header )
	defocusU_index, defocusV_index, defocusAngle_index = starparse.getDefocus( header )
	OriginXYZAngstWrtParticleCenter_index = starparse.getOriginXYZAngstWrtParticleCenter( header )

	for x in range(0,len(reverse_defocus_ndarray)):

		### Find the UID of the current subparticle
		my_UID = reverse_defocus_ndarray[x][ UID_index ]

		### Set the defocus equal to the flipped value from dictionary
		reverse_defocus_ndarray[x][ defocusU_index ] = FlippedDefocusU[ my_UID ]
		reverse_defocus_ndarray[x][ defocusV_index ] = FlippedDefocusV[ my_UID ]
		reverse_defocus_ndarray[x][ OriginXYZAngstWrtParticleCenter_index ] = FlippedXYZ_string[ my_UID ]

	my_output = output

	f = open( my_output, 'w' )
	np.savetxt( f, fullheader, delimiter=' ', fmt="%s" )
	f.close()

	f = open( my_output, 'a' )
	np.savetxt( f, reverse_defocus_ndarray, delimiter=' ', fmt="%s" )
	f.close()

	return


def random_subsample( starfile, ROI ) :

	desired_subparticles = int(10000)

	if ROI == 'null' :		my_sample_size = int( desired_subparticles )
	if ROI == 'fivefold' :		my_sample_size = math.ceil( desired_subparticles / 12 )
	if ROI == 'threefold' :		my_sample_size = math.ceil( desired_subparticles / 20 )
	if ROI == 'twofold' :		my_sample_size = math.ceil( desired_subparticles / 30 )
	if ROI == 'fullexpand' :	my_sample_size = math.ceil( desired_subparticles / 60 )


	if int( starfile.size ) < my_sample_size :
		sample_size = starfile.size
	else:
		sample_size = my_sample_size

	indices = np.arange( len(starfile) )
	random_indices = np.random.choice( indices, size = sample_size )

	new_starfile = starfile[random_indices]

	return new_starfile


def assess3dDistance( point1, point2 ):

	### stackoverflow.com/questions/1401712
	distance = np.linalg.norm( point1 - point2 )

	### equivalent below
#	distance = np.sqrt( ( point1[0] - point2[0] )**2 + ( point1[1] - point2[1] )**2 + ( point1[2] - point2[2] )**2 )

	return distance

def calculateAngpix( detectorPixelSize, Magnification ):
	detectorPixelSize = float( detectorPixelSize )
	magnification = float( Magnification )
	apix = np.true_divide ( detectorPixelSize, magnification ) * 10000
	return apix

def slowPrint( text ) :
	for character in text:
		sys.stdout.write( character )
		sys.stdout.flush()
		random_delay = np.true_divide( np.random.randint(2,15), 100 )
		time.sleep( random_delay )
	print( "\n" )
	return
