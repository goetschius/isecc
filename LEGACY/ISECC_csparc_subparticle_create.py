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

def random_subsample( generic_csfile, generic_ptfile, ROI ) :

	if ROI == 'null' or ROI == 'just_unbin' : my_sample_size = int( 10000 )
	if ROI == 'fivefold' :		my_sample_size = math.ceil( 10000 / 12 )
	if ROI == 'threefold' :		my_sample_size = math.ceil( 10000 / 20 )
	if ROI == 'twofold' :		my_sample_size = math.ceil( 10000 / 30 )

	if int( generic_csfile.size ) < my_sample_size :
		sample_size = generic_csfile.size
	else:
		sample_size = my_sample_size

	new_csfile = np.random.choice( generic_csfile, size = sample_size, replace = False )
	new_ptfile = np.random.choice( generic_ptfile, size = sample_size, replace = False )
	## Never use this new ptfile for anything real ##
	return new_csfile, new_ptfile

def remove_field_name(a, name):                 # stackoverflow.com/questions/15575878
	names = list(a.dtype.names)
	if name in names:
		names.remove(name)
	b = a[names]
	return b

def quat2aa(q):         # from pyQuaternion
	my_quaternion = Quaternion(q)		# pyQuaternion
	ax = my_quaternion.get_axis()
	theta = my_quaternion.radians
	return theta * ax

def aa2quat(ax, theta=None):			# calls pyQuaternion after determining theta
	if theta is None:				# pyem/0.4, Daniel Asarnow
		theta = np.linalg.norm(ax)
		if theta != 0:
			ax = ax / theta
	q = Quaternion(axis=ax, angle=theta)	# pyQuaternion
	return q

def defineAreaOfInterest(ROI, user_vector, user_fudge, higher_order_sym):
	area_of_interest = ROI
	my_vector = user_vector
	my_vector = my_vector * user_fudge		# allows you to bump vector in or out a bit
	my_sym = higher_order_sym	

	checkVector(user_vector, ROI, higher_order_sym)

	## If I2, swap x and y
	## This is required as script was written for I1 default
	if higher_order_sym == 'I2':
		my_vector = np.array( [ my_vector[1], my_vector[0], my_vector[2] ] )

	if area_of_interest == "just_unbin" :
		sym_index = np.array( [ 0 ] )
		my_arctan = 0
		quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
		model_sym = 'c1'
		my_vector = np.array( [0, 0, 0] )

	if area_of_interest == "null" :
		sym_index = np.array( [ 0 ] )
		my_arctan = 0
		quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
		model_sym = 'c1'

	if area_of_interest == "fivefold" :
		sym_index = np.array( [ 0,1,2,3,4,5,6,7,9,11,12,31 ] )
		### Ideal vector is 0.000, 0.618, 1.000
		my_arctan = np.arctan( np.true_divide( my_vector[1], my_vector[2] ) )
		quaternion_toZ = Quaternion(axis=(1.0,0.0,0.0), radians=my_arctan)
		model_sym = 'c5'

	if area_of_interest == "threefold" :
		sym_index = np.array( [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22 ] )
		### Ideal vector is 0.382, 0.000, 1.000
		my_arctan = np.arctan( np.true_divide( my_vector[0], my_vector[2] ) )
		quaternion_toZ = Quaternion(axis=(0.0,-1.0,0.0), radians=my_arctan)
		model_sym = 'c3'

	if area_of_interest == "twofold" :
		sym_index = np.array( [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,28,29,30,31,32 ] )
		### Ideal vector is 0.000, 0.000, 1.000
		my_arctan = 0
		quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
		model_sym = 'c2'

	if area_of_interest == "fullexpand" :
		sym_index = np.arange( 60 ) 
		my_vector = 1.0 * my_vector
#		my_arctan_vert = np.arctan( np.true_divide( my_vector[1], my_vector[2] ) )	### From FIVEFOLD
#		my_arctan_horiz = np.arctan( np.true_divide( my_vector[0], my_vector[2] ) )	### From THREEFOLD
#		my_inplane_angle = 0								### optional in-plane rotation
		my_arctan = 0
		quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
#		quaternion_to_vert = Quaternion(axis=(1.0,0.0,0.0), radians=my_arctan_vert)  	### From FIVEFOLD
#		quaternion_to_horiz = Quaternion(axis=(0.0,-1.0,0.0), radians=my_arctan_horiz)	### From THREEFOLD
#		quaternion_in_plane = Quaternion(axis=(0.0, 0.0, 1.0), radians=my_inplane_angle )	### optional in-plane rotation
#		quaternion_toZ = quaternion_to_horiz * quaternion_to_vert			### Combine the two rotations
#		quaternion_toZ = quaternion_in_plane * quaternion_toZ				### Apply in-plane rotation
		model_sym = 'c1'

	return { 'area_of_interest': area_of_interest, 'sym_index': sym_index, 'my_vector': my_vector, 'quaternion_toZ': quaternion_toZ, 'model_sym': model_sym }


def checkVector(user_vector, ROI, higher_order_sym):
	if ROI == 'fivefold' and higher_order_sym == 'I1' :
		vector_check = np.true_divide( user_vector[1], user_vector[2] )
		my_check = np.isclose( vector_check, 0.618, atol=0.01 )
		my_check2 = np.isclose( user_vector[0], 0, atol=0.01 )

	elif ROI == 'fivefold' and higher_order_sym == 'I2' :
		vector_check = np.true_divide( user_vector[0], user_vector[2] )
		my_check = np.isclose( vector_check, 0.618, atol=0.01 )
		my_check2 = np.isclose( user_vector[1], 0, atol=0.01 )

	elif ROI == 'threefold' and higher_order_sym == 'I1' :
		vector_check = np.true_divide( user_vector[0], user_vector[2] )
		my_check = np.isclose( vector_check, 0.382, atol=0.01 )
		my_check2 = np.isclose( user_vector[1], 0, atol=0.01 )

	elif ROI == 'threefold' and higher_order_sym == 'I2' :
		vector_check = np.true_divide( user_vector[1], user_vector[2] )
		my_check = np.isclose( vector_check, 0.382, atol=0.01 )
		my_check2 = np.isclose( user_vector[0], 0, atol=0.01 )

	elif ROI == 'twofold' and higher_order_sym == 'I1' :
		my_check = np.isclose( user_vector[0], 0, atol=0.01 )
		my_check2 = np.isclose( user_vector[1], 0, atol=0.01 )

	elif ROI == 'twofold' and higher_order_sym == 'I2' :
		my_check = np.isclose( user_vector[0], 0, atol=0.01 )
		my_check2 = np.isclose( user_vector[1], 0, atol=0.01 )

	elif ROI == 'fullexpand' or ROI == 'null' or ROI == 'just_unbin' :
		my_check = True
		my_check2 = True

	if str(my_check) == 'False' or str(my_check2) == 'False':
		print( 'ERROR: Vector is not valid for roi', ROI, 'and symmetry', higher_order_sym )
		print( '       Please provide a valid vector and try again.\n' )

		print( '  For I1 symmetry,' )
		print( '    Idealized Fivefold:   0.000, 0.618, 1.000')
		print( '    Idealized Threefold:  0.382, 0.000, 1.000')
		print( '    Idealized Twofold:    0.000, 0.000, 1.000\n')

		print( '  For I1 symmetry,' )
		print( '    Idealized Fivefold:   0.618, 0.000, 1.000')
		print( '    Idealized Threefold:  0.000, 0.382, 1.000')
		print( '    Idealized Twofold:    0.000, 0.000, 1.000\n')

		sys.exit()

	elif str(my_check) == 'True' or str(my_check2) == 'True':
		print( "Vector is valid for roi", ROI, "and symmetry", higher_order_sym, "\n" )
		
	return


def applyParameters(generic_csfile, generic_ptfile, x, reference_csfile, new_pose_as_aa, local_defocus1_A, local_defocus2_A, rotated_vector, ROI, BATCH_MODE, padded_particle_number):

	## Apply Pose
	generic_csfile[x]['alignments3D/pose'] = new_pose_as_aa   # the pose, i.e. orientations

	## Apply Defocus
	generic_csfile[x]['ctf/df1_A'] = local_defocus1_A         # correct local defocus
	generic_csfile[x]['ctf/df2_A'] = local_defocus2_A         # correct local defocus

	## Apply Offsets
	generic_csfile[x]['alignments3D/shift'][0] = reference_csfile[x]['alignments3D/shift'][0] - rotated_vector[0]	# the x-coord
	generic_csfile[x]['alignments3D/shift'][1] = reference_csfile[x]['alignments3D/shift'][1] - rotated_vector[1]	# the y-coord
	if 'location/micrograph_path' in generic_ptfile.dtype.names:
		generic_ptfile[x]['location/micrograph_path'] = ''.join( ['subparticles/', ROI, '/particle', padded_particle_number, '.mrc'] )
		if BATCH_MODE:
			generic_ptfile[x]['location/micrograph_path'] = ''.join( ['subparticles/', ROI, '/batch', padded_particle_number, '.mrc'] )

	return 


def checkDataSign( a, b=None ):		# Checks whether datasign is relion or cryoSPARC standard
	if int( a ) == int( 1 ):
		b = int( 1 )
		print( "Data sign is positive:", b, ": which is cryoSPARC standard.\n" )
	elif int( a ) == int( -1 ):
		b = int( -1 )
		print( "Data sign is negative :", b, ": which is relion standard.\n" )
	else:
		print( "Can't parse data sign:", a )
		sys.exit()
	return b


### Generate array containing I1 rotations ready for pyQuaternion in format [ a, bi, cj, dk ]
I1Quaternions = np.array(   [   [ 1.000, 0.000, 0.000, 0.000 ],
				[ 0.000, 1.000, 0.000, 0.000 ],
				[ 0.809, -0.500, 0.000, 0.309 ],
				[ -0.309, 0.809, 0.000, -0.500 ],
				[ 0.309, 0.809, 0.000, -0.500 ],
				[ 0.809, 0.500, 0.000, -0.309 ],
				[ -0.500, 0.809, 0.309, 0.000 ],
				[ 0.500, 0.809, 0.309, 0.000 ],
				[ 0.500, 0.809, -0.309, 0.000 ],
				[ 0.809, 0.309, -0.500, 0.000 ],
				[ 0.809, 0.309, 0.500, 0.000 ],
				[ 0.809, -0.309, -0.500, 0.000 ],
				[ 0.809, -0.309, 0.500, 0.000 ],
				[ -0.500, 0.809, -0.309, 0.000 ],
				[ 0.000, 0.809, 0.500, -0.309 ],
				[ 0.500, 0.500, 0.500, -0.500 ],
				[ 0.809, 0.000, 0.309, -0.500 ],
				[ 0.809, -0.500, 0.000, -0.309 ],
				[ 0.809, 0.500, 0.000, 0.309 ],
				[ -0.500, 0.500, 0.500, -0.500 ],
				[ 0.809, 0.000, -0.309, 0.500 ],
				[ 0.809, 0.000, 0.309, 0.500 ],
				[ -0.500, 0.500, -0.500, -0.500 ],
				[ 0.000, 0.809, -0.500, -0.309 ],
				[ -0.309, 0.809, 0.000, 0.500 ],
				[ 0.809, 0.000, -0.309, -0.500 ],
				[ 0.500, -0.309, 0.000, 0.809 ],
				[ 0.000, -0.500, 0.309, 0.809 ],
				[ 0.500, 0.500, -0.500, -0.500 ],
				[ -0.309, -0.500, 0.809, 0.000 ],
				[ 0.000, 0.809, -0.500, 0.309 ],
				[ 0.309, 0.809, 0.000, 0.500 ],
				[ -0.500, 0.500, 0.500, 0.500 ],
				[ 0.000, 0.809, 0.500, 0.309 ],
				[ 0.309, 0.500, 0.809, 0.000 ],
				[ 0.000, -0.500, -0.309, 0.809 ],
				[ -0.500, -0.309, 0.000, 0.809 ],
				[ -0.500, 0.000, 0.809, 0.309 ],
				[ -0.309, 0.500, 0.809, 0.000 ],
				[ -0.500, 0.000, 0.809, -0.309 ],
				[ 0.500, 0.500, -0.500, 0.500 ],
				[ 0.500, 0.500, 0.500, 0.500 ],
				[ 0.500, 0.000, 0.809, 0.309 ],
				[ 0.309, -0.500, 0.809, 0.000 ],
				[ 0.500, 0.000, 0.809, -0.309 ],
				[ -0.500, 0.500, -0.500, 0.500 ],
				[ 0.000, 0.309, 0.809, -0.500 ],
				[ -0.309, 0.000, -0.500, 0.809 ],
				[ -0.500, 0.309, 0.000, 0.809 ],
				[ 0.309, 0.000, -0.500, 0.809 ],
				[ 0.500, 0.309, 0.000, 0.809 ],
				[ 0.309, 0.000, 0.500, 0.809 ],
				[ 0.000, -0.309, 0.809, 0.500 ],
				[ 0.000, 0.000, 0.000, 1.000 ],
				[ -0.309, 0.000, 0.500, 0.809 ],
				[ 0.000, 0.500, 0.309, 0.809 ],
				[ 0.000, 0.309, 0.809, 0.500 ],
				[ 0.000, -0.309, 0.809, -0.500 ],
				[ 0.000, 0.500, -0.309, 0.809 ],
				[ 0.000, 0.000, 1.000, 0.000 ] ] )

def generate_subparticle_csfile(csfile, ROI, user_vector, user_fudge, user_subbox, higher_order_sym, user_testmode, user_batch_size, user_batch=None, passthrough=None):
	print( "\nCSPARC SUBPARTICLE DEFINE v 2019.10.18.1" )
	print( "This program will define subparticle centers and poses in cryoSPARC format\n" )
	print( "  Note: Vector must be supplied in Angstroms." )
	print( "  Note: User indicated that refinement was done in", higher_order_sym, "orientation.\n" )
	if user_testmode:
		print( "  Note: Running in test mode. Only ~10k subparticles will be generated.\n" )
		
	cs = csfile if type(csfile) is np.ndarray else np.load(csfile)
	pt = passthrough if type(passthrough) is np.ndarray else np.load(passthrough)

	if user_testmode:
		cs, pt = random_subsample( cs, pt, ROI )
		pt = cs

	BATCH_MODE = None
	if user_batch == True:
		BATCH_MODE = True
		batch_size = user_batch_size
		print( "  Note: Batch mode will be used to speed subparticle generation in relion." )
		print( "  Note: Requested batch size is", batch_size, "\n" )

	if ROI == 'just_unbin':
		## 'just_unbin' should maintain unbinned box size
		print( "  User has indicated to only correct for OTF downsampling during refinement!" )
		print( "  NOTE: User specified box:", user_subbox, user_subbox )

		blob_shape = cs[0]['blob/shape']        # e.g. [800 800]
		user_subbox = int(blob_shape[0])       # e.g. [800]
		print( "  NOTE: Will correct to:", user_subbox, user_subbox, "\n" )

	if 'location/micrograph_path' not in pt.dtype.fields:
		## If the field 'location/micrograph_path' is not in ptfile, we need to add it 
		## See stackoverflow.com/questions/25427197
		new_dt = np.dtype(pt.dtype.descr + [('location/micrograph_path', 'S100')])
		updated_pt = np.zeros( pt.shape, dtype=new_dt, order='C' )
		names = list(pt.dtype.names)
		for name in names:
			updated_pt[name] = pt[name]
		pt = updated_pt


	if ".mrcs" not in str( cs[0]['blob/path'] ):
		## Make new array with greater string length if .mrc rather than .mrcs
		cs_restring = cs.copy(order='C')
		cs_restring = remove_field_name(cs_restring, 'blob/path')

		my_dtype = str( cs['blob/path'].dtype ) 
		new_string_length = int( my_dtype.split('S')[1] ) + 1
		new_string_dtype = str( ''.join( [ 'S', str( new_string_length ) ] ) ) 
		new_dt = np.dtype(cs_restring.dtype.descr + [('blob/path', new_string_dtype)])		

		updated_cs = np.zeros( cs_restring.shape, dtype=new_dt, order='C' )
		names = list(cs.dtype.names)
		for name in names:
			updated_cs[name] = cs[name]
		cs = updated_cs

		## replace mrc with mrcs
		for index in range(cs.size):
			index = int(index)
			cs[index]['blob/path'] = cs[index]['blob/path'].replace(b'.mrc', b'.mrcs')


	PRUNE_FIELDS = True
	if PRUNE_FIELDS:
		### Remove data fields that may be problematic re: 2d classification
		remove_list = [ 'alignments2D/split', 'alignments2D/shift', 'alignments2D/shift_ess', 'alignments2D/pose', 'alignments2D/psize_A', 'alignments2D/error', 'alignments2D/error_min', 'alignments2D/resid_pow', 'alignments2D/slice_pow', 'alignments2D/image_pow', 'alignments2D/cross_cor', 'alignments2D/alpha', 'alignments2D/weight', 'alignments2D/pose_ess', 'alignments2D/pose_ess', 'alignments2D/class_posterior', 'alignments2D/class', 'alignments2D/class_ess', 'motion/psize_A' ]

		for item in remove_list:
			cs = remove_field_name( cs, item )
			pt = remove_field_name( pt, item )


	## Check data sign
	data_sign = checkDataSign( pt[0]['blob/sign'] )


	### Check to see if data was binned for refinement
	IS_BINNED = None 
	if cs[0]['alignments3D/psize_A'] != cs[0]['blob/psize_A']:
		IS_BINNED = True


	if IS_BINNED: 
		### Convert alignments shift to unbinned pixel size
		print( "ATTENTION:\n  Data was binned during refinement. Will convert to unbinned values." )
		print( "  Refinement pixel size:", cs[0]['alignments3D/psize_A'])
		print( "  Unbinned pixel size:", cs[0]['blob/psize_A'], "\n" )
		print( "  Particle 100 shift before correction", cs[100]['alignments3D/shift'])

		### Calculate the bin correction factor
		bin_correction = np.true_divide( cs[0]['alignments3D/psize_A'], cs[0]['blob/psize_A'] )

		### Correct the csfile
		if 'alignments3D/shift' in cs.dtype.fields:
			cs['alignments3D/shift'] =  cs['alignments3D/shift'] * bin_correction
			print( "  Correcting parameter alignments3D/shift in csfile" )
		if 'alignments3D/psize_A' in cs.dtype.fields:
			cs['alignments3D/psize_A'] = cs['blob/psize_A']
			print( "  Correcting parameter alignments3D/psize_A in csfile" )

		### Correct the ptfile also, if values exist there
		if 'alignments3D/shift' in pt.dtype.fields:
			pt['alignments3D/shift'] =  pt['alignments3D/shift'] * bin_correction
			print( "  Correcting parameter alignments3D/shift in passthrough" )
		if 'alignments3D/psize_A' in pt.dtype.fields:
			pt['alignments3D/psize_A'] = pt['blob/psize_A']
			print( "  Correcting parameter alignments3D/psize_A in passthrough" )

		print( "  Particle 100 shift after correction ", cs[100]['alignments3D/shift'], "\n" )


	### Make a copy of structured arrays before I mess with it any more
	cs_sacred = cs.copy(order='C')		# Reference containing original values
	cs_new = cs.copy(order='C')		# This will become our new csfile
	pt_sacred = pt.copy(order='C')		# Reference containing original values
	pt_new = pt.copy(order='C')		# This will become our new ptfile


	### Get parameters for requested symmetry operation
	expand_parameters = defineAreaOfInterest(ROI, user_vector, user_fudge, higher_order_sym)
	area_of_interest = expand_parameters['area_of_interest']
	sym_index = expand_parameters['sym_index']
	my_vector = expand_parameters['my_vector']
	quaternion_toZ = expand_parameters['quaternion_toZ']
	model_sym = expand_parameters['model_sym']


	### Convert vector from vectors from Angstroms to pixels
	print( "My vector in Angstroms: ", my_vector )
	unbinned_pixel_size = cs[0]['blob/psize_A']
	my_vector = np.true_divide( my_vector, unbinned_pixel_size )
	print( "My vector in pixels:    ", my_vector, "\n" )
	print( "Area of interest is", str.upper(area_of_interest) )


	### Iterate through the entire array, modify values for a single rotation operation
	for y in range(0,len(sym_index)):		# Iterate over all symmetry ops

		### Set the current I1 rotation
		Sym_as_quat = Quaternion( I1Quaternions[ sym_index[y] ] )	# removed scipy dependency

		print( "  Applying symmetry rotation", Sym_as_quat, "  (", y+1, "of", len(sym_index), ")"  )

		for x in range(cs.size):		# Iterate through all particles

			## Assign new uids to prevent subparticle ID clash
			subparticle_uid = np.random.randint(1,9223372036854775000,dtype='<u8')         # generate a random uid
			cs[x]['uid'] = subparticle_uid
			pt[x]['uid'] = subparticle_uid         # passthrough version needs same uid as csfile

			## Use 'location/micrograph_path' to store particle number
			## to keep track of which particle each subparticles came from
			particle_number = str(x + 1)					# start at 1, not 0
			padded_particle_number = particle_number.rjust(9, '0')		# pad with zeros to 9 digits

			## Check if in batch mode
			if BATCH_MODE:
				## Subparticles from a given particle will share the same batch
				particle_number = int( particle_number ) - 1		# Convert back to int
				batch_size_symop = int( batch_size / len(sym_index) )	# Subpart/batch for this symop
				particle_batch = int( particle_number / batch_size_symop )	# Calculate batch number

				## Repurpose 'particle_number' to carry 'particle_batch'
				particle_number = str( particle_batch + 1 )			# start at 1, not 0
				padded_particle_number = particle_number.rjust(6, '0')	# pad with zeros to 6 digits
                   
				if user_testmode:	# to speed test mode, randomly assign to one of 5 batches
					particle_number = str( np.random.randint( 1, 5 ) )
					padded_particle_number = particle_number.rjust(6, '0')
 
			## Obtain current pose
			current_pose_as_AA = cs_sacred[x]['alignments3D/pose']
			current_pose_as_quat = aa2quat(current_pose_as_AA)

			## Convert to I2 if requested
			if higher_order_sym == 'I2':
				I2_to_I1 = Quaternion(axis=(0.0,0.0,1.0), degrees=90)       # I2-specific code
				current_pose_as_quat = I2_to_I1 * current_pose_as_quat      # I2-specific code

			### Calculate the symmetry-related pose
			new_pose_as_quat = Quaternion(Sym_as_quat) * Quaternion(current_pose_as_quat)
			## Note, operation is expressed as Rotation 2 * Rotation 1
			## i.e. (symmetry matrix) * (original pose)

			### Calculate the vector offset for subparticles, i.e. OFFSETS
			inverted_new_pose_as_quat = new_pose_as_quat.inverse
			rotated_vector = Quaternion(inverted_new_pose_as_quat).rotate( my_vector )

			### This line rotates subparticle pose to align to z-axis. 
			### Note, this must occur *after* my_vector is defined for subparticle origin determination
			new_pose_as_quat = quaternion_toZ * new_pose_as_quat

			### Convert to axis-angle format as required for csfile
			new_pose_as_aa = quat2aa( new_pose_as_quat )

			### Calculate the local defocus
			local_defocus_offset_A = rotated_vector[2] * cs_sacred[x]['blob/psize_A']
			local_defocus1_A = cs_sacred[x]['ctf/df1_A'] + local_defocus_offset_A	# subtract or add offset?
			local_defocus2_A = cs_sacred[x]['ctf/df2_A'] + local_defocus_offset_A	# subtract or add offset?

			## Actually apply the new values below ##
			if y == 0:		# Run only on the null symop
				applyParameters(cs_new, pt_new, x, cs_sacred, new_pose_as_aa, local_defocus1_A, local_defocus2_A, rotated_vector, ROI, BATCH_MODE, padded_particle_number)

			if y != -1:		# Run on all particles
				applyParameters(cs, pt, x, cs_sacred, new_pose_as_aa, local_defocus1_A, local_defocus2_A, rotated_vector, ROI, BATCH_MODE, padded_particle_number)

		if y != 0:
			### Concatenate the modified array (cs) to the growing new array (cs_new)
			cs_new = np.concatenate((cs_new,cs),axis=0)
			pt_new = np.concatenate((pt_new,pt),axis=0)

	### end loop through the symmetry-operators

	### Save the output csfile and ptfile
	np.save("csfile_4_localreconstruction.cs", cs_new)
	np.save("ptfile_4_localreconstruction.cs", pt_new)


	### Rename to remove .npy suffix
	os.rename( 'csfile_4_localreconstruction.cs.npy', 'csfile_4_localreconstruction.cs' )
	os.rename( 'ptfile_4_localreconstruction.cs.npy', 'ptfile_4_localreconstruction.cs' )
	print( "\nOutput files have been saved at:" )
	print( "  csfile_4_localreconstruction.cs" )
	print( "  ptfile_4_localreconstruction.cs" )
	print( "\nSubparticle centers and orientations have successfully been defined in cryoSPARC format.\n" )
	print( "WARNING! Please verify that you refined with", higher_order_sym, "symmetry" )


	## Convert to star using csparc2star
	cmd = ''.join( ['csparc2star.py csfile_4_localreconstruction.cs ptfile_4_localreconstruction.cs ', ROI, 'subparticle_alignments.star' ] )
	print( "Executing command:", cmd, "\n" )
	os.system( cmd )


	## Print user message if batch mode
	if BATCH_MODE:
		num_batches = math.ceil( np.true_divide ( cs_new.size, batch_size ) )
		print( "  Note: Relion will process subparticles in", int(num_batches), "batches of size", batch_size, "\n" )


	## Setting up test mode vs. real mode 
	if user_testmode:
		## Send message to user		
		print( "  Note: Test Mode selected. Will only generate ~10k subparticles so you can check initial model.\n" )

		## Take only 10k particles
		fullstar = ''.join( [ ROI, 'subparticle_alignments.star' ] )
		shortstar = ''.join( [ ROI, 'subparticle_alignments_abbrev.star' ] )
		cmd = ''.join( [ 'head -n10022 ', fullstar, ' > ', shortstar ] )
		print( "Executing command:", cmd )
		os.system( cmd )

		input = shortstar
	else:
		fullstar = ''.join( [ ROI, 'subparticle_alignments.star' ] )
		input = fullstar


	## This step recenters on subparticle origins
	cmd = ''.join( ['relion_stack_create --i ', input, ' --o ', ROI, ' --apply_rounded_offsets_only --split_per_micrograph'] )
	print( "Executing command:", cmd, "\n" )
	os.system( cmd )

	
	## Fix data sign, crop to subparticle box size
	new_box = str(user_subbox)
	if int( data_sign ) == int( -1 ):	# relion standard
		cmd = ''.join( [ 'relion_image_handler --i ', ROI,'.star --o subpart --new_box ', new_box ] )
	elif int( data_sign ) == int( 1 ):	# cryoSPARC standard, must invert
		cmd = ''.join( [ 'relion_image_handler --i ', ROI,'.star --o subpart --new_box ', new_box,' --multiply_constant -1' ] )
	else:					# unreachable state
		print( "ERROR: xkcd/2200" )
		sys.exit()
	print( "\nExecuting command:", cmd, "\n" )
	os.system( cmd )


	## Delete whole particle images
	## Example path is fivefold_subparticles/fivefold/particle000000001.mrcs
	if BATCH_MODE:
		particle_files = ''.join( [ ROI, '_subparticles/', ROI, '/batch??????.mrcs' ] )
	else:
		particle_files = ''.join( [ ROI, '_subparticles/', ROI, '/particle?????????.mrcs' ] )

	cmd = ''.join( ['rm ', particle_files ] )
	print( "\nExecuting command:", cmd )
	os.system( cmd )


	## Make initial model star file from 1st 10k lines of ROI_subpart.star
	fullfile = ''.join( [ ROI, '_subpart.star' ] )
	initialmodel_star = ''.join( [ ROI, '_initialmodel.star' ] )
	cmd = ''.join( [ 'head -n10000 ', fullfile, ' > ', initialmodel_star ] )
	print( "Executing command:", cmd )
	os.system( cmd )


	## Make initial model
	initialmodel_mrc = ''.join( [ ROI, '_initialmodel_', model_sym,'.mrc' ] )
	cmd = ''.join( [ 'relion_reconstruct --i ', initialmodel_star, ' --o ', initialmodel_mrc ,' --ctf --maxres 5 --sym ', model_sym ] )
	print( "Executing command:", cmd, "\n" )
	os.system( cmd )


	## Change relion metadata labels to include Prior
	filename = ''.join( [ ROI, '_subpart.star' ] )
	filename_PRIOR = ''.join( [ ROI, '_subpart_PRIOR.star' ] )

	cmd = ' '.join( [ 'cp', filename, filename_PRIOR ] )
	print( "\nExecuting command:", cmd )
	os.system( cmd )
	cmd = ''.join( [ 'sed -i \'s/_rlnAnglePsi/_rlnAnglePsiPrior/g\' ', filename_PRIOR ] )
	print( "Executing command:", cmd )
	os.system( cmd )
	cmd = ''.join( [ 'sed -i \'s/_rlnAngleRot/_rlnAngleRotPrior/g\' ', filename_PRIOR ] )
	print( "Executing command:", cmd )
	os.system( cmd )
	cmd = ''.join( [ 'sed -i \'s/_rlnAngleTilt/_rlnAngleTiltPrior/g\' ', filename_PRIOR ] )
	print( "Executing command:", cmd )
	os.system( cmd )
	cmd = ''.join( [ 'sed -i \'s/_rlnOriginX/_rlnOriginXPrior/g\' ', filename_PRIOR ] )
	print( "Executing command:", cmd )
	os.system( cmd )
	cmd = ''.join( [ 'sed -i \'s/_rlnOriginY/_rlnOriginYPrior/g\' ', filename_PRIOR ] )
	print( "Executing command:", cmd, "\n" )
	os.system( cmd )

	print( "Success!\n" )

	return 


def main(args):

	user_batch = None
	if args.batch == 'true':
		user_batch = True

	user_testmode = None
	if args.testmode == 'true':
		user_testmode = True

	if args.input.endswith(".cs"):
		cs = np.load(args.input)
		generate_subparticle_csfile(cs, args.roi, np.array(args.vector), args.fudge, args.subpart_box, args.supersym, user_testmode, args.batchsize, user_batch, passthrough=args.passthrough)
	else:
		print( "Please provide a valid input file." )
		sys.exit()

	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="path to cryosparc_P#_J#_###_particles.cs file")
	parser.add_argument("--passthrough", "-p", required=True, help="path to passthrough_particles.cs file")
	parser.add_argument("--roi", choices=['null', 'fivefold', 'threefold', 'twofold', 'fullexpand', 'just_unbin'], type=str.lower, required=True)
	parser.add_argument("--vector", type=float, nargs=3, required=True, help="X Y Z in Angstroms (space delimited)")
	parser.add_argument("--fudge", type=float, nargs=1, default='1', help="scale vector a bit, e.g. 0.8")
	parser.add_argument("--supersym", choices=['I1', 'I2'], type=str.upper, default='I1')
	parser.add_argument("--subpart_box", type=int, required=True, help="box size for subparticles")
	parser.add_argument("--batch", choices=['true', 'false'], type=str.lower, default='true', help="relion will process in batches rather than per-particle")
	parser.add_argument("--batchsize", type=int, default=3000) 
	parser.add_argument("--testmode", choices=['true', 'false'], type=str.lower, default='false', help="Generate ~10k subparticles so you can verify settings by inspecting the initial model")
	sys.exit(main(parser.parse_args()))

