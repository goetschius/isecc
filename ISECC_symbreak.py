#!/usr/bin/env python3.5
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
import time
import math
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime


def idealizeUserVector( user_vector, ROI ) :

        norm_factor = np.amax( np.absolute(user_vector) )

        I1_fivefold  = np.array( [0.000, 0.618, 1.000] ) * norm_factor
        I1_threefold = np.array( [0.382, 0.000, 1.000] ) * norm_factor
        I1_twofold   = np.array( [0.000, 0.000, 1.000] ) * norm_factor

        my_dtype = np.dtype( [  ( 'vector', '<f4', (3,) ), 
                                ( '5f_distance', '<f4' ),
                                ( '3f_distance', '<f4' ),
                                ( '2f_distance', '<f4' )  ]  )

        expanded_user_vector = np.zeros( (60), dtype=my_dtype )

        for index, numpy_quat in enumerate(I1Quaternions, start=0 ):

                quat = Quaternion( numpy_quat )
                rotated_vector = np.around( quat.rotate( user_vector ), decimals=3 )

                expanded_user_vector[index]['vector'] = rotated_vector
                expanded_user_vector[index]['5f_distance'] = np.around( assess3dDistance( rotated_vector, I1_fivefold  ), decimals=3 )
                expanded_user_vector[index]['3f_distance'] = np.around( assess3dDistance( rotated_vector, I1_threefold ), decimals=3 )
                expanded_user_vector[index]['2f_distance'] = np.around( assess3dDistance( rotated_vector, I1_twofold   ), decimals=3 )


        min_5f_distance = np.amin( expanded_user_vector['5f_distance'] )
        min_3f_distance = np.amin( expanded_user_vector['3f_distance'] )
        min_2f_distance = np.amin( expanded_user_vector['2f_distance'] )

        ideal_index = 999

        for index, item in enumerate(expanded_user_vector):

                if np.isclose(expanded_user_vector[index]['5f_distance'], min_5f_distance, atol=1) and np.isclose(expanded_user_vector[index]['3f_distance'], min_3f_distance, atol=1) and np.isclose(expanded_user_vector[index]['2f_distance'], min_2f_distance, atol=1):

                        ideal_index = index


        if ideal_index != 999:
                idealized_vector = np.around( expanded_user_vector[ideal_index]['vector'], decimals=3 )
        else:
                print( 'Error idealizing user vector. Exiting now.' )
                sys.exit()


        if not np.allclose( idealized_vector, user_vector, atol=0.5 ):
                print( '\nUser vector', user_vector, "has been modified to be within desired asymmetric unit." )
                print( 'User vector is now', idealized_vector )

        return idealized_vector


def formatVertexAssignment( particle, sym_index, ROI ):

        fivefold_general  = str( int( particle['Vertex_5f_general'][sym_index] ) ).rjust(2, '0')
        threefold_general = str( int( particle['Vertex_3f_general'][sym_index] ) ).rjust(2, '0')
        twofold_general   = str( int( particle['Vertex_2f_general'][sym_index] ) ).rjust(2, '0')

        fivefold_specific  = extractRotationAssignment( particle['Vertex_5f_specific'][sym_index] )
        threefold_specific = extractRotationAssignment( particle['Vertex_3f_specific'][sym_index] )
        twofold_specific   = extractRotationAssignment( particle['Vertex_2f_specific'][sym_index] )

        ### Formating
        my_fivefold  =  ''.join( [ '5f', fivefold_general,  fivefold_specific  ] )
        my_threefold =  ''.join( [ '3f', threefold_general, threefold_specific ] )
        my_twofold   =  ''.join( [ '2f', twofold_general,   twofold_specific   ] )

        ### Combine them
        my_vertex_string = '.'.join( [ my_fivefold, my_threefold, my_twofold ] )

        if ROI == 'fivefold':   my_vertex_string = ''.join( [ '5f', fivefold_general ] )
        if ROI == 'threefold':  my_vertex_string = ''.join( [ '3f', threefold_general ] )
        if ROI == 'twofold':    my_vertex_string = ''.join( [ '2f', twofold_general ] )

        return my_vertex_string



def vertexExpand( particle_index, particle_pose, transformed_pose_array, vertex_vector ):

        my_dtype = np.dtype( [  ( 'IdealVector', '<f4', (3,) ),
                                ( 'TransformedVector', '<f4', (3,) ),
                                ( 'OriginalPose' , '<f4', (4,) ),
                                ( 'ExpandQuat', '<f4', (4,) ),
                                ( 'TransformedPose', '<f4', (4,) ),
                                ( 'Unique', 'bool' ) ] )

        ### Make table of length 60
        vertex_table = np.zeros( len(I1Quaternions), my_dtype)

        ### Given
        vertex_table['IdealVector'] = vertex_vector
        vertex_table['ExpandQuat'] = I1Quaternions
        vertex_table['OriginalPose'] = particle_pose

        ### Pose of the particle
        particle_pose = Quaternion( particle_pose )

        for index, vertex in enumerate( vertex_table, start=0) :

                ### Transformed pose for this particle
                transformed_pose_numpy = transformed_pose_array[index]
                transformed_pose = Quaternion(transformed_pose_array[index])
                vertex_table[index]['TransformedPose'] = transformed_pose_numpy

                ### Invert for the vector transformation
                transformed_pose_inverse = transformed_pose.inverse

                ### Transform the vertex vector
                transformed_vector = np.around( transformed_pose_inverse.rotate( vertex_vector ), decimals=3)

                # Assess whether tranformed point is unique
                unique_bool = True
                for loop_index, item in enumerate( vertex_table, start=0) :
                        if np.allclose( transformed_vector, vertex_table[loop_index]['TransformedVector'], atol=0.01) :
                                unique_bool = False

                # Store the information
                vertex_table[index]['TransformedVector'] = transformed_vector
                vertex_table[index]['Unique'] = unique_bool


        ### Gather the unique vertices
        unique_vertex_indices = np.where( vertex_table['Unique'] )[0]
        unique_vertex_table = vertex_table[ np.where( vertex_table['Unique'] ) ]


        return vertex_table, unique_vertex_indices



def prepareSubparticleTable( star_array, user_vector, header, ROI ):


        I1_fivefold  = np.array( [0.000, 0.618, 1.000] )
        I1_threefold = np.array( [0.382, 0.000, 1.000] )
        I1_twofold   = np.array( [0.000, 0.000, 1.000] )


        ####
        ## Manual override for debug
        ####

#       user_vector = np.array( [0, 0.618, 1] )

        my_dtype = np.dtype( [  ( 'UserVector', '<f4', (3,) ),
                                ( 'ExpandQuatIndex', 'i4', (60,1) ), ( 'ExpandQuat', '<f4', (60,4) ),
                                ( 'OriginalPose', '<f4', (4,) ), ( 'TransformedPose', '<f4', (60,4) ),
                                ( 'TransformedVector', '<f4', (60,3) ),
                                ( 'Unique', 'bool', (60,1) ),
                                ( 'Vertex_5f_general', 'i4', (60,1) ), ( 'Vertex_5f_specific', '|S1', (60,1)), ( 'Vertex_5f_coords', '<f4', (60,3,) ),
                                ( 'Vertex_3f_general', 'i4', (60,1) ), ( 'Vertex_3f_specific', '|S1', (60,1)), ( 'Vertex_3f_coords', '<f4', (60,3) ),
                                ( 'Vertex_2f_general', 'i4', (60,1) ), ( 'Vertex_2f_specific', '|S1', (60,1)), ( 'Vertex_2f_coords', '<f4', (60,3) ) ] )

        master_table = np.zeros( len(star_array), dtype=my_dtype )


        ### For parsing star file
        rot_index, tilt_index, psi_index = getEulers( header )


        ### Load up the table
        master_table['UserVector'] = user_vector
        master_table['ExpandQuat'] = I1Quaternions
        master_table['ExpandQuatIndex'] = np.arange(60).reshape(60,1)


        ### Per particle values
        for particle_index, particle in enumerate(star_array):

                my_rotrad  = np.radians( float( star_array[particle_index][rot_index]  ) )
                my_tiltrad = np.radians( float( star_array[particle_index][tilt_index] ) )
                my_psirad  = np.radians( float( star_array[particle_index][psi_index]  ) )

                ### Original pose for calculations
                original_pose = Quaternion( myEuler2Quat( my_rotrad, my_tiltrad, my_psirad ) )

                ### Original pose for master_table
                original_pose_numpy = original_pose.elements
                master_table[particle_index]['OriginalPose'] = original_pose_numpy


                ### Array that we'll send to vertex determiner
                transformed_pose_array = np.zeros( (60,4), dtype='<f4' )
                        

                ### Per subparticle values
                for subparticle_index, subparticle in enumerate(master_table[particle_index]['TransformedPose']):

                        ####
                        ## Quaternion/pose
                        ####

                        expand_quat = master_table[particle_index]['ExpandQuat'][subparticle_index]
                        expand_quat = Quaternion( expand_quat )

                        ### Transformed pose for calculations. Inverse for vector rotation.
                        transformed_pose = expand_quat * original_pose
                        transformed_pose_inverse = transformed_pose.inverse

                        ### Transformed pose for master_table
                        transformed_pose_numpy = transformed_pose.elements
                        master_table[particle_index]['TransformedPose'][subparticle_index] = transformed_pose_numpy

                        ### Store to send for vertex determination
                        transformed_pose_array[subparticle_index] = transformed_pose_numpy


                        ####
                        ## Transformed user_vector
                        ####

                        transformed_vector = np.around( transformed_pose_inverse.rotate( user_vector ), decimals=3)


                        # Assess whether tranformed point is unique
                        unique_bool = True
                        for loop_index, item in enumerate( master_table[particle_index]['TransformedVector'], start=0) :
                                if np.allclose( transformed_vector, master_table[particle_index]['TransformedVector'][loop_index], atol=0.01) :
                                        unique_bool = False

                        # Store the information
                        master_table[particle_index]['TransformedVector'][subparticle_index] = transformed_vector
                        master_table[particle_index]['Unique'][subparticle_index] = unique_bool


                ### Determine the vertices
                if particle_index == 0:
                        print( "Making nearest-vertex assignments for all symops." )

                fivefold_vertices, unique_5f_indices  = vertexExpand( particle_index, original_pose_numpy, transformed_pose_array, I1_fivefold  )
                threefold_vertices, unique_3f_indices = vertexExpand( particle_index, original_pose_numpy, transformed_pose_array, I1_threefold )
                twofold_vertices, unique_2f_indices   = vertexExpand( particle_index, original_pose_numpy, transformed_pose_array, I1_twofold   )

                if particle_index == 0:
                        print( "Vertex assignments complete!\n" )


                ####
                ## To use my returnNearestVertex function, we need to make a subarray
                ## ...containing only the subparticle values for a single particle
                ## ...i.e., it'll be 60-long
                ####


                my_dtype = np.dtype( [  ( 'ExpandQuatIndex', 'i4' ),         ( 'TransformedVector', '<f4', (3,) ),
                                        ( 'Vertex_5f_general', 'i4' ),       ( 'Vertex_5f_specific', '|S1' ), 
                                        ( 'Vertex_5f_coords', '<f4', (3,) ), ( 'Vertex_3f_general', 'i4' ), 
                                        ( 'Vertex_3f_specific', '|S1' ),     ( 'Vertex_3f_coords', '<f4', (3,) ),
                                        ( 'Vertex_2f_general', 'i4' ),       ( 'Vertex_2f_specific', '|S1' ), 
                                        ( 'Vertex_2f_coords', '<f4', (3,) ) ] )


                this_particle = np.zeros( 60, dtype=my_dtype )
                for index, item in enumerate(this_particle):

                        ### Copy the minimal number of values over to avoid schenanigans

                        this_particle[index]['TransformedVector'] = master_table[particle_index]['TransformedVector'][index]
                        this_particle[index]['ExpandQuatIndex'] = master_table[particle_index]['ExpandQuatIndex'][index]

                ####
                ## Nearest vertex
                ####

                if (ROI != 'threefold') and (ROI != 'twofold'):
                        this_particle = returnNearestVertex( this_particle, fivefold_vertices,  'fivefold',  ROI )
                if (ROI != 'fivefold') and (ROI != 'twofold'):
                        this_particle = returnNearestVertex( this_particle, threefold_vertices, 'threefold', ROI )
                if (ROI != 'fivefold') and (ROI != 'threefold'):
                        this_particle = returnNearestVertex( this_particle, twofold_vertices,   'twofold',   ROI )


                ####
                ## Copy the values back to the real array. Ugly, but it works
                ###

                for index, item in enumerate(this_particle):
                        master_table[particle_index]['Vertex_5f_general'][index] = this_particle[index]['Vertex_5f_general']
                        master_table[particle_index]['Vertex_3f_general'][index] = this_particle[index]['Vertex_3f_general']
                        master_table[particle_index]['Vertex_2f_general'][index] = this_particle[index]['Vertex_2f_general']

                        master_table[particle_index]['Vertex_5f_specific'][index] = this_particle[index]['Vertex_5f_specific']
                        master_table[particle_index]['Vertex_3f_specific'][index] = this_particle[index]['Vertex_3f_specific']
                        master_table[particle_index]['Vertex_2f_specific'][index] = this_particle[index]['Vertex_2f_specific']

                        master_table[particle_index]['Vertex_5f_coords'][index] = this_particle[index]['Vertex_5f_coords']
                        master_table[particle_index]['Vertex_3f_coords'][index] = this_particle[index]['Vertex_3f_coords']
                        master_table[particle_index]['Vertex_2f_coords'][index] = this_particle[index]['Vertex_2f_coords']
                        
        if ROI == 'fullexpand':         return master_table
        elif ROI == 'fivefold':         return master_table, unique_5f_indices
        elif ROI == 'threefold':        return master_table, unique_3f_indices
        elif ROI == 'twofold':          return master_table, unique_2f_indices


        else:   return master_table


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

        defocusU = np.around(defocusU,6)        # round to 6 decimals
        defocusV = np.around(defocusV,6)        # round to 6 decimals


        return defocusU, defocusV


def reverseDefocus( input, output, FlippedDefocusU, FlippedDefocusV, FlippedXYZ_string ) :

        filename = input
        header, fullheader = getStarHeader( filename, 'null' )
        fullheader = fullheader[3:]

        a = getStarData( filename, len( header ) )

        reverse_defocus_ndarray = np.asarray( a, order='C' )

        ### Get the necessary indices
        UID_index = getUID( header )
        defocusU_index, defocusV_index, defocusAngle_index = getDefocus( header )
        OriginXYZAngstWrtParticleCenter_index = getOriginXYZAngstWrtParticleCenter( header )

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


def myEuler2Quat( phi, theta, psi ) :           # doi.org/10.1101/733881
        # Rot, Tilt, Psi
        # Phi, Theta, Psi
        # convention is for paper above is q = q0 +q1i +q2j +q3k
                
        q_phi = np.zeros(4)
        q_phi[0] = np.cos( np.true_divide( phi, 2 ) )
        q_phi[1] = 0
        q_phi[2] = 0
        q_phi[3] = np.sin( np.true_divide( phi, 2 ) ) 
        quaternion_phi = Quaternion( array=q_phi )

        q_theta = np.zeros(4)
        q_theta[0] = np.cos( np.true_divide( theta, 2 ) )
        q_theta[1] = 0
        q_theta[2] = np.sin( np.true_divide( theta, 2 ) )
        q_theta[3] = 0
        quaternion_theta = Quaternion( array=q_theta )

        q_psi = np.zeros(4)
        q_psi[0] = np.cos( np.true_divide( psi, 2 ) )
        q_psi[1] = 0
        q_psi[2] = 0
        q_psi[3] = np.sin( np.true_divide( psi, 2 ) )
        quaternion_psi = Quaternion( array=q_psi )

        ## Note, operation is expressed as Rotation 2 * Rotation 1
        quaternion_composite = ( quaternion_phi * quaternion_theta ) * quaternion_psi
        ## This expression is confirmed functional

        ######## The below statement is equivalent to the above
        #quaternion_composite = quaternion_theta * quaternion_phi
        #quaternion_composite = quaternion_psi * quaternion_composite
        ########

        return quaternion_composite 


## From relion source
        ## alpha = rot
        ## beta = tilt
        ## gamma = psi


def rot2euler(r):                      # pyem, Daniel Asarnow
        """Decompose rotation matrix into Euler angles"""
        # assert(isrotation(r))
        # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
        epsilon = np.finfo(np.double).eps
        abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
        if abs_sb > 16 * epsilon:
                gamma = np.arctan2(r[1, 2], -r[0, 2])
                alpha = np.arctan2(r[2, 1], r[2, 0])
                if np.abs(np.sin(gamma)) < epsilon:
                        sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
                else:
                        sign_sb = np.sign(r[1, 2]) if np.sin(gamma) > 0 else -np.sign(r[1, 2])
                beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
        else:
                if np.sign(r[2, 2]) > 0:
                        alpha = 0
                        beta = 0
                        gamma = np.arctan2(-r[1, 0], r[0, 0])
                else:
                        alpha = 0
                        beta = np.pi
                        gamma = np.arctan2(r[1, 0], -r[0, 0])

        ## Convert to degrees before returning values
        alpha = math.degrees( alpha )
        beta = math.degrees( beta )
        gamma = math.degrees( gamma )

        return alpha, beta, gamma


def random_subsample( starfile, ROI ) :

        desired_subparticles = int(10000)

        if ROI == 'null' :              my_sample_size = int( desired_subparticles )
        if ROI == 'fivefold' :          my_sample_size = math.ceil( desired_subparticles / 12 )
        if ROI == 'threefold' :         my_sample_size = math.ceil( desired_subparticles / 20 )
        if ROI == 'twofold' :           my_sample_size = math.ceil( desired_subparticles / 30 )
        if ROI == 'fullexpand' :        my_sample_size = math.ceil( desired_subparticles / 60 )


        if int( starfile.size ) < my_sample_size :
                sample_size = starfile.size
        else:
                sample_size = my_sample_size

        indices = np.arange( len(starfile) )
        random_indices = np.random.choice( indices, size = sample_size )

        new_starfile = starfile[random_indices]

        return new_starfile


def defineAreaOfInterest(ROI, user_vector, user_fudge, higher_order_sym):
        area_of_interest = ROI
        my_vector = user_vector
        my_vector = my_vector * user_fudge            # allows you to bump vector in or out a bit
        my_sym = higher_order_sym

        checkVector(user_vector, ROI, higher_order_sym)

        ## If I2, swap x and y
        ## This is required as script was written for I1 default
        if higher_order_sym == 'I2':
                my_vector = np.array( [ my_vector[1], my_vector[0], my_vector[2] ] )

        if area_of_interest == "null" :
                my_arctan = 0
                quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
                model_sym = 'c1'
                short_roi = 'null'

        if area_of_interest == "fivefold" :
                ### Ideal vector is 0.000, 0.618, 1.000
                my_arctan = np.arctan( np.true_divide( my_vector[1], my_vector[2] ) )
                quaternion_toZ = Quaternion(axis=(1.0,0.0,0.0), radians=my_arctan)
                model_sym = 'c5'
                short_roi = '5f'

        if area_of_interest == "threefold" :
                ### Ideal vector is 0.382, 0.000, 1.000
                my_arctan = np.arctan( np.true_divide( my_vector[0], my_vector[2] ) )
                quaternion_toZ = Quaternion(axis=(0.0,-1.0,0.0), radians=my_arctan)
                model_sym = 'c3'
                short_roi = '3f'

        if area_of_interest == "twofold" :
                ### Ideal vector is 0.000, 0.000, 1.000
                my_arctan = 0
                quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
                model_sym = 'c2'
                short_roi = '2f'

        if area_of_interest == "fullexpand" :
                my_vector = 1.0 * my_vector
                my_arctan = 0
                quaternion_toZ = Quaternion(axis=(0.0,0.0,1.0), radians=my_arctan)
                model_sym = 'c1'
                short_roi = 'ex'

        return { 'area_of_interest': area_of_interest, 'my_vector': my_vector, 'quaternion_toZ': quaternion_toZ, 'model_sym': model_sym, 'short_roi': short_roi }


def checkVector(user_vector, ROI, higher_order_sym):
        if ROI == 'fivefold' and higher_order_sym == 'I1' :
                vector_check = np.true_divide( user_vector[1], user_vector[2] )
                my_check = np.isclose( vector_check, 0.618, atol=0.1 )
                my_check2 = np.isclose( user_vector[0], 0, atol=0.1 )

        elif ROI == 'fivefold' and higher_order_sym == 'I2' :
                vector_check = np.true_divide( user_vector[0], user_vector[2] )
                my_check = np.isclose( vector_check, 0.618, atol=0.1 )
                my_check2 = np.isclose( user_vector[1], 0, atol=0.1 )

        elif ROI == 'threefold' and higher_order_sym == 'I1' :
                vector_check = np.true_divide( user_vector[0], user_vector[2] )
                my_check = np.isclose( vector_check, 0.382, atol=0.1 )
                my_check2 = np.isclose( user_vector[1], 0, atol=0.1 )

        elif ROI == 'threefold' and higher_order_sym == 'I2' :
                vector_check = np.true_divide( user_vector[1], user_vector[2] )
                my_check = np.isclose( vector_check, 0.382, atol=0.1 )
                my_check2 = np.isclose( user_vector[0], 0, atol=0.1 )

        elif ROI == 'twofold' and higher_order_sym == 'I1' :
                my_check = np.isclose( user_vector[0], 0, atol=0.1 )
                my_check2 = np.isclose( user_vector[1], 0, atol=0.1 )

        elif ROI == 'twofold' and higher_order_sym == 'I2' :
                my_check = np.isclose( user_vector[0], 0, atol=0.1 )
                my_check2 = np.isclose( user_vector[1], 0, atol=0.1 )

        elif ROI == 'fullexpand' or ROI == 'null' :
                my_check = True
                my_check2 = True

        if str(my_check) == 'False' or str(my_check2) == 'False':
                print( 'ERROR: Vector is not valid for roi', ROI, 'and symmetry', higher_order_sym )
                print( '       ', user_vector )
                print( '       Please provide a valid vector and try again.\n' )

                print( '  For I1 symmetry,' )
                print( '    Idealized Fivefold:   0.000, 0.618, 1.000')
                print( '    Idealized Threefold:  0.382, 0.000, 1.000')
                print( '    Idealized Twofold:    0.000, 0.000, 1.000\n')

                print( '  For I2 symmetry,' )
                print( '    Idealized Fivefold:   0.618, 0.000, 1.000')
                print( '    Idealized Threefold:  0.000, 0.382, 1.000')
                print( '    Idealized Twofold:    0.000, 0.000, 1.000\n')

                sys.exit()

        elif str(my_check) == 'True' or str(my_check2) == 'True':
                print( "\nVector is valid for roi", ROI, "and symmetry", higher_order_sym, "\n" )

        return


def getStarHeader( my_star ):
        header = []
        header_index = int( 0 )
        full_header = []
        
        # Add run info to output star file
        date = str( datetime.now() )
        append_this = ''.join( [ '# SCRIPT_RUN_DATE: ', date ] )
        full_header.append( append_this.strip() )

        append_this = ''.join( [ '# SCRIPT_VERSION: ', sys.argv[0] ] )
        full_header.append( append_this.strip() )

        arguments = ' '.join( sys.argv[1:] )
        append_this = ''.join( [ '# SCRIPT_ARGS: ', arguments ] )
        full_header.append( append_this.strip() )

        with open(my_star, "r") as my_star:
                
                START_PARSE = None      # Needed for relion 3.1 star format

                for line in my_star:

                        if START_PARSE == None:
                                ### Store full header for final output
                                full_header.append( line.strip() )

                        if line.strip() == 'data_particles':
                                START_PARSE = True

                        if START_PARSE:
                                if line.startswith('loop_'):
                                        full_header.append( '' )
                                        full_header.append( line.strip() )
                                if line.startswith('_rln'):
                                        full_header.append( line.strip() )
                                        header_index = int( header_index + 1 )
                                        header.append( line[1:].split()[0] )

        if 'rlnImageOriginalName' not in header:
                header.append( 'rlnImageOriginalName' )
                header_index = int( header_index + 1 )
                append_this = ''.join( [ '_rlnImageOriginalName #', str(header_index) ] )
                full_header.append( append_this )


        if 'rlnCustomUID' not in header:
                header.append( 'rlnCustomUID' )
                header_index = int( header_index + 1 )
                append_this = ''.join( [ '_rlnCustomUID #', str(header_index) ] )
                full_header.append( append_this )


        if 'rlnCustomVertexGroup' not in header:
                header.append( 'rlnCustomVertexGroup' )
                header_index = int( header_index + 1 )
                append_this = ''.join( [ '_rlnCustomVertexGroup #', str(header_index) ] )
                full_header.append( append_this )


        if 'rlnCustomOriginXYZAngstWrtParticleCenter' not in header:
                header.append( 'rlnCustomOriginXYZAngstWrtParticleCenter' )
                header_index = int( header_index + 1 )
                append_this = ''.join( [ '_rlnCustomOriginXYZAngstWrtParticleCenter #', str(header_index) ] )
                full_header.append( append_this )


        if 'rlnCustomRelativePose' not in header:
                header.append( 'rlnCustomRelativePose' )
                header_index = int( header_index + 1 )
                append_this = ''.join( [ '_rlnCustomRelativePose #', str(header_index) ] )
                full_header.append( append_this )

        return header, full_header


def getStarData( my_star, header_length, header ):

        # Get the index of the micrograph name. Need this for sorting later
        micrographname_indexP, imagename_indexP, particleID_indexP = getMicrographName( header )



        with open(my_star, "r") as my_star:
                stardata = []

                START_PARSE = None      # Needed for relion 3.1 star format

                for line in my_star:

                        if line.strip() == 'data_particles':
                                START_PARSE = True

                        if START_PARSE:

                                linesplit = line.split()
                                if len( linesplit ) < header_length:
                                        ### Needed to add rlnImageOriginalName and rlnCustomUID
                                        line = ' '.join( [ line, 'ImageOriginalName', 'CustomUID', 'CustomVertexGroup', 'CustomOriginXYZAngst', 'CustomRelativePose' ] )

                                linesplit = line.split()
                                if len( linesplit ) == header_length:
                                        if line[0] != '#':      # avoid the stupid comment line
                                                stardata.append( linesplit )

#        print( micrographname_indexP )


#        print( "  Sorting by micrograph name." )
#        stardata = stardata[stardata[:,micrographname_indexP].argsort()]


#        print( stardata[0] )

#        stardata.sort(key = micrographname_indexP )

        return stardata


def getMinimalStarData( my_star, subparticle_header, subparticle_header_length, num_particles ):

        my_dtype = np.dtype( [  ( 'ParticleImageName', 'U200'),
                                ( 'Class', '<i4' ),
                                ( 'EulerRot', '<f4' ), ( 'EulerTilt', '<f4' ), ( 'EulerPsi', '<f4' ) ] ) 
            
        minimal_subparticle_array    =  np.zeros( ( (num_particles * 60),), dtype=my_dtype ) 
        subparticle_index = 0

        ## This function should only return class and eulers
        rot_indexSP, tilt_indexSP, psi_indexSP = getEulers( subparticle_header )
        class_indexSP = getClass( subparticle_header )
        micrographname_indexSP, imagename_indexSP, particleID_indexSP = getMicrographName( subparticle_header )
    
        with open(my_star, "r") as my_star:
                stardata = []

                START_PARSE = None      # Needed for relion 3.1 star format

                for line in my_star:

                        if line.strip() == 'data_particles':
                                START_PARSE = True

                        if START_PARSE:

                                linesplit = line.split()

                                if len( linesplit ) > 4:    # get beyond the metadata labels

                                    line = ' '.join( [ linesplit[particleID_indexSP], linesplit[class_indexSP], linesplit[rot_indexSP], linesplit[tilt_indexSP], linesplit[psi_indexSP] ] )

                                    if line[0] != '#':      # avoid the stupid comment line
                                        stardata.append( line )

                                        minimal_subparticle_array[subparticle_index]['ParticleImageName'] = linesplit[particleID_indexSP] 
                                        minimal_subparticle_array[subparticle_index]['Class']     = linesplit[ class_indexSP ]
                                        minimal_subparticle_array[subparticle_index]['EulerRot']  = linesplit[ rot_indexSP   ]
                                        minimal_subparticle_array[subparticle_index]['EulerTilt'] = linesplit[ tilt_indexSP  ]
                                        minimal_subparticle_array[subparticle_index]['EulerPsi']  = linesplit[ psi_indexSP   ]

                                        subparticle_index = subparticle_index + 1

        print( "  --> sorting based on particle image name." )
        minimal_subparticle_array.sort(order='ParticleImageName')
        print( "  --> done!" )

#        sys.exit()

        return minimal_subparticle_array



def getApix( header ):
        det_pixelsize_index= header.index('rlnDetectorPixelSize')
        mag_index = header.index('rlnMagnification')
        return det_pixelsize_index, mag_index

def calculateAngpix( detectorPixelSize, Magnification ):
        detectorPixelSize = float( detectorPixelSize )
        magnification = float( Magnification )
        apix = np.true_divide ( detectorPixelSize, magnification ) * 10000
        return apix

def getEulers( header ):
        rot_index = header.index('rlnAngleRot')
        tilt_index = header.index('rlnAngleTilt')
        psi_index = header.index('rlnAnglePsi')
        return rot_index, tilt_index, psi_index

def getOffsets( header ):
        originX_index = header.index('rlnOriginX')
        originY_index = header.index('rlnOriginY')
        originXPrior_index = header.index('rlnOriginXPrior') 
        originYPrior_index = header.index('rlnOriginYPrior')

        return originX_index, originY_index, originXPrior_index, originYPrior_index

def getOffsetAngst( header ):
        originXAngst_index = header.index('rlnOriginXAngst')
        originYAngst_index = header.index('rlnOriginYAngst')

        return originXAngst_index, originYAngst_index

def getDefocus( header ):
        defocusU_index = header.index('rlnDefocusU')
        defocusV_index = header.index('rlnDefocusV')
        defocusAngle_index = header.index('rlnDefocusAngle')
        return defocusU_index, defocusV_index, defocusAngle_index

def getClass( header ):
        class_index = header.index('rlnClassNumber')
        return class_index

def getUID( header ) :
        uid_index = header.index('rlnCustomUID')
        return uid_index

def getVertexGroup( header ) :
        vertexGroup_index = header.index('rlnCustomVertexGroup')
        return vertexGroup_index

def getOriginXYZAngstWrtParticleCenter( header ) :
        OriginXYZAngstWrtParticleCenter_index = header.index('rlnCustomOriginXYZAngstWrtParticleCenter')
        return OriginXYZAngstWrtParticleCenter_index

def getCustomRelativePose( header ) :
        relativePose_index = header.index('rlnCustomRelativePose')
        return relativePose_index


def getMicrographName( header ) :
        # rlnMicrographName
        micrographname_index = header.index('rlnMicrographName')
        imagename_index = header.index('rlnImageName')
        particleID_index = header.index('rlnImageOriginalName')

        return micrographname_index, imagename_index, particleID_index


def getSymRelatedVertexGroup( index ) :
        I1_fivefold  = np.array( [0.000, 0.618, 1.000] )
        I1_threefold = np.array( [0.382, 0.000, 1.000] )
        I1_twofold   = np.array( [0.000, 0.000, 1.000] )

        current_quaternion = Quaternion( I1Quaternions[index] )

        current_5F = np.around(current_quaternion.rotate( I1_fivefold  ), decimals=3)
        current_3F = np.around(current_quaternion.rotate( I1_threefold ), decimals=3)
        current_2F = np.around(current_quaternion.rotate( I1_twofold   ), decimals=3)

        this_5f_symgroup = list()
        this_3f_symgroup = list()
        this_2f_symgroup = list()

        for loop_index, pose in enumerate(I1Quaternions, start=0):

                compare_quaternion = Quaternion( pose )

                rotated_5F = np.around(compare_quaternion.rotate( I1_fivefold  ), decimals=3)
                rotated_3F = np.around(compare_quaternion.rotate( I1_threefold ), decimals=3)
                rotated_2F = np.around(compare_quaternion.rotate( I1_twofold   ), decimals=3)

                if np.array_equal( rotated_5F, current_5F  ):   
                        this_5f_symgroup.append( int(loop_index) )
                if np.array_equal( rotated_3F, current_3F ):    
                        this_3f_symgroup.append( int(loop_index) )
                if np.array_equal( rotated_2F, current_2F   ):  
                        this_2f_symgroup.append( int(loop_index) )



#       print( this_5f_symgroup )
#       sys.exit()


        return this_5f_symgroup, this_3f_symgroup, this_2f_symgroup



def extractRotationAssignment( rotation_assignment ):

        if str(rotation_assignment) == str([b'a'])   :  return str('a')
        elif str(rotation_assignment) == str([b'b']) :  return str('b')
        elif str(rotation_assignment) == str([b'c']) :  return str('c')
        elif str(rotation_assignment) == str([b'd']) :  return str('d')
        elif str(rotation_assignment) == str([b'e']) :  return str('e')
        elif str(rotation_assignment) == str([b''])  :  return str('z')

        else:
                print( 'rotation_assignment not understood.' )
                print( rotation_assignment )
                sys.exit()

        return


def assess3dDistance( point1, point2 ):

        ### stackoverflow.com/questions/1401712
        distance = np.linalg.norm( point1 - point2 )

        ### equivalent below
#       distance = np.sqrt( ( point1[0] - point2[0] )**2 + ( point1[1] - point2[1] )**2 + ( point1[2] - point2[2] )**2 )

        return distance



def returnNearestVertex( expanded_user_vector, expanded_vertex, my_vertex, ROI ):

        ####
        ## First, make array containing unique vertices as points
        ####

        if my_vertex == 'fivefold'  :
                array_size = 12
                per_vertex = 5
                rot_designation = np.array(['a','b','c','d','e'], dtype='|S1')
        if my_vertex == 'threefold' :
                array_size = 20
                per_vertex = 3
                rot_designation = np.array(['a','b','c'], dtype='|S1')
        if my_vertex == 'twofold'   :
                array_size = 30
                per_vertex = 2
                rot_designation = np.array(['a','b'], dtype='|S1')
        
        my_dtype = np.dtype( [  ( 'VertexNumber', 'i4'),
                                ( 'VertexCoords', '<f4', (3,) ) ] )

        unique_vertices    =  np.zeros( (array_size,), dtype=my_dtype )
        index = 0

        for vertex in expanded_vertex:

                if vertex['Unique'] == True:
                        unique_vertices[index]['VertexNumber'] = index + 1
                        unique_vertices[index]['VertexCoords'] = vertex['TransformedVector'] 
                        index = index + 1

        ####
        ## Now, assess distance between transformed user_vector and unique vertices
        ####

        for point in expanded_user_vector:

                closest_distance = 999

                for vertex in unique_vertices:

                        distance_check = assess3dDistance( point['TransformedVector'], vertex['VertexCoords'])
                        distance_check = np.around(distance_check, decimals=3)
                        if distance_check < closest_distance:
                                closest_distance = distance_check

                                if my_vertex == 'fivefold'  :
                                        point['Vertex_5f_general'] = vertex['VertexNumber']
                                        point['Vertex_5f_coords']  = vertex['VertexCoords']

                                if my_vertex == 'threefold' :
                                        point['Vertex_3f_general'] = vertex['VertexNumber']
                                        point['Vertex_3f_coords']  = vertex['VertexCoords']

                                if my_vertex == 'twofold'   :
                                        point['Vertex_2f_general'] = vertex['VertexNumber']
                                        point['Vertex_2f_coords']  = vertex['VertexCoords']


        ####
        ## Finally, assess the vertex rotational assignment
        ####

        assigned_rotations = np.array([])       # Empty array to store list of assigned rotations

        for point_index, point in enumerate( expanded_user_vector, start=0 ):

                if my_vertex == 'fivefold'  :
                        my_unique_vertex = point['Vertex_5f_general']
                        my_vertex_partners = np.where( expanded_user_vector['Vertex_5f_general'] == my_unique_vertex )
                if my_vertex == 'threefold' :
                        my_unique_vertex = point['Vertex_3f_general']
                        my_vertex_partners = np.where( expanded_user_vector['Vertex_3f_general'] == my_unique_vertex )
                if my_vertex == 'twofold'   :
                        my_unique_vertex = point['Vertex_2f_general']
                        my_vertex_partners = np.where( expanded_user_vector['Vertex_2f_general'] == my_unique_vertex )

                if not np.isin( point_index, assigned_rotations ):

                        ####
                        ## Make subarray containing only this vertex group
                        ####

                        sub_array = np.zeros( per_vertex, dtype=expanded_user_vector.dtype )
                        for index, item in enumerate(sub_array):
                                sub_array[index] = expanded_user_vector[ my_vertex_partners[0][index] ]

                        ####
                        ## Add these points into the assigned rotations group
                        ##   ...we don't want to assign them again
                        ####

                        for item in my_vertex_partners[0]:
                                assigned_rotations = np.append( assigned_rotations, item )


                        # Definitions for the rotation testing
                        if my_vertex == 'fivefold'  :
                                rotation_axis = sub_array[0]['Vertex_5f_coords']
                                rotation_degrees = 72
                        if my_vertex == 'threefold'  :
                                rotation_axis = sub_array[0]['Vertex_3f_coords']
                                rotation_degrees = 120
                        if my_vertex == 'twofold'  :
                                rotation_axis = sub_array[0]['Vertex_2f_coords']
                                rotation_degrees = 180


                        my_dtype = np.dtype( [  ( 'Point', '<f4', (3,) ),
                                                ( 'Rotation', 'i4' ),
                                                ( 'Designation', '|S1' ) ] )

                        rotated_points=np.zeros( (per_vertex), dtype=my_dtype )


                        rotation_order = np.zeros( len(rot_designation), dtype='|S1' )

                        for x in range(0, per_vertex):

                                my_rotation = int( rotation_degrees * x )
                                rotation_quaternion = Quaternion(axis=rotation_axis, degrees=my_rotation)

                                new_point = rotation_quaternion.rotate(sub_array[0]['TransformedVector'])
                                new_point = np.around(new_point, decimals=3)

                                rotated_points[x]['Point']       = new_point
                                rotated_points[x]['Rotation']    = my_rotation
                                rotated_points[x]['Designation'] = rot_designation[x]


                        for x in range(0, per_vertex):

                                for y in range(0, per_vertex):

                                        if np.allclose( sub_array[x]['TransformedVector'], rotated_points[y]['Point'], atol=1):

                                                # atol allows up to 1 angstrom math error
                                                rotation_order[x] = rot_designation[y]



                ####
                ## Now just need to apply these to expanded_user_vector
                ## Store them in sub_array, then copy to expanded_user_vector
                ####

                for z in range(0, len(sub_array) ):

                        if my_vertex == 'fivefold'  :
                                sub_array[z]['Vertex_5f_specific'] = rotation_order[z]
                        if my_vertex == 'threefold'  :
                                sub_array[z]['Vertex_3f_specific'] = rotation_order[z]
                        if my_vertex == 'twofold'  :
                                sub_array[z]['Vertex_2f_specific'] = rotation_order[z]

                        insert_where = np.where( expanded_user_vector['ExpandQuatIndex'] == sub_array[z]['ExpandQuatIndex'] )
                        insert_here = insert_where[0][0]

                        expanded_user_vector[insert_here] = sub_array[z]

        return expanded_user_vector



def slowPrint( text ) :
        for character in text:
                sys.stdout.write( character )
                sys.stdout.flush()
                random_delay = np.true_divide( np.random.randint(2,15), 100 )
                time.sleep( random_delay )
        print( "\n" )
        return


### Generate array containing I1 rotations ready for pyQuaternion in format [ a, bi, cj, dk ]
I1Quaternions = np.array(   [   [ 1.000, 0.000, 0.000, 0.000 ],    [ 0.000, 1.000, 0.000, 0.000 ],
                                [ 0.809, -0.500, 0.000, 0.309 ],   [ -0.309, 0.809, 0.000, -0.500 ],
                                [ 0.309, 0.809, 0.000, -0.500 ],   [ 0.809, 0.500, 0.000, -0.309 ],
                                [ -0.500, 0.809, 0.309, 0.000 ],   [ 0.500, 0.809, 0.309, 0.000 ],
                                [ 0.500, 0.809, -0.309, 0.000 ],   [ 0.809, 0.309, -0.500, 0.000 ],
                                [ 0.809, 0.309, 0.500, 0.000 ],    [ 0.809, -0.309, -0.500, 0.000 ],
                                [ 0.809, -0.309, 0.500, 0.000 ],   [ -0.500, 0.809, -0.309, 0.000 ],
                                [ 0.000, 0.809, 0.500, -0.309 ],   [ 0.500, 0.500, 0.500, -0.500 ],
                                [ 0.809, 0.000, 0.309, -0.500 ],   [ 0.809, -0.500, 0.000, -0.309 ],
                                [ 0.809, 0.500, 0.000, 0.309 ],    [ -0.500, 0.500, 0.500, -0.500 ],
                                [ 0.809, 0.000, -0.309, 0.500 ],   [ 0.809, 0.000, 0.309, 0.500 ],
                                [ -0.500, 0.500, -0.500, -0.500 ], [ 0.000, 0.809, -0.500, -0.309 ],
                                [ -0.309, 0.809, 0.000, 0.500 ],   [ 0.809, 0.000, -0.309, -0.500 ],
                                [ 0.500, -0.309, 0.000, 0.809 ],   [ 0.000, -0.500, 0.309, 0.809 ],
                                [ 0.500, 0.500, -0.500, -0.500 ],  [ -0.309, -0.500, 0.809, 0.000 ],
                                [ 0.000, 0.809, -0.500, 0.309 ],   [ 0.309, 0.809, 0.000, 0.500 ],
                                [ -0.500, 0.500, 0.500, 0.500 ],   [ 0.000, 0.809, 0.500, 0.309 ],
                                [ 0.309, 0.500, 0.809, 0.000 ],    [ 0.000, -0.500, -0.309, 0.809 ],
                                [ -0.500, -0.309, 0.000, 0.809 ],  [ -0.500, 0.000, 0.809, 0.309 ],
                                [ -0.309, 0.500, 0.809, 0.000 ],   [ -0.500, 0.000, 0.809, -0.309 ],
                                [ 0.500, 0.500, -0.500, 0.500 ],   [ 0.500, 0.500, 0.500, 0.500 ],
                                [ 0.500, 0.000, 0.809, 0.309 ],    [ 0.309, -0.500, 0.809, 0.000 ],
                                [ 0.500, 0.000, 0.809, -0.309 ],   [ -0.500, 0.500, -0.500, 0.500 ],
                                [ 0.000, 0.309, 0.809, -0.500 ],   [ -0.309, 0.000, -0.500, 0.809 ],
                                [ -0.500, 0.309, 0.000, 0.809 ],   [ 0.309, 0.000, -0.500, 0.809 ],
                                [ 0.500, 0.309, 0.000, 0.809 ],    [ 0.309, 0.000, 0.500, 0.809 ],
                                [ 0.000, -0.309, 0.809, 0.500 ],   [ 0.000, 0.000, 0.000, 1.000 ],
                                [ -0.309, 0.000, 0.500, 0.809 ],   [ 0.000, 0.500, 0.309, 0.809 ],
                                [ 0.000, 0.309, 0.809, 0.500 ],    [ 0.000, -0.309, 0.809, -0.500 ],
                                [ 0.000, 0.500, -0.309, 0.809 ],   [ 0.000, 0.000, 1.000, 0.000 ]     ]    )

def doSymbreak( particle_ndarray, particle_header, particle_fullheader, subparticle_ndarray ) :

        print( '\nInitializing.' )

        #### Keep unaltered version of original values
        pristineP = particle_ndarray.copy(order='C')
        pristineSP = subparticle_ndarray.copy(order='C')

        total_particles = len(pristineP)
        total_subparticles = total_particles * 60

        ### Parse particle star file
        rot_indexP, tilt_indexP, psi_indexP = getEulers( particle_header )
        originXAngst_indexP, originYAngst_indexP = getOffsetAngst( particle_header )
        defocusU_indexP, defocusV_indexP, defocusAngle_indexP = getDefocus( particle_header )
        class_indexP = getClass( particle_header )
        micrographname_indexP, imagename_indexP, particleID_indexP = getMicrographName( particle_header )
        uid_indexP = getUID( particle_header )
        vertexGroup_indexP = getVertexGroup( particle_header )
        OriginXYZAngstWrtParticleCenter_indexP = getOriginXYZAngstWrtParticleCenter( particle_header )
        relativePose_indexP = getCustomRelativePose( particle_header )

        ## Initialize subparticle index at 0. Will be used to generate subparticle UID
        subparticle_index = 0
       
        subpart_per_particle = int( np.true_divide( total_subparticles, total_particles ) )


        ## Sort the particle array by image name
        print( "  Sorting particle array by particle image name." )
#        print( "  Original item 568:", particle_ndarray[568][imagename_indexP] )
        particle_ndarray = particle_ndarray[ particle_ndarray[:,imagename_indexP].argsort()]
#        print( "  Sorted item   568:", particle_ndarray[568][imagename_indexP] )
        print( "  --> done!\n" )


        print( "Looping through all particles and subparticles." )
        slowPrint( "  --> Back-applying eulers and classes to parent particles.\n" )

        ## We need to copy the subparticle Eulers over the particle Eulers
        ## We also need the class
        for outer_index in range(0, subpart_per_particle):          # 0 thru 59

                ## Loop through all the particles
                for particle_index in range(0, total_particles):    # 0 thru max particle number

                    subparticle_index = ( 60 * particle_index ) + outer_index
                    this_subpart = subparticle_index

#                    print( subparticle_index )

#                    child_subparticles = np.where( subparticle_ndarray['ParticleImageName'] == particle_ndarray[particle_index][imagename_indexP] )

#                    print( child_subparticles )

                    ### Assign values from child to parent ###
                    particle_ndarray[particle_index][class_indexP ] = subparticle_ndarray[this_subpart]['Class'] 
                    particle_ndarray[particle_index][rot_indexP ] = subparticle_ndarray[this_subpart]['EulerRot'] 
                    particle_ndarray[particle_index][tilt_indexP] = subparticle_ndarray[this_subpart]['EulerTilt'] 
                    particle_ndarray[particle_index][psi_indexP ] = subparticle_ndarray[this_subpart]['EulerPsi'] 

                    ### Blank out placeholder values. Remnants of an earlier script
                    particle_ndarray[particle_index][ relativePose_indexP ] = ""
                    particle_ndarray[particle_index][ OriginXYZAngstWrtParticleCenter_indexP ] = ""
                    particle_ndarray[particle_index][ vertexGroup_indexP ] = ""
                    particle_ndarray[particle_index][ particleID_indexP ] = ""
                    particle_ndarray[particle_index][ uid_indexP ] = ""


                    if ( particle_index == 0 ) :
                        print( 'Iteration:', outer_index )
                        print( '  Particle:', particle_index, ':', particle_ndarray[particle_index][class_indexP ], particle_ndarray[particle_index][rot_indexP ], particle_ndarray[particle_index][tilt_indexP ], particle_ndarray[particle_index][psi_indexP ]  )

                ### Make the file, add the header, only once per run
                if outer_index == 0:
                        filename = ''.join( [ 'symbreak', '.star' ])
                        f = open( filename, 'w' )
                        np.savetxt( f, particle_fullheader, delimiter=' ', fmt="%s" )
                        f.close()

                ### Write current batch of particles to the star file each iteration
                f = open( filename, 'a' )       # open in append mode
                np.savetxt( f, particle_ndarray, delimiter=' ', fmt="%s")
                f.close()       

        return

def main(args):


        if args.particle_file.endswith(".star"):
                print( "  Parsing particle star file." )
                particle_filename = args.particle_file
                particle_header, particle_fullheader = getStarHeader( particle_filename )
                particle_stardata = getStarData( particle_filename, len( particle_header ), particle_header )
                particle_ndarray = np.asarray( particle_stardata, order='C' )

                num_particles = ( len(particle_ndarray) )

        if args.subparticle_file.endswith(".star"):
                print( "  Parsing subparticle star file." )
                subparticle_filename = args.subparticle_file
                subparticle_header, subparticle_fullheader = getStarHeader( subparticle_filename )
                print( "  --> done!" )
                print( "  Making ndarray with minimal necessary items." )
                subparticle_ndarray = getMinimalStarData( subparticle_filename, subparticle_header, len( subparticle_header ), num_particles )

        else:
                print( "Please provide valid input files." )

        doSymbreak( particle_ndarray, particle_header, particle_fullheader, subparticle_ndarray )

        return 0


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--particle_file", type=str, required=True)
        parser.add_argument("--subparticle_file", type=str, required=True)
        sys.exit(main(parser.parse_args()))

