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
from isecc import transform
from isecc import starparse
from isecc import symops
from isecc import checks
from isecc import utils
from isecc.isecc_classes import Particle
from isecc.isecc_classes import AreaOfInterest

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

    if ROI == 'fivefold':    my_vertex_string = ''.join( [ '5f', fivefold_general ] )
    if ROI == 'threefold':    my_vertex_string = ''.join( [ '3f', threefold_general ] )
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
            if np.allclose( transformed_vector, vertex_table[loop_index]['TransformedVector'], rtol=0.1) :
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

#    user_vector = np.array( [0, 0.618, 1] )

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
    rot_index, tilt_index, psi_index = starparse.getEulers( header )


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
        original_pose = Quaternion( transform.myEuler2Quat( my_rotrad, my_tiltrad, my_psirad ) )

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
                if np.allclose( transformed_vector, master_table[particle_index]['TransformedVector'][loop_index], rtol=0.01) :
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
            
    if ROI == 'fullexpand':        return master_table, np.arange(0,60)
    elif ROI == 'fivefold':        return master_table, unique_5f_indices
    elif ROI == 'threefold':    return master_table, unique_3f_indices
    elif ROI == 'twofold':        return master_table, unique_2f_indices

    else:    return master_table


def ZZdefineAreaOfInterest(ROI, user_vector, user_fudge, higher_order_sym):
    """ This code may be not currently called """

    area_of_interest = ROI
    my_vector = user_vector
    my_vector = my_vector * user_fudge          # allows you to bump vector in or out a bit
    my_sym = higher_order_sym


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

    return this_5f_symgroup, this_3f_symgroup, this_2f_symgroup


def extractRotationAssignment( rotation_assignment ):

    if str(rotation_assignment) == str([b'a'])   :  return str('a')
    elif str(rotation_assignment) == str([b'b']) :  return str('b')
    elif str(rotation_assignment) == str([b'c']) :  return str('c')
    elif str(rotation_assignment) == str([b'd']) :  return str('d')
    elif str(rotation_assignment) == str([b'e']) :  return str('e')
    elif str(rotation_assignment) == str([b''])  :    return str('z')

    else:
        print( 'rotation_assignment not understood.' )
        print( rotation_assignment )
        sys.exit()

    return


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

            distance_check = utils.assess3dDistance( point['TransformedVector'], vertex['VertexCoords'])
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

    assigned_rotations = np.array([])    # Empty array to store list of assigned rotations

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
                try:
                    sub_array[index] = expanded_user_vector[ my_vertex_partners[0][index] ]
                except:
                    print("Ambiguous Yet Fatal Error. Is your vector roughly equidistant to two or more", ROI, "vertices?" )
                    print("That would be break ASU addresses. Try shifting your vector a bit.")
                    sys.exit()

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


### Generate array containing I1 rotations ready for pyQuaternion in format [ a, bi, cj, dk ]
I1Quaternions = symops.getSymOps()



def defineSubparticles( my_ndarray, ROI, user_vector, user_fudge, user_subbox, higher_order_sym, user_testmode, user_batch_size, header, fullheader, RUN_ID, regen_string, user_batch=None ) :

    print( '\nInitializing.' )
    print( '  Note: Sign of local defocus adjustment has been corrected as of 20 Dec 2019.' )

    """ Intercept user_vector and return idealized_vector within desired asymmetric unit. """
    """ This is essential for proper nearest_vertex assignment. """
    user_vector = checks.idealizeUserVector( user_vector, ROI )

    checks.checkVector(user_vector, ROI, higher_order_sym)


    """ Testmode generates 10k subparticles from a random subset of particles """
    if user_testmode:
        my_ndarray = utils.random_subsample( my_ndarray , ROI)

    """ Keep unaltered version of original values """
    pristine = my_ndarray.copy(order='C')
    total_particles = len(pristine)

    """ Get parameters for requested symmetry operation """
    expand_params = AreaOfInterest( ROI, user_vector )
    expand_params.apply_fudge( user_fudge )

    """ Stupid assignments """
    my_vector = expand_params.vector
    model_sym = expand_params.model_sym
    short_roi = expand_params.short_roi

    """ Parse star file header """
    rot_index, tilt_index, psi_index = starparse.getEulers( header )
    originXAngst_index, originYAngst_index = starparse.getOffsetAngst( header )
    defocusU_index, defocusV_index, defocusAngle_index = starparse.getDefocus( header )
    class_index = starparse.getClass( header )
    micrographname_index, imagename_index, particleID_index = starparse.getMicrographName( header )
    uid_index = starparse.getUID( header )
    vertexGroup_index = starparse.getVertexGroup( header )
    OriginXYZAngstWrtParticleCenter_index = starparse.getOriginXYZAngstWrtParticleCenter( header )
    relativePose_index = starparse.getCustomRelativePose( header )


    BATCH_MODE = None
    if user_batch == True:
        BATCH_MODE = True
        batch_size = user_batch_size
        print( "  Note: Batch mode will be used to speed subparticle generation in relion." )
        print( "  Note: Requested batch size is", batch_size, "\n" )

    ## Initialize subparticle index at 0. Will be used to generate subparticle UID
    subparticle_index = 0


    ####
    ## Generate vertex assignments from a single particle.
    ## These will then be applied to all particles.
    ## Any given symop will always result in the same vertex assignment.
    ####

    vertex_check_array = pristine[0:1]

    """ Fetch the indices for non-redundant symmetry operations """
    if (expand_params.roi == 'fivefold') or (expand_params.roi == 'threefold') or (expand_params.roi == 'twofold') or (expand_params.roi == 'fullexpand'):
        master_array, unique_indices = prepareSubparticleTable( vertex_check_array, user_vector, header, ROI )
    else:
        """ when roi is null """
        master_array = prepareSubparticleTable( vertex_check_array, user_vector, header, ROI )
        unique_indices = np.arange(1)

    """ Update to use only the unique symmetry operations """
    expand_params.updateSymIndices( unique_indices )
    symops = I1Quaternions[ expand_params.symindices ]

    # Start iterating through the star file
    for index, symop in enumerate(symops, start=0):        # Iterate over all symmetry ops

        """ Refresh the star array with the original values """
        my_ndarray = pristine.copy(order='C')

        """ Set the current I1 rotation, make string for relativePose """
        symop_quat = Quaternion( symop )

        """ Format items for the current symop """
        relativePose_string = str(symop_quat).strip().replace(" ", ",")
        my_vertexGroup = formatVertexAssignment( master_array[0], expand_params.symindices[index], ROI )

        """ Inform user of progress """
        print( "  Applying symmetry rotation", symop_quat, "  (", index+1, "of", len(symops), ") (", my_vertexGroup, ")"  )

        """ Begin symmetry operations """
        for x in range(0,len(my_ndarray)):          # Iterate over all particles


            """ Create Particle instance for the current particle """
            particle = Particle( imageName = my_ndarray[x][imagename_index] ,
                originXAngst = my_ndarray[x][originXAngst_index] ,
                originYAngst = my_ndarray[x][originYAngst_index] ,
                angleRot  = my_ndarray[x][rot_index] ,
                angleTilt = my_ndarray[x][tilt_index] ,
                anglePsi  = my_ndarray[x][psi_index]  )

            particle.add_defocus_info( defocusU = my_ndarray[x][defocusU_index] ,
                defocusV = my_ndarray[x][defocusV_index] ,
                defocusAngle = my_ndarray[x][defocusAngle_index],
                phaseShift = 0 )

            """ Renaming hack to implement batchmode """
            if BATCH_MODE:

                """ Subparticles from a given particle will share the same batch """
                particle_number = int(x)                    
                batch_size_symop = int(batch_size / len(symops))

                """ Calculate and format batch number """
                batch_num = int(particle_number / batch_size_symop) + 1
                batch_num = str(batch_num).rjust(6, '0')    # pad to 6 digits
            
                """ Store original name; assign new name """
                my_originalname = my_ndarray[x][imagename_index]
                my_ndarray[x][particleID_index] = my_originalname
                new_name = ''.join(['subparticles/', RUN_ID, '/Micrographs/batch', batch_num, '.mrcs']) 
                my_ndarray[x][micrographname_index] = new_name
            
                """ Store a customUID in rlnCustomUID. Unique for each subparticle. """
                subparticleUID = str(subparticle_index+1).rjust(9, '0')
                subparticleUID = ''.join( [ 'subparticleUID_', subparticleUID ] )
                particleUID = int(x+1)  # never used
                my_UID = subparticleUID


            """ Rotate Particle, define subparticles """
            my_quat = particle.pose
            particle.rotateParticle( symop_quat, expand_params.quatZ )
            particle.defineSubparticle( my_vector )

            """ Assign for writing to star file """
            my_ndarray[x][rot_index]  = np.around( particle.subpartRot,  decimals=6 )
            my_ndarray[x][tilt_index] = np.around( particle.subpartTilt, decimals=6 )
            my_ndarray[x][psi_index]  = np.around( particle.subpartPsi,  decimals=6 )
            my_ndarray[x][originXAngst_index] = np.around( particle.subpartX, decimals=6 )
            my_ndarray[x][originYAngst_index] = np.around( particle.subpartY, decimals=6 )
            my_ndarray[x][defocusU_index] =  np.around( particle.subpartZU, decimals=6 )
            my_ndarray[x][defocusV_index] = np.around( particle.subpartZV, decimals=6 )

            """ Assign CustomUID, VertexGroup, relativePose """
            my_ndarray[x][uid_index] = my_UID
            my_ndarray[x][vertexGroup_index]  = my_vertexGroup
            my_ndarray[x][relativePose_index] = relativePose_string

            """ Format subparticle XYZ offset with respect to the particle origin """
            rounded_x = str( np.around( particle.rotated_vector[0], decimals=4 ) )
            rounded_y = str( np.around( particle.rotated_vector[1], decimals=4 ) )
            rounded_z = str( np.around( particle.rotated_vector[2], decimals=4 ) )
            XYZ_string = ','.join( [rounded_x, rounded_y, rounded_z] )

            """ Assign XYZ string """
            my_ndarray[x][OriginXYZAngstWrtParticleCenter_index] = XYZ_string

            """ Increment subparticle index """
            subparticle_index = subparticle_index + 1

        if index == 0:
            expanded_star = my_ndarray.copy(order='C')
        if index != 0:
            expanded_star = np.concatenate( ( expanded_star, my_ndarray ) ,axis=0)


    ### end loop through the symmetry-operators

    """ Write star file, begin operations in relion """

    ### Prepare the output star file
    filename = ''.join( [ ROI, 'subparticle_alignments.star' ])
    f = open( filename, 'w' )
    np.savetxt( f, fullheader, delimiter=' ', fmt="%s" )
    f.close()

    ### Write the particles to the star file
    f = open( filename, 'a' )
    np.savetxt( f, expanded_star, delimiter=' ', fmt="%s")
    f.close()    

    ### DEBUG
    print( '\nWrote', filename )

    print( "\nSubparticle centers and orientations have successfully been defined in star 3.1 format.\n" )
    print( "WARNING! Please verify that you refined with", higher_order_sym, "symmetry" )

    ## Setting up test mode vs. real mode
    if user_testmode:
        ## Send message to user
        print( "  Note: Test Mode selected. Will only generate ~10k subparticles so you can check initial model.\n" )

        ## Take only 10k particles
        fullstar = ''.join( [ ROI, 'subparticle_alignments.star' ] )
        shortstar = ''.join( [ ROI, 'subparticle_alignments_abbrev.star' ] )
        cmd = ''.join( [ 'head -n10099 ', fullstar, ' > ', shortstar ] )
        print( "Executing command:", cmd )
        os.system( cmd )

        input = shortstar
    else:
        fullstar = ''.join( [ ROI, 'subparticle_alignments.star' ] )
        input = fullstar


    ## This step recenters on subparticle origins
    cmd = ''.join( ['relion_stack_create --i ', input, ' --o ', ROI, ' --apply_rounded_offsets_only --split_per_micrograph > /dev/null' ] )
    print( "Executing command:", cmd )
    print( "\n  NOTE: This will take some time..." )
    utils.slowPrint( str('             ...isn\'t it a nice day for a bike ride?') )
    os.system( cmd )

    new_box = str(user_subbox)
    cmd = ''.join( [ 'relion_image_handler --i ', ROI,'.star --o subpart --new_box ', new_box ] )
    print( "\nExecuting command:", cmd )
    os.system( cmd )


    ### Add comments to roi_subpart.star file
    filename = ''.join( [ ROI, '_subpart.star' ] )
    file_comments = fullheader[:3]
    ### Add info for how to regenerate subparticles
    if '--timestamp_run' not in file_comments[2] :
        file_comments[2] = ' '.join( [ file_comments[2], '--timestamp_run', regen_string ] )
    new_filename = ''.join( [ ROI, '_subpart.star.temp' ] )

    f = open( new_filename, 'w' )
    np.savetxt( f, file_comments, delimiter=' ', fmt="%s" )
    f.close()

    cmd = ' '.join( [ 'cat', filename, ">>", new_filename ] )
    os.system( cmd )
    cmd = ' '.join( [ 'mv', new_filename, filename ] )
    os.system( cmd )
#    filename = new_filename


    ## Delete whole particle images
    ## Example path is fivefold_subparticles/fivefold/particle000000001.mrcs
    if BATCH_MODE:
        particle_files = ''.join( [ ROI, '_subparticles/', RUN_ID, '/Micrographs/batch??????.mrcs' ] )
    else:
        particle_files = ''.join( [ ROI, '_subparticles/', RUN_ID, '/Micrographs/particle?????????.mrcs' ] )

    cmd = ''.join( ['rm ', particle_files ] )
    print( "Executing command:", cmd )
    os.system( cmd )


    ## Make initial model star file from 1st 10k lines of ROI_subpart.star
    fullfile = filename
    initialmodel_star = ''.join( [ ROI, '_initialmodel.star' ] )
    cmd = ''.join( [ 'head -n10000 ', fullfile, ' > ', initialmodel_star ] )
    print( "Executing command:", cmd )
    os.system( cmd )


    ## Make initial model
    initialmodel_mrc = ''.join( [ ROI, '_initialmodel_', model_sym,'.mrc' ] )
    cmd = ''.join( [ 'relion_reconstruct --i ', initialmodel_star, ' --o ', initialmodel_mrc ,' --ctf --maxres 10 --sym ', model_sym ] )
    print( "\nGenerating initial model" )
    print( "Executing command:", cmd, "\n" )
    os.system( cmd )

    ## Make a copy of the subparticle file for the Priors
    filename = ''.join( [ ROI, '_subpart.star' ] )
    filename_PRIOR = ''.join( [ ROI, '_subpart_PRIOR.star' ] )
    cmd = ' '.join( [ 'cp', filename, filename_PRIOR ] )
    print( "\nExecuting command:", cmd )
    os.system( cmd )


    ### Modify the Priors file
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


    ## Move files to the job directory
    job_directory = ''.join( [ ROI, '_subparticles/', RUN_ID, '/' ] )
    print( "  NOTE: Moving files to output directory:", job_directory )

    roi_star = ''.join( [ ROI, '.star' ] )
    roi_alignments = ''.join( [ ROI, 'subparticle_alignments.star' ] )
    roi_alignments_abbrev = ''.join( [ ROI, 'subparticle_alignments_abbrev.star' ] )

    to_move = np.array( [filename, filename_PRIOR, initialmodel_mrc, initialmodel_star, roi_star, roi_alignments ] )

    for file in to_move :
        cmd = ' '.join( [ 'mv', file, job_directory ] )
        os.system( cmd )
    if user_testmode:
        cmd = ''.join( [ 'mv ', ROI, 'subparticle_alignments_abbrev.star ', job_directory ] )
        os.system( cmd )

    print( "\nSuccess!\n" )


    return



def main(args):

    user_batch = None
    if args.batch == 'true':
        user_batch = True
    
    user_testmode = None
    if args.testmode == 'true':
        user_testmode = True

    ### Today's date will be used in the subparticle path
    date = str( datetime.now().strftime("%Y%m%d") )
    hour = str( datetime.now().strftime("%H%M") )
    RUN_ID = '_'.join( [ args.roi, date, hour ] )
    regen_string = '_'.join( [ date, hour ] )

    if args.timestamp_run :
        RUN_ID = '_'.join( [ args.roi, args.timestamp_run ] )
        regen_string = args.timestamp_run

    if args.input.endswith(".star"):
        filename = args.input
        header, fullheader = starparse.getStarHeader( filename, regen_string )
        stardata = starparse.getStarData( filename, len( header ) )
        my_ndarray = np.asarray( stardata, order='C' )
        defineSubparticles(  my_ndarray, args.roi, np.array(args.vector), args.fudge, args.subpart_box, args.supersym, user_testmode, args.batchsize, header, fullheader, RUN_ID, regen_string, user_batch )
    else:
        print( "Please provide a valid input file." )
    sys.exit()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to cryosparc_P#_J#_###_particles.cs file")
    parser.add_argument("--roi", choices=['null', 'fivefold', 'threefold', 'twofold', 'fullexpand'], type=str.lower, required=True)
    parser.add_argument("--vector", type=float, nargs=3, required=True, help="X Y Z in Angstroms (space delimited)")
    parser.add_argument("--fudge", type=float, nargs=1, default='1', help="scale vector a bit, e.g. 0.8")
    parser.add_argument("--supersym", choices=['I1', 'I2'], type=str.upper, default='I1')
    parser.add_argument("--subpart_box", type=int, required=True, help="box size for subparticles")
    parser.add_argument("--batch", choices=['true', 'false'], type=str.lower, default='true', help="relion will process in batches rather than per-particle")
    parser.add_argument("--batchsize", type=int, default=3000)
    parser.add_argument("--testmode", choices=['true', 'false'], type=str.lower, default='false', help="Generate ~10k subparticles so you can verify settings by inspecting the initial model")
    parser.add_argument("--timestamp_run", type=str, required=False, help="Allows you to re-create subparticles from a previous run of the script. Manually sets the timestamp string in the output path.")
    sys.exit(main(parser.parse_args()))

