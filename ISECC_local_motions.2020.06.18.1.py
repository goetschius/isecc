#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime


def getMinimalStarData( my_star, subparticle_header, subparticle_header_length, num_subparticles, subparticle_type=None ):

    ### Get index for icosahedral (priors) and locally refined X and Y values

    ## Pentavalent
    icos_X_index,  icos_Y_index  =  getOffsetAngstPriors( subparticle_header )
    local_X_index, local_Y_index =  getOffsetAngst( subparticle_header )

    ### Get index for the local XYZ relative to particle origin
    relativeXYZ_index = getOriginXYZAngstWrtParticleCenter( subparticle_header )

    ### Get index for the vertex group assignment
    vertexGroup_index = getVertexGroup( subparticle_header )

    ### Get index for whole particle unique identifier
    micrographname_index, imagename_index, particleID_index = getMicrographName( subparticle_header )


    subparticle_array_dtype = np.dtype( [   ( 'ParticleSpecifier', '|U200' ),
                        ( 'Subparticle_type', '|S11' ),
                        ( 'Vertex5f_general', 'i4' ), ( 'Vertex5f_specific', '|S1' ),
                        ( 'Vertex3f_general', 'i4' ), ( 'Vertex3f_specific', '|S1' ),
                        ( 'Vertex2f_general', 'i4' ), ( 'Vertex2f_specific', '|S1' ),
                        ( 'SubparticleOrigin_icos',  '<f4', (2,) ),
                        ( 'SubparticleOrigin_local', '<f4', (2,) ),
                        ( 'SubparticleRelative_XYZ', '<f4', (3,) )  ] )

    minimal_subparticle_array = np.zeros( num_subparticles, dtype=subparticle_array_dtype )


    subparticle_index = 0

    with open(my_star, "r") as my_star:
        stardata = []

        START_PARSE = None      # Needed for relion 3.1 star format

        for line in my_star:

            if line.strip() == 'data_particles':
                START_PARSE = True

            if START_PARSE:

                linesplit = line.split()

                if len( linesplit ) > 4:    # get beyond the metadata labels

                    ### Not sure below line is really needed
                    line = ' '.join( [ linesplit[particleID_index], linesplit[icos_X_index], linesplit[icos_Y_index], linesplit[local_X_index], linesplit[local_Y_index] ] )

                    if line[0] != '#':      # avoid the stupid comment line
                        #stardata.append( line )


                        # Particle Specifier
                        minimal_subparticle_array[subparticle_index]['ParticleSpecifier'] = linesplit[ particleID_index ]
                        # Subparticle Type
                        minimal_subparticle_array[subparticle_index]['Subparticle_type'] = subparticle_type
                        # X,Y Priors
                        minimal_subparticle_array[subparticle_index]['SubparticleOrigin_icos'] = linesplit[ icos_X_index ], linesplit[ icos_Y_index ]
                        # X,Y Locally Refined
                        minimal_subparticle_array[subparticle_index]['SubparticleOrigin_local'] = linesplit[ local_X_index ], linesplit[ local_Y_index ]
                        # Relative X,Y,Z with respect to particle center
                        minimal_subparticle_array[subparticle_index]['SubparticleRelative_XYZ'] = linesplit[ relativeXYZ_index ].split(',')

                        # Vertex Assignment
                        if subparticle_type == 'pentavalent':
                            vertex_assignment = int( str( linesplit[vertexGroup_index] )[2:4] )  #example: '5f01'
                            minimal_subparticle_array[subparticle_index]['Vertex5f_general'] = vertex_assignment

                        elif subparticle_type == 'hexavalent':
                                    vertex_assignment = str( linesplit[vertexGroup_index] )    # example: '5f01a.3f01a.2f01a'
                                    minimal_subparticle_array[subparticle_index]['Vertex5f_general' ] = vertex_assignment[2:4]      # character 3-4
                                    minimal_subparticle_array[subparticle_index]['Vertex5f_specific'] = vertex_assignment[4]        # character 5
                                    minimal_subparticle_array[subparticle_index]['Vertex3f_general' ] = vertex_assignment[8:10]     # character 9-10
                                    minimal_subparticle_array[subparticle_index]['Vertex3f_specific'] = vertex_assignment[10]       # character 11
                                    minimal_subparticle_array[subparticle_index]['Vertex2f_general' ] = vertex_assignment[14:16]    # character 15-16
                                    minimal_subparticle_array[subparticle_index]['Vertex2f_specific'] = vertex_assignment[16]       # character 17


                        subparticle_index = subparticle_index + 1

    print( "  --> sorting based on particle image name." )
    minimal_subparticle_array.sort(order='ParticleSpecifier')
    print( "  --> done!" )


#    print('\nExample:')
#    print( minimal_subparticle_array.dtype )
#    print( minimal_subparticle_array[0], '\n' )


    return minimal_subparticle_array



def myEuler2Quat( phi, theta, psi ) :        # doi.org/10.1101/733881
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


def rot2euler(r):               # pyem, Daniel Asarnow
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


def getStarHeader( my_star, regen_string ):
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
        
        START_PARSE = None    # Needed for relion 3.1 star format

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
#        header.append( 'rlnCustomOriginXYZAngstWrtParticleCenter' )
#        header_index = int( header_index + 1 )
#        append_this = ''.join( [ '_rlnCustomOriginXYZAngstWrtParticleCenter #', str(header_index) ] )
#        full_header.append( append_this )

        print( "rlnCustomOriginXYZAngstWrtParticleCenter not in header" )


    if 'rlnCustomRelativePose' not in header:
#        header.append( 'rlnCustomRelativePose' )
#        header_index = int( header_index + 1 )
#        append_this = ''.join( [ '_rlnCustomRelativePose #', str(header_index) ] )
#        full_header.append( append_this )

        print( "rlnCustomRelativePose not in header" ) 


    ### Add info for how to regenerate subparticles
    if '--timestamp_run' not in full_header[2] :
        full_header[2] = ' '.join( [ full_header[2], '--timestamp_run', regen_string ] )

    return header, full_header


def getStarData( my_star, header_length ):
    with open(my_star, "r") as my_star:
        stardata = []

        START_PARSE = None      # Needed for relion 3.1 star format

        for line in my_star:

            if line.strip() == 'data_particles':
                START_PARSE = True

            if START_PARSE:

                linesplit = line.split()

                #print( len( linesplit ), header_length )

                if len( linesplit ) == header_length:
                    if line[0] != '#':      # avoid the stupid comment line
                        stardata.append( linesplit )
    return stardata


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

def getOffsetAngstPriors( header ) :
    # rlnOriginXPriorAngst, rlnOriginYPriorAngst
    originXPriorAngst_index = header.index('rlnOriginXPriorAngst')
    originYPriorAngst_index = header.index('rlnOriginYPriorAngst')

    return originXPriorAngst_index, originYPriorAngst_index


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

def slowPrint( text ) :
    for character in text:
        sys.stdout.write( character )
        sys.stdout.flush()
        random_delay = np.true_divide( np.random.randint(2,15), 100 )
        time.sleep( random_delay )
    print( "\n" )
    return

def assess3dDistance( point1, point2 ):

    ### stackoverflow.com/questions/1401712
    distance = np.linalg.norm( point1 - point2 )

    ### equivalent below
    #distance = np.sqrt( ( point1[0] - point2[0] )**2 + ( point1[1] - point2[1] )**2 + ( point1[2] - point2[2] )**2 )
    return distance


def generateSubparticleArray( ndarray_pent, header_pent, ndarray_hex, header_hex ):

    ### Get index for icosahedral (priors) and locally refined X and Y values

    ## Pentavalent
    #icos_X_index_pent,  icos_Y_index_pent  =  getOffsetAngstPriors( header_pent )
    #local_X_index_pent, local_Y_index_pent =  getOffsetAngst( header_pent )

    ## Hexavalent
    #icos_X_index_hex,  icos_Y_index_hex  =  getOffsetAngstPriors( header_hex )
    #local_X_index_hex, local_Y_index_hex =  getOffsetAngst( header_hex )


    ### Get index for the local XYZ relative to particle origin
    #relativeXYZ_index_pent = getOriginXYZAngstWrtParticleCenter( header_pent )
    #relativeXYZ_index_hex = getOriginXYZAngstWrtParticleCenter( header_hex )

    ### Get index for the vertex group assignment
    #vertexGroup_index_pent = getVertexGroup( header_pent )
    #vertexGroup_index_hex = getVertexGroup( header_hex )

    ### Get index for whole particle unique identifier
    #micrographname_index_pent, imagename_index_pent, particleID_index_pent = getMicrographName( header_pent )
    #micrographname_index_hex, imagename_index_hex, particleID_index_hex = getMicrographName( header_hex )


    ### Example vertex specifier from hexavalent:    5f01a.3f01a.2f01a
    ### Example vertex specifier from pentavalent:    5f01

    total_particle_number = np.true_divide( len( ndarray_pent ), 12 )
    total_particle_number2 = np.true_divide( len( ndarray_hex ), 60 )
    total_subparticle_number = int( len( ndarray_pent ) + len( ndarray_hex ) )

    if total_particle_number != total_particle_number2:
        print( 'ERROR: Total particles from pentavalent file does not equal particle number from hexavalent!' )
        print( total_particle_number, '!=', total_particle_number2 )
        sys.exit()


    subparticle_array_dtype = np.dtype( [   ( 'ParticleSpecifier', '|S200' ), 
                        ( 'Subparticle_type', '|S11' ), 
                        ( 'Vertex5f_general', 'i4' ), ( 'Vertex5f_specific', '|S1' ),
                        ( 'Vertex3f_general', 'i4' ), ( 'Vertex3f_specific', '|S1' ),
                        ( 'Vertex2f_general', 'i4' ), ( 'Vertex2f_specific', '|S1' ),
                        ( 'SubparticleOrigin_icos',  '<f4', (2,) ),
                        ( 'SubparticleOrigin_local', '<f4', (2,) ),
                        ( 'SubparticleRelative_XYZ', '<f4', (3,) )  ] )

    subparticle_array = np.zeros( total_subparticle_number, dtype=subparticle_array_dtype )

    ### Read everything into a subparticle array
    print( '\nAdding all pentavalent capsomers into subparticle array.' )
    
    ### Start with the pentavalent
    for index in range(0, len( ndarray_pent ) ):

        subparticle_array[index] = ndarray_pent[index]

    ### Here we want to also find the max and min z values.
    ### Below syntax gets the 3rd column (z) of field 'SubparticleRelative_XYZ'
    max_z = np.amax( subparticle_array[:]['SubparticleRelative_XYZ'][:,2] )
    min_z = np.amin( subparticle_array[:]['SubparticleRelative_XYZ'][:,2] )
    print( '  Note: Relative z range is from', min_z, 'to', max_z )
    

    ### Add in the hexavalents
    print( 'Adding all hexavalent capsomers into subparticle array.' )

    for index in range( 0, len( ndarray_hex ) ):

        # Sort out the two indexes
        hex_index = index
        subpart_index = int( len( ndarray_pent ) + hex_index )

        subparticle_array[subpart_index] = ndarray_hex[hex_index]

    return subparticle_array, max_z


def calculateNeighborDistance( reference, neighbor, relationship ):

    """ Grab the parameters for the reference hexavalent """
    ### Relevant parameters
    reference_x_icos  = reference['SubparticleRelative_XYZ'][0]
    reference_y_icos  = reference['SubparticleRelative_XYZ'][1]
    reference_z_icos  = reference['SubparticleRelative_XYZ'][2]
    ### Value >0 if localOrigin > icosOrigin. Verify the sign!
    reference_delta_x = reference['SubparticleOrigin_local'][0] - reference['SubparticleOrigin_icos'][0]
    reference_delta_y = reference['SubparticleOrigin_local'][1] - reference['SubparticleOrigin_icos'][1]
    ### Add in the deltas
    reference_x_local = reference['SubparticleRelative_XYZ'][0] + reference_delta_x
    reference_y_local = reference['SubparticleRelative_XYZ'][1] + reference_delta_y
    ### Flatten along z
    reference_xy_icos  = np.array( [ reference_x_icos,  reference_y_icos,  0 ] )
    reference_xy_local = np.array( [ reference_x_local, reference_y_local, 0 ] )
    ### Maintain z
    reference_xyz_icos = np.array( [ reference_x_icos, reference_y_icos, reference_z_icos ] )

    ideal_distance_list=[]
    real_distance_list=[]
    delta_list=[]    

    """ Grab parameters for neighbor """
    ### Relevant parameters
    neighbor_x_icos  = neighbor['SubparticleRelative_XYZ'][0]
    neighbor_y_icos  = neighbor['SubparticleRelative_XYZ'][1]
    neighbor_z_icos  = neighbor['SubparticleRelative_XYZ'][2]
    ## Value >0 if localOrigin > icosOrigin. Verify the sign!
    neighbor_delta_x = neighbor['SubparticleOrigin_local'][0] - neighbor['SubparticleOrigin_icos'][0]
    neighbor_delta_y = neighbor['SubparticleOrigin_local'][1] - neighbor['SubparticleOrigin_icos'][1]
    ### Add in the deltas
    neighbor_x_local = neighbor['SubparticleRelative_XYZ'][0] + neighbor_delta_x
    neighbor_y_local = neighbor['SubparticleRelative_XYZ'][1] + neighbor_delta_y
    ### Flatten along z
    neighbor_xy_icos  = np.array( [ neighbor_x_icos,  neighbor_y_icos, 0 ] )
    neighbor_xy_local = np.array( [ neighbor_x_local, neighbor_y_local, 0 ] )
    ### Maintain z
    neighbor_xyz_icos = np.array( [ neighbor_x_icos, neighbor_y_icos, neighbor_z_icos ] )
    
    ### Run distance calculations
    ideal_distance = np.around( assess3dDistance( reference_xyz_icos,  neighbor_xyz_icos ), decimals=1 )
    ideal_distance_flattened = np.around( assess3dDistance( reference_xy_icos,  neighbor_xy_icos ), decimals=1 )
    real_distance_flattened  = np.around( assess3dDistance( reference_xy_local, neighbor_xy_local ), decimals=1 )
    delta_flattened = np.around( assess3dDistance( reference_xy_icos,  neighbor_xy_icos ) - assess3dDistance( reference_xy_local, neighbor_xy_local ), decimals=1 )

    print( relationship )
    print("This capsomer evaluated:", ideal_distance_flattened, real_distance_flattened, delta_flattened, "\n" )
    
    
    #ideal_distance_list.append(ideal_distance_flattened)
    #real_distance_list.append(real_distance_flattened)
    #delta_list.append(delta_flattened)

    return

def correlateRefinedCapsomers( subparticle_array, max_z, inclusion_threshold, diameter_threshold ) :

    particle_names = np.array( subparticle_array['ParticleSpecifier'] )

    unique_particles = np.unique( particle_names )

    z_threshold = np.around( (inclusion_threshold * max_z), decimals = 2 )
    central_slice_threshold = np.around( ( diameter_threshold * max_z), decimals = 2 )

    print( '\nNote: Current threshold is', inclusion_threshold, 'of particle radius. Will only consider vertices where:' )
    print( '   pentavalent relative Z >', z_threshold, 'or' )
    print( '   pentavalent relative Z <', (-1*z_threshold), '\n' )

    print( 'Will report deltas in distance between hexavalent and pentavalent capsomer' )
    print( '   ...as compared to icosahedral refinement. Values in Angstroms.' )
    print( '   Note: Z-dimension is flattened for this analysis.\n' )

    print( 'For diameter analysis, will consider deltas in z-range of:' )
    print( '   ',(-1*central_slice_threshold),'< pentavalent relative Z <', central_slice_threshold )
    print( '   This represents a threshold of:', diameter_threshold )
    print( '   Note: Z-dimension is flattened for this analysis.\n' )


#    sys.exit()

    ### iterate through all the unique particles
    for index, item in enumerate(unique_particles, start=0):

        ### Particle specifier for printout
        particle_specifier = str(index).rjust(5, '0')

        # make temporary array with all subparticles from current particle
        condition = subparticle_array['ParticleSpecifier'] == item

        #subparticles = np.extract( condition, subparticle_array )    # Don't need to use np.extract
        subparticles = subparticle_array[condition]

        #### Check distances along central plane (Z~=0)
        assessed_diameter = []
        max_diameter = 0


#        print( 'DEBUG', len(subparticles) )

        for assessed_index, subparticle in enumerate(subparticles, start=0):

            ### Icos coords
            reference_x_icos  = subparticle['SubparticleRelative_XYZ'][0]
            reference_y_icos  = subparticle['SubparticleRelative_XYZ'][1]
            reference_z_icos  = subparticle['SubparticleRelative_XYZ'][2]
            reference_xy_icos = np.array( [ reference_x_icos, reference_y_icos, 0 ] )
            reference_xyz_icos = np.array( [ reference_x_icos, reference_y_icos, reference_z_icos ] )

            ### Local coords
            reference_delta_x = subparticle['SubparticleOrigin_local'][0] - subparticle['SubparticleOrigin_icos'][0]
            reference_delta_y = subparticle['SubparticleOrigin_local'][1] - subparticle['SubparticleOrigin_icos'][1]
            reference_x_local = subparticle['SubparticleRelative_XYZ'][0] + reference_delta_x
            reference_y_local = subparticle['SubparticleRelative_XYZ'][1] + reference_delta_y
            reference_xy_local = np.array( [ reference_x_local, reference_y_local, 0 ] )

            """ This code is for diameter assessment """
            if (reference_z_icos > (-1*central_slice_threshold)) and (reference_z_icos < central_slice_threshold) and (assessed_index not in assessed_diameter):

                for compare_index, compare in enumerate(subparticles, start=0):

                    ### Icos coords
                    compare_x_icos  = compare['SubparticleRelative_XYZ'][0]
                    compare_y_icos  = compare['SubparticleRelative_XYZ'][1]
                    compare_z_icos  = compare['SubparticleRelative_XYZ'][2]
                    compare_xy_icos = np.array( [ compare_x_icos, compare_y_icos, 0 ] )
                    compare_xyz_icos = np.array( [ compare_x_icos, compare_y_icos, compare_z_icos ] )
                    negate_compare_xyz_icos = -1 * compare_xyz_icos

                    ### Local coords
                    compare_delta_x = compare['SubparticleOrigin_local'][0] - compare['SubparticleOrigin_icos'][0]
                    compare_delta_y = compare['SubparticleOrigin_local'][1] - compare['SubparticleOrigin_icos'][1]
                    compare_x_local = compare['SubparticleRelative_XYZ'][0] + compare_delta_x
                    compare_y_local = compare['SubparticleRelative_XYZ'][1] + compare_delta_y
                    compare_xy_local = np.array( [ compare_x_local, compare_y_local, 0 ] )


                    this_diameter = assess3dDistance( reference_xyz_icos, compare_xyz_icos )
                    if this_diameter > max_diameter:
                        max_diameter = this_diameter
                        ideal_polar_distance = np.around( assess3dDistance( reference_xyz_icos, compare_xyz_icos ), decimals=2 )
                        flattened_ideal_polar_distance = np.around( assess3dDistance( reference_xy_icos, compare_xy_icos ), decimals=2 )
                        flattened_real_polar_distance  = np.around( assess3dDistance( reference_xy_local, compare_xy_local ), decimals=2 )

                    if np.allclose(reference_xyz_icos, negate_compare_xyz_icos, atol=1):

                        assessed_diameter.append(compare_index)

                        """ Run calculations """
                        ideal_polar_distance = np.around( assess3dDistance( reference_xyz_icos, compare_xyz_icos ), decimals=2 )
                        flattened_ideal_polar_distance = np.around( assess3dDistance( reference_xy_icos, compare_xyz_icos ), decimals=2 )
                        flattened_real_polar_distance = np.around( assess3dDistance( reference_xy_local, compare_xy_local ), decimals=2 )

                        print( 'Difference ratio: ', np.around( np.true_divide(flattened_real_polar_distance,flattened_ideal_polar_distance), decimals=4) )

#        sys.exit()

        """ Here we'll get into the relative motions of neighboring capsomers """
        unique_5f_vertices = np.unique( np.array( subparticles['Vertex5f_general'] ) )
        unique_3f_vertices = np.unique( np.array( subparticles['Vertex3f_general'] ) )
        unique_2f_vertices = np.unique( np.array( subparticles['Vertex2f_general'] ) )


        pentavalent_condition = subparticles[ 'Subparticle_type' ] == b'pentavalent'     # b is from bytestring
        hexavalent_condition  = subparticles[ 'Subparticle_type' ] == b'hexavalent'    # b is from bytestring

        pentavalents = subparticles[ pentavalent_condition ]
        hexavalents  = subparticles[ hexavalent_condition ]


        """ New strategy is to attack this from the hexavalent side """
        for index, hexavalent in enumerate(hexavalents):

            """ If within a tolerance from zmax or zmin """
            if (hexavalent['SubparticleRelative_XYZ'][2] > z_threshold) or (hexavalent['SubparticleRelative_XYZ'][2] < (-1*z_threshold)):

                
                """ Prepare conditions for twofold scenario first """
                condition_2f_1 = hexavalents['Vertex2f_general']  == hexavalent['Vertex2f_general']     # Get this twofold
                condition_2f_2 = hexavalents['Vertex2f_specific'] != hexavalent['Vertex2f_specific']    # Avoid chosing self
                neighbor_twofold = hexavalents[ (condition_2f_1) & (condition_2f_2) ]                  # Store in array
                
                
                """ Prepare conditions for threefold scenario next """
                condition_3f_1 = hexavalents['Vertex3f_general']  == hexavalent['Vertex3f_general']     # Get this threefold
                condition_3f_2 = hexavalents['Vertex3f_specific'] != hexavalent['Vertex3f_specific']    # Avoid chosing self
                neighbors_threefold = hexavalents[ (condition_3f_1) & (condition_3f_2) ]                # Store in array
                
                
                """ Prepare conditions for fivefold scenario next. Hexavalents only for now.
                    Condition 1 returns all five hexavalents about the shared pentavalent.
                    Conditions 2 and 3 will provide the two neighbors. """
                condition_5f_1 = hexavalents['Vertex5f_general']  == hexavalent['Vertex5f_general']     # Get this fivefold

                """ Sloppy definitions to get a-e neighbors """
                if hexavalent['Vertex5f_specific'] == b'a':                     # if a, grab only b and e
                    condition_5f_2 = hexavalents['Vertex5f_specific'] == b'b'   
                    condition_5f_3 = hexavalents['Vertex5f_specific'] == b'e'
                elif hexavalent['Vertex5f_specific'] == b'b':                   # if b, grab only a and c
                    condition_5f_2 = hexavalents['Vertex5f_specific'] == b'a'
                    condition_5f_3 = hexavalents['Vertex5f_specific'] == b'c'
                elif hexavalent['Vertex5f_specific'] == b'c':                   # if c, grab only b and d
                    condition_5f_2 = hexavalents['Vertex5f_specific'] == b'b'
                    condition_5f_3 = hexavalents['Vertex5f_specific'] == b'd'
                elif hexavalent['Vertex5f_specific'] == b'd':                   # if d, grab only c and e
                    condition_5f_2 = hexavalents['Vertex5f_specific'] == b'c'
                    condition_5f_3 = hexavalents['Vertex5f_specific'] == b'e'
                elif hexavalent['Vertex5f_specific'] == b'e':                   # if b, grab only d and a
                    condition_5f_2 = hexavalents['Vertex5f_specific'] == b'd'
                    condition_5f_3 = hexavalents['Vertex5f_specific'] == b'a'
                else:
                    print("You have a logic error there bud.")
                    sys.exit()
                neighbors_fivefold = hexavalents[ (condition_5f_1) & ( condition_5f_2 or condition_5f_3 ) ]  # Store in array


                """ Finally, grab the pentavalent neighbor """
                condition_pentavalent = hexavalent['Vertex5f_general'] == pentavalents['Vertex5f_general']
                neighbor_pentavalent = pentavalents[ condition_pentavalent ]
        

                """ Now we're ready to do the math """
                """ Run the calculations for each neighbor """
                # Twofold hexavalent
                calculateNeighborDistance( hexavalent, neighbor_twofold, 'Twofold-hex' )
                # Threefold hexavalents
                for neighbor in neighbors_threefold:
                    calculateNeighborDistance( hexavalent, neighbor, 'Threefold-hex' )
                # Fivefold hexavalents
                for neighbor in neighbors_fivefold:
                    calculateNeighborDistance( hexavalent, neighbor, 'Fivefold-hex' )
                # Fivefold pentavalent
                calculateNeighborDistance( hexavalent, neighbor_pentavalent, 'Fivefold-pent' )

            if index > 99:
                sys.exit()
    
    return



def main(args):

    ### Today's date will be used in the subparticle path
    date = str( datetime.now().strftime("%Y%m%d") )
    hour = str( datetime.now().strftime("%H%M") )
    RUN_ID = '_'.join( [ 'assess', date, hour ] )
    regen_string = '_'.join( [ date, hour ] )


    if args.pentavalent.endswith(".star"):
        filename = args.pentavalent
        print( "\nReading locally refined coordinates from:", filename )

        pent_header, fullheader = getStarHeader( filename, regen_string )
        stardata = getStarData( filename, len( pent_header ) )
        num_particles = len(stardata)

        print( "  --> done!" )
        print( "  Making ndarray with minimal necessary items." )

        pentavalent_ndarray = getMinimalStarData( filename, pent_header, len( pent_header ), num_particles, subparticle_type='pentavalent' )


    else:
        print( "Please provide a valid star file" )
        sys.exit()


    if args.hexavalent.endswith(".star"):
        filename = args.hexavalent
        print( "Reading locally refined coordinates from:", filename )

        hex_header, fullheader = getStarHeader( filename, regen_string )
        stardata = getStarData( filename, len( hex_header ) )
        num_particles = len(stardata)

        print( "  --> done!" )
        print( "  Making ndarray with minimal necessary items." )

        hexavalent_ndarray = getMinimalStarData( filename, hex_header, len( hex_header ), num_particles, subparticle_type='hexavalent' )

    else:
        print( "Please provide a valid star file" )
        sys.exit()

    subparticle_array, max_z = generateSubparticleArray( pentavalent_ndarray, pent_header, hexavalent_ndarray, hex_header  )

#    print( '\n', subparticle_array.dtype )
#    print( '\n', subparticle_array )


    correlateRefinedCapsomers( subparticle_array, max_z, args.zmax_threshold, args.centsec_threshold )

    #generateSubparticleArray

    sys.exit()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument("input", help="path to cryosparc_P#_J#_###_particles.cs file")

    parser.add_argument("--pentavalent", required=True, help="Locally refined pentavalent capsomers")
    parser.add_argument("--hexavalent", required=True, help="Locally refined hexavalent capsomers")
    parser.add_argument("--zmax_threshold", type=float, default='0.90', help="Threshold for inclusion in motions analysis")
    parser.add_argument("--centsec_threshold", type=float, default='0.05', help="Threshold for inclusion in diameter analysis")

    sys.exit(main(parser.parse_args()))

