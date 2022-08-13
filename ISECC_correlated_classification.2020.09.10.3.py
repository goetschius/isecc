#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime


def getMinimalStarData( my_star, subparticle_header, subparticle_header_length, num_subparticles, occupied_class, subparticle_type=None ):

    ## Subparticles
    icos_X_index,  icos_Y_index  =  getOffsetAngst( subparticle_header )
    subpart_class_index = getClass( subparticle_header )

    ### Get index for the local XYZ relative to particle origin
    relativeXYZ_index = getOriginXYZAngstWrtParticleCenter( subparticle_header )

    ### Get index for the vertex group assignment
    vertexGroup_index = getVertexGroup( subparticle_header )

    ### Get index for whole particle unique identifier
    micrographname_index, imagename_index, particleID_index = getMicrographName( subparticle_header )


    subparticle_array_dtype = np.dtype( [   ( 'ParticleSpecifier', '|U200' ),
                        ( 'Subparticle_type', '|S11' ),
                        ( 'Vertex5f_general', 'i4' ), ( 'Vertex5f_specific', '|U1' ),
                        ( 'Vertex3f_general', 'i4' ), ( 'Vertex3f_specific', '|U1' ),
                        ( 'Vertex2f_general', 'i4' ), ( 'Vertex2f_specific', '|U1' ),
                        ( 'SubparticleOrigin_icos',  '<f4', (2,) ),
                        ( 'SubparticleRelative_XYZ', '<f4', (3,) ),
                        ( 'OccupiedStatus', '?' ),
                        ( 'SubparticleClass', 'i4' ) ] )

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
                    line = ' '.join( [ linesplit[particleID_index], linesplit[icos_X_index], linesplit[icos_Y_index] ] )

                    if line[0] != '#':      # avoid the stupid comment line
                        #stardata.append( line )


                        # Particle Specifier
                        minimal_subparticle_array[subparticle_index]['ParticleSpecifier'] = linesplit[ particleID_index ]
                        # Subparticle Type
                        minimal_subparticle_array[subparticle_index]['Subparticle_type'] = subparticle_type
                        # Subparticle Class
                        minimal_subparticle_array[subparticle_index]['SubparticleClass'] = linesplit[ subpart_class_index ] 


                        # X,Y Priors (icosahedrally derived)
                        minimal_subparticle_array[subparticle_index]['SubparticleOrigin_icos'] = linesplit[ icos_X_index ], linesplit[ icos_Y_index ]
                        # Relative X,Y,Z with respect to particle center
                        minimal_subparticle_array[subparticle_index]['SubparticleRelative_XYZ'] = linesplit[ relativeXYZ_index ].split(',')

                        # Vertex Assignment
                        if subparticle_type == 'Fab':
                            vertex_assignment = str( linesplit[vertexGroup_index] )    # example: '5f01a.3f01a.2f01a'
                            minimal_subparticle_array[subparticle_index]['Vertex5f_general' ] = vertex_assignment[2:4]      # character 3-4
                            minimal_subparticle_array[subparticle_index]['Vertex5f_specific'] = vertex_assignment[4]        # character 5
                            minimal_subparticle_array[subparticle_index]['Vertex3f_general' ] = vertex_assignment[8:10]     # character 9-10
                            minimal_subparticle_array[subparticle_index]['Vertex3f_specific'] = vertex_assignment[10]       # character 11
                            minimal_subparticle_array[subparticle_index]['Vertex2f_general' ] = vertex_assignment[14:16]    # character 15-16
                            minimal_subparticle_array[subparticle_index]['Vertex2f_specific'] = vertex_assignment[16]       # character 17

                        """ Assign boolean ('?') to indicate occupied status """
                        if int(linesplit[subpart_class_index]) == int(occupied_class):
                            minimal_subparticle_array[subparticle_index]['OccupiedStatus'] = True
                        elif int(linesplit[subpart_class_index]) != int(occupied_class):
                            minimal_subparticle_array[subparticle_index]['OccupiedStatus'] = False

                        subparticle_index = subparticle_index + 1


    print( "  --> sorting based on particle image name." )
    minimal_subparticle_array.sort(order='ParticleSpecifier')
    print( "  --> done!" )


    print('\nExample:')
    print( minimal_subparticle_array.dtype )
    print( minimal_subparticle_array[0], '\n' )


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


def assessVertexOfType( subparticles, vertex_type, unique_indices ):

    """ Only for N>3 does the order around a vertex actually matter """
    if vertex_type == 'twofold':
        output = np.ones([30,2], dtype=bool)

        for x in range(0,len(unique_indices)):
            this_index = unique_indices[x]
            condition = subparticles['Vertex2f_general'] == subparticles[this_index]['Vertex2f_general']

            this_vertex = subparticles[condition]

            """ Unnecessary, but sort the vertex by a,b rotational parameter """
            #this_vertex.sort(order='Vertex2f_specific')

            output[x] = this_vertex['OccupiedStatus']


    if vertex_type == 'threefold':
        output = np.ones([20,3], dtype=bool)

        for x in range(0,len(unique_indices)):
            this_index = unique_indices[x]
            condition = subparticles['Vertex3f_general'] == subparticles[this_index]['Vertex3f_general']

            this_vertex = subparticles[condition]

            """ Unnecessary, but sort the vertex by a,b,c rotational parameter """
            #this_vertex.sort(order='Vertex3f_specific')

            output[x] = this_vertex['OccupiedStatus']


    """ Now I have to worry about the Vertex5f_specific parameter for order """
    if vertex_type == 'fivefold':
        output = np.ones([12,5], dtype=bool)

        for x in range(0,len(unique_indices)):
            this_index = unique_indices[x]
            condition = subparticles['Vertex5f_general'] == subparticles[this_index]['Vertex5f_general']

            this_vertex = subparticles[condition]

            """ Sort the vertex by a,b,c,d,e rotational parameter """
            this_vertex.sort(order='Vertex5f_specific')

            output[x] = this_vertex['OccupiedStatus']

    """ Convert to int for easier parsing """
    output = output.astype(np.int)

    return output


def reportVertices( particle_2fs, particle_3fs, particle_5fs ):

    """ Sum across the rows """
    twofold_sums   = np.sum(particle_2fs, axis=1)
    threefold_sums = np.sum(particle_3fs, axis=1)
    fivefold_sums  = np.sum(particle_5fs, axis=1)

    """ Setup the twofold reporters """
    twofold_00 = 0
    twofold_01 = 0
    twofold_11 = 0

    for x in range(0,30):
        if twofold_sums[x] == 0:
            twofold_00 +=1
        elif twofold_sums[x] == 1:
            twofold_01 +=1
        elif twofold_sums[x] == 2:
            twofold_11 +=1
        else:
            print("BROKEN")
            sys.exit()

    """ Setup the threefold reporters """
    threefold_000 = 0
    threefold_001 = 0
    threefold_011 = 0
    threefold_111 = 0

    for x in range(0,20):
        if threefold_sums[x] == 0:
            threefold_000 += 1
        if threefold_sums[x] == 1:
            threefold_001 += 1
        if threefold_sums[x] == 2:
            threefold_011 += 1
        if threefold_sums[x] == 3:
            threefold_111 += 1



    """ Setup the fivefold reporters """
    ### 0 or 5 ###
    fivefold_00000 = 0
    fivefold_11111 = 0

    ### 1 or 4 ###
    fivefold_00001 = 0
    fivefold_11110 = 0

    ### 2 or 3 cis ###
    fivefold_00011 = 0
    fivefold_11100 = 0

    ### 2 or 3 trans ###
    fivefold_00101 = 0
    fivefold_11010 = 0

    for x in range(0,12):

        ### 0 or 5 ###
        if fivefold_sums[x] == 0:
            fivefold_00000 += 1
        elif fivefold_sums[x] == 5:
            fivefold_11111 += 1

        ### 1 or 4 ###
        elif fivefold_sums[x] == 1:
            fivefold_00001 += 1
        elif fivefold_sums[x] == 4:
            fivefold_11110 += 1

        ### 2 ###
        elif fivefold_sums[x] == 2:
            con1 = particle_5fs[x][0] == 1
            con2 = particle_5fs[x][1] == 1
            con3 = particle_5fs[x][2] == 1
            con4 = particle_5fs[x][3] == 1
            con5 = particle_5fs[x][4] == 1

            ### Cis arrangments ###
            if   (con1) and (con2) :
                fivefold_00011 +=1
            elif (con2) and (con3) :
                fivefold_00011 +=1
            elif (con3) and (con4) :
                fivefold_00011 +=1
            elif (con4) and (con5) :
                fivefold_00011 +=1
            elif (con5) and (con1) :
                fivefold_00011 +=1

            ### Trans arrangement ###
            else:
                fivefold_00101 +=1

        ### 3 ###
        elif fivefold_sums[x] == 3:
            con1 = particle_5fs[x][0] == 0
            con2 = particle_5fs[x][1] == 0
            con3 = particle_5fs[x][2] == 0
            con4 = particle_5fs[x][3] == 0
            con5 = particle_5fs[x][4] == 0

            ### Cis arrangments ###
            if   (con1) and (con2) :
                fivefold_11100 +=1
            elif (con2) and (con3) :
                fivefold_11100 +=1
            elif (con3) and (con4) :
                fivefold_11100 +=1
            elif (con4) and (con5) :
                fivefold_11100 +=1
            elif (con5) and (con1) :
                fivefold_11100 +=1

            ### Trans arrangement ###
            else:
                fivefold_11010 +=1


        ### Catch logic error ###
        else:
            print("Fatal logic error in script")
            sys.exit()


    print( "GEOM-2F_EMPTY ", twofold_00 )    
    print( "GEOM-2F_MONO  ", twofold_01 )    
    print( "GEOM-2F_FULL  ", twofold_11 )    

    print( "GEOM-3F_EMPTY ", threefold_000 )
    print( "GEOM-3F_MONO  ", threefold_001 )
    print( "GEOM-3F_DI    ", threefold_011 )
    print( "GEOM-3F_FULL  ", threefold_111 )

    ### 0  ###
    print( "GEOM-5F_EMPTY ", fivefold_00000 )

    ### 1  ###
    print( "GEOM-5F_MONO  ", fivefold_00001 )

    ### 2 or 3 cis ###
    print( "GEOM-5F_DI_cis", fivefold_00011 )
    print( "GEOM-5F_DI_trans", fivefold_00101 )

    ### 2 or 3 trans ###
    print( "GEOM-5F_TRI_cis", fivefold_11100 )
    print( "GEOM-5F_TRI_trans", fivefold_11010 )

    ### 4 ###
    print( "GEOM-5F_TETRA ", fivefold_11110 )

    ### 5 ###
    print( "GEOM-5F_FULL  ", fivefold_11111 )


    return



def correlateSuparticleClasses( subparticle_array ):

    particle_names = np.array( subparticle_array['ParticleSpecifier'] )
    unique_particles = np.unique( particle_names )


    # iterate through all the unique particles
    for index, item in enumerate(unique_particles, start=0):

        # Particle specifier for printout
        particle_specifier = str(index).rjust(5, '0')

        # Setting conditions
        condition_1 = subparticle_array['ParticleSpecifier'] == item 
        condition_2 = subparticle_array['OccupiedStatus'] == True

        # Grab all of the subparticles belonging to the current particle that are also occupied
        subparticles = subparticle_array[ (condition_1) & ( condition_2 ) ] 
        subparticles_all = subparticle_array[ (condition_1) ]

        print( len(subparticles), "meet RDF conditions for particle", particle_specifier )
        print( len(subparticle_array[(condition_1)]), "subparticles will be analysed for particle", particle_specifier )

        rdf_current_particle = []


        """ Only want the indices. Must happen outside subparticle loop """
        unique_2f, unique_2f_indices = np.unique(subparticles_all['Vertex2f_general'], return_index=True)
        unique_3f, unique_3f_indices = np.unique(subparticles_all['Vertex3f_general'], return_index=True)
        unique_5f, unique_5f_indices = np.unique(subparticles_all['Vertex5f_general'], return_index=True)

        particle_2fs = assessVertexOfType( subparticles_all, 'twofold',   unique_2f_indices )
        particle_3fs = assessVertexOfType( subparticles_all, 'threefold', unique_3f_indices )
        particle_5fs = assessVertexOfType( subparticles_all, 'fivefold',  unique_5f_indices )

        reportVertices( particle_2fs, particle_3fs, particle_5fs )


        """ Let's generate the radial distribution functions """
        for assessed_index, subparticle in enumerate(subparticles, start=0):

            assessed_rdf = []

            ### Icos coords
            reference_x_icos  = subparticle['SubparticleRelative_XYZ'][0]
            reference_y_icos  = subparticle['SubparticleRelative_XYZ'][1]
            reference_z_icos  = subparticle['SubparticleRelative_XYZ'][2]
            reference_xyz_icos = np.array( [ reference_x_icos, reference_y_icos, reference_z_icos ] )


            """ This code is for rdf assessment """
            if (assessed_index not in assessed_rdf):

                for compare_index, compare in enumerate(subparticles, start=0):

                    """ Don't assess self. """
                    """ Also, only assess index > self to prevent dupes """
                    if (compare_index > assessed_index):

                        ### Icos coords
                        compare_x_icos  = compare['SubparticleRelative_XYZ'][0]
                        compare_y_icos  = compare['SubparticleRelative_XYZ'][1]
                        compare_z_icos  = compare['SubparticleRelative_XYZ'][2]
                        compare_xyz_icos = np.array( [ compare_x_icos, compare_y_icos, compare_z_icos ] )
                        negate_compare_xyz_icos = -1 * compare_xyz_icos

                        rdf_component = assess3dDistance( reference_xyz_icos, compare_xyz_icos )
                        rdf_component = np.around( rdf_component, decimals=1 )
                        assessed_rdf.append(compare_index)
                        rdf_current_particle.append(rdf_component)
                        rdf_current_particle.sort()

        if rdf_current_particle:
            print( "RDF for current particle is:", rdf_current_particle, "\n" )


    return



def main(args):

    ### Today's date will be used in the subparticle path
    date = str( datetime.now().strftime("%Y%m%d") )
    hour = str( datetime.now().strftime("%H%M") )
    RUN_ID = '_'.join( [ 'assess', date, hour ] )
    regen_string = '_'.join( [ date, hour ] )

    if args.starfile.endswith(".star"):
        filename = args.starfile
        print( "\nReading Class3D star file from:", filename )

        subpart_header, fullheader = getStarHeader( filename, regen_string )
        stardata = getStarData( filename, len( subpart_header ) )
        num_particles = len(stardata)

        print( "  --> done!" )
        print( "  Making ndarray with minimal necessary items." )

        subparticle_ndarray = getMinimalStarData( filename, subpart_header, len( subpart_header ), num_particles, args.occupied_class, subparticle_type='Fab' )


    else:
        print( "Please provide a valid star file" )
        sys.exit()


    correlateSuparticleClasses( subparticle_ndarray )

    #generateSubparticleArray

    sys.exit()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--starfile", required=True, help="Class3D star file")
    parser.add_argument("--occupied_class", required=True, help="Which class number is occupied?")

    sys.exit(main(parser.parse_args()))

