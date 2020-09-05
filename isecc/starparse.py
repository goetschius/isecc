#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from datetime import datetime


def getStarHeader( my_star, regen_string ):     # original version
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


        ### Add info for how to regenerate subparticles
        if '--timestamp_run' not in full_header[2] :
                full_header[2] = ' '.join( [ full_header[2], '--timestamp_run', regen_string ] )

        return header, full_header


def getStarHeader2( my_star ):
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


def getStarData( my_star, header_length ):      # original
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
        return stardata


def getStarData2( my_star, header_length, header ):

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
