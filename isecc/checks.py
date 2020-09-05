#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime
from . import transform
from . import starparse
from . import symops
from . import utils

I1Quaternions = symops.getSymOps()

def idealizeUserVector( user_vector, ROI ) :

    norm_factor = np.amax( np.absolute(user_vector) )

    I1_fivefold   = np.array( [ 0.000,  0.618, 1.000] ) * norm_factor
    I1_fivefold2  = np.array( [ 0.000, -0.618, 1.000] ) * norm_factor
    I1_threefold  = np.array( [ 0.382,  0.000, 1.000] ) * norm_factor
    I1_threefold2 = np.array( [-0.382,  0.000, 1.000]) * norm_factor
    I1_twofold    = np.array( [ 0.000,  0.000, 1.000] ) * norm_factor

    """ Distance to ASU center is currently not used """
    """ ASU definition from 5f:3f:3f """
    ASU_center  = np.mean( np.array( [I1_fivefold, I1_threefold, I1_threefold2 ] ), axis=0 )
    """ Alternate ASU definition from 5f:5f:3f """
    ASU_center2 = np.mean( np.array( [I1_fivefold, I1_fivefold2, I1_threefold  ] ), axis=0 )

    my_dtype = np.dtype( [    ( 'vector', '<f4', (3,) ), 
                ( '5f_distance',   '<f4' ),
                ( '3f_distance',   '<f4' ),
                ( '2f_distance',   '<f4' ),
                ( 'ASU_distance',  '<f4' ),
                ( 'ASU2_distance', '<f4') ]  )

    expanded_user_vector = np.zeros( (60), dtype=my_dtype )

    for index, numpy_quat in enumerate(I1Quaternions, start=0 ):

        quat = Quaternion( numpy_quat )
        rotated_vector = np.around( quat.rotate( user_vector ), decimals=3 )

        expanded_user_vector[index]['vector'] = rotated_vector
        expanded_user_vector[index]['5f_distance']   = np.around( utils.assess3dDistance( rotated_vector, I1_fivefold  ), decimals=3 )
        expanded_user_vector[index]['3f_distance']   = np.around( utils.assess3dDistance( rotated_vector, I1_threefold ), decimals=3 )
        expanded_user_vector[index]['2f_distance']   = np.around( utils.assess3dDistance( rotated_vector, I1_twofold   ), decimals=3 )
        expanded_user_vector[index]['ASU_distance']  = np.around( utils.assess3dDistance( rotated_vector, ASU_center  ), decimals=3 )
        expanded_user_vector[index]['ASU2_distance'] = np.around( utils.assess3dDistance( rotated_vector, ASU_center2 ), decimals=3 )

    min_5f_distance = np.amin( expanded_user_vector['5f_distance'] )
    min_3f_distance = np.amin( expanded_user_vector['3f_distance'] )
    min_2f_distance = np.amin( expanded_user_vector['2f_distance'] )

    ideal_index = 999

    for index, item in enumerate(expanded_user_vector):

        """ Removed check for nearest threefold. """
        """ Having that check unintentionally limits you to half of the ASU """

        #if np.isclose(expanded_user_vector[index]['5f_distance'], min_5f_distance, atol=1) and np.isclose(expanded_user_vector[index]['3f_distance'], min_3f_distance, atol=1) and np.isclose(expanded_user_vector[index]['2f_distance'], min_2f_distance, atol=1):
        if np.isclose(expanded_user_vector[index]['5f_distance'], min_5f_distance, rtol=0.001) and np.isclose(expanded_user_vector[index]['2f_distance'], min_2f_distance, rtol=0.001):
            ideal_index = index


    if ideal_index != 999:
        idealized_vector = np.around( expanded_user_vector[ideal_index]['vector'], decimals=3 )
    else:
        print( 'Error idealizing user vector. Exiting now.' )
        sys.exit()

    """ Hack to catch incorrect threefold """
    if ROI == 'threefold':
        idealized_vector = I1_threefold
        print("Threefolds are tricky. Ensure that", I1_threefold, "is the vector you want.")

    if not np.allclose( idealized_vector, user_vector, atol=0.5 ):
        print( '\nUser vector', user_vector, "has been modified to be within desired asymmetric unit." )
        print(' You are responsible for ensuring that the new vector is appropriate.')
        print( 'User vector is now', idealized_vector )

    return idealized_vector


def checkVector(user_vector, ROI, higher_order_sym):

    print("\nRunning checkVector")
    
    if ROI == 'fivefold' and higher_order_sym == 'I1' :
        vector_check = np.true_divide( user_vector[1], user_vector[2] )
        my_check = np.isclose( vector_check, 0.618, rtol=0.01 )
        my_check2 = np.isclose( user_vector[0], 0, rtol=0.01 )

    elif ROI == 'fivefold' and higher_order_sym == 'I2' :
        vector_check = np.true_divide( user_vector[0], user_vector[2] )
        my_check = np.isclose( vector_check, 0.618, rtol=0.01 )
        my_check2 = np.isclose( user_vector[1], 0, rtol=0.01 )

    elif ROI == 'threefold' and higher_order_sym == 'I1' :
        vector_check = np.true_divide( user_vector[0], user_vector[2] )
        my_check = np.isclose( vector_check, 0.382, rtol=0.01 )
        my_check2 = np.isclose( user_vector[1], 0, rtol=0.01 )

    elif ROI == 'threefold' and higher_order_sym == 'I2' :
        vector_check = np.true_divide( user_vector[1], user_vector[2] )
        my_check = np.isclose( vector_check, 0.382, rtol=0.01 )
        my_check2 = np.isclose( user_vector[0], 0, rtol=0.01 )

    elif ROI == 'twofold' and higher_order_sym == 'I1' :
        my_check = np.isclose( user_vector[0], 0, rtol=0.01 )
        my_check2 = np.isclose( user_vector[1], 0, rtol=0.01 )

    elif ROI == 'twofold' and higher_order_sym == 'I2' :
        my_check = np.isclose( user_vector[0], 0, rtol=0.01 )
        my_check2 = np.isclose( user_vector[1], 0, rtol=0.01 )

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

