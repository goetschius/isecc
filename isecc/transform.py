#!/usr/bin/env python3.5

import argparse
import sys
import os
import time
import math
import numpy as np
from pyquaternion import Quaternion
from . import symops


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


def rot2euler(r):                      # function modified from pyem, Daniel Asarnow, GPL3
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

