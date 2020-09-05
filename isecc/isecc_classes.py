#!/usr/bin/env python3.5
import numpy as np
from isecc import transform
from pyquaternion import Quaternion

class Particle:
    """ Contains items from data_particles """

    """ Init with minimal necessary items for a particle """
    def __init__(self, imageName, originXAngst, originYAngst, angleRot, angleTilt, anglePsi):
        self.rlnImageName    = imageName
        self.rlnOriginXAngst = float(originXAngst)
        self.rlnOriginYAngst = float(originYAngst)
        self.rlnAngleRot     = float(angleRot)
        self.rlnAngleTilt    = float(angleTilt)
        self.rlnAnglePsi     = float(anglePsi)

        """ define particle pose as quaternion """
        rotRad  = np.radians(self.rlnAngleRot)
        tiltRad = np.radians(self.rlnAngleTilt)
        psiRad  = np.radians(self.rlnAnglePsi)

        self.pose = Quaternion( transform.myEuler2Quat( rotRad, tiltRad, psiRad ) )

    def add_defocus_info(self, defocusU, defocusV, defocusAngle, phaseShift):
        self.rlnDefocusU     = float(defocusU)
        self.rlnDefocusV     = float(defocusV)
        self.rlnDefocusAngle = float(defocusAngle)
        self.rlnPhaseShift   = float(phaseShift)

    def add_class(self, classNum):
        self.rlnClassNumber = int(classNum)

    def add_micrograph_params(self, micrographName=None, coordX=None, coordY=None, originalMicrograph=None):
        self.rlnMicrographName = micrographName
        self.rlnCoordinateX    = float(coordX)
        self.rlnCoordinateY    = float(coordY)
        self.rlnImageOriginalName = originalMicrograph

    def add_isecc_params(self, uid, vertexGroup, originWRTparticle, relativePose):
        self.rlnCustomUID         = uid
        self.rlnCustomVertexGroup = vertexGroup
        self.rlnCustomOriginXYZAngstWrtParticleCenter = originWRTparticle
        self.rlnRelativePose      = relativePose

    def add_priors(self, priorXAngst, priorYAngst, priorRot, priorTilt, priorPsi):
        self.rlnOriginXPriorAngst = float(priorXAngst)
        self.rlnOriginYPriorAngst = float(priorYAngst)
        self.rlnAngleRotPrior     = float(priorRot)
        self.rlnAngleTiltPrior    = float(priorTilt)
        self.rlnAnglePsiPrior     = float(priorPsi)

    def rotateParticle(self, quat, quatZ):
        """ define quat as Quaternion object """
        quat = Quaternion( quat )

        """ Apply the rotation """
        self.newpose  = quat  * self.pose
        self.newposeZ = quatZ * self.newpose

    def defineSubparticle(self, vector):

        inverted_pose  = self.newpose.inverse
        self.rotated_vector = inverted_pose.rotate( vector )

        """ Must invert for conversion to Eulers """
        rot_matrix = self.newposeZ.inverse.rotation_matrix

        """ subparticle eulers """
        self.subpartRot, self.subpartTilt, self.subpartPsi = transform.rot2euler( rot_matrix )

        """ subparticle origins """
        self.subpartX = self.rlnOriginXAngst - self.rotated_vector[0]
        self.subpartY = self.rlnOriginYAngst - self.rotated_vector[1]

        self.subpartZU = self.rlnDefocusU + self.rotated_vector[2]
        self.subpartZV = self.rlnDefocusV + self.rotated_vector[2]
        


class Optics:

    """ Contains the items from data_optics """
    def __init__(self, opticsGroup, opticsGroupName, ampContrast, sphAb, voltage, pixSize, imageSize, imageDim):
        self.rlnOpticsGroup = opticsGroup
        self.rlnOpticsGroupName = opticsGroupName
        self.rlnAmplitudeContrast = ampContrast
        self.rlnSphericalAberration = sphAb
        self.rlnVoltage = voltage
        self.rlnImagePixelSize = pixSize
        self.rlnImageSize = imageSize
        self.rlnImageDimensionality = imageDim


class AreaOfInterest:

    """ As written, this only works for I1 symmetry """
    def __init__(self, roi, vector):
        self.roi = roi
        self.vector = np.array( [vector[0], vector[1], vector[2]] )
        self.symindices = None

        if self.roi == 'fivefold':
            angle = np.arctan( np.true_divide( self.vector[1], self.vector[2] ) )
            self.quatZ = Quaternion( axis=(1,0,0), radians=angle )
            self.model_sym = 'c5'
            self.short_roi = '5f'

        elif self.roi == 'threefold':
            angle = np.arctan( np.true_divide( self.vector[0], self.vector[2] ) )
            self.quatZ = Quaternion( axis=(0,-1,0), radians=angle )
            self.model_sym = 'c3'
            self.short_roi = '3f'

        elif self.roi == 'twofold':
            self.quatZ = Quaternion( axis=(0,0,1), radians=0 )
            self.model_sym = 'c2'
            self.short_roi = '2f'

        elif self.roi == 'fullexpand':
            self.quatZ = Quaternion( axis=(0,0,1), radians=0 )
            self.model_sym = 'c1'
            self.short_roi = 'ex'
            #self.symindices = np.arange(0,60)

        elif self.roi == 'null':
            self.quatZ = Quaternion( axis=(0,0,1), radians=0 )
            self.model_sym = 'c1'
            self.short_roi = 'null'
            #self.symindices = np.arange(0,1)

        else:
            print( "roi not understood" )


    def apply_fudge(self, fudge_factor):
        try:    self.vector = self.vector * float(fudge_factor[0])
        except: self.vector = self.vector * float(fudge_factor)


    def updateSymIndices(self, unique_indices):
        """ This provides the unique symmetry indices for symops """

        if not self.symindices:
            self.symindices = list(map(int, unique_indices))

