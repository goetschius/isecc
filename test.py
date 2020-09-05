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
from isecc import transform
from isecc import starparse
from isecc import symops
from isecc import checks
from isecc import utils
from isecc.isecc_classes import Particle

I1Quaternions = symops.getSymOps()
my_vector = np.array( [0, 0.618, 1] )


my_arctan = np.arctan( np.true_divide( my_vector[1], my_vector[2] ) ) 
quaternion_toZ = Quaternion(axis=(1.0,0.0,0.0), radians=my_arctan)


my_particle = Particle( 'dummy.mrc', 0.256, -0.432, 58.46, 126.2, -70.5 )
my_particle.add_defocus_info( 32350.43, 30611.52, 13.18, 0 )

print( my_particle )

my_particle.rotateParticle(I1Quaternions[0], quaternion_toZ)
print( vars(my_particle) )

my_particle.defineSubparticle(my_vector)
print( vars(my_particle) )

