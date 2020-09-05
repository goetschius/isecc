#!/usr/bin/env python3.5

import mrcfile
import numpy as np
from isecc import *
import scipy.ndimage

###
my_file = mrcfile.open('../MRC/symbreak_binned.mrc')
my_ndimage = np.copy(my_file.data)
iseccFFT_v2.plot2dimage( my_ndimage[:][:][180] )
del my_file

rotated_data = iseccFFT_v2.rotateVolume3d( my_ndimage, my_ndimage.shape[0] )

iseccFFT_v2.plot2dimage( my_ndimage[:][:][180] )
iseccFFT_v2.plot2dimage( rotated_data[:][:][180] )
