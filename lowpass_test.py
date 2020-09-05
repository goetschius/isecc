#!/usr/bin/env python3.5

import mrcfile
import numpy as np
from isecc import *
import scipy.ndimage

###
my_file = mrcfile.open('../MRC/simulated_particles.mrcs')
my_ndimage = np.copy(my_file.data[0])
iseccFFT_v2.plot2dimage( my_ndimage )
del my_file

simulated_lp3 = iseccFFT_v2.fourierFilter( my_ndimage, 3, lowpass=True).astype(np.float32)
simulated_lp5 = iseccFFT_v2.fourierFilter( my_ndimage, 5, lowpass=True).astype(np.float32)
simulated_lp10 = iseccFFT_v2.fourierFilter( my_ndimage, 10, lowpass=True).astype(np.float32)
simulated_lp15 = iseccFFT_v2.fourierFilter( my_ndimage, 15, lowpass=True).astype(np.float32)
simulated_lp20 = iseccFFT_v2.fourierFilter( my_ndimage, 20, lowpass=True).astype(np.float32)
simulated_lp25 = iseccFFT_v2.fourierFilter( my_ndimage, 25, lowpass=True).astype(np.float32)
simulated_lp30 = iseccFFT_v2.fourierFilter( my_ndimage, 30, lowpass=True).astype(np.float32)
simulated_lp35 = iseccFFT_v2.fourierFilter( my_ndimage, 35, lowpass=True).astype(np.float32)
simulated_lp40 = iseccFFT_v2.fourierFilter( my_ndimage, 40, lowpass=True).astype(np.float32)
simulated_lp45 = iseccFFT_v2.fourierFilter( my_ndimage, 45, lowpass=True).astype(np.float32)
simulated_lp50 = iseccFFT_v2.fourierFilter( my_ndimage, 50, lowpass=True).astype(np.float32)
simulated_lp100 = iseccFFT_v2.fourierFilter( my_ndimage, 100, lowpass=True).astype(np.float32)
simulated_lp200 = iseccFFT_v2.fourierFilter( my_ndimage, 200, lowpass=True).astype(np.float32)

iseccFFT_v2.plot2dimage(simulated_lp3)
iseccFFT_v2.plot2dimage(simulated_lp5)
iseccFFT_v2.plot2dimage(simulated_lp10)
iseccFFT_v2.plot2dimage(simulated_lp15)
iseccFFT_v2.plot2dimage(simulated_lp20)
iseccFFT_v2.plot2dimage(simulated_lp25)
iseccFFT_v2.plot2dimage(simulated_lp30)
iseccFFT_v2.plot2dimage(simulated_lp35)
iseccFFT_v2.plot2dimage(simulated_lp40)
iseccFFT_v2.plot2dimage(simulated_lp45)
iseccFFT_v2.plot2dimage(simulated_lp50)
iseccFFT_v2.plot2dimage(simulated_lp100)
iseccFFT_v2.plot2dimage(simulated_lp200)


