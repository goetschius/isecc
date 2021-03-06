#!/usr/bin/env python3.5

import mrcfile
import numpy as np
from isecc import *
import scipy.ndimage
from pyem import *
import matplotlib.pyplot as plt

def plot2dimage( ndimage ):
    plt.imshow( ndimage )
    plt.show()
    return

###
my_file = mrcfile.open('../MRC/output_stack.mrc').data

plot2dimage( my_file[0] )
plot2dimage( my_file[1])

