#!/usr/bin/env python3.5

import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from numpy import fft
from scipy.spatial.transform import Rotation as R
import scipy.interpolate
import mrcfile

def plotPSD2d( ndimage, plot=False ):
    ### See Jessica Lu, "Fourier Transforms of Images in Python", AstroBetter ###
    """This takes a numpy ndimage (2d) and plots the power spectrum"""

    """Take the fft"""
    myFFT = fft.fft2(ndimage)

    """Shift low spatial frequencies to the center"""
    myFFT_shifted = fft.fftshift(myFFT)

    """Generate power spectrum"""
    psd2D = np.abs(myFFT_shifted)**2

    if plot:
        """Plot using matplotlib"""
        plt.imshow( np.log10(psd2D) )
        plt.show()

    return

def fourierFilter( ndimage, mask_radius, lowpass=True, highpass=False ):
    """This is a lowpass filter by default"""
    if lowpass and highpass:
        print( "Cannot lowpass and highpass filter simultaneously" )
        print( "Will return unfiltered image" )
        return ndimage

    """Take fft, shift frequencies to center"""
    myFFT = fft.fft2(ndimage)
    myFFT_shifted = fft.fftshift(myFFT)

    ## Build the filter function here
    myMask = createCircularMask( myFFT.shape[0], radius=mask_radius )

    if highpass:
        myMask = np.invert(myMask)

    myMask = myMask.astype(np.int)

    """Apply the mask"""
    maskedFFT_shifted = myMask * myFFT_shifted

    """Shift the frequencies back"""
    filteredFFT = fft.ifftshift( maskedFFT_shifted )

    """Inverse FFT"""
    filtered_ndimage = fft.ifft2( filteredFFT )

    print( filtered_ndimage.shape, filtered_ndimage.dtype )

    return filtered_ndimage

def plot2dimage( ndimage ):

    plt.imshow( ndimage )
    plt.show()

    return

def createCircularMask(box_size, center=None, radius=None):
    """See stackoverflow.com/questions/44865023"""

    if center is None: # use the middle of the image
        center = (int(box_size/2), int(box_size/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], box_size-center[0], box_size-center[1])

    Y, X = np.ogrid[:box_size, :box_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def swapAxes_ndimage( my_ndimage ):
    """The data array is indexed in C style, 
       so data values can be accessed using mrc.data[z][y][x] """
    ## From mrcfile.readthedocs.io

    my_ndimage = np.swapaxes( my_ndimage, 0, 2 )

    return my_ndimage


def rotateVolume3d( my_ndimage, box_size ):
    """Input boxsize as int"""
    print( my_ndimage.shape )

    my_ndimage = swapAxes_ndimage( my_ndimage )

    """Your rotation"""     # Hardcoded for debug
#    r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
#    r = R.from_quat( [ 0.809, -0.500, 0.000, 0.309 ] )
    r = R.from_quat( [ 0.000, 0.000, 0.000, 1.000] )
    """scipy notation is bi, cj, dk, w"""   ### Confirm this!
    """ docs.scipy.org: in scalar-last format """

#    box_size = float(box_size)

    x_max = np.true_divide(box_size, 2) - 0.5
    x_min = (-1*x_max)

    x = np.linspace( x_min, x_max, int(box_size) )
    y = np.copy(x)
    z = np.copy(x)

#    print( x_max, x_min, box_size)
#    print( x )

    coordStack_original = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T 

    print( "Making interpolator" )
    f = scipy.interpolate.RegularGridInterpolator((x,y,z),my_ndimage,bounds_error=False, fill_value=0)
    # 'fill_value=0' required to avoid nan, which will screw up downstream processing


    """ Rotate the coordinate system """
    coordStack_rotated=r.apply(coordStack_original)

    """ Interpolate data on the rotated coordinate system """
    print( "Interpolating data" )
    rotated_data = f(coordStack_rotated)
    print( rotated_data.shape )

    """ Reshape the data to the original shape """
    rotated_data = rotated_data.reshape( box_size, box_size, box_size )

    ### These two lines are a hack. ### 
    ### Output map has been z-flipped and rotated ###
    ### compared to the original ###
#    rotated_data = np.flip( rotated_data, axis=2 )
#    rotated_data = np.rot90( rotated_data )
    print( rotated_data.shape )

    return rotated_data

def saveMRC( ndimage, filename ):
    """ Simple function to save as mrc. WILL OVERWRITE EXISTING FILES """

    with mrcfile.new( filename, overwrite=True) as mrc:
        mrc.set_data( ndimage.astype(np.float32))

    return

