#!/usr/bin/env python3.5

import math
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from numpy import fft
from scipy.spatial.transform import Rotation as R
import scipy.interpolate
import mrcfile
from pyfftw.interfaces.numpy_fft import rfftn
import warnings
from pyem import *
import sys
warnings.filterwarnings('ignore', message='Casting complex values')


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

def fourierFilter( ndimage, filter_res, lowpass=True, highpass=False ):
    """This is a lowpass filter by default"""
    if lowpass and highpass:
        print( "Cannot lowpass and highpass filter simultaneously" )
        print( "Will return unfiltered image" )
        return ndimage

#    ndimage = padEdges(ndimage)

    box_size = ndimage.shape[0]
#    print( 'Box is:', box_size )


    """ See docs.scipy.org """
    """ This is the angpix. Hardcoded for debug"""
    sample_rate = 1.1
    """ Generate the fft frequency curve """
    freq = np.fft.fftfreq(box_size, d=sample_rate)
    freq_shift = np.fft.fftshift(freq)
    freq_shift_angst = np.reciprocal( freq_shift )

    """ Take the first half of the shifted freq array """
    n = int( np.true_divide(box_size,2) )
    pos_freq = np.abs(freq_shift_angst[:n])     # take the first half
    x = np.linspace( 0, n-1, n )

    f = np.interp( filter_res, pos_freq, x )
    filter_radius = n-f

    """ Message for sanity check """
    print( "Applying filter to", filter_res, "Å (",np.around(filter_radius, decimals=2), "pixel radius in FFT)." )

    """Take fft, shift frequencies to center"""
    myFFT = np.fft.fft2(ndimage)
    myFFT_shifted = np.fft.fftshift(myFFT)

    """ Make a mask, invert if highpass """
    myMask = createCircularMask( myFFT.shape[0], radius=(filter_radius) )

    if highpass:
        myMask = np.invert(myMask)

    myMask = myMask.astype(np.int)

    """ Apply the mask """
    maskedFFT_shifted = myMask * myFFT_shifted

    """ Shift the frequencies back """
    filteredFFT = np.fft.ifftshift( maskedFFT_shifted )

    """ Inverse FFT """
    filtered_ndimage = np.fft.ifft2( filteredFFT )

#    print( filtered_ndimage.shape, filtered_ndimage.dtype )

    return filtered_ndimage

def plot2dimage( ndimage ):

    plt.imshow( ndimage )
    plt.show()

    return

def padEdges( ndimage ):
    """ Pad ndimage with edge values to odd box size """

    ndimage = np.pad( ndimage, (1,2), mode='edge')

    return ndimage

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

def createSphericalMask(box_size, center=None, radius=None):
    """See stackoverflow.com/questions/44865023"""

    if center is None: # use the middle of the image
        center = (int(box_size/2), int(box_size/2), int(box_size/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], center[2], box_size-center[0], box_size-center[1], box_size-center[2])

    Z, Y, X = np.ogrid[:box_size, :box_size, :box_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)

    mask = dist_from_center <= radius
    return mask


def swapAxes_ndimage( my_ndimage ):
    """The data array is indexed in C style, 
       so data values can be accessed using mrc.data[z][y][x] """
    ## From mrcfile.readthedocs.io

    my_ndimage = np.swapaxes( my_ndimage, 0, 2 )

    return my_ndimage

def scipy2pyquat( scipy_quat ):
    """ Takes scipy formated quaternion and returns pyquaternion format """
    """ pyquaternion is SCALAR FIRST """
    """ scipy quaternion is SCALAR LAST """

    pyquat_quat = scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]

    return pyquat_quat  # as numpy array

def pyquat2scipy( pyquat_quat ):
    """ Takes pyquaternion formated quaternion and returns scipy format """
    """ pyquaternion is SCALAR FIRST """
    """ scipy quaternion is SCALAR LAST """

    scipy_quat = pyquat_quat[1], pyquat_quat[2], pyquat_quat[3], pyquat_quat[0]

    return scipy_quat   # as numpy array

def grid_correct(vol, pfac=2, order=1):     # DIRECTLY FROM PYEM!!!
    n = vol.shape[0]
    nhalf = n // 2
    x, y, z = np.meshgrid(*[np.arange(-nhalf, nhalf)] * 3, indexing="xy")
    r = np.sqrt(x**2 + y**2 + z**2, dtype=vol.dtype) / (n * pfac)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sin(np.pi * r) / (np.pi * r)  # Results in 1 NaN in the center.
    sinc[nhalf, nhalf, nhalf] = 1.
    if order == 0:
        cordata = vol / sinc
    elif order == 1:
        cordata = vol / sinc**2
    else:
        raise NotImplementedError("Only nearest-neighbor and trilinear grid corrections are available")
    return cordata

def vol_ft(vol, pfac=2, threads=1, normfft=1):
    """ Returns a centered, Nyquist-limited, zero-padded, interpolation-ready 3D Fourier transform.
    :param vol: Volume to be Fourier transformed.
    :param pfac: Size factor for zero-padding.
    :param threads: Number of threads for pyFFTW.
    :param normfft: Normalization constant for Fourier transform.
    """
    vol = grid_correct(vol, pfac=pfac, order=1)
    padvol = np.pad(vol, int((vol.shape[0] * pfac - vol.shape[0]) // 2), "constant")
    ft = rfftn(np.fft.ifftshift(padvol), padvol.shape, threads=threads)
#    ft = np.fft.fftn(np.fft.ifftshift(padvol), padvol.shape )
    ftc = np.zeros((ft.shape[0] + 3, ft.shape[1] + 3, ft.shape[2]), dtype=ft.dtype)
    vop.fill_ft(ft, ftc, vol.shape[0], normfft=normfft)
    print( ftc.shape )
    plot2dimage(np.log10(np.abs(ftc[:][:][230])).astype(np.float))
    sys.exit()
    return ftc


def get2dsection( my_ndimage, my_rotation ) :
    """ This function gets a 2d section through the center of a 3d volume """
    """ my_rotation must be scipy-ordered numpy array! """

    plot2dimage( my_ndimage[:][:][int(int(my_ndimage.shape[0])/2)])
    my_ndimage = grid_correct(my_ndimage, pfac=1, order=1)
    plot2dimage( my_ndimage[:][:][int(int(my_ndimage.shape[0])/2)])

#    padded_ndimage = np.pad( my_ndimage, ((my_ndimage.shape[0]//2),), 'constant', constant_values=(0))
#    my_ndimage = padded_ndimage

    my_ndimage = padEdges( my_ndimage )

    box_size = my_ndimage.shape[0]
#    myFFT_shifted = vol_ft( my_ndimage )
#    box_size = myFFT_shifted.shape[0]

    mask = createSphericalMask( box_size, radius=100 )
    mask = mask.astype(np.int)
    plot2dimage(mask[:][:][int(box_size/2)])

    #### TESTING 3DFFT ####
    """Take fft, shift frequencies to center"""
    myFFT = np.fft.fftn(my_ndimage, norm='ortho')
    myFFT_shifted = np.fft.fftshift(myFFT)
    myFFT_shifted = mask * myFFT_shifted
    plot2dimage(np.log10(np.abs(myFFT_shifted[:][:][int(box_size/2)].astype(np.float)**2)))

    """ This is the angpix. Hardcoded for debug"""
    sample_rate = 1.1
    """ Generate the fft frequency curve """
    freq = np.fft.fftfreq(box_size, d=sample_rate)
    freq_shift = np.fft.fftshift(freq)
   ####

    half_box = int( np.true_divide( box_size, 2 ) )

    """ Assumes x,y,z ordering of ndimage """
    x_max = np.true_divide(box_size, 2) - 1
    x_min = (-1*x_max) -1

    x = np.linspace( x_min, x_max, int(box_size) )
    y,z = np.copy(x), np.copy(x)

    print(x)

    """ This will contain the x,y,z for all points on the central plane """
    coordStack_original = np.vstack(np.meshgrid(x,y,0)).reshape(3,-1).T
    freqStack_original = np.vstack(np.meshgrid(freq_shift,freq_shift,0)).reshape(3,-1).T

    q = R.from_quat( my_rotation )  # in scipy format!

    """ Make the interpolator """
    f = scipy.interpolate.RegularGridInterpolator((x,y,z),my_ndimage,bounds_error=False, fill_value=0)
    # 'fill_value=0' required to avoid nan, which will screw up downstream processing
    f_fft = scipy.interpolate.RegularGridInterpolator((x,y,z),myFFT_shifted,bounds_error=False, fill_value=0)
    #f_fft = scipy.interpolate.RegularGridInterpolator((freq_shift,freq_shift,freq_shift),myFFT_shifted,bounds_error=False, fill_value=0, method='nearest')


    """ Rotate the coordinate system """
    coordStack_rotated=q.apply(coordStack_original)
    freqStack_rotated=q.apply(freqStack_original)
#    coordStack_rotated = coordStack_original
    print( np.average(coordStack_rotated[2]) )
    print( np.average( x ))
    print( freq_shift )

    """ Interpolate data on the rotated coordinate system """
    rotated_section = f(coordStack_rotated)
    rotated_section_fft = f_fft(coordStack_rotated)
#    rotated_section_fft = rotated_mag * np.exp( 1j * rotated_phase )
    ### Uncomment line below for debugging
#    rotated_section_fft = myFFT_shifted[:][:][half_box]
#    rotated_mag = scipy.ndimage.interpolation.rotate(mag_shifted,45,reshape=False)[:][:][half_box]
#    rotated_phase = scipy.ndimage.interpolation.rotate(phase_shifted,45,reshape=False)[:][:][half_box]
#    rotated_section_fft = rotated_mag * np.exp( 1j * rotated_phase )
#    rotated_section_fft = rotated_mag + np.imag(rotated_phase)
    rotated_section_fft = np.fft.ifftshift(rotated_section_fft)

    """ Reshape the data to the original shape """
    rotated_section = rotated_section.reshape( box_size, box_size )
    rotated_section = np.rot90( rotated_section, axes=(0,1) )
    rotated_section_fft = rotated_section_fft.reshape( box_size, box_size )
    rotated_section_fft = np.rot90( rotated_section_fft, axes=(0,1) )

    """ Display the image, for debug """
    plot2dimage(my_ndimage[:][:][half_box])
    plot2dimage(rotated_section)
    plot2dimage( np.fft.ifftn(rotated_section_fft).astype(np.float) )

    """ Apply grid correction """
#    my_image = np.fft.ifftn(rotated_section_fft).astype(np.float)
#    my_image = 
#    plot2dimage

    """ May require more thorough testing!! """

    ### Must take an initial plane, as defined by x,y,z coordinates ###
    ### Will be 3d ndarray of format ( box_size, box_size, 1 )
    ### Then must reshape and transpose to numpy array of shape ( 3, )
    ### Then apply rotation to each of those coordinates

    ### Interpolation function is built off of 3darray and original coordinates
    ### This is not really the expensive part
    ### But could consider cropping to a smaller box, if in fourier space

    ### Finally, interpolate the values at your rotated plane coordinates
    ### Be careful to properly reshape your 2darray afterwards
    ### See rotateVolume3d function for those details

    return  rotated_section

def rotateVolume3d( my_ndimage, box_size ):
    """Input boxsize as int"""
    print( my_ndimage.shape )

    my_ndimage = swapAxes_ndimage( my_ndimage )

    """Your rotation"""     # Hardcoded for debug
    q = R.from_quat( [ -0.500, 0.000, 0.309, 0.809 ] )
    r = q.inv()     # invert, or not?
#    r = R.from_quat( [ 0.000, 0.000, 0.000, 1.000] )   ## identity for testing
    """scipy notation is bi, cj, dk, w"""   ### Confirm this!
    """ docs.scipy.org: in scalar-last format """

    x_max = np.true_divide(box_size, 2) - 1
    x_min = (-1*x_max) - 1

    x = np.linspace( x_min, x_max, int(box_size) )
    y = np.copy(x)
    z = np.copy(x)

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

    ### The following lines are a hack. ### 
    ### The reshape operation does not result ###
    ### in the desired organization of the data. ###
    """ Rotate 90 degrees, flip along z, swap axes to z,y,x ordering """
    rotated_data = np.rot90( rotated_data, axes=(0,1) )
    rotated_data = np.flip( rotated_data, axis=0 )
    """Swap the axes back to z,y,x """
    rotated_data = swapAxes_ndimage( rotated_data ) 
    print( rotated_data.shape )

    return rotated_data

def saveMRC( ndimage, filename ):
    """ Simple function to save as mrc. WILL OVERWRITE EXISTING FILES """

    with mrcfile.new( filename, overwrite=True) as mrc:
        mrc.set_data( ndimage.astype(np.float32))

    return

