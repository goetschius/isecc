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
import mrcfile
from scipy import ndimage
from pyquaternion import Quaternion
from datetime import datetime
from isecc import transform
from isecc import starparse
from isecc import symops
from isecc import checks
from isecc import utils
from isecc.isecc_classes import Particle
from isecc.isecc_classes import AreaOfInterest
from isecc.isecc_display2d import plot2dimage
from isecc.iseccFFT_v2 import *
from isecc.iseccFFT_v2 import swapAxes_ndimage

### Generate array containing I1 rotations ready for pyQuaternion in format [ a, bi, cj, dk ]
I1Quaternions = symops.getSymOps()


def rotate3d( my_ndimage, box_size, quat, residual ):
    """Input boxsize as int"""

    my_ndimage = swapAxes_ndimage( my_ndimage )
    quat =  pyquat2scipy( quat )    # Convert from pyquaternion to scipy format

    """Your rotation"""
    q = R.from_quat( quat )         # Scipy
    r = q.inv()     # invert, or not?
    #r = q

    x_max = np.true_divide(box_size, 2) - 1
    x_min = (-1*x_max) - 1

    x = np.linspace( x_min, x_max, int(box_size) )
    y = np.copy(x)
    z = np.copy(x)

    """ Debug """
    #print(x,y,z)

    """ Add in the fraction of a pixel (or subtract?) """
    x = x + residual[0]
    y = y + residual[1]
    z = z + residual[2]
    #print(x,y,z)

    coordStack_original = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T 

    print( "Making interpolator" )
    f = scipy.interpolate.RegularGridInterpolator((x,y,z),my_ndimage,bounds_error=False, fill_value=0)
    # 'fill_value=0' required to avoid nan, which will screw up downstream processing

    """ Rotate the coordinate system """
    coordStack_rotated=r.apply(coordStack_original)

    """ Interpolate data on the rotated coordinate system """
    print( "Interpolating data" )
    rotated_data = f(coordStack_rotated)

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

    return rotated_data


def openMRCfile( my_file ):
    ###
    my_file = mrcfile.open(my_file).data
    my_ndarray = np.copy(my_file)
    del my_file
    return my_ndarray


def getUniqueVertices( vertex_vector ):

    my_dtype = np.dtype( [  ( 'IdealVector', '<f4', (3,) ),
                ( 'ExpandVector', '<f4', (3,) ),
                ( 'ExpandQuat', '<f4', (4,) ),
                ( 'Unique', 'bool' ) ] )

    ### Make table of length 60
    vertex_table = np.zeros( len(I1Quaternions), my_dtype)

    ### Given
    vertex_table['IdealVector'] = vertex_vector
    vertex_table['ExpandQuat'] = I1Quaternions

    for index, vertex in enumerate( vertex_table, start=0) :

        ### Invert for the vector transformation
        expand_pose_inverse = Quaternion(I1Quaternions[index]).inverse

        ### Transform the vertex vector
        expand_vector = np.around( expand_pose_inverse.rotate( vertex_vector ), decimals=3)

        # Assess whether tranformed point is unique
        unique_bool = True
        for loop_index, item in enumerate( vertex_table, start=0) :
            if np.allclose( expand_vector, vertex_table[loop_index]['ExpandVector'], atol=0.1) :
                unique_bool = False

        # Store the information
        vertex_table[index]['Unique'] = unique_bool
        vertex_table[index]['ExpandVector'] = expand_vector

    ### Gather the unique vertices
    unique_vertex_indices = np.where( vertex_table['Unique'] )[0]

    return unique_vertex_indices


def padCapsomer( capsomer, vector, subbox, mapbox ):

    """ vector, subbox, and mapbox should all be in pixels """
    pad_pix = int( mapbox - subbox )    # total number of pixels to pad

    """ convert vector to array offsets """
    offset_x = vector[0] + np.true_divide(mapbox,2)
    offset_y = (-1*vector[1]) + np.true_divide(mapbox,2)
    offset_z = (-1*vector[2]) + np.true_divide(mapbox,2)

    """ Must use offset to determine pixels to pad on the 'left' side """
    pad_x1 = int( offset_x - np.true_divide(subbox,2) )
    pad_y1 = int( offset_y - np.true_divide(subbox,2) )
    pad_z1 = int( offset_z - np.true_divide(subbox,2) )

    """ Determine the padding for the 'right' side """
    pad_x2 = int( pad_pix - pad_x1 )
    pad_y2 = int( pad_pix - pad_y1 )
    pad_z2 = int( pad_pix - pad_z1 )

    """ Do the padding """
    capsomer = np.pad(capsomer, ( (pad_x1,pad_x2), (pad_y1,pad_y2), (pad_z1,pad_z2) ), 'constant', constant_values=(0,0))

    """ Some printing for debug purposes """
    #print("Pad left:",  pad_x1, pad_y1, pad_z1)
    #print("Pad right:", pad_x2, pad_y2, pad_z2)

    return capsomer


def maskCapsomer( capsomer, correction_mask, vector, capsomer_centers ):

    capsomer_mask = np.ones_like( capsomer )
    #equidistance_mask = np.zeros_like( capsomer, dtype=np.bool )

    nhalf = len(capsomer) // 2
    x0, y0, z0 = np.meshgrid(*[np.arange(-nhalf - 0.5, nhalf - 0.5)] * 3, indexing="ij")
    x = x0
    y = -1*y0
    z = -1*z0
    print("Preparing to create mask for current subvolume")
    gold_distance = np.sqrt((x - vector[0])**2 + (y-vector[1])**2 + (z-vector[2])**2)
    
    for index, center in enumerate(capsomer_centers):

        """ Don't compare this subvolume center against itself """
        if not np.array_equal(vector,center.astype(int)):

            """ This makes the evaluator """
            x0, y0, z0 = np.meshgrid(*[np.arange(-nhalf - 0.5, nhalf - 0.5)] * 3, indexing="ij")
            x = x0
            y = (-1*y0)
            z = (-1*z0)

            """ A volume containing distances to the currently evaluated center """
            dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2 + (z-center[2])**2)

            """ Update overall mask """
            mask = gold_distance <= dist_from_center
            #mask_edge = gold_distance == dist_from_center
            #equidistance_mask = equidistance_mask + mask_edge
            capsomer_mask = capsomer_mask * mask.astype(np.int)
            #print( np.unique(equidistance_mask) )

    """ At this point, capsomer_mask contains all voxels belonging to the given capsomer """
    """ In the case of equidistance, those voxels are overrepresented """
    """ We now want to dilate and soften that mask to reduce nyquist artifacts """
    soft_1 = ndimage.binary_dilation(capsomer_mask, iterations=1) - capsomer_mask
    soft_2 = ndimage.binary_dilation(capsomer_mask, iterations=2) - (capsomer_mask + soft_1)
    soft_3 = ndimage.binary_dilation(capsomer_mask, iterations=3) - (capsomer_mask + soft_1 + soft_2)
    soft_1 = np.cos((1*np.pi)/8) * soft_1.astype(np.float32)
    soft_2 = np.cos((2*np.pi)/8) * soft_2.astype(np.float32)
    soft_3 = np.cos((3*np.pi)/8) * soft_3.astype(np.float32)

    """ Add the soft mask to the capsomer mask """
    capsomer_mask = capsomer_mask.astype(np.float32) + soft_1 + soft_2 + soft_3
    #print( "Unique values in capsomer mask", np.unique(capsomer_mask) )

    """ Only take the capsomer_mask where capsomer map has values """
    capsomer_bool = (capsomer!=0).astype(np.int)
    capsomer_mask = capsomer_mask * capsomer_bool

    """ Multiply by mask """
    capsomer = capsomer * capsomer_mask.astype(np.float32)

    """ Find intersection of equidistance_mask and capsomer_mask """
    #equidistance_mask = equidistance_mask * capsomer_mask.astype(np.bool)

    """ Update the correction_mask """
    correction_mask = correction_mask + capsomer_mask.astype(np.float32)
    #print( "Unique values in correction mask", np.unique(correction_mask) )

    return capsomer, correction_mask


def adjustCapsomer( capsomer, correction_mask, symop, quatZ, vector, capsomer_centers, subbox, mapbox ):
    """ Must invert the symop """
    quaternion = Quaternion(symop).inverse

    """ Return to location in capsid before expanding """
    qpose = quaternion * quatZ
    pose = np.array( [ qpose.scalar, qpose.vector[0], qpose.vector[1], qpose.vector[2]] )

    """ This will be used on the vector """
    quaternion_translate = Quaternion(symop).inverse
    vector = quaternion_translate.rotate(vector)
    vector_decimal = vector - vector.astype(int)
    vector = vector.astype(int)
    print("New is",vector, "residual is", vector_decimal )

    """ Change vector to mrc ZYX order """
    vector = np.array( [(1*(vector[2])), (-1*vector[1]), (-1*vector[0])] )
    vector_decimal = np.array( [vector_decimal[2], (-1*vector_decimal[1]), (-1*vector_decimal[0])] )

    #print("DEBUG:", vector, capsomer_centers[index])

    """ Will need to rotate/interpolate the capsomer here """
    rotated_data = rotate3d( capsomer, capsomer.shape[0], qpose, vector_decimal )
    """ Move the capsomer to the appropriate location """
    capsomer = padCapsomer(rotated_data, vector, subbox, mapbox)
    """ Mask the capsomer to only voxels closest to this capsomer center"""
    capsomer, correction_mask = maskCapsomer( capsomer, correction_mask, vector, capsomer_centers )

    return capsomer, correction_mask


def stitchCapsomers( mrc_pentavalent, mrc_hexavalent, vector_pent, vector_hex, mapbox, angpix, output ):

    """ Make ndarray to store the output map """
    capsid = np.zeros( (mapbox,mapbox,mapbox), dtype=np.float32)
    correction_mask = np.zeros_like(capsid, dtype=np.float32)

    """ Take input """
    vector_pentavalent = np.array([ vector_pent[0], vector_pent[1], vector_pent[2] ])
    vector_hexavalent  = np.array([ vector_hex[0],  vector_hex[1],  vector_hex[2]  ])

    """ Grab the image values """
    ndarray_pentavalent = openMRCfile( mrc_pentavalent )
    ndarray_hexavalent  = openMRCfile( mrc_hexavalent )
    print("dtype of mrc input is", ndarray_pentavalent.dtype)

    """ Get box size for capsomers """
    subbox_pentavalent = ndarray_pentavalent.shape[0]
    subbox_hexavalent  = ndarray_hexavalent.shape[0]

    """ Get parameters for requested symmetry operation """
    params_pent = AreaOfInterest( 'fivefold',   vector_pentavalent )
    params_hex  = AreaOfInterest( 'fullexpand', vector_hexavalent ) 

    """ Quaternions to return to original orientation """
    pent_quat = Quaternion(params_pent.quatZ).inverse
    hex_quat  = Quaternion(params_hex.quatZ ).inverse

    """ Fetch the indices for non-redundant symmetry operations """
    fivefold_indices = getUniqueVertices( vector_pent )
    fullexpand_indices = getUniqueVertices( vector_hex )

    """ Update to use only the unique symmetry operations """
    params_pent.updateSymIndices( fivefold_indices )
    symops_pent = I1Quaternions[ params_pent.symindices ]
    symops_hex  = I1Quaternions

    """ Faster debugging by reducing number of subvolumes relocated """
    #symops_pent = symops_pent[0:2]
    #symops_hex = symops_hex[0:10]
    total_symops = int(len(symops_pent)) + int(len(symops_hex))


    """ Make array to store the x,y,z of each capsomer wrt the capsid """
    capsomer_centers = np.zeros( (total_symops,3) )   # Works for papillomavirus and polyomavirus

    """ Catch bad pentavalent vector, while still allowing for debug """
    if (len(symops_pent) > 12):
        print(len(symops_pent),"symops for pentavalent suggests bad vector. Ensure ratio is 0 0.618 1")
        sys.exit()

    """ Store the centers. Rotate before converting to ZYX ordering """
    for index, symop in enumerate(symops_pent, start=0):    # papillomavirus and polyomavirus
        center = Quaternion(symop).inverse.rotate(vector_pentavalent)
        capsomer_centers[index] = np.array( [(1*center[2]), (-1*center[1]), (-1*center[0])] ).astype(np.int)
    for index, symop in enumerate(symops_hex, start=len(symops_pent)):    # papillomavirus and polyomavirus
        center = Quaternion(symop).inverse.rotate(vector_hexavalent)
        capsomer_centers[index] = np.array( [(1*center[2]), (-1*center[1]), (-1*center[0])] ).astype(np.int)

    """ Adjust the pentavalent capsomers """
    for index, symop in enumerate(symops_pent, start=0): 
        capsomer, correction_mask = adjustCapsomer( ndarray_pentavalent, correction_mask, symop, pent_quat, vector_pentavalent, capsomer_centers, subbox_pentavalent, mapbox )
        capsid = np.add(capsid,capsomer)
        print("Pentavalent capsomer", index, "completed." )

    """ Adjust the hexavalent capsomers """
    for index, symop in enumerate(symops_hex, start=0): 
        capsomer, correction_mask = adjustCapsomer( ndarray_hexavalent, correction_mask, symop, hex_quat, vector_hexavalent, capsomer_centers, subbox_hexavalent, mapbox )
        capsid = np.add(capsid,capsomer)
        print("Hexavalent capsomer", index, "completed.")
        #print("DEBUG: Correction_mask currently contains values", np.unique(correction_mask) )

    """ Cast to float32. Might be unneccessary """
    capsid = capsid.astype(np.float32)

    """ correction_mask currently contains values >1 at overlaps """
    """ should convert zeros to ones before dividing """
    #correction_mask_offset = correction_mask == 0
    #correction_mask = correction_mask.astype(np.float32) + correction_mask_offset.astype(np.int)

    """ Correct the voxels that are overweighted. Avoid divide-by-zero error """
    capsid = np.divide( capsid, correction_mask.astype(np.float32), out=np.zeros_like(capsid), where=correction_mask!=0 )

    """ Write the output mrc file """
    mrc = mrcfile.new_mmap(output, shape=(mapbox,mapbox,mapbox), mrc_mode=2, overwrite=True)
    mrc.set_data(capsid)
    mrc.voxel_size = angpix
    mrc.header.label[1] = str('Created using ISECC_recombine, Goetschius DJ 2020')
    mrc.close()

    return

def main(args):

    if not args.pentavalent.endswith(".mrc"):
        print("Please provide an mrc file for pentavalent capsomer")
        sys.exit()
    if not args.hexavalent.endswith(".mrc"):
        print("Please provide an mrc file for hexavalent capsomer")
        sys.exit

    print("Please recombine subparticle maps using Chimera instead, i.e. fitting into a consensus map, followed by the command vop maximum.")
    print("This script can produce improper recombined maps if precise assumptions are not met.")
    sys.exit
        
    """ Convert vectors from Angstroms to pixels """
    vector_pentavalent = np.true_divide( args.pentavalent_vector, args.angpix )
    vector_hexavalent  = np.true_divide( args.hexavalent_vector,  args.angpix )

    """ Pass parameters to main program """
    stitchCapsomers(args.pentavalent, args.hexavalent, vector_pentavalent, vector_hexavalent, args.output_box, args.angpix, args.output )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pentavalent", type=str, help="pentavalent capsomer mrc file", required=True)
    parser.add_argument("--hexavalent",  type=str, help="hexavalent capsomer mrc file",  required=True)
    parser.add_argument("--pentavalent_vector", type=float, nargs=3, help="vector for pentavalent capsomer", required=True)
    parser.add_argument("--hexavalent_vector",  type=float, nargs=3, help="vector for hexavalent capsomer",  required=True)
    parser.add_argument("--output_box", type=int, help="box size for output map", required=True)
    parser.add_argument("--output", type=str, help="name for output mrc file", default='icosahedron.mrc')
    parser.add_argument("--angpix", type=float, help="pixel size in the input maps", default='1.1')
    sys.exit(main(parser.parse_args()))

