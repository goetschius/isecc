import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import mrcfile
from isecc import *
import scipy.ndimage
from pyem import *
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
from matplotlib import cm
import sys
from isecc import utils
from isecc import iseccFFT_v3

def plot3dimage( ndimage ):
    angpix = 2.2
    correction = np.array([angpix,angpix,angpix])
    axis_pix = int(ndimage.shape[0])
    axis = axis_pix
    center = np.array([ axis/2, axis/2, axis/2 ])
    zero_center = np.array([0,0,0])
#    zero_center = center
    max_radius = utils.assess3dDistance( center, np.array([0,0,0]) )
    x,y,z = np.ogrid[0:axis_pix, 0:axis_pix, 0:axis_pix]
#    x,y,z = np.ogrid[0:axis:axis_pix, 0:axis:axis_pix, 0:axis:axis_pix]

    radii = np.zeros( [1,], dtype=np.float )

    threshold = np.around( ( (np.amax(ndimage) / 10) + 0.004 ), decimals=4 )

    colormap = np.array(cm.viridis.colors)
    colormap = np.array(cm.plasma.colors)

    verts, faces, normals, values = measure.marching_cubes_lewiner( ndimage, step_size=2, level=threshold )
    #verts, faces, normals, values = measure.marching_cubes_lewiner( ndimage, step_size=4, level=0.0078 )

    fig = plt.figure(figsize=(7.5,7.5))
    ax = fig.add_subplot(111, projection='3d')
    #mesh = Poly3DCollection(verts[faces], linewidth=0.1, facecolors='xkcd:aqua', edgecolors='xkcd:indigo')
    num_faces = len(faces)
    radii = np.zeros( [num_faces,], dtype=float )

    for index, face in enumerate(faces):
        triangle = verts[face[0]] - center, verts[face[1]] - center, verts[face[2]] - center
        radius = utils.assess3dDistance( triangle[0], zero_center )
        if index == 0:  radii[0] = radius
#        else:   radii = np.append(radii, radius)
        else:   radii[index] = radius 
        
        if index % (10**5) == 0:
            print( index, 'of', len(faces))

    min_radius = np.amin(radii)
    max_radius = np.amax(radii)

    for index, face in enumerate(faces):

        triangle = verts[face[0]] - center, verts[face[1]] - center, verts[face[2]] - center
        radius = utils.assess3dDistance( triangle[0], zero_center )
        color_value = np.true_divide( (radius-min_radius), (max_radius-min_radius) ) * (len(colormap) - 1)
        face_color = colormap[int(color_value)]
        triangle *= correction

        if index % (10**5) == 0:
            print( face_color, index, 'of', len(faces))

        face = Poly3DCollection([triangle], linewidth=0.1, facecolor=face_color, edgecolors='k')
#        face.set_facecolor(face_color)
#        face.set_edgecolor('k')
#        face.set_linewidth(0.1)
        ax.add_collection(face)

#    ax.add_collection3d(mesh)
    maxrad = int((max_radius*angpix)/10) * 10
    ax.set_xlim(-maxrad,maxrad)
    ax.set_ylim(-maxrad,maxrad)
    ax.set_zlim(-maxrad,maxrad)
#    ax.set_xlim(0,maxrad*2)
#    ax.set_ylim(0,maxrad*2)
#    ax.set_zlim(0,maxrad*2)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
#    ax.view_init(elev=100, azim=0)
#    plt.title(''.join([ 'Isosurface at ', str(threshold) ]) )
    plt.tight_layout()
    plt.axis('off')
#    plt.show()

    plt.savefig( 'mupyv.png', bbox_inches='tight')

    return



###
my_file = mrcfile.open('../MRC/MuPyV_scale.mrc').data
my_ndarray = np.copy(my_file)
my_ndarray = np.flipud(my_ndarray)
#my_ndarray = np.rot90(my_ndarray,k=-1,axes=(0,2))
my_ndarray = iseccFFT_v3.swapAxes_ndimage( my_ndarray )
#my_ndarray = np.rot90(my_ndarray, k=1, axes=(0,1))
del my_file

plot3dimage( my_ndarray )
