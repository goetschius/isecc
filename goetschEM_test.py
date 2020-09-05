from scipy.spatial.transform import Rotation as R
import scipy.interpolate
import numpy as np
import mrcfile

r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])



x=np.linspace(0,5,6)
y=np.linspace(0,5,6)
z=np.linspace(0,5,6)

### Output of print(x)
### [0. 1. 2. 3. 4. 5.] 

a = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T 
b = r.apply(a) 	# where a is the rotation previously defined


### Do I have to put the coordinate 0,0,0 at the center?
### I think so...
### Want rotations to be about the origin
### This is weird because I'm rotating the actual coordinate system...
### ...by treating coordinates as vectors

### Will apctually want to do something along the lines of:
x = np.linspace(-229.5,229.5,460)
y = np.copy(x)
z = np.copy(x)


f=scipy.interpolate.LinearNDInterpolator(a,data)

### a should have the shape ( LEN, 3 )
### data should have the shape ( LEN, )

print(f(0,0,0))
### This will report the interpolated value at point 0,0,0



#### To make this viable (TAKES TOO LONG), must instead use RegularGridInterpolator
x = np.linspace(-229.5,229.5,460)
y = np.copy(x)
z = np.copy(x)

### Make meshgrid and reshape
a = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T 

my_file = mrcfile.open('MRC/symbreak_it039_class001.mrc')
my_ndimage = np.copy(my_file.data)
del my_file


### For RegularGridInterpolator, you want to pass
# x,y,z as 1d arrays, basically your axes
# my_ndimage as the straight ndarray of values (460,460,460)
f=scipy.interpolate.RegularGridInterpolator((x,y,z),my_ndimage,bounds_error=False)

### Rotate the grid
b=r.apply(a)

rotated_data=f(b)

### Unfortunately, accessing the data from the interpolator takes too long... 0.022 seconds per coordinate. Or 600 hours for a box size 460,460,460
### Actually, I might just be hitting the RAM limit on my laptop


import timeit
print(f( (100,100,100) ), timeit.timeit(stmt='a=10;b=10;sum=a+b'))
-0.0009630394735040682 0.022180300089530647
