import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=4,suppress=True)

### Custom Array
array = np.array([[-90.144, -157.144, -92.415]])

############
#my_matrix = np.array([[-0.30902, 0.5,  0.80902],
#        [ 0.5,       0.80902,      -0.30902],
#        [-0.80902,       0.30902,          -0.5]],
############
I2Matrix_1 = np.array([
[1,0,0],
[0,1,0],
[0,0,1]], dtype=np.float)

I2Matrix_2 = np.array([
[-1,0,0],
[0,-1,0],
[0,0,1]], dtype=np.float)

I2Matrix_3 = np.array([
[0.5,0.80902,0.30902],
[-0.80902,0.30902,0.5],
[0.30902,-0.5,0.80902]])

I2Matrix_4 = np.array([
[-0.30902,0.5,0.80902],
[-0.5,-0.80902,0.30902],
[0.80902,-0.30902,0.5]])

I2Matrix_5 = np.array([
[-0.30902,-0.5,0.80902],
[0.5,-0.80902,-0.30902],
[0.80902,0.30902,0.5]])

I2Matrix_6 = np.array([
[0.5,-0.80902,0.30902],
[0.80902,0.30902,-0.5],
[0.30902,0.5,0.80902]])

I2Matrix_7 = np.array([
[-0.5,0.80902,-0.30902],
[-0.80902,-0.30902,0.5],
[0.30902,0.5,0.80902]])

I2Matrix_8 = np.array([
[-0.5,-0.80902,0.30902],
[0.80902,-0.30902,0.5],
[-0.30902,0.5,0.80902]])

I2Matrix_9 = np.array([
[-0.5,-0.80902,-0.30902],
[0.80902,-0.30902,-0.5],
[0.30902,-0.5,0.80902]])

I2Matrix_10 = np.array([
[0.30902,-0.5,-0.80902],
[0.5,0.80902,-0.30902],
[0.80902,-0.30902,0.5]])

I2Matrix_11 = np.array([
[0.30902,-0.5,0.80902],
[0.5,0.80902,0.30902],
[-0.80902,0.30902,0.5]])

I2Matrix_12 = np.array([
[0.30902,0.5,-0.80902],
[-0.5,0.80902,0.30902],
[0.80902,0.30902,0.5]])

I2Matrix_13 = np.array([
[0.30902,0.5,0.80902],
[-0.5,0.80902,-0.30902],
[-0.80902,-0.30902,0.5]])

I2Matrix_14 = np.array([
[-0.5,0.80902,0.30902],
[-0.80902,-0.30902,-0.5],
[-0.30902,-0.5,0.80902]])

I2Matrix_15 = np.array([
[-0.80902,0.30902,0.5],
[0.30902,-0.5,0.80902],
[0.5,0.80902,0.30902]])

I2Matrix_16 = np.array([
[0,0,1],
[1,0,0],
[0,1,0]], dtype=np.float)

I2Matrix_17 = np.array([
[0.80902,0.30902,0.5],
[0.30902,0.5,-0.80902],
[-0.5,0.80902,0.30902]])

I2Matrix_18 = np.array([
[0.5,0.80902,-0.30902],
[-0.80902,0.30902,-0.5],
[-0.30902,0.5,0.80902]])

I2Matrix_19 = np.array([
[0.5,-0.80902,-0.30902],
[0.80902,0.30902,0.5],
[-0.30902,-0.5,0.80902]])

I2Matrix_20 = np.array([
[0,1,0],
[0,0,1],
[1,0,0]], dtype=np.float)

I2Matrix_21 = np.array([
[0.80902,0.30902,-0.5],
[0.30902,0.5,0.80902],
[0.5,-0.80902,0.30902]])

I2Matrix_22 = np.array([
[0.80902,-0.30902,0.5],
[-0.30902,0.5,0.80902],
[-0.5,-0.80902,0.30902]])

I2Matrix_23 = np.array([
[0,0,1],
[-1,0,0],
[0,-1,0]], dtype=np.float)

I2Matrix_24 = np.array([
[-0.80902,-0.30902,0.5],
[-0.30902,-0.5,-0.80902],
[0.5,-0.80902,0.30902]])

I2Matrix_25 = np.array([
[-0.30902,0.5,-0.80902],
[-0.5,-0.80902,-0.30902],
[-0.80902,0.30902,0.5]])

I2Matrix_26 = np.array([
[0.80902,-0.30902,-0.5],
[-0.30902,0.5,-0.80902],
[0.5,0.80902,0.30902]])

I2Matrix_27 = np.array([
[0.80902,0.30902,0.5],
[-0.30902,-0.5,0.80902],
[0.5,-0.80902,-0.30902]])

I2Matrix_28 = np.array([
[0.30902,-0.5,0.80902],
[-0.5,-0.80902,-0.30902],
[0.80902,-0.30902,-0.5]])

I2Matrix_29 = np.array([
[0,-1,0],
[0,0,-1],
[1,0,0]], dtype=np.float)

I2Matrix_30 = np.array([
[-0.80902,-0.30902,-0.5],
[0.30902,0.5,-0.80902],
[0.5,-0.80902,-0.30902]])

I2Matrix_31 = np.array([
[-0.80902,0.30902,-0.5],
[0.30902,-0.5,-0.80902],
[-0.5,-0.80902,0.30902]])

I2Matrix_32 = np.array([
[-0.30902,-0.5,-0.80902],
[0.5,-0.80902,0.30902],
[-0.80902,-0.30902,0.5]])

I2Matrix_33 = np.array([
[0,0,-1],
[-1,0,0],
[0,1,0]], dtype=np.float)

I2Matrix_34 = np.array([
[-0.80902,-0.30902,-0.5],
[-0.30902,-0.5,0.80902],
[-0.5,0.80902,0.30902]])

I2Matrix_35 = np.array([
[-0.80902,-0.30902,0.5],
[0.30902,0.5,0.80902],
[-0.5,0.80902,-0.30902]])

I2Matrix_36 = np.array([
[0.30902,0.5,0.80902],
[0.5,-0.80902,0.30902],
[0.80902,0.30902,-0.5]])

I2Matrix_37 = np.array([
[0.80902,-0.30902,0.5],
[0.30902,-0.5,-0.80902],
[0.5,0.80902,-0.30902]])

I2Matrix_38 = np.array([
[-0.30902,-0.5,-0.80902],
[-0.5,0.80902,-0.30902],
[0.80902,0.30902,-0.5]])

I2Matrix_39 = np.array([
[-0.80902,0.30902,-0.5],
[-0.30902,0.5,0.80902],
[0.5,0.80902,-0.30902]])

I2Matrix_40 = np.array([
[-0.30902,0.5,-0.80902],
[0.5,0.80902,0.30902],
[0.80902,-0.30902,-0.5]])

I2Matrix_41 = np.array([
[0,0,-1],
[1,0,0],
[0,-1,0]], dtype=np.float)

I2Matrix_42 = np.array([
[0,-1,0],
[0,0,1],
[-1,0,0]], dtype=np.float)

I2Matrix_43 = np.array([
[-0.30902,-0.5,0.80902],
[-0.5,0.80902,0.30902],
[-0.80902,-0.30902,-0.5]])

I2Matrix_44 = np.array([
[-0.80902,0.30902,0.5],
[-0.30902,0.5,-0.80902],
[-0.5,-0.80902,-0.30902]])

I2Matrix_45 = np.array([
[-0.30902,0.5,0.80902],
[0.5,0.80902,-0.30902],
[-0.80902,0.30902,-0.5]])

I2Matrix_46 = np.array([
[0,1,0],
[0,0,-1],
[-1,0,0]], dtype=np.float)

I2Matrix_47 = np.array([
[-0.5,0.80902,0.30902],
[0.80902,0.30902,0.5],
[0.30902,0.5,-0.80902]])

I2Matrix_48 = np.array([
[0.5,0.80902,0.30902],
[0.80902,-0.30902,-0.5],
[-0.30902,0.5,-0.80902]])

I2Matrix_49 = np.array([
[0.80902,0.30902,-0.5],
[-0.30902,-0.5,-0.80902],
[-0.5,0.80902,-0.30902]])

I2Matrix_50 = np.array([
[0.5,0.80902,-0.30902],
[0.80902,-0.30902,0.5],
[0.30902,-0.5,-0.80902]])

I2Matrix_51 = np.array([
[0.80902,-0.30902,-0.5],
[0.30902,-0.5,0.80902],
[-0.5,-0.80902,-0.30902]])

I2Matrix_52 = np.array([
[0.5,-0.80902,0.30902],
[-0.80902,-0.30902,0.5],
[-0.30902,-0.5,-0.80902]])

I2Matrix_53 = np.array([
[-0.5,-0.80902,0.30902],
[-0.80902,0.30902,-0.5],
[0.30902,-0.5,-0.80902]])

I2Matrix_54 = np.array([
[1,0,0],
[0,-1,0],
[0,0,-1]], dtype=np.float)

I2Matrix_55 = np.array([
[0.5,-0.80902,-0.30902],
[-0.80902,-0.30902,-0.5],
[0.30902,0.5,-0.80902]])

I2Matrix_56 = np.array([
[0.30902,-0.5,-0.80902],
[-0.5,-0.80902,0.30902],
[-0.80902,0.30902,-0.5]])

I2Matrix_57 = np.array([
[-0.5,-0.80902,-0.30902],
[-0.80902,0.30902,0.5],
[-0.30902,0.5,-0.80902]])

I2Matrix_58 = np.array([
[-0.5,0.80902,-0.30902],
[0.80902,0.30902,-0.5],
[-0.30902,-0.5,-0.80902]])

I2Matrix_59 = np.array([
[0.30902,0.5,-0.80902],
[0.5,-0.80902,-0.30902],
[-0.80902,-0.30902,-0.5]])

I2Matrix_60 = np.array([
[-1,0,0],
[0,1,0],
[0,0,-1]], dtype=np.float)
#########################

my_quat = np.around( np.array( R.from_matrix(I2Matrix_1).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_2).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_3).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_4).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_5).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_6).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_7).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_8).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_9).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_10).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_11).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_12).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_13).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_14).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_15).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_16).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_17).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_18).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_19).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_20).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_21).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_22).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_23).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_24).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_25).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_26).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_27).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_28).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_29).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_30).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_31).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_32).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_33).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_34).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_35).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_36).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_37).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_38).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_39).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_40).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_41).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_42).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_43).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_44).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_45).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_46).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_47).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_48).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_49).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_50).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_51).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_52).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_53).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_54).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_55).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_56).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_57).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_58).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_59).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )
my_quat = np.around( np.array( R.from_matrix(I2Matrix_60).as_quat() ), decimals=3 )
print( my_quat[3], my_quat[0], my_quat[1], my_quat[2]  )

