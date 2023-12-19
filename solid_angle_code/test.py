from solid_angle.sage import *
from sage.all_cmdline import *   # import sage library

_sage_const_2 = Integer(2); _sage_const_0 = Integer(0); _sage_const_1 = Integer(1); _sage_const_3 = Integer(3); _sage_const_4 = Integer(4); _sage_const_1en6 = RealNumber('1e-6'); _sage_const_100 = Integer(100); _sage_const_0p5 = RealNumber('0.5')# load("~/ma611-code/logging.sage")

A = matrix([[1, 0, 0], [0, 1]])
print(solid_angle_simplicial_2d(A))