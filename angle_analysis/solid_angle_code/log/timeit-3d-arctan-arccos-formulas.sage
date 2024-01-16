sage: def solid3(A): #lots of numerical approximations 
....:     v_0=A.row(0) 
....:     v_1=A.row(1) 
....:     v_2=A.row(2) 
....:     c_01=v_0.cross_product(v_1) 
....:     c_02=v_0.cross_product(v_2) 
....:     c_12=v_1.cross_product(v_2) 
....:     n_01=c_01.norm().n() 
....:     n_02=c_02.norm().n() 
....:     n_12=c_12.norm().n() 
....:     d_0=c_01.dot_product(c_02) 
....:     d_1=c_01.dot_product(c_12) 
....:     d_2=c_02.dot_product(c_12) 
....:     a_0=arccos(d_0/(n_01*n_02)).n() 
....:     a_1=arccos(-d_1/(n_01*n_12)).n() 
....:     a_2=arccos(d_2/(n_02*n_12)).n() 
....:     sum=a_0+a_1+a_2 
....:     denom = (4*pi).n() 
....:     omega=(sum-pi)/denom 
....:     return (omega).n() 
....:                                                                           
sage: def solid_angle_3d(v, normalized=True): #Dr.Zhou's with atan 
....:     assert(v.nrows() == 3) 
....:     vnorm = [v[i].norm().n() for i in range(3)] 
....:     angle = 2 * atan2(abs(v.determinant()), vnorm[0]*vnorm[1]*vnorm[2]+(v[
....: 0]*v[1])*vnorm[2]+(v[0]*v[2])*vnorm[1]+(v[1]*v[2])*vnorm[0])   # atan2(y, 
....: x) in [-pi, pi]  
....:     if normalized: 
....:         return (angle/(4*pi)).n() 
....:     else: 
....:         return (angle).n() 
....:                                                                           
sage: v = matrix([[1,0,1],[0,1,1],[0,0,1]])                                     
sage: solid3(v)                                                                 
0.0270433619923482
sage: solid_angle_3d(v)                                                         
0.0270433619923482
sage: timeit.eval("solid3(v)")                                                  
625 loops, best of 3: 541 μs per loop
sage:                                                                           
sage: timeit.eval("solid3(v)")                                                  
625 loops, best of 3: 530 μs per loop
sage:                                                                           
sage: timeit.eval("solid_angle_3d(v)")                                          
625 loops, best of 3: 538 μs per loop
sage: timeit.eval("solid_angle_3d(v)")                                          
625 loops, best of 3: 539 μs per loop
sage: timeit.eval("solid3(v)")                                                  
625 loops, best of 3: 540 μs per loop
sage: timeit.eval("solid_angle_3d(v)")                                          
625 loops, best of 3: 541 μs per loop
sage: w=matrix([[1,0,1],[0,1,1],[0,0,-1]])                                      
sage: solid3(w)                                                                 
0.222956638007652
sage: solid_angle_3d(w)                                                         
0.222956638007652
sage: timeit.eval("solid3(w)")                                                  
625 loops, best of 3: 534 μs per loop
sage: timeit.eval("solid_angle_3d(w)")                                          
625 loops, best of 3: 541 μs per loop
sage: timeit.eval("solid3(w)")                                                  
625 loops, best of 3: 527 μs per loop
sage: timeit.eval("solid_angle_3d(w)")                                          
625 loops, best of 3: 541 μs per loop
sage: timeit.eval("solid3(w)")                                                  
625 loops, best of 3: 529 μs per loop
sage: timeit.eval("solid_angle_3d(w)")                                          
625 loops, best of 3: 555 μs per loop
sage: x=matrix([[1,0,0],[0,1,0],[0,0,1]])                                       
sage: solid3(x)                                                                 
0.125000000000000
sage: solid_angle_3d(x)                                                         
0.125000000000000
sage: timeit.eval("solid3(x)")                                                  
625 loops, best of 3: 372 μs per loop
sage: timeit.eval("solid_angle_3d(x)")                                          
625 loops, best of 3: 233 μs per loop
sage: timeit.eval("solid3(x)")                                                  
625 loops, best of 3: 362 μs per loop
sage: timeit.eval("solid_angle_3d(x)")                                          
625 loops, best of 3: 229 μs per loop
sage: timeit.eval("solid3(x)")                                                  
625 loops, best of 3: 365 μs per loop
sage: timeit.eval("solid_angle_3d(x)")                                          
625 loops, best of 3: 228 μs per loop
sage: y=matrix([[2,0,0],[0,3,0],[-4,-4,0]])                                     
sage: solid3(y)                                                                 
0.500000000000000
sage: solid_angle_3d(y)                                                         
0.500000000000000
sage: timeit.eval("solid3(y)")                                                  
625 loops, best of 3: 385 μs per loop
sage: timeit.eval("solid_angle_3d(y)")                                          
625 loops, best of 3: 360 μs per loop
sage: timeit.eval("solid3(y)")                                                  
625 loops, best of 3: 382 μs per loop
sage: timeit.eval("solid_angle_3d(y)")                                          
625 loops, best of 3: 357 μs per loop
sage: timeit.eval("solid3(y)")                                                  
625 loops, best of 3: 390 μs per loop
sage: timeit.eval("solid_angle_3d(y)")                                          
625 loops, best of 3: 361 μs per loop
sage: timeit("solid_angle_3d(y)")                                               
625 loops, best of 3: 361 μs per loop
sage: timeit("solid3(y)")                                                       
625 loops, best of 3: 366 μs per loop
sage: timeit("solid3(v)")                                                       
625 loops, best of 3: 529 μs per loop
sage: timeit("solid3(w)")                                                       
625 loops, best of 3: 530 μs per loop
sage: timeit("solid_angle_3d(v)")                                               
625 loops, best of 3: 540 μs per loop
sage: timeit("solid_angle_3d(w)")                                               
625 loops, best of 3: 535 μs per loop
sage: timeit("solid_angle_3d(v)")                                               
625 loops, best of 3: 541 μs per loop
sage: timeit("solid3(v)")                                                       
625 loops, best of 3: 530 μs per loop
sage: timeit("solid3(x)")                                                       
625 loops, best of 3: 360 μs per loop
sage: timeit("solid_angle_3d(x)")                                               
625 loops, best of 3: 224 μs per loop
