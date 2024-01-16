sage: Delta2Delta1 = Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]]) * Polyhedron(vertices=[[-1], [1]])
sage: Delta2Delta1.vertices()
(A vertex at (0, 0, -1),
 A vertex at (0, 0, 1),
 A vertex at (0, 1, -1),
 A vertex at (0, 1, 1),
 A vertex at (1, 0, -1),
 A vertex at (1, 0, 1))
sage: Delta2Delta1.gale_transform()
[(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]
## equals (up to order) Ribando thesis page 26

sage: C3 = polytopes.cube()
sage: C3.gale_transform?
sage: C3.gale_transform()
[(1, 0, 0, 0),
 (0, 1, 0, 0),
 (0, 0, 1, 0),
 (-1, -1, -1, 0),
 (0, 0, 0, 1),
 (-1, -1, 0, -1),
 (-1, 0, -1, -1),
 (2, 1, 1, 1)]
## different from Ribando thesis page 29.
## Ribando: "Note  that in this case, the matrix representation has been chosen so that Y has orthogonal rows (and Y^T has orthogonal columns)."

sage: v = matrix([[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]]) #type large in table 4.1
sage: solid_angle(v, eps=1e-6, deg=40)
1/16
sage: v = matrix([[1,-1,-1,-1],[0,2,0,0],[0,0,2,0],[0,0,0,2]]) #type corner
sage: solid_angle(v, eps=1e-6, deg=40)
0.228644891452286
sage: solid_angle(v, eps=1e-6, deg=40, tridiag=False) # long time
0.234053001243693
sage: solid_angle(v, eps=1e-6, deg=100)
0.234344408607669
sage: v = matrix([[1,-1,-1,-1],[-1,-1,-1,1],[0,0,2,0],[0,0,0,2]]) #type standard
sage: solid_angle(v, eps=1e-6, deg=40)
0.104008507434958
sage: solid_angle(v, eps=1e-6, deg=100)
0.104166725557257
sage: v = matrix([[1,-1,-1,-1],[2,0,0,0],[0,0,2,0],[0,0,0,2]]) #type opposite
sage: solid_angle(v, eps=1e-6, deg=40)
0.0572561709065628
sage: solid_angle(v, eps=1e-6, deg=100)
0.0572856071113725

sage: 2*1/16+8*0.234344408607669+24*0.104166725557257+24*0.0572856071113725
5.87461125290846
sage: RR(5.87461125290846/5.875*100)
99.9933830282291

GT = [[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1],[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]]
Y = matrix(GT)
d = Y.nrows(); n = Y.ncols()
sqnormGT = [sum(i*i for i in x) for x in GT]
VtVs = []
sas = []
for i in Combinations(range(d),n):
    Vt = Y.matrix_from_rows(i)
    if Vt.rank() < n:
        sa = 0
    else:
        sa = None
        for j in Permutations(i):
            Vt_j = Y.matrix_from_rows(j)
            # Vt_j = matrix(AA, Y.matrix_from_rows(j))
            # for k in range(n):
            #     Vt_j[k] = Vt_j[k] / sqrt(sqnormGT[j[k]])
            VtV_j = Vt_j*Vt_j.transpose()
            for k in range(len(VtVs)):
                if VtV_j == VtVs[k]:
                    sa = sas[k]
                    break
            if sa is not None:
                break
        if sa is None:
            sa = solid_angle(Vt, eps=1e-6, deg=100, normalized=True, tridiag=True)       
    VtV = Vt*Vt.transpose()
    # Vt_i = matrix(AA, Vt)
    # for k in range(n):
    #     Vt_i[k] = Vt_i[k] / sqrt(sqnormGT[i[k]])
    # VtV = Vt_i*Vt_i.transpose()
    VtVs.append(VtV)
    sas.append(sa)
    print i, Vt, VtV, sa
sage: len(sas)
70
sage: sum(sas)
5.87453707327138
sage: set(sas)
{0, 0.0572856071113725, 1/16, 0.104163634739045, 0.234344408607669}

# If we use unnormalized GT, symmetry is lost. #thesis page 32 bottom
C3 = polytopes.cube()
GT = C3.gale_transform()
Y = .......
....
sage: len(sas)
70
sage: sum(sas)
5.91330083610103
sage: set(sas)
{0,
 0.0113273741611140,
 0.0118305089027963,
 0.0270071375032462,
 0.0383883061402848,
 0.0519535194829534,
 1/16,
 0.0761100471150076,
 0.0866614670413439,
 0.0951423362255349,
 0.105695168004360,
 0.145831009936207,
 0.199029261790698,
 0.273852232567164,
 0.294757061985102}

# https://ask.sagemath.org/question/9039/calculating-an-orthonormal-basis/
C3 = polytopes.cube()
A = matrix(C3.gale_transform()).transpose()
G, M = A.gram_schmidt()     # no normalization in G
g=G*G.transpose()
gsqrt=g.apply_map(sqrt)
q=gsqrt.inverse()
gcdq = gcd(q.diagonal())
GT= (q/gcdq*G).transpose()
Y = matrix(QQ, GT)
......
sage: sum(sas)
5.87468275564102
sage: set(sas)
{0, 0.0572885863918958, 1/16, 0.104166725557257, 0.234344408607669}


# thesis section 4.5.2
sage: Delta1Delta1Delta2 =  Polyhedron(vertices=[[-1], [1]]) *  Polyhedron(vertices=[[-1], [1]]) * Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])
sage: Delta1Delta1Delta2.vertices()
(A vertex at (-1, -1, 0, 0),
 A vertex at (-1, -1, 0, 1),
 A vertex at (-1, -1, 1, 0),
 A vertex at (-1, 1, 0, 0),
 A vertex at (-1, 1, 0, 1),
 A vertex at (-1, 1, 1, 0),
 A vertex at (1, -1, 0, 0),
 A vertex at (1, -1, 0, 1),
 A vertex at (1, -1, 1, 0),
 A vertex at (1, 1, 0, 0),
 A vertex at (1, 1, 0, 1),
 A vertex at (1, 1, 1, 0))


Delta1Delta1Delta2 =  Polyhedron(vertices=[[-1], [1]]) *  Polyhedron(vertices=[[-1], [1]]) * Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])
A = matrix(Delta1Delta1Delta2.gale_transform()).transpose()
G, M = A.gram_schmidt()     # no normalization in G
g=G*G.transpose()
gsqrt=g.apply_map(sqrt)
#gsqrt=matrix(RR,g.apply_map(sqrt)) #mistake in Vt.rank()
q=gsqrt.inverse()
#gcdq = gcd(q.diagonal())
GT= (q*G).transpose()
Y = GT
#TypeError: cannot convert (1/4*sqrt(2), -3/14*sqrt(7/2), -1/3*sqrt(3/7), -1/4*sqrt(2), 1/30*sqrt(15/2), -1/2*sqrt(7/15), 0) to Vector space of dimension 7 over Rational Field!
#No more error after changing Cone c to matrix v.
diffangles=0
d = Y.nrows(); n = Y.ncols()
sqnormGT = [sum(i*i for i in x) for x in GT]
VtVs = []
sas = []
num_var_num_subc = []
for i in Combinations(range(d),n):
    Vt = Y.matrix_from_rows(i)
    numsubcones = []
    if Vt.rank() < n:
        sa = 0
    else:
        sa = None
        for j in Permutations(i):
            Vt_j = Y.matrix_from_rows(j)
            # Vt_j = matrix(AA, Y.matrix_from_rows(j))
            # for k in range(n):
            #     Vt_j[k] = Vt_j[k] / sqrt(sqnormGT[j[k]])
            VtV_j = Vt_j*Vt_j.transpose()
            for k in range(len(VtVs)):
                if VtV_j == VtVs[k]:
                    sa = sas[k]
                    break
            if sa is not None:
                break
        if sa is None:
            diffangles += 1
            sa = diffangles
            numsubcones =[(orth_m.nrows(),len(list(generate_cones_decomposition(orth_m, h=None, w=None, s=1, tridiag=True)))) for orth_m in generate_orthogonal_parts(Vt)]
            num_var_num_subc += numsubcones
            #sa = solid_angle(Vt, eps=1e-6, deg=10, normalized=True, tridiag=True)       
    VtV = Vt*Vt.transpose()
    # Vt_i = matrix(AA, Vt)
    # for k in range(n):
    #     Vt_i[k] = Vt_i[k] / sqrt(sqnormGT[i[k]])
    # VtV = Vt_i*Vt_i.transpose()
    VtVs.append(VtV)
    sas.append(sa)
    print i, sa, numsubcones

# [0, 1, 2, 3, 4, 5, 6] 0
# [0, 1, 2, 3, 4, 5, 7] 0
# [0, 1, 2, 3, 4, 5, 8] 0
# [0, 1, 2, 3, 4, 5, 9] 0
# [0, 1, 2, 3, 4, 5, 10] 0
# [0, 1, 2, 3, 4, 5, 11] 0
# [0, 1, 2, 3, 4, 6, 7] 0.00691903860242367
# [0, 1, 2, 3, 4, 6, 8] 0.00343345669212583
# [0, 1, 2, 3, 4, 6, 9] 0
# [0, 1, 2, 3, 4, 6, 10] -0.00535326444254445
# [0, 1, 2, 3, 4, 6, 11] -0.00632645872815816
# [0, 1, 2, 3, 4, 7, 8] 0.00343345669212583
# [0, 1, 2, 3, 4, 7, 9] -0.00535326444254445
# [0, 1, 2, 3, 4, 7, 10] 0
# [0, 1, 2, 3, 4, 7, 11] -0.00632645872815816
# ......

# table 4.1 page 53 for Delta1Delta1Delta2
#type of cone, num of repetitions,(dim, num subcones), [ray indices]
# 0 288 lower dim cone
# 1 12 (7, 120) [0, 1, 2, 3, 4, 6, 7]
# 2 24 (7, 250) [0, 1, 2, 3, 4, 6, 8] 
# 3 48 (7, 176) [0, 1, 2, 3, 4, 6, 10]
# 4 48 (7, 164) [0, 1, 2, 3, 4, 6, 11]
# 5 48 (7, 250) [0, 1, 2, 3, 4, 8, 9]
# 6 24 (7, 204) [0, 1, 2, 3, 4, 9, 10]
# 7 48 (7, 148) [0, 1, 2, 3, 4, 9, 11]
# 8 24 (7, 204) [0, 1, 2, 3, 7, 9, 10]
# 9 48 (7, 148) [0, 1, 2, 3, 7, 9, 11]
# 10 12 (7, 120) [0, 1, 3, 4, 6, 7, 11]
# 11 48 (7, 190) [0, 1, 3, 4, 6, 8, 10]
# 12 24 (7, 52)  [0, 1, 3, 5, 6, 8, 10]
# 13 24 (7, 120) [0, 1, 3, 5, 6, 8, 11]
# 14 48 (7, 164) [0, 1, 3, 5, 6, 10, 11]
# 15 24 (7, 250) [0, 1, 3, 5, 7, 8, 11]

# type of cone, VtV
# 1
# [ 7/12  -1/6  -1/6  -1/4     0     0     0]
# [ -1/6  7/12  -1/6     0  -1/4     0  -1/4]
# [ -1/6  -1/6  7/12     0     0  -1/4     0]
# [ -1/4     0     0  7/12  -1/6  -1/6   1/6]
# [    0  -1/4     0  -1/6  7/12  -1/6 -1/12]
# [    0     0  -1/4  -1/6  -1/6  7/12   1/6]
# [    0  -1/4     0   1/6 -1/12   1/6  7/12]
# 2
# [ 7/12  -1/6  -1/6  -1/4     0     0     0]
# [ -1/6  7/12  -1/6     0  -1/4     0     0]
# [ -1/6  -1/6  7/12     0     0  -1/4  -1/4]
# [ -1/4     0     0  7/12  -1/6  -1/6   1/6]
# [    0  -1/4     0  -1/6  7/12  -1/6   1/6]
# [    0     0  -1/4  -1/6  -1/6  7/12 -1/12]
# [    0     0  -1/4   1/6   1/6 -1/12  7/12]
# 3
# [ 7/12  -1/6  -1/6  -1/4     0     0 -1/12]
# [ -1/6  7/12  -1/6     0  -1/4     0   1/6]
# [ -1/6  -1/6  7/12     0     0  -1/4   1/6]
# [ -1/4     0     0  7/12  -1/6  -1/6  -1/4]
# [    0  -1/4     0  -1/6  7/12  -1/6     0]
# [    0     0  -1/4  -1/6  -1/6  7/12     0]
# [-1/12   1/6   1/6  -1/4     0     0  7/12]
# 4
# [ 7/12  -1/6  -1/6  -1/4     0     0   1/6]
# [ -1/6  7/12  -1/6     0  -1/4     0 -1/12]
# [ -1/6  -1/6  7/12     0     0  -1/4   1/6]
# [ -1/4     0     0  7/12  -1/6  -1/6     0]
# [    0  -1/4     0  -1/6  7/12  -1/6  -1/4]
# [    0     0  -1/4  -1/6  -1/6  7/12     0]
# [  1/6 -1/12   1/6     0  -1/4     0  7/12]
# 5
# [ 7/12  -1/6  -1/6  -1/4     0     0   1/6]
# [ -1/6  7/12  -1/6     0  -1/4     0   1/6]
# [ -1/6  -1/6  7/12     0     0  -1/4 -1/12]
# [ -1/4     0     0  7/12  -1/6  -1/6     0]
# [    0  -1/4     0  -1/6  7/12  -1/6     0]
# [    0     0  -1/4  -1/6  -1/6  7/12  -1/4]
# [  1/6   1/6 -1/12     0     0  -1/4  7/12]
# 6
# [ 7/12  -1/6  -1/6  -1/4     0  -1/4     0]
# [ -1/6  7/12  -1/6     0  -1/4     0  -1/4]
# [ -1/6  -1/6  7/12     0     0     0     0]
# [ -1/4     0     0  7/12  -1/6 -1/12   1/6]
# [    0  -1/4     0  -1/6  7/12   1/6 -1/12]
# [ -1/4     0     0 -1/12   1/6  7/12  -1/6]
# [    0  -1/4     0   1/6 -1/12  -1/6  7/12]
# 7
# [ 7/12  -1/6  -1/6  -1/4     0  -1/4     0]
# [ -1/6  7/12  -1/6     0  -1/4     0     0]
# [ -1/6  -1/6  7/12     0     0     0  -1/4]
# [ -1/4     0     0  7/12  -1/6 -1/12   1/6]
# [    0  -1/4     0  -1/6  7/12   1/6   1/6]
# [ -1/4     0     0 -1/12   1/6  7/12  -1/6]
# [    0     0  -1/4   1/6   1/6  -1/6  7/12]
# 8
# [ 7/12  -1/6  -1/6  -1/4     0  -1/4 -1/12]
# [ -1/6  7/12  -1/6     0  -1/4     0   1/6]
# [ -1/6  -1/6  7/12     0     0     0   1/6]
# [ -1/4     0     0  7/12  -1/6 -1/12  -1/4]
# [    0  -1/4     0  -1/6  7/12   1/6     0]
# [ -1/4     0     0 -1/12   1/6  7/12  -1/4]
# [-1/12   1/6   1/6  -1/4     0  -1/4  7/12]
# 9
# [ 7/12  -1/6  -1/6  -1/4     0  -1/4   1/6]
# [ -1/6  7/12  -1/6     0  -1/4     0 -1/12]
# [ -1/6  -1/6  7/12     0     0     0   1/6]
# [ -1/4     0     0  7/12  -1/6 -1/12     0]
# [    0  -1/4     0  -1/6  7/12   1/6  -1/4]
# [ -1/4     0     0 -1/12   1/6  7/12     0]
# [  1/6 -1/12   1/6     0  -1/4     0  7/12]
# 10
# [ 7/12  -1/6  -1/6  -1/4     0  -1/4   1/6]
# [ -1/6  7/12  -1/6     0  -1/4     0   1/6]
# [ -1/6  -1/6  7/12     0     0     0 -1/12]
# [ -1/4     0     0  7/12  -1/6 -1/12     0]
# [    0  -1/4     0  -1/6  7/12   1/6     0]
# [ -1/4     0     0 -1/12   1/6  7/12     0]
# [  1/6   1/6 -1/12     0     0     0  7/12]
# 11
# [ 7/12  -1/6  -1/6  -1/4     0     0     0]
# [ -1/6  7/12  -1/6     0  -1/4  -1/4     0]
# [ -1/6  -1/6  7/12     0     0     0  -1/4]
# [ -1/4     0     0  7/12  -1/6   1/6   1/6]
# [    0  -1/4     0  -1/6  7/12 -1/12   1/6]
# [    0  -1/4     0   1/6 -1/12  7/12  -1/6]
# [    0     0  -1/4   1/6   1/6  -1/6  7/12]
# 12
# [ 7/12  -1/6  -1/6  -1/4     0     0 -1/12]
# [ -1/6  7/12  -1/6     0  -1/4  -1/4   1/6]
# [ -1/6  -1/6  7/12     0     0     0   1/6]
# [ -1/4     0     0  7/12  -1/6   1/6  -1/4]
# [    0  -1/4     0  -1/6  7/12 -1/12     0]
# [    0  -1/4     0   1/6 -1/12  7/12     0]
# [-1/12   1/6   1/6  -1/4     0     0  7/12]
# 13
# [ 7/12  -1/6  -1/6  -1/4     0     0   1/6]
# [ -1/6  7/12  -1/6     0  -1/4  -1/4 -1/12]
# [ -1/6  -1/6  7/12     0     0     0   1/6]
# [ -1/4     0     0  7/12  -1/6   1/6     0]
# [    0  -1/4     0  -1/6  7/12 -1/12  -1/4]
# [    0  -1/4     0   1/6 -1/12  7/12  -1/4]
# [  1/6 -1/12   1/6     0  -1/4  -1/4  7/12]
# 14
# [ 7/12  -1/6  -1/6  -1/4     0     0   1/6]
# [ -1/6  7/12  -1/6     0  -1/4  -1/4   1/6]
# [ -1/6  -1/6  7/12     0     0     0 -1/12]
# [ -1/4     0     0  7/12  -1/6   1/6     0]
# [    0  -1/4     0  -1/6  7/12 -1/12     0]
# [    0  -1/4     0   1/6 -1/12  7/12     0]
# [  1/6   1/6 -1/12     0     0     0  7/12]
# 15
# [ 7/12  -1/6  -1/6  -1/4     0     0 -1/12]
# [ -1/6  7/12  -1/6     0  -1/4     0   1/6]
# [ -1/6  -1/6  7/12     0     0  -1/4   1/6]
# [ -1/4     0     0  7/12  -1/6   1/6  -1/4]
# [    0  -1/4     0  -1/6  7/12   1/6     0]
# [    0     0  -1/4   1/6   1/6  7/12     0]
# [-1/12   1/6   1/6  -1/4     0     0  7/12]

# not orthogonal basis
Delta1Delta1Delta2 =  Polyhedron(vertices=[[-1], [1]]) *  Polyhedron(vertices=[[-1], [1]]) * Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])
GT = Delta1Delta1Delta2.gale_transform()
Y = matrix(GT)
d = Y.nrows(); n = Y.ncols()
sqnormGT = [sum(i*i for i in x) for x in GT]
VtVs = []
sas = []
for i in Combinations(range(d),n):
    Vt = Y.matrix_from_rows(i)
    if Vt.rank() < n:
        sa = 0
    else:
        sa = None
        for j in Permutations(i):
            # Vt_j = Y.matrix_from_rows(j) 
            Vt_j = matrix(AA, Y.matrix_from_rows(j))
            for k in range(n):
                Vt_j[k] = Vt_j[k] / sqrt(sqnormGT[j[k]])
            VtV_j = Vt_j*Vt_j.transpose()
            for k in range(len(VtVs)):
                if VtV_j == VtVs[k]:
                    sa = sas[k]
                    break
            if sa is not None:
                break
        if sa is None:
            sa = solid_angle(Vt, eps=1e-6, deg=10, normalized=True, tridiag=True)       
    # VtV = Vt*Vt.transpose()
    Vt_i = matrix(AA, Vt)
    for k in range(n):
        Vt_i[k] = Vt_i[k] / sqrt(sqnormGT[i[k]])
    VtV = Vt_i*Vt_i.transpose()
    VtVs.append(VtV)
    sas.append(sa)
    print i, sa

#### 11 variables. TOO lONG ####
# [0, 1, 2, 3, 4, 5, 6] 0
# [0, 1, 2, 3, 4, 5, 7] 0
# [0, 1, 2, 3, 4, 5, 8] 0
# [0, 1, 2, 3, 4, 5, 9] 0
# [0, 1, 2, 3, 4, 5, 10] 0
# [0, 1, 2, 3, 4, 5, 11] 0
# [0, 1, 2, 3, 4, 6, 7] 1/128
# [0, 1, 2, 3, 4, 6, 8] 0.00765875031528974
# [0, 1, 2, 3, 4, 6, 9] 0
# [0, 1, 2, 3, 4, 6, 10] 0.0170013721316332
# [0, 1, 2, 3, 4, 6, 11] -0.00107702508992040
# [0, 1, 2, 3, 4, 7, 8] 0.00765875031528974
# [0, 1, 2, 3, 4, 7, 9] 0.0170013721316332
# [0, 1, 2, 3, 4, 7, 10] 0
# [0, 1, 2, 3, 4, 7, 11] -0.00107702508992040
# ......


Delta1power4 =  Polyhedron(vertices=[[-1], [1]]) *  Polyhedron(vertices=[[-1], [1]]) * Polyhedron(vertices=[[-1], [1]]) *  Polyhedron(vertices=[[-1], [1]])

Delta2power3 =  Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])*Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])*Polyhedron(vertices=[[1, 0], [0, 1], [0, 0]])
