def solid_angle(v, eps=1e-6, deg=100, normalized=True, tridiag=True):
    r"""
    stops adding if total deg > given deg or if abs(sum of terms of current deg)<eps.

    sage: v = matrix([[1,0],[0,1]])
    sage: solid_angle(v)
    1/4

    sage: v = matrix([[1,0],[1,1]])
    sage: solid_angle(v, eps=1e-15, deg=100)
    0.125000000000000

    sage: v = matrix([[1,0],[-1,1.732]])
    sage: solid_angle(v)
    0.333334416287660

    sage: v = matrix([[1,0],[0,1],[-1,0]])
    sage: solid_angle(v)
    1/2

    sage: v = matrix([[1,0,0],[0,1,0],[0,0,1]])
    sage: solid_angle(v)
    1/8

    sage: v = matrix([[1,0],[0,1],[-1,-1]])
    sage: solid_angle(v)
    1

    sage: v = matrix([[1,0,0],[0,1,0],[-1,-1,0],[0,0,1]])
    sage: solid_angle(v)
    1/2

    sage: v = matrix([[1,0,1],[0,1,1],[0,1,0],[1,0,0]])
    sage: solid_angle(v)
    0.0979565682372290

    # divergent!
    # It is expected to 0.0270384862516054 or 0.0270433270209179
    # which is 1/8 - 0.0979615137483946 (see above)
    # or  0.0540866540418358 / 2 (see below)
    # old code without decomposition shows
    #Warning: the matrix M is not positive definite. Uncertain about the convergence of the series.
    #1752.2874627804
    # NOW with decomposition, it can compute the solid angle.
    sage: v = matrix([[1,0,1],[0,1,1],[0,0,1]])
    sage: solid_angle(v)
    0.0270413925010829

    sage: v = matrix([[1,0,1],[0,1,1],[0,-1,1]])
    sage: solid_angle(v)
    0.0540866540418358

    sage: v = matrix([[1,0,1],[0,1,1],[-1,0,1],[0,-1,1]])
    sage: solid_angle(v)
    0.108173308083672

    # another divergent example
    # It is expected to be 0.222946054977479
    # which is 1/8 + 0.0979460549774794
    # old code without decomposition shows
    #Warning: the matrix M is not positive definite. Uncertain about the convergence of the series.
    #0.222953630900661
    sage: v = matrix([[1,0,1],[0,1,1],[0,0,-1]])
    sage: solid_angle(v)
    0.222955514064721

    # Example 3.4 from 2010-Gourion-Seeger-Deterministic and stochastic methods
    sage: v = matrix([[1,-1,-1,1],[5,1,7,5],[-4,4,1,4],[-4,-5,8,4]])
    sage: solid_angle(v, tridiag=True) #long
    0.0404254670338585
    sage: solid_angle(v, deg=20, tridiag=False)
    0.0404388468353361  # = 0.0809/2
    # Notice that smallest_eigenvalue_of_M(v) is 0.252559457921473, and 
    # min(smallest_eigenvalue_of_M(vi) for (vi,si) in generate_cones_decomposition(v))
    # is only 0.0214062627868244. 
    # If decompose even when M is positive definite, then have slow convergence.
    # sage: solid_angle_simplicial_cone(v,deg=10)
    # 0.0202082251727200
    # sage: solid_angle_simplicial_cone(v,deg=20)
    # 0.0324844036921302
    # sage: solid_angle_simplicial_cone(v,deg=30)
    # 0.0370565170573890
    # sage: solid_angle_simplicial_cone(v,deg=40, tridiag=True)
    # 0.0389435025144103
    # with option tridiag=True, we can afford to compute more terms (default deg=100) in the series.


    # Example 4.2 from 2010-Gourion-Seeger-Deterministic and stochastic methods
    sage: u = 0.01
    sage: v = matrix([[1,1,0],[1,0,1],[u,sqrt((1-u^2)/2),sqrt((1-u^2)/2)]])
    # Although M is not pos def, we can compute the solid angle using series,
    sage: solid_angle(v, deg=20)
    0.0433399927117563
    # and the result is quite good, compared to the exact 3d formula.
    sage: solid_angle_3d(v)
    0.0433406323314552
    # the relative error is 0.0000147579687810834.
    # Now let u=-0.01, M is pos def, but the smallest eigenvalue of M is 0.095,
    # The paper claims that convergence is slow.
    sage: u = -0.01
    sage: v = matrix([[1,1,0],[1,0,1],[u,sqrt((1-u^2)/2),sqrt((1-u^2)/2)]])
    sage: solid_angle_3d(v)
    0.0444016967251110
    sage: solid_angle(v)
    0.0443997230025628
    # record the relative error 
    # (don't decompose when M is pos def tridiag=False v.s. decompose anyway tridiag=True).
    # we see that decomposition has better convergence.
    # this make sense because M associated with subcones has smallest eigenvalues 0.28778 and 0.29789
    # deg, abs(solid_angle(c, deg=deg)- solid_angle_3d(c))/solid_angle_3d(c) with tridiag=True or False
    # 10 0.0118547949479619    0.00293322605736128
    # 11 0.0104410758149342    0.00212645629649379
    # 12 0.00928992971455234   0.00136652243617361
    # 13 0.00833874184016855   0.00100936839215565
    # 14 0.00754235951792652   0.000612062229270231
    # 15 0.00686769940732550   0.000497752771936430
    # 16 0.00629013091949063   0.000263874393781494
    # 17 0.00579103067077005   0.000260028918173213
    # 18 0.00535610952452586   0.000101106252088077
    # 19 0.00497425277974064   0.000148298732580314
    # 20 0.00463670466415151   0.0000242357298114218
    # 21 0.00433648665755115   0.0000953036286059360
    # 22 0.00406797671581536   0.0000123666635513566
    # 23 0.00382660072125993   0.0000699813159216678
    # 24 0.00360860327543324   0.0000299115009007078
    # 25 0.00341087533400895   0.0000578085946092526
    # 26 0.00323082308765735   0.0000383675168851596
    # 27 0.00306626713966474   0.0000519277823539351
    # 28 0.00291536419644375   0.0000424616428966114
    # 29 0.00277654567074576   0.0000490747870307488
    # 30 0.00264846912146094   0.0000444515118526205 
    """
    ans = 0
    for subcone_matrix in simplicial_subcones_matrices(v):
        ans += solid_angle_simplicial_cone(subcone_matrix, eps, deg, normalized,tridiag)
    return ans

def simplicial_subcones_matrices(v):
    if v.rank() == v.nrows():
        return [v]
    else:
        from sage.geometry.triangulation.point_configuration \
            import PointConfiguration
        origin = v.nrows()
        pc = PointConfiguration(v.stack(vector([0]*v.ncols())), star=origin)
        triangulation = pc.triangulate()
        matrices = []
        for simplex in triangulation:
            matrices.append(matrix(v[i] for i in simplex if i!=origin))
        return matrices

def solid_angle_simplicial_cone(v, eps=1e-6, deg=100, normalized=True,tridiag=True):
    # check condition for convergence
    # The following M is not normalized to have 1 on the diagonal
    M = v * v.transpose()
    d = v.nrows()
    for i in range(d):
        for j in range(d):
            if i != j:
                M[i,j] = - abs(M[i,j])
    if (not tridiag) and M.is_positive_definite():
        return product(solid_angle_by_convergent_series(orth_m, eps, deg, normalized,tridiag)
                       for orth_m in generate_orthogonal_parts(v))
        #return solid_angle_by_convergent_series(v, eps, deg, normalized,tridiag)
        #print "The matrix M is positive definite. Convergent series."
    # else:
    #     #print M
    #     print "Warning: the matrix M is not positive definite. Uncertain about the convergence of the series."
    answer = 1
    for orth_m in generate_orthogonal_parts(v):    
        angle = 0
        for (vi, si) in generate_cones_decomposition(orth_m, h=None, w=None, s=1, tridiag=tridiag):
            anglei = si
            for  orth_vi in generate_orthogonal_parts(vi):
                anglei *= solid_angle_by_convergent_series(orth_vi, eps, deg, normalized,tridiag)
            #anglei = solid_angle_by_convergent_series(vi, eps, deg, normalized,tridiag)
            angle += anglei
        answer = answer * angle
    return answer

def solid_angle_by_convergent_series(v, eps=1e-6, deg=100, normalized=True, tridiag=False):
    r"""
    # Example 3.4 from 2010-Gourion-Seeger-Deterministic and stochastic methods
    sage: v = matrix([[1,-1,-1,1],[5,1,7,5],[-4,4,1,4],[-4,-5,8,4]])
    sage: solid_angle_by_convergent_series(v, deg=10)
    0.0404638509737549  # = 0.0809/2

    If tridiag=True, we assume v*v.transpose is a symmetric tridiagonal matrix.
    """
    if (v.base_ring() == SR) or (v.base_ring() == AA):
        v = matrix(RR, v)
    d = v.nrows()
    if d == 1:
        if normalized:
            return 1/2
        else:
            return 1
    da = int(d*(d-1)/2)
    vnorm = [v[i].norm().n() for i in range(d)]
    if normalized:
        # we don't consider \Omega_d, then
        const = sqrt((v.n()*v.transpose()).determinant()) / prod(vnorm) / ((4*pi.n()) ** (d/2))
    else:
        # we consider the solid angle of sphere in the factor, then
        const = sqrt((v.n()*v.transpose()).determinant()) / prod(vnorm) / (gamma(0.5*d) * (2**(d-1)))
    if tridiag:
        beta = [v[i] * v[i+1] / (vnorm[i]*vnorm[i+1]) for i in range(d-1)]
        partial_sum = 0
        for n in range(deg+1):
            sum_deg_n = 0
            for b in composition_of_n_into_k_parts(n, d-1):
                betatob = prod([beta[k]**b[k] for k in range(d-1)])
                coef = (-2)**(sum(b)) / prod([factorial(b[k]) for k in range(d-1)])
                coef *= gamma(0.5*(b[0]+1))
                for i in range(d-2):
                    coef *= gamma(0.5*(b[i]+b[i+1]+1))
                coef *= gamma(0.5*(b[d-2]+1))
                sum_deg_n  += coef * betatob
            partial_sum += sum_deg_n
            if abs(const * sum_deg_n) < eps:
                break
    else:
        alpha = [0]*da
        for i in range(d-1):
            for j in range(i+1, d):
                k = (2*d-i-1)*i/2 + j-i-1
                alpha[k] = v[i] * v[j] / (vnorm[i]*vnorm[j])
        partial_sum = 0
        for n in range(deg+1):
            sum_deg_n = 0
            for a in composition_of_n_into_k_parts(n, da):
                alphatoa = 1
                for k in range(da):
                    alphatoa *= alpha[k]**a[k]
                    if alphatoa == 0:
                        break
                if alphatoa == 0:
                    continue
                coef = (-2)**(sum(a)) / prod([factorial(a[k]) for k in range(da)])
                for i in range(d):
                    s = 1
                    for j in range(i):
                        k = (2*d-j-1)*j/2+i-j-1
                        s += a[k]
                    for j in range(i+1,d):
                        k = (2*d-i-1)*i/2 + j-i-1
                        s += a[k]
                    coef = coef * gamma(0.5*s)
                sum_deg_n  += coef * alphatoa
            partial_sum += sum_deg_n
            if abs(const * sum_deg_n) < eps:
                break
     # sum_deg_n has alternate signs, return the mean of the partial sums of deg n and n-1. 
    return const * (partial_sum - sum_deg_n/2)

def composition_of_n_into_k_parts(n, k):
    if k == 1:
        yield [n]
    elif n == 0:
        yield [0] * k
    else:
        for i in range(n+1):
            for c in composition_of_n_into_k_parts(n-i, k-1):
                yield [i]+c

def check_sign_consistency(v):
    r"""
    input matrix v, whose rows are v[0], ..., v[n-1].
    Mn(v)_{i,j} =  -|vi \cdot vj| for i \neq j, and = vi\cdot vj if i=j.
    Check if Mn(v) be written as (v')^T * (v'),
    with v'[i] = + v'[i] or - v'[i].

    # Example 3.4 from 2010-Gourion-Seeger-Deterministic and stochastic methods
    sage: v = matrix([[1,-1,-1,1],[5,1,7,5],[-4,4,1,4],[-4,-5,8,4]])
    sage: check_sign_consistency(v)
    False
    """
    n = v.nrows()
    s = [0]*n
    for i in range(n):
        if s[i] == 0:
            s[i] = 1
        for j in range(i+1, n):
            if v[i] * v[j] < 0:
                if s[j] == -s[i]:
                    return False
                s[j] = s[i]
            elif v[i] * v[j] > 0:
                if s[j] == s[i]:
                    return False
                s[j] = -s[i]
    return True
        
def generate_cones_decomposition(v, h=None, w=None, s=1, tridiag=True):
    r"""
    # Example 3.4 from 2010-Gourion-Seeger-Deterministic and stochastic methods
    sage: v = matrix([[1,-1,-1,1],[5,1,7,5],[-4,4,1,4],[-4,-5,8,4]])
    sage: list(generate_cones_decomposition(v, tridiag=False))
     [(
    [      1      -1      -1       1]   
    [      5       1       7       5]   
    [   17/2    13/2    37/2    33/2]   
    [-265/87 -740/87  370/87  -35/29], 1
    ),
     (
    [      1      -1      -1       1]    
    [      5       1       7       5]    
    [    7/2    -7/2    37/2    23/2]    
    [-265/67 -740/67  370/67 -105/67], -1
    ),
     (
    [    1    -1    -1     1]    
    [   -4     4     1     4]    
    [ 17/5  13/5  37/5  33/5]    
    [  8/5  37/5 -37/5  -8/5], -1
    ),
     (
    [     1     -1     -1      1]   
    [    -4     -5      8      4]   
    [   7/3   -7/3   37/3   23/3]   
    [465/79 720/79 370/79 625/79], 1
    ),
     (
    [      1      -1      -1       1]   
    [     -4      -5       8       4]   
    [    8/3    37/3   -37/3    -8/3]   
    [465/109 720/109 370/109 625/109], 1
    )]
    sage: list(generate_cones_decomposition(v, tridiag=True))
    [(
    [      1      -1      -1       1]   
    [      5       1       7       5]   
    [   17/2    13/2    37/2    33/2]   
    [-265/87 -740/87  370/87  -35/29], 1
    ),
     (
    [      1      -1      -1       1]    
    [      5       1       7       5]    
    [    7/2    -7/2    37/2    23/2]    
    [-265/67 -740/67  370/67 -105/67], -1
    ),
     (
    [        1        -1        -1         1]    
    [       -4         4         1         4]    
    [     17/5      13/5      37/5      33/5]    
    [      5/9  1010/153 -1480/153   -185/51], -1
    ),
     (
    [       1       -1       -1        1]   
    [      -4        4        1        4]   
    [     8/5     37/5    -37/5     -8/5]   
    [   85/47  1010/47 -1480/47  -555/47], 1
    ),
     (
    [     1     -1     -1      1]   
    [    -4     -5      8      4]   
    [   7/3   -7/3   37/3   23/3]   
    [465/79 720/79 370/79 625/79], 1
    ),
     (
    [      1      -1      -1       1]   
    [     -4      -5       8       4]   
    [    8/3    37/3   -37/3    -8/3]   
    [465/109 720/109 370/109 625/109], 1
    )]
    """
    #print v, h, w, s
    #if v.nrows() == 2: ? if check_sign_consistency(v)): ?
    if (v.nrows() <= 2) or ((not tridiag) and (check_sign_consistency(v))):
        if w is None:
            yield((v, s))
        else:
            yield((w.stack(v), s))
    else:
        n = v.nrows()
        if h is None:
            max_num_orth = -1
            for i in range(n):
                num_orth = [v[i]*v[j] for j in range(n)].count(0)
                if num_orth > max_num_orth:
                    max_num_orth = num_orth
                    h = i
        if w is None:
            ww = matrix(v[h])
        else:
            ww = w.stack(v[h])
        num_orth = [v[h]*v[j] for j in range(n)].count(0)
        if num_orth == n-1:
            u = v.delete_rows([h])
            for vs in generate_cones_decomposition(u, h=None, w=ww, s=s, tridiag=tridiag):
                yield vs
        else:
            for i in range(n):
                if (i == h) or (v[i]*v[h]==0):
                    continue
                u = matrix(v[i])
                if v[i]*v[h] > 0:
                    si = s
                    for j in range(i):
                        if (j != h) and (v[j]*v[h]>0):
                            si = -si
                    for k in range(n):
                        if (k == h) or (k == i):
                            continue
                        if (k < i) and (v[k]*v[h]>0):
                            eik = -1
                        else:
                            eik = 1
                        projvk = v[k]-(v[k]*v[h])/(v[i]*v[h]) * v[i]
                        u = u.stack(eik * projvk)
                    for vs in generate_cones_decomposition(u, h=0, w=ww, s=si, tridiag=tridiag):
                        yield vs
                elif v[i]*v[h] < 0:
                    si = s
                    for j in range(i+1, n):
                        if (j != h) and (v[j]*v[h]<0):
                            si = -si
                    for k in range(n):
                        if (k == h) or (k == i):
                            continue
                        if (k > i) and (v[k]*v[h]<0):
                            eik = -1
                        else:
                            eik = 1
                        projvk = v[k]-(v[k]*v[h])/(v[i]*v[h]) * v[i]
                        u = u.stack(eik * projvk)
                    for vs in generate_cones_decomposition(u, h=0, w=ww, s=si, tridiag=tridiag):
                        yield vs


# [Cone([ 1 -1 -1  1],     [Cone([ 1 -1 -1  1],       [Cone([ 1 -1  -1  1],     [Cone([ 1 -1  -1  1],
#       [ 5  1  7  5],  =        [ 5  1  7  5],   -         [-4  4   1  4],  +        [-4 -5   8  4],  
#       [-4  4  1  4],           [17 13 37 33],             [17 13  37 33],           [ 7 -7  37 23],
#       [-4 -5  8  4])]          [ 7 -7 37 23])]            [ 8 37 -37 -8])]          [ 8 37 -37 -8])]

# c0 = Cone([[1,-1,-1,1],[ 5,1,7,5],[17,13,37,33],[ 7,-7,37,23]]); s0=1;
# c1 = Cone([[1,-1,-1,1],[-4,4,1,4],[17,13,37,33],[ 8,37,-37,-8]]); s1=-1;
# c2 = Cone([[1,-1,-1,1],[-4,-5,8,4],[ 7,-7,37,23],[ 8,37,-37,-8]]); s2=1;


def solid_angle_3d(v, normalized=True):
    r"""
    sage: v = matrix([[1,0,1],[0,1,1],[0,0,1]])
    sage: solid_angle_3d(v)
    0.0270433619923482
    sage: v = matrix([[1,0,1],[0,1,1],[0,0,-1]])
    sage: solid_angle_3d(v)
    0.222956638007652
    """
    assert(v.nrows() == 3)
    vnorm = [v[i].norm().n() for i in range(3)]
    angle = 2 * atan2(abs(v.determinant()), vnorm[0]*vnorm[1]*vnorm[2]+(v[0]*v[1])*vnorm[2]+(v[0]*v[2])*vnorm[1]+(v[1]*v[2])*vnorm[0])   # atan2(y, x) in [-pi, pi] 
    if normalized:
        return (angle/(4*pi)).n()
    else:
        return (angle).n()

def generate_orthogonal_parts(v):
    r"""
    sage: v = matrix([[1,0,0],[0,1,0],[0,0,1]])
    sage: list(generate_orthogonal_parts(v))
    [[1 0 0], [0 1 0], [0 0 1]]
    sage: v = matrix([[1,1,0],[0,1,0],[0,0,1]])
    sage: list(generate_orthogonal_parts(v))
    [
    [1 1 0]         
    [0 1 0], [0 0 1]
    ]"""
    n = v.nrows()
    k = 0
    u_indice_list = [0]
    w_indice_set = set(range(1,n))
    while k < len(u_indice_list):
        i = u_indice_list[k]
        for j in range(n):
            if (j in w_indice_set) and (v[i]*v[j] != 0):
                u_indice_list.append(j)
                w_indice_set.remove(j)
        k += 1
    u = v.delete_rows(w_indice_set)
    yield u
    if w_indice_set:
        w = v.matrix_from_rows(w_indice_set)
        for result in generate_orthogonal_parts(w):
            yield result


def smallest_eigenvalue_of_M(v):
    r"""
    sage: u = 0.01
    sage: v = matrix([[1,1,0],[1,0,1],[u,sqrt((1-u^2)/2),sqrt((1-u^2)/2)]])
    sage: smallest_eigenvalue_of_M(v)
    -0.00940202034184155
    sage: u = -0.01
    sage: v = matrix([[1,1,0],[1,0,1],[u,sqrt((1-u^2)/2),sqrt((1-u^2)/2)]])
    sage: smallest_eigenvalue_of_M(v)
    0.00945386943099301
    """
    d = v.nrows()
    ## Exact symbolic computation
    # M = matrix(SR, d)
    # diag = [sqrt(v[i]*v[i]) for i in range(d)]
    # for i in range(d):
    #     for j in range(d):
    #         if i <> j:
    #             M[i,j] = - abs(v[i]*v[j])/(diag[i] * diag[j])
    #         else:
    #             M[i,j] = 1
    M = v.n()*v.transpose()
    diag = [sqrt(x) for x in M.diagonal()]
    for i in range(d):
        for j in range(d):
            if i != j:
                M[i,j] = - abs(M[i,j])/ (diag[i] * diag[j])
            else:
                M[i,j] = 1
    return min(M.eigenvalues())