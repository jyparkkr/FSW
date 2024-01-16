def solid2(A):
    r"""
    Return the normalized solid angle measure of the solid angle spanned by the
    vectors given by the two rows of A.

    INPUT:

    - ``A`` -- 2x2 matrix; A is a 2x2 matrix which should be input as
    A=matrix([[a,b],[c,d]]) where [a,b] and [c,d] represent the two extreme
    rays/vectors of the cone in R^2. The vectors should be nonzero.

    OUTPUT: the solid angle spanned by the two vectors, as a decimal

    EXAMPLES:

    This example shows the solid angle spanned by the vectors [0,1] and [1,0]::

        sage: solid2(A=matrix([[0,1],[1,0]]))
        0.250000000000000

    We now show the solid angle spanned by the vectors [1,0], [-1, sqrt(3)]::

        sage: solid2(A= matrix([[1,0],[-1,sqrt(3)]]))
        0.333333333333333

    This example illustrates how the solid angle measure will not greater than
    0.5 as the function always outputs the minimal angle between the two rays::

        sage: solid2(A= matrix([[1,0],[-1,-1]]))
        0.375000000000000


    .. NOTE::

        This function uses the dot product of two vectors to determine the
        angle between them.

    It is an error to input the vectors directly, instead of a matrix::

        sage: solid2([1,0],[0,1])
        Traceback (most recent call last):
        ...
        TypeError: solid2() takes 1 positional argument but 2 were given


    The following tests check for corner cases where the vectors are
    antiparallel, parallel and perpendicular, respectively.

        sage: solid2(A=matrix([[1,1],[-1,-1]]))
        0.500000000000000

        sage: solid2(A=matrix([[1,2],[2,4]]))
        0.000000000000000

        sage: solid2(A=matrix([[2,2],[-1,1]]))
        0.250000000000000
    """
    u = A.row(0)
    v = A.row(1)
    p = u.dot_product(v)
    a = u.norm()
    b = v.norm()
    cs = p/(a*b)
    final_calc = arccos(cs) / (2*pi)
    return final_calc.numerical_approx()


def solid_angle3(v): #arctan method
    r"""
    Return the normalized solid angle measure of the solid angle spanned by
    three vectors given by the rows of v.

    INPUT:

    - ``v`` -- 3x3 matrix; v is a 3x3 matrix which should be input as
      A=matrix([[a,b],[c,d],[e,f]]) where [a,b], [c,d], and [e,f] represent
      the three extreme rays/vectors of the cone in R^3. Any two vectors should
      not be scalar multiples of each other.

    OUTPUT:

    - the normalized solid angle measure spanned by the three vectors,
      as a decimal

    EXAMPLES:

    This example shows the measure of the solid angle spanned by the vectors
    [1,0,0],[0,1,0], and [0,0,1]::

        sage: solid_angle3(v=matrix([[1,0,0],[0,1,0],[0,0,1]]))
        0.125000000000000

    This example shows the solid angle spanned by a set of linearly
    dependent vectors [2,0,0], [0,3,0] and [-4,-4,0]::

        sage: solid_angle3(v=matrix([[2,0,0],[0,3,0],[-4,-4,0]]))
        0.500000000000000

    It is an error to input a matrix A where one row is a multiple of another::

        sage: solid_angle3(v=matrix([[-1,0,1],[3,0,0],[-1,0,0]]))
        NaN

    It is an error to input vectors from R^2 into this function::

        sage: solid_angle3(v=matrix([[1,0],[3,4],[-1,2]]))
        Traceback (most recent call last):
        ...
        ValueError: self must be a square matrix

    .. NOTE::

        This function uses the formula given in Ribando's
        2006 paper entitled "Measuring Solid Angles Beyond
        Dimension Three." Refer to Oosterom and Strackee (1983)
        for more information.

    Check corner case vectors mutually orthogonal::

        sage: solid_angle3(v=matrix([[0,0,3],[-1,-1,0],[-2,2,0]]))
        0.125000000000000
    """
    vnorm = [v[i].norm().n() for i in range(3)]
    a = v[0]/vnorm[0]
    b = v[1]/vnorm[1]
    c = v[2]/vnorm[2]
    w = matrix([a, b, c])
    det = abs(w.determinant())  # same as det of matrix [abc]
    dab = a.dot_product(b)
    dac = a.dot_product(c)
    dbc = b.dot_product(c)
    denom = 1+dab+dac+dbc
    omega = 2*atan2(det, denom)
    return (omega/(4*pi)).n()


def solid3(A): #arccos method
    r"""
    Return the normalized solid angle measure of the solid angle spanned by
    three vectors given by the rows of A.

    INPUT:

    - ``A`` -- 3x3 matrix; A is a 3x3 matrix which should be input as
      A=matrix([[a,b],[c,d],[e,f]]) where [a,b], [c,d], and [e,f] represent
      the three extreme rays/vectors of the cone in R^3. Any two vectors should
      not be scalar multiples of each other.

    OUTPUT:

    - the normalized solid angle measure spanned by the three vectors,
      as a decimal

    EXAMPLES:

    This example shows the measure of the solid angle spanned by the vectors
    [1,0,0],[0,1,0], and [0,0,1]::

        sage: solid3(A=matrix([[1,0,0],[0,1,0],[0,0,1]]))
        0.125000000000000

    This example shows the solid angle spanned by a set of linearly
    dependent vectors [2,0,0], [0,3,0] and [-4,-4,0]::

        sage: solid3(A=matrix([[2,0,0],[0,3,0],[-4,-4,0]]))
        0.500000000000000

    It is an error to input a matrix A where A^t is not full rank
    (i.e it is an error for the vectors to be linearly dependent)::

        sage: solid3(A=matrix([[-1,0,1],[3,0,0],[-1,0,0]]))
        NaN

    It is an error to input vectors from R^2 into this function::

        sage: solid3(A=matrix([[1,0],[3,4],[-1,2]]))
        Traceback (most recent call last):
        ...
        TypeError: Cross product only defined for vectors of length three
        or seven, not (2 and 2)


    .. NOTE::

        This function uses the formula given in Proposition 6 of
        Beck et. al.'s 2015 paper entitled "Positivity Theorems
        for Solid-Angle Polynomials."

    Check corner case vectors mutually orthogonal::

        sage: solid3(A=matrix([[0,0,3],[-1,-1,0],[-2,2,0]]))
        0.125000000000000
    """
    v_0 = A.row(0)
    v_1 = A.row(1)
    v_2 = A.row(2)
    c_01 = v_0.cross_product(v_1)
    c_02 = v_0.cross_product(v_2)
    c_12 = v_1.cross_product(v_2)
    n_01 = c_01.norm().n()
    n_02 = c_02.norm().n()
    n_12 = c_12.norm().n()
    d_0 = c_01.dot_product(c_02)
    d_1 = c_01.dot_product(c_12)
    d_2 = c_02.dot_product(c_12)
    a_0 = arccos(d_0/(n_01*n_02)).n()
    a_1 = arccos(-d_1/(n_01*n_12)).n()
    a_2 = arccos(d_2/(n_02*n_12)).n()
    sum = a_0+a_1+a_2
    denom = (4*pi).n()
    omega = (sum-pi)/denom
    return (omega).n()

def simplicial_subcones_solidangle2(v):
    r"""
    Return the normalized solid angle measure of the solid angle spanned
    by at least 2 vectors in R^2.

    INPUT:

    - ``v`` -- v is a matrix whose columns are the vectors spanning
    the cone in R^2. The matrix should be input as
    v = matrix([[a,b],[c,d], ... [e,f]]) where [a,b], [c,d],...,[e,f]
    are the rays that span the cone.

    OUTPUT: The normalized solid angle measure of each of the simplices
    produced via the triangulation of the cone, and the normalized solid
    angle spanned by the column vectors of the matrix v, as a decimal

    EXAMPLES:

    This example shows the solid angle spanned by the vectors [1,0],[0,1],
    and [-1,0] in R^2::

        sage: v = matrix([[1,0],[0,1],[-1,0]])
        sage: simplicial_subcones_solidangle2(v)
        [0.250000000000000, 0.250000000000000]
        0.500000000000000

    We now show the solid angle spanned by 4 vectors [1,1],[1,2],[-1,1],
    and [-3,0]::

        sage: v = matrix([[1,1],[1,2],[-1,1],[-3,0]])
        sage: simplicial_subcones_solidangle2(v)
        [0.0512081911747833, 0.198791808825217, 0.125000000000000]
        0.375000000000000

    This example illustrates how the solid angle measure can equal 1. That is,
    the span of the rays is all of space::

        sage: v = matrix([[1,1],[0,-1],[-1,-1],[-3,0]])
        sage: simplicial_subcones_solidangle2(v)
        [0.375000000000000, 0.375000000000000, 0.125000000000000,
        0.125000000000000]
        1.00000000000000

    This example shows that when two vectors are given that are not
    parallel, only the normalized measure of the solid angle is given
    as the cone the vectors span is already simplicial.

        sage: v = matrix([[2,3],[-3,-7]])
        sage: simplicial_subcones_solidangle2(v)
        0.470857008299079


    .. NOTE::

        This function is based on Dr. Yuan Zhou's code. It also uses the solid2
        function which is defined above.

        It is an error to input a 2 x 1 matrix, i.e. one vector::

        sage: v = matrix([[1,3]])
        sage: simplicial_subcones_solidangle2(v)
        Traceback (most recent call last):
        ...
        IndexError: row index out of range

    Check corner case where the input gives 2 vectors that are parallel

        sage: v = matrix([[1,0],[2,0]])
        sage: simplicial_subcones_solidangle2(v)
        0.000000000000000

        sage: solid2(A=matrix([[1,2],[2,4]]))
        0.000000000000000

        sage: solid2(A=matrix([[2,2],[-1,1]]))
        0.250000000000000
    """
    if v.nrows() == 2:
        return solid2(v)
    else:
        from sage.geometry.triangulation.point_configuration \
            import PointConfiguration
        origin = v.nrows()
        pc = PointConfiguration(v.stack(vector([0]*v.ncols())), star=origin)
        triangulation = pc.triangulate()
        matrices = []
        for simplex in triangulation:
            matrices.append(matrix(v[i] for i in simplex if i != origin))
        n = len(matrices)
        results = []
        for i in range(n):
            results.append(solid2(matrices[i]))
        print(results)
        return sum([results[k] for k in range(len(results))])

def simplicial_subcones_solidangle3(v):
    r"""
    Return the normalized solid angle measure of the solid angle spanned
    by at least 3 vectors in R^3.

    INPUT:

    - ``v`` -- v is a matrix whose columns are the vectors spanning
    the cone in R^3. The matrix should be input as
    v = matrix([[a,b,c],[d,e,f], ... [p,q,r]]) where [a,b,c],[d,e,f],
    ...[p,q,r] represent the rays of the cone.

    OUTPUT: The normalized solid angle measure of each of the simplices
    produced via the triangulation of the cone, and the normalized solid
    angle spanned by the column vectors of the matrix v, as a decimal

    EXAMPLES:

    This example shows the solid angle spanned by the vectors [[1,0,0],
    [-1,0,0], and [-1,3,1],[1,0,-1]::

        sage: v = matrix([[1,0,0],[-1,0,0],[-1,3,1],[1,0,-1]])
        sage: simplicial_subcones_solidangle3(v)
        [0.0920896306410492, 0.209118560533734]
        0.301208191174783

    We now show the solid angle spanned by the vectors [1,0,0],[0,1,0],
    [-1,0,0],[0,0,-1], and [1,1,1]::

        sage: v = matrix([[1,0,0],[0,1,0],[-1,0,0],[0,0,-1],[1,1,1]])
        sage: simplicial_subcones_solidangle3(v)
        [0.0833333333333334, 0.125000000000000, 0.0833333333333334,
         0.0833333333333334]
        0.375000000000000

    This example illustrates how the solid angle measure can equal 1. That is,
    the span of the rays is all of space::

        sage: v = matrix([[1,0,0],[0,1,0],[-1,0,0],[0,0,1],[0,0,-1],[0,-1,0]])
        sage: simplicial_subcones_solidangle3(v)
        [0.125000000000000, 0.125000000000000, 0.125000000000000,
        0.125000000000000, 0.125000000000000, 0.125000000000000,
        0.125000000000000, 0.125000000000000]
        1.00000000000000

    This example shows that when three vectors are given that are not
    parallel, only the normalized measure of the solid angle is given
    as the cone the vectors span is already simplicial.

        sage: v = matrix([[1,0,2],[-1,3,1],[1,0,-1]])
        sage: simplicial_subcones_solidangle3(v)
        0.181768746434821


    .. NOTE::

        This function is based on Dr. Yuan Zhou's code. It also uses the solid3
        function which is defined above.

        It is an error to input less than 3 vectors::

        sage: v = matrix([[1,0,0],[0,2,3]])
        sage: simplicial_subcones_solidangle3(v)
        Traceback (most recent call last):
        ...
        IndexError: row index out of range

        It is an error to input a matrix with rank less than 3::

        sage: v=matrix([[1,0,0],[2,0,0],[-1,1,0],[-2,2,0]])
        sage: simplicial_subcones_solidangle3(v)
        Traceback (most recent call last):
        ...
        IndexError: row index out of range
    """
    if v.nrows() == v.rank():
        return solid3(v)
    else:
        from sage.geometry.triangulation.point_configuration \
            import PointConfiguration
        origin = v.nrows()
        pc = PointConfiguration(v.stack(vector([0]*v.ncols())), star=origin)
        triangulation = pc.triangulate()
        matrices = []
        for simplex in triangulation:
            matrices.append(matrix(v[i] for i in simplex if i != origin))
        n = len(matrices)
        results = []
        for i in range(n):
            results.append(solid3(matrices[i]))
        print(results)
        return sum([results[k] for k in range(len(results))])
