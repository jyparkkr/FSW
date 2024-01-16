# load("~/ma611-code/logging.sage")
import os
load(os.path.join(os.getcwd(), "logging.sage"))


# **********************************************************
#         Formulas for 2d and 3d simplicial cones
# **********************************************************
def solid_angle_simplicial_2d(A):
    r"""
    Return the normalized solid angle measure of the cone spanned by the
    two row vectors of ``A``.

    INPUT:

    - ``A`` -- 2x2 matrix in the form of ``matrix([[a,b],[c,d]])`` or
      simply ``[[a,b],[c,d]]``, where the nonzero ``[a,b]`` and ``[c,d]``
      represent the two extreme rays/vectors of the cone in `\RR^2`.

    OUTPUT:

    - the normalized solid angle measure of the cone spanned by the
      two vectors, as a decimal

    EXAMPLES:

    This example shows the normalized measure of the solid angle spanned
    by the rows of the matrix::

        sage: solid_angle_simplicial_2d(matrix([[0,1],[1,0]])) # abs tol 1e-15
        0.25

    The input can be a list of vectors instead of a matrix as well::

        sage: solid_angle_simplicial_2d([[1,0], [-1,sqrt(3)]]) # abs tol 1e-15
        0.3333333333333333

    This example illustrates how the solid angle measure will not be
    greater than 0.5 as the function always outputs the minimal angle
    between the two rays::

        sage: solid_angle_simplicial_2d([[1,0],[-1,-1]])       # abs tol 1e-15
        0.375

    .. NOTE::

        This function uses the dot product of two vectors to determine the
        angle between them.

    The following tests check for corner cases where the vectors are
    antiparallel, parallel and perpendicular, respectively::

        sage: solid_angle_simplicial_2d([[1,1],[-1,-1]])       # abs tol 1e-15
        0.5

        sage: solid_angle_simplicial_2d([[1,2],[2,4]])         # abs tol 1e-15
        0.0

        sage: solid_angle_simplicial_2d([[2,2],[-1,1]])        # abs tol 1e-15
        0.25
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if A.nrows() != 2 or A.ncols() != 2:
        raise ValueError("input matrix has incorrect dimension.")
    u = A.row(0)
    v = A.row(1)
    p = u.dot_product(v)
    a = u.norm()
    b = v.norm()
    cs = p/(a*b)
    final_calc = arccos(cs) / (2*pi)
    return RDF(final_calc)


def solid_angle_simplicial_arccos_3d(A):
    r"""
    Return the normalized solid angle measure of the cone spanned by
    the three row vectors of ``A``.

    INPUT:

    - ``A`` -- 3x3 matrix or a list of three vectors in `\RR^3` where the row
      vectors represent the three extreme rays/vectors of the cone in `\RR^3`.
      Any two vectors should not be scalar multiples of each other.

    OUTPUT:

    - the normalized solid angle measure spanned by the three vectors,
      as a decimal

    EXAMPLES:

    This example shows the measure of the solid angle spanned by the vectors::

        sage: A = matrix([[1,0,0],[0,1,0],[0,0,1]])
        sage: solid_angle_simplicial_arccos_3d(A)              # abs tol 1e-15
        0.125

    The input can be a list of vectors instead of a matrix::

        sage: solid_angle_simplicial_arccos_3d(
        ....:         [[0,0,3],[-1,-1,0],[-2,2,0]])            # abs tol 1e-15
        0.125

    This example shows the solid angle of a cone in 3d with affine dimension 2.
    In contrast to ``solid_angle_3d``, this formula gives a non-zero angle::

        sage: A = matrix([[2,0,0],[0,3,0],[-4,-4,0]])
        sage: solid_angle_simplicial_arccos_3d(A)              # abs tol 1e-15
        0.5

    It is an error to input a matrix ``A``, which has two vectors
    that are scalar multiples of each other::

        sage: A = matrix([[-1,0,1],[3,0,0],[-1,0,0]])
        sage: solid_angle_simplicial_arccos_3d(A)
        Traceback (most recent call last):
        ...
        ZeroDivisionError: rational division by zero

    It is an error to input vectors from `\RR^2` into this function::

        sage: solid_angle_simplicial_arccos_3d(A=matrix([[1,0],[3,4],[-1,2]]))
        Traceback (most recent call last):
        ...
        ValueError: input matrix has incorrect dimension.

    .. NOTE::

        This function uses the formula given in Proposition 6 of
        Beck et. al.'s 2015 paper entitled "Positivity Theorems
        for Solid-Angle Polynomials."
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if A.nrows() != 3 or A.ncols() != 3:
        raise ValueError("input matrix has incorrect dimension.")
    v_0 = A.row(0)
    v_1 = A.row(1)
    v_2 = A.row(2)
    c_01 = v_0.cross_product(v_1)
    c_02 = v_0.cross_product(v_2)
    c_12 = v_1.cross_product(v_2)
    n_01 = c_01.norm()
    n_02 = c_02.norm()
    n_12 = c_12.norm()
    d_0 = c_01.dot_product(c_02)
    d_1 = c_01.dot_product(c_12)
    d_2 = c_02.dot_product(c_12)
    a_0 = arccos(d_0/(n_01*n_02))
    a_1 = arccos(-d_1/(n_01*n_12))
    a_2 = arccos(d_2/(n_02*n_12))
    return RDF((a_0+a_1+a_2 - pi)/(4*pi))


def solid_angle_simplicial_arctan_3d(A):
    r"""
    Return the normalized solid angle measure of the cone spanned by
    the three row vectors of ``A``.

    INPUT:

    - ``A`` -- 3x3 matrix or a list of three vectors in `\RR^3` where the row
      vectors represent the three extreme rays/vectors of the cone in `\RR^3`.
      Any two vectors should not be scalar multiples of each other.

    OUTPUT:

    - the normalized solid angle measure spanned by the three vectors,
      as a decimal

    EXAMPLES:

    This example shows the measure of the solid angle spanned by the vectors::

        sage: A = matrix([[1,0,0],[0,1,0],[0,0,1]])
        sage: solid_angle_simplicial_arctan_3d(A)              # abs tol 1e-15
        0.125

    The input can be a list of vectors instead of a matrix::

        sage: solid_angle_simplicial_arctan_3d(
        ....:         [[0,0,3],[-1,-1,0],[-2,2,0]])            # abs tol 1e-15
        0.125

    This example shows the solid angle of a cone in 3d with affine dimension 2.
    In contrast to ``solid_angle_3d``, this formula gives a non-zero angle::

        sage: A = matrix([[2,0,0],[0,3,0],[-4,-4,0]])
        sage: solid_angle_simplicial_arctan_3d(A)              # abs tol 1e-15
        0.5

    It is an error to input a matrix ``A``, which has two vectors
    that are scalar multiples of each other::

        sage: A = matrix([[-1,0,1],[3,0,0],[-1,0,0]])
        sage: solid_angle_simplicial_arctan_3d(A)
        NaN

    It is an error to input vectors from `\RR^2` into this function::

        sage: solid_angle_simplicial_arctan_3d(matrix([[1,0],[3,4],[-1,2]]))
        Traceback (most recent call last):
        ...
        ValueError: input matrix has incorrect dimension.

    .. NOTE::

        This function uses the formula given in Ribando's
        2006 paper entitled "Measuring Solid Angles Beyond
        Dimension Three." Refer to Oosterom and Strackee (1983)
        for more information.
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if A.nrows() != 3 or A.ncols() != 3:
        raise ValueError("input matrix has incorrect dimension.")
    vnorm = [A[i].norm() for i in range(3)]
    a = A[0]/vnorm[0]
    b = A[1]/vnorm[1]
    c = A[2]/vnorm[2]
    w = matrix([a, b, c])
    det = abs(w.determinant())  # same as det of matrix [abc]
    dab = a.dot_product(b)
    dac = a.dot_product(c)
    dbc = b.dot_product(c)
    denom = 1+dab+dac+dbc
    omega = 2*atan2(det, denom)
    return RDF(omega/(4*pi))


# **********************************************************
#        Main functions of solid angle in 2d and 3d
# **********************************************************
def solid_angle_2d(A):
    r"""
    Return the normalized solid angle measure of the cone spanned by two
    vectors in `\RR^2`.

    INPUT:

    - ``A`` -- `n\times 2` matrix whose rows vectors span the cone in `\RR^2`
      of which we look for the solid angle. The input can be in the form of a
      matrix or as a list of vectors in `\RR^2`.

    OUTPUT: The normalized solid angle measure spanned by the row vectors, as
    a decimal

    EXAMPLES:

    The following three examples show the solid angle measures of the cones
    spanned by the given two, three or four vectors in `\RR^2`, respectively::

        sage: logging.disable(logging.WARNING)
        sage: A = matrix([[2,3],[-3,-7]])
        sage: solid_angle_2d(A)                                # abs tol 1e-15
        0.4708570082990789

        sage: A = matrix([[1,0],[0,1],[-1,0]])
        sage: solid_angle_2d(A)                                # abs tol 1e-15
        0.5

        sage: A = matrix([[1,1],[1,2],[-1,1],[-3,0]])
        sage: solid_angle_2d(A)                                # abs tol 1e-15
        0.375


    This example illustrates how the solid angle measure can equal `1`.
    That is, the span of the rays is all of space::

        sage: A = matrix([[1,1],[0,-1],[-1,-1],[-3,0]])
        sage: solid_angle_2d(A)                                # abs tol 1e-15
        1.0

    Check examples where the where cones have affine dimension less than `2`::

        sage: A = matrix([[1,0],[2,0]])
        sage: solid_angle_2d(A)
        0
        sage: A = matrix([[1,2],[-2,-4]])
        sage: solid_angle_2d(A)
        0
        sage: A = matrix([[-2,5],[-4,10],[-1,5/2],[-2/5,1]])
        sage: solid_angle_2d(A)
        0
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if A.rank() < 2:
        logging.warning("cone not full-dimensional")
        return 0
    if A.nrows() == 2:
        return solid_angle_simplicial_2d(A)
    A_list = simplicial_subcones_decomposition(A)
    logging.info('Decompose into simplicial subcones\n' +
                 ',\n'.join('{}'.format(Ai) for Ai in A_list))
    results = [solid_angle_simplicial_2d(Ai) for Ai in A_list]
    logging.info("Solid angles of the subcones are %s" % results)
    return sum(results)


def solid_angle_3d(A, method="arctan"):
    r"""
    Return the normalized solid angle measure of the cone spanned
    by three vectors in `\RR^3`.

    INPUT:

    - ``A`` -- `n\times 3` matrix whose rows vectors span the cone in `\RR^3`
      of which we look for the solid angle. The input can be in the form of a
      matrix or as a list of vectors in `\RR^3`.

    - ``method`` -- (optional) Either ``arctan`` or ``arccos``.

    OUTPUT: The normalized solid angle measure of the cone spanned by the row
    vectors, as a decimal

    EXAMPLES:

    The following three examples show the solid angles spanned by the given
    three, four or five vectors in `\RR^3`, respectively::

        sage: logging.disable(logging.WARNING)
        sage: solid_angle_3d([[1,0,2],[-1,3,1],[1,0,-1]])      # abs tol 1e-15
        0.1817687464348209

        sage: A = matrix([[1,0,0],[-1,0,0],[-1,3,1],[1,0,-1]])
        sage: solid_angle_3d(A)                                # abs tol 1e-15
        0.30120819117478337

        sage: A = matrix([[1,0,0],[0,1,0],[-1,0,0],[0,0,-1],[1,1,1]])
        sage: solid_angle_3d(A)                                # abs tol 1e-15
        0.375

    This example illustrates how using the `arccos` method instead of the
    default `arctan` method gives the same result::

        sage: A = matrix([[1,0,0],[0,1,0],[-1,0,0],[0,0,-1],[1,1,1]])
        sage: solid_angle_3d(A, method="arccos")               # abs tol 1e-15
        0.375

    This example illustrates how the solid angle measure can equal 1. That is,
    the span of the rays is all of space::

        sage: A = matrix([[1,0,0],[0,1,0],[-1,0,0],[0,0,1],[0,0,-1],[0,-1,0]])
        sage: solid_angle_3d(A)                                # abs tol 1e-15
        1.0

    Check corner case where the where cones have affine dimension less than 3::

        sage: solid_angle_3d([[1,0,0],[2,0,0],[-1,1,0],[-2,2,0]])
        0
        sage: solid_angle_3d(matrix([[1,0,0],[0,2,3]]))
        0
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if A.rank() < 3:
        logging.warning("cone not full-dimensional")
        return 0
    if method == "arctan":
        solid_angle_function = solid_angle_simplicial_arctan_3d
    elif method == "arccos":
        solid_angle_function = solid_angle_simplicial_arccos_3d
    else:
        raise ValueError("method %s of solid_angle_3d is unknown" % method)
    if A.nrows() == 3:
        return solid_angle_function(A)
    A_list = simplicial_subcones_decomposition(A)
    logging.info('Decompose into simplicial subcones\n' +
                 ',\n'.join('{}'.format(Ai) for Ai in A_list))
    results = [solid_angle_function(Ai) for Ai in A_list]
    logging.info("Solid angles of the subcones are %s" % results)
    return sum(results)


# **********************************************************
#      Main function of solid angle in higher dimensions
# **********************************************************
def solid_angle_general(A, eps=1e-6, deg=100,
                        simplicial=None, space="ambient"):
    r"""
    Return an estimate of the normalized solid angle measure of the
    cone spanned by the row vectors of the given matrix ``A``,
    based on a truncated form of Jason Ribando's formula (see note).

    INPUT:

    - ``A`` -- a matrix or a list that is convertible to a matrix; the row
      vectors of ``A`` span the cone for which we compute its solid angle.

    - ``eps`` -- positive real number (default: ``1e-6``); this parameter
      is used to determine when the summation stops. In terms of the partial
      sum, when `s_n-s_{n-1} < \epsilon`, we stop adding terms to the partial
      sum sequence.

    - ``deg`` -- integer (default: `100`); ``deg`` is the maximum sum of the
      powers of the `\alpha_{ij}`'s in the summation (i.e. it is the maximum
      sum of the terms in the multiexponent.)

    - ``simplicial`` -- ``None`` (by default), or a Boolean. You can provide
      ``simplicial=True`` to skip some checks if the row vectors of ``A`` are
      known to represent the extreme rays of a simplicial cone.

    - ``space`` -- either "ambient" (by default) or "affine", indicating with
      respect to which space the solid angle of the cone is considered.

    OUTPUT:

    - an estimate of the normalized solid angle measure spanned by the row
      vectors given in ``A``.

    EXAMPLES:

    This example shows the measure of the solid angle spanned by the vectors
    ``[1,0]`` and ``[-1,-1]``. Note that it agrees with the value obtained by
    the arctan formula.::

        sage: logging.disable(logging.INFO)
        sage: A = matrix([[1,0],[-1,-1]])
        sage: solid_angle_general(A, eps=1e-9, simplicial=True) # abs tol 2e-9
        0.375

    This example shows that when the vectors are linearly dependent,
    the measure of the solid angle with respect to the ambient space is 0::

        sage: A = matrix([[2,0,0], [0,3,0], [-4,-4,0]])
        sage: solid_angle_general(A, space="ambient")
        WARNING: cone not full-dimensional
        0

    In contrast, when considered in the affine space, the solid angle is 1::

        sage: solid_angle_general(A, eps=1e-16, space="affine") # abs tol 1e-15
        1

    This example shows the measure of the solid angle spanned by
    the vectors ``[2, sqrt(2), 3], [-1, 1, 2]``, and ``[-3, 0, 5/4]``, with
    ``deg`` set to ``20`` and ``eps`` set to ``1e-6``. The relative error
    compared to value ``0.01183`` obtained by the arctan formula is <0.5%.::

        sage: A = matrix([[2, sqrt(2), 3], [-1, 1, 2], [-3, 0, 5/4]])
        sage: a = solid_angle_general(A, deg=20, eps=1e-6)
        sage: b = solid_angle_3d(A)
        sage: abs(a-b)/b < 0.005
        True

    This example shows an estimation of the measure of the solid angle
    spanned by vectors `\RR^5`, with different ``deg`` values.::

        sage: A = [[1,1,0,0,0],[-1,3,0,-4,1],[5,0,0,-1,0],
        ....:            [0,0,-2,1,4],[0,0,0,0,1]]
        sage: solid_angle_general(A, deg=10)                   # abs tol 1e-15
        0.005330879073359687

        sage: solid_angle_general(A, deg=12) # long time (18 s), abs tol 1e-15
        0.004870472360500353

    This example demonstrates that the method works even when the input
    is a matrix that does not correspond to a simplicial cone. The expected
    result based on the ``solid_angle_3d`` function) is ``0.3012081...``::

        sage: A = matrix([[1,0,0],[-1,0,0],[-1,3,1],[1,0,-1]])
        sage: solid_angle_general(A)                           # abs tol 1e-15
        0.3012056062147818

    This example illustrates that when the input matrix has an
    associated matrix that is not positive definite, a warning appears::

        sage: A = matrix([[1,2,-1,2,0],[-3,0,0,0,2],[1,-2,-2/5,0,0],
        ....:            [0,0,0,-2,1],[-1,-1,0,-1,0]])
        sage: solid_angle_general(A, deg=5)                    # abs tol 1e-15
        WARNING: Associated matrix NOT positive definite, series NOT converge
        0.027044019290803845

    TESTS:

    The example below is based on Example 3.4 in Gourion and Seeger (see
    notes). For the matrix ``A`` below, the authors used truncated forms
    of Ribando's formula, testing deg = 0,1,2,5,10,20, and 40.
    The estimates they obtained were 0.097403, 0.067204, 0.082871, 0.079939,
    0.080930, 0.080878, and 0.080878 respectively. The authors normalized
    their measurement with respect to a half space. Thus, the function should
    return estimates that are half of the above values. Below, we show that
    this is the case. We observe that the last two returns are equal, showing
    that eps=1e-6 is too large when deg=40.::

        sage: A = matrix([[1/2, -1/2, -1/2, 1/2],[1/2, 1/10, 7/10, 1/2],
        ....:     [-4/7, 4/7, 1/7, 4/7], [-4/11, -5/11, 8/11, 4/11]])
        sage: solid_angle_general(A, deg=0)                    # abs tol 1e-15
        0.04870129870129871

        sage: solid_angle_general(A, deg=1)                    # abs tol 1e-15
        0.03360184592862353

        sage: solid_angle_general(A, deg=2)                    # abs tol 1e-15
        0.04319218542971285

        sage: solid_angle_general(A, deg=5)                    # abs tol 1e-15
        0.03996966211891789

        sage: solid_angle_general(A, deg=10)                   # abs tol 1e-15
        0.0404638509737549

        sage: solid_angle_general(A, deg=20)  # long time (28s), abs tol 1e-15
        0.04043924941007705

        sage: solid_angle_general(A, deg=40)  # long time (28s), abs tol 1e-15
        0.04043924941007705

    .. NOTE::

        This function uses the formula given in Ribando's
        2006 paper entitled "Measuring Solid Angles Beyond
        Dimension Three." More specifically, it is a truncated
        form of the multi-variate power series given in Theorem
        2.2.

        In Gourion and Seeger's 2010 paper entitled "Deterministic
        and stochastic methods for computing volumetric moduli of
        convex cones," the authors look at the volumetric modulus/
        normalized volume of convex polyhedral cones, in comparison
        to a half space. See Theorem 4.1 and Remark 4.2.
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    if space == "ambient" and A.rank() < A.ncols():
        logging.warning("cone not full-dimensional")
        return 0
    if simplicial is True:
        t = is_M_alpha_posdef(A)
        if t is False:
            logging.warning("Associated matrix NOT positive definite, "
                            "series NOT converge")
        d = A.nrows()
        v = matrix(RDF, [A[i]/A[i].norm() for i in range(d)])
        da = int(d * (d-1) / 2)
        const = sqrt((v * v.transpose()).determinant()) / (RDF(4*pi) ** (d/2))
        alpha = [0] * da
        for i in range(d - 1):
            for j in range(i + 1, d):
                k = (2*d - i - 1) * i/2 + j - i - 1
                alpha[k] = v[i] * v[j]
        partial_sum = 0
        for n in range(deg + 1):
            sum_deg_n = 0
            for a in composition_of_n_into_k_parts(n, da):
                alphatoa = 1
                for k in range(da):
                    alphatoa = alpha[k] ** a[k] * alphatoa
                    if alphatoa == 0:
                        break
                if alphatoa == 0:
                    continue
                t = (-2) ** (sum(a))
                fact_denom = prod([factorial(a[k]) for k in range(da)])
                coef = t / fact_denom
                for i in range(d):
                    s_i = 0
                    for j in range(d):
                        if j != i:
                            m_1 = max(i, j)
                            m_0 = min(i, j)
                            k = (2*d - m_0 - 1) * m_0 / 2 + m_1 - m_0 - 1
                            s_i += a[k]
                    coef = coef * gamma(0.5 * (s_i + 1))
                sum_deg_n += coef * alphatoa
            partial_sum += sum_deg_n
            if abs(const * sum_deg_n) < eps:
                break
        return RDF(const * (partial_sum))
    else:
        A_list = simplicial_subcones_decomposition(A)
        n = len(A_list)
        results = []
        for i in range(n):
            results.append(
                solid_angle_general(A_list[i], eps=eps, deg=deg,
                                    simplicial=True, space=space))
        logging.info("Solid angle(s) of cones in Decomposition: %s" % results)
        return sum(results)


# **********************************************************
#                    Helper functions
# **********************************************************
def simplicial_subcones_decomposition(A):
    r"""
    Return a list of matrices that give the extreme rays
    of the simplicial cones formed from the triangulation
    of ``A``.

    INPUT:

    - ``A`` -- matrix; ``A`` is a matrix whose row vectors are a generating set
      for a cone (not necessarily simplicial.)

    OUTPUT:

    - a list of matrices that corresponds to the dissection of the cone
      spanned by the rows of ``A`` into simplicial cones.
      Each matrix represents a simplicial cone in the dissection.

    EXAMPLES:

    This example shows that the cone spanned by ``[1,0,0], [0,1,0], [0,0,1]``,
    and ``[-1,0,0]`` can be dissected into two simplicial cones, one with
    extreme rays ``[1,0,0], [0,1,0], [0,0,1]`` and the other with extreme
    rays ``[0,1,0], [0,0,1], [-1,0,0]``::

        sage: logging.disable(logging.NOTSET)
        sage: A = matrix([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        sage: simplicial_subcones_decomposition(A)
        [
        [1 0 0]  [ 0  1  0]
        [0 1 0]  [ 0  0  1]
        [0 0 1], [-1  0  0]
        ]

    This example shows that if the input corresponds to a simplicial cone,
    the function returns [input matrix itself]::

        sage: A = matrix([[1,0,0],[1,2,3],[-5,4,2]])
        sage: simplicial_subcones_decomposition(A)
        [
        [ 1  0  0]
        [ 1  2  3]
        [-5  4  2]
        ]

    This example shows that the function works in higher dimensions, such as
    `\RR^4`. Note that the input can also be in the form of a list of vectors::

        sage: A_in = [[1,0,-2,0],[1,2,3,-2],[-1,3,4,4],[-2,-1,0,0],[1,1,1,3]]
        sage: simplicial_subcones_decomposition(A_in)
        [
        [ 1  0 -2  0]  [ 1  0 -2  0]  [ 1  0 -2  0]  [ 1  2  3 -2]
        [ 1  2  3 -2]  [ 1  2  3 -2]  [-1  3  4  4]  [-1  3  4  4]
        [-1  3  4  4]  [-1  3  4  4]  [-2 -1  0  0]  [-2 -1  0  0]
        [-2 -1  0  0], [ 1  1  1  3], [ 1  1  1  3], [ 1  1  1  3]
        ]

    This example shows when the vectors in ``A`` are in `\RR^n`, but the
    cone spanned by the vectors lives in a lower dimensional space::

        sage: A = matrix([[1,0,0],[0,1,0],[3,2,0]])
        sage: simplicial_subcones_decomposition(A)
        [
        [1 0 0]  [0 1 0]
        [3 2 0], [3 2 0]
        ]

    This example shows that the cone in `\RR^4` spanned by the rows of ``A``
    (which is input as a list of lists) is actually a halfspace of affine
    dimension `2`. The triangulation dissects it into three 2-d subcones::

        sage: A_in = [[-3,0,5,0],[0,0,1,0],[-4,0,0,0],[-1,0,0,0],[0,0,-4,0]]
        sage: simplicial_subcones_decomposition(A_in)
        [
        [-3  0  5  0]  [-3  0  5  0]  [-4  0  0  0]
        [ 0  0  1  0], [-4  0  0  0], [ 0  0 -4  0]
        ]
    """
    if not hasattr(A, 'nrows'):
        A = matrix(A)
    r = A.rank()
    if A.rank() == A.nrows():
        return [A]
    else:
        from sage.geometry.triangulation.point_configuration \
            import PointConfiguration
        origin = A.nrows()
        pc = PointConfiguration(A.stack(vector([0]*A.ncols())), star=origin)
        triangulation = pc.triangulate()
        matrices = []
        for simplex in triangulation:
            matrices.append(matrix(A[i] for i in simplex if i != origin))
        return matrices


def composition_of_n_into_k_parts(n, k):
    r"""
    Return a generator of the weak integer compositions of ``n``
    into ``k`` parts.

    INPUT:

    - ``n`` -- integer; ``n`` is the integer that we want to find the weak
      compositions of.

    - ``k`` -- integer; ``k`` is the positive integer giving the number of
      parts we want in a composition of n.


    OUTPUT:

    - a generator object containing the ``k`` tuples that are weak compositions
      of ``n``.

    EXAMPLES:

    This example shows the weak compositions of `3` into `2` parts::

        sage: list(composition_of_n_into_k_parts(3,2))
        [[0, 3], [1, 2], [2, 1], [3, 0]]

    This example illustrates how the function can be used to find the number of
    the weak compositions of `11` into `2` parts::

        sage: len(list(composition_of_n_into_k_parts(11,2)))
        12

    This example shows that when `k=1`, the list returned has only the
    entry ``[n]``::

        sage: list(composition_of_n_into_k_parts(4,1))
        [[4]]

    This example shows that when `n=0`, the list returned has only one
    entry, a `k`-tuple of `0`'s::

        sage: list(composition_of_n_into_k_parts(0,6))
        [[0, 0, 0, 0, 0, 0]]


    .. NOTE::

        This function is used to develop a truncation form for the
        multivariate power series `T_{\alpha}` in Ribando's paper
        "Measuring solid angles beyond dimension three."
    """
    if k == 1:
        yield [n]
    elif n == 0:
        yield [0] * k
    else:
        for i in range(n+1):
            for c in composition_of_n_into_k_parts(n-i, k-1):
                yield [i]+c


def is_M_alpha_posdef(A):
    r"""
    Return whether the associated matrix `M(1, -|\alpha_{ij}|)` for the given
    matrix ``A`` is positive definite or not (see note).

    INPUT:

    - ``A`` -- matrix whose row vectors span a simplicial cone.

    OUTPUT: ``True`` or ``False``.

    EXAMPLES:

    This example shows that the associated matrix of ``[[1,-1,0],[2,1,1],
    [-1,0,0]]`` is not positive definite.::

        sage: logging.disable(logging.INFO)
        sage: A = matrix([[1,-1,0],[2,1,1],[-1,0,0]])
        sage: is_M_alpha_posdef(A)
        False

    In this example, we use the matrix given in Example 3.4 of Gourion and
    Seeger. The authors note that the associated matrix is positive definite.
    We see that our output aligns with this result.::

        sage: A = matrix([[1/2, -1/2, -1/2, 1/2],[1/2, 1/10, 7/10, 1/2],
        ....:   [-4/7, 4/7, 1/7, 4/7],[-4/11, -5/11, 8/11, 4/11]])
        sage: is_M_alpha_posdef(A)
        True

    The following examples illustrate that the function works in higher
    dimensions::

        sage: A = matrix([[1,2,3,4,5],[-1,3,0,-4,1],[5,0,0,-1,0],
        ....:   [0,0,-2,1,4],[0,0,0,0,1]])
        sage: is_M_alpha_posdef(A)
        False

        sage: A = matrix([[1,1,0,0,0],[-1,3,0,-4,1],[5,0,0,-1,0],
        ....:   [0,0,-2,1,4],[0,0,0,0,1]])
        sage: is_M_alpha_posdef(A)
        True

    .. NOTE::

        Here, `M(1, -|\alpha_{ij}|)` denotes the symmetric matrix with diagonal
        entries 1 and off-diagonal entries `-|\alpha_{ij}|` for `i\neq j`,
        where `\alpha_{ij}` is the normal vector of the dot product of the
        i-th and the j-th rows of the given matrix ``A``.

        This function is used as a test for the convergence of
        Ribando's power series for the solid angle of a cone. By
        Corollary 3.2 in Ribando's paper "Measuring solid angles
        beyond dimension three," the series converges if and only if
        the associated matrix is positive definite.
    """
    d = A.nrows()
    M_exact = A * A.transpose()  # unnormalized matrix with diag entries not 1
    vnorm = [A[i].norm() for i in range(d)]
    M = matrix(RDF, d)
    for i in range(d):
        for j in range(d):
            if i != j:
                M_exact[i, j] = - abs(M_exact[i, j])
            M[i, j] = RDF(M_exact[i, j] / (vnorm[i] * vnorm[j]))
    try:  # We prefer testing pos-def on M_exact over exact ring
        t = M_exact.is_positive_definite()
    except ValueError:  # Note that M is pos-def iff M_exact is pos-def
        t = M.is_positive_definite()
    logging.info("Associated Matrix:\n%s" % M)
    return t
