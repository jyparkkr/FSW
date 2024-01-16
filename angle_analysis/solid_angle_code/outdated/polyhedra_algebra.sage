## sage: from sage.geometry.polyhedron.parent import Polyhedra
## sage: P1 = Polyhedra(QQ, 1)

def closed_interval(a,b):
    return Polyhedron(vertices=[[a], [b]])

# Patch Polyhedra_base so that it knows about its cardinality.
# This needs to run before the first polyhedron is constructed...
from sage.geometry.polyhedron.parent import Polyhedra_base

def _Polyhedra_base_init(self, base_ring, ambient_dim, backend):
    self._backend = backend
    self._ambient_dim = ambient_dim
    from sage.categories.polyhedra import PolyhedralSets
    category = PolyhedralSets(base_ring)
    if ambient_dim == 0:
        category = category.Finite()
        ## should mix in another category to get the cardinality method
    else:
        category = category.Infinite()
    Parent.__init__(self, base=base_ring, category=category)
    self._Inequality_pool = []
    self._Equation_pool = []
    self._Vertex_pool = []
    self._Ray_pool = []
    self._Line_pool = []

Polyhedra_base.__init__ = _Polyhedra_base_init

from sage.geometry.polyhedron.parent import Polyhedra
from sage.geometry.polyhedron.backend_normaliz import Polyhedron_QQ_normaliz
from sage.structure.element import coerce_binop

class Polyhedron_halfopen_QQ_normaliz(Polyhedron_QQ_normaliz):

    def __init__(self, parent, Vrep, Hrep, normaliz_cone=None,
                 normaliz_data=None, normaliz_field=None,
                 open_rays=None, **kwds):
        """
        If provided, ``open_rays`` should be a list of indices into the rays list of ``Vrep``.

        TESTS::

            sage: parent = HalfOpenPolyhedra_base(QQ, 3)
            sage: P = Polyhedron_halfopen_QQ_normaliz(parent, [[[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], []], None, open_rays=[])
            sage: P._open_rays
            frozenset()
            sage: P._open_facets
            frozenset()
            sage: P = Polyhedron_halfopen_QQ_normaliz(parent, [[[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], []], None, open_rays=[0])
            sage: P._open_rays
            frozenset({A ray in the direction (1, 0, 0)})
            sage: P._open_facets
            frozenset({An inequality (1, 0, 0) x + 0 >= 0})

        Here the open redundant ray is eliminated::

            sage: P = Polyhedron_halfopen_QQ_normaliz(parent, [[[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], []], None, open_rays=[2])    # known bug

        """

        super(Polyhedron_halfopen_QQ_normaliz, self).__init__(parent, Vrep, Hrep, normaliz_cone,
                                                              normaliz_data, normaliz_field, **kwds)
        # The given generators are not necessarily preserved by the backend in this form.
        # So we have to reconstruct the openness information.
        # FIXME: This logic is only good enough for our simplicial cones.
        new_open_facets = []
        new_open_rays = []
        if Vrep is not None and open_rays:
            # Given Vrep, we reconstruct which computed facets are open from the given open rays
            for index in open_rays:
                ray_vector = vector(self.base_ring(), Vrep[1][index])
                new_open_facets += [ inequality for inequality in self.inequality_generator()
                                     if inequality.eval(ray_vector) != 0 ]
            # Now reconstruct the open rays
            for inequality in new_open_facets:
                new_open_rays += [ ray for ray in self.ray_generator()
                                   if ray.evaluated_on(inequality) != 0 ]
        # FIXME: We should detect inconsistencies of the openness information
        # and raise an error in this case.
        self._open_facets = frozenset(new_open_facets)
        self._open_rays = frozenset(new_open_rays)

    def closure(self):
        """
        EXAMPLES::

            sage: parent = HalfOpenPolyhedra_base(QQ, 3)
            sage: P = Polyhedron_halfopen_QQ_normaliz(parent, [[[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], []], None, open_rays=[0])
            sage: PC = Polyhedron_halfopen_QQ_normaliz(parent, [[[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], []], None)
            sage: P == PC
            False
            sage: P.closure() == PC
            True
        """
        parent = self.parent()
        return Polyhedron_halfopen_QQ_normaliz( ##parent.element_class(   <---- This gets the wrong class and causes _is_subpolyhedron to fail
                        parent, [self.vertices(), self.rays(), self.lines()],
                        None)

    def _repr_(self):
        s = super(Polyhedron_halfopen_QQ_normaliz, self)._repr_()
        if self._open_rays or self._open_facets:
            s += " (half-open)"
        return s

    @coerce_binop
    def _is_subpolyhedron(self, other):
        ### This does not really work yet because of the coercion system!
        ### We need to create a parent.
        """
        Test whether ``self`` is a (not necessarily strict)
        sub-polyhedron of ``other``.

        INPUT:

        - ``other`` -- a :class:`Polyhedron`

        OUTPUT:

        Boolean

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,0), (0,1)], rays=[(1,1)])
            sage: Q = Polyhedron(vertices=[(1,0), (0,1)])
            sage: P._is_subpolyhedron(Q)
            False
            sage: Q._is_subpolyhedron(P)
            True
        """
        def contains(other_H, self_V):
            if not other_H.contains(self_V):
                return False
            if self_V not in self._open_rays:
                try:
                    if other_H in other._open_facets:
                        return False
                except AttributeError:
                    pass
            return True
        ##print("### is_sub")
        return all(contains(other_H, self_V)
                   for other_H in other.Hrepresentation()
                   for self_V in self.Vrepresentation())

## Families of polyhedra.

from sage.structure.unique_representation import UniqueRepresentation
from sage.geometry.polyhedron.parent import Polyhedra_normaliz

class HalfOpenPolyhedra_base(Polyhedra_normaliz):

    def __init__(self, base_ring, ambient_dim, backend=None):
        if backend is not None and backend != "normaliz":
            raise NotImplementedError
        super(HalfOpenPolyhedra_base, self).__init__(base_ring, ambient_dim, backend)

    Element = Polyhedron_halfopen_QQ_normaliz

    def _repr_(self):
        return "Half-open " + super(HalfOpenPolyhedra_base, self)._repr_() + " (non-facade parent)"

class HalfOpenPolyhedra(UniqueRepresentation, Parent):

    def __init__(self, base_ring, ambient_dim, category=None):
        if category is None:
            category = Semigroups()  # but not a monoid, with respect to Minkowski sums
        if ambient_dim == 0:
            category = category.Finite()
        else:
            category = category.Infinite()
        self._polyhedra = HalfOpenPolyhedra_base(base_ring, ambient_dim, backend='normaliz')
        Parent.__init__(self, facade=self._polyhedra,
                        category=category)

    def _element_constructor_(self, *args, **kwds):
        r"""
        Construction of elements

        This is also used for the membership test.
        """
        polyhedron = self._polyhedra._element_constructor_(*args, **kwds)
        self._check_polyhedron(polyhedron)
        return polyhedron

    def _check_polyhedron(self, polyhedron):
        pass

    def _repr_(self):
        return "Half-open Polyhedra in " + self._polyhedra._repr_ambient_module()

    def universe(self):
        return self._polyhedra.universe()

    def ambient_dim(self):
        return self._polyhedra.ambient_dim()


class ClosedPolyhedra(HalfOpenPolyhedra):

    def _check_polyhedron(self, polyhedron):
        super(ClosedPolyhedra, self)._check_polyhedron(polyhedron)
        if (hasattr(polyhedron, "_open_rays") and polyhedron._open_rays) or (hasattr(polyhedron, "_open_facets") and polyhedron._open_facets):
            raise ValueError("{} should be a closed polyhedron".format(polyhedron))

    def _repr_(self):
        return "Closed Polyhedra in " + self._polyhedra._repr_ambient_module()

class RelativelyOpenPolyhedra(HalfOpenPolyhedra):

    def _check_polyhedron(self, polyhedron):
        super(RelativelyOpenPolyhedra, self)._check_polyhedron(polyhedron)
        raise NotImplementedError()

    def _repr_(self):
        return "Relatively Open Polyhedra in " + self._polyhedra._repr_ambient_module()

class FullDimensionalPolyhedra(HalfOpenPolyhedra):

    def _repr_(self):
        return "Full-dimensional " + super(FullDimensionalPolyhedra, self)._repr_()

    def _check_polyhedron(self, polyhedron):
        super(FullDimensionalPolyhedra, self)._check_polyhedron(polyhedron)
        if not polyhedron.is_full_dimensional():
            raise ValueError("{} should be full-dimensional".format(polyhedron))

class LowerDimensionalPolyhedra(HalfOpenPolyhedra):

    def _repr_(self):
        return "Lower-dimensional " + super(LowerDimensionalPolyhedra, self)._repr_()

    def _check_polyhedron(self, polyhedron):
        super(LowerDimensionalPolyhedra, self)._check_polyhedron(polyhedron)
        if polyhedron.is_full_dimensional():
            raise ValueError("{} should be lower-dimensional".format(polyhedron))

class NonPointedPolyhedra(HalfOpenPolyhedra):
    """
    EXAMPLES::

        sage: Polyhedron(vertices=[[0, 0]]) in NonPointedPolyhedra(QQ, 2)
        False
        sage: Polyhedron(lines=[[1, 1]]) in NonPointedPolyhedra(QQ, 2)
        True
    """
    def _repr_(self):
        return "Non-pointed " + super(NonPointedPolyhedra, self)._repr_()

    def _check_polyhedron(self, polyhedron):
        super(NonPointedPolyhedra, self)._check_polyhedron(polyhedron)
        if not polyhedron.lines():
            raise ValueError("{} should be non-pointed".format(polyhedron))

from sage.categories.category_types import Category_over_base_ring
from sage.categories.filtered_modules_with_basis import FilteredModulesWithBasis

class PolyhedraModules(Category_over_base_ring):

    @cached_method
    def super_categories(self):
        R = self.base_ring()
        return [FilteredModulesWithBasis(R)]

    class ParentMethods:

        def degree_on_basis(self, m):
            r"""
            Polyhedra generate a module filtered by dimension.

            Modulo the linear relations of polyhedra, this is only a filtration,
            not a grading, as the following example shows.

            EXAMPLES::

                sage: M1 = FormalPolyhedraModule(QQ, 1)
                sage: X = M1(closed_interval(0, 1)) + M1(closed_interval(1, 2)) - M1(closed_interval(0, 2))
                sage: X.degree()
                1

                sage: Y = M1(closed_interval(1, 1))
                sage: Y.degree()
                0

            In the FormalPolyhedraModule, this is actually a grading.
            So we can extract homogeneous components::

                sage: O = M1(closed_interval(0, 1)) + M1(closed_interval(0, 0)) + M1(closed_interval(1, 1))
                sage: O.homogeneous_component(0)
                [ {[0]} ] + [ {[1]} ]
                sage: O.homogeneous_component(1)
                [ conv([0], [1]) ]

            """
            return m.dimension()

    class ElementMethods:

        def dual(self):
            r"""
            EXAMPLES::

                sage: M2 = FormalPolyhedraModule(QQ, 2)
                sage: C = Polyhedron(rays=[[1, 0], [1, 1]])
                sage: IC = M2(C)
                sage: IC.dual()
                [ cone([0, 1], [1, -1]) ]
                sage: R = Polyhedron(rays=[[1, 0]])
                sage: IC_halfopen = IC - M2(R); IC_halfopen
                -[ cone([1, 0]) ] + [ cone([1, 0], [1, 1]) ]
                sage: IC_halfopen.dual()
                [ cone([0, 1], [1, -1]) ] - [ cone([1, 0]) + lin([0, 1]) ]

            """
            return self.parent().duality_valuation()(self)

        def triangulate(self, codomain=None):
            r"""
            EXAMPLES::

                sage: M3 = FormalPolyhedraModule(QQ, 3)
                sage: M3_mod_lower = PolyhedraModule(QQ, 3, mod_lower_dimensional=True)
                sage: egyptian = Polyhedron(rays=[[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], backend='normaliz')
                sage: M3(egyptian).triangulate(codomain=M3_mod_lower)
                [ cone([1, 0, 0], [1, 0, 1], [1, 1, 0]) ] + [ cone([1, 0, 1], [1, 1, 0], [1, 1, 1]) ]
            """
            return self.parent().triangulation_morphism(codomain=codomain)(self)

        def dual_decomposition(self, new_facet_normal, codomain=None):
            # Could add more keywords such as "mod_lower_dimensional",
            # which select "codomain".
            r"""
            Decompose into cones that have a facet with inner or outer normal vector ``new_facet_normal``.

            This is a special case of ``brion_vergne_decomposition`` where ``subspace``
            is orthogonal to ``new_facet_normal``.

            If ``codomain.is_mod_non_pointed()``, this is the decomposition of Proposition 14 in
            Baldoni et al. https://arxiv.org/pdf/1011.6002.pdf

            If ``codomain.is_mod_lower_dimensional()``, this is the decomposition used for the solid angles.

            EXAMPLES::

                sage: M2 = PolyhedraModule(QQ, 2, mod_lower_dimensional=True)
                sage: C = Polyhedron(rays=[[1, 0], [1, 1]])
                sage: IC = M2(C)
                sage: IC.dual_decomposition([[0, 1]])

            """
            def poly_decompose_mod_lower_dimensional(m):
                ## FIXME: Needs an actual implementation
                ## m is a Polyhedron.
                parent = self.parent()
                return parent(Polyhedron(rays=[[5, 28], [28, 5]])) - parent(Polyhedron(rays=[[11, 1], [1, 11]]))

            if codomain is None:
                codomain = self.parent()
            if codomain.is_mod_lower_dimensional():
                if self.parent().is_mod_non_pointed() and not codomain.is_mod_non_pointed():
                    raise TypeError("cannot map from {} to {}".format(self.parent(), codomain))
                morphism = self.parent().module_morphism(on_basis=poly_decompose_mod_lower_dimensional, codomain=codomain)
                return morphism(self)
            raise NotImplementedError()

        def brion_vergne_decomposition(self, subspace, codomain=None):
            raise NotImplementedError("Brion-Vergne is not implemented")

        def stellar_decomposition(self, new_ray, codomain=None):
            r"""
            Decompose into cones that have a ray in the direction of the vector plus/minus ``new_ray``.

            This is a special case of ``brion_vergne_decomposition`` where ``subspace``
            is generated by ``new_ray``.
            """
            raise NotImplementedError("stellar decomposition is not implemented")

from sage.sets.family import LazyFamily

def PolyhedraModule(base_ring, dimension, basis=None,
                    mod_lower_dimensional=False, mod_non_pointed=False):
    M = FormalPolyhedraModule(base_ring, dimension, basis)
    ## if mod_lower_dimensional:
    ##     M = M / M.lower_dimensional_polyhedra_submodule()
    ## if mod_non_pointed:
    ##     M = M / M.non_pointed_polyhedra_submodule()
    return M

class FormalPolyhedraModule(CombinatorialFreeModule):
    """
    Formal module generated by polyhedra.

    It is formal because it is free -- it does not know
    about linear relations of polyhedra.

    EXAMPLES:

    Finite basis::

        sage: I01 = closed_interval(0, 1)
        sage: I11 = closed_interval(1, 1)
        sage: I12 = closed_interval(1, 2)
        sage: basis = [I01, I11, I12]
        sage: M = FormalPolyhedraModule(QQ, 1, basis=basis)
        sage: M.get_order()
        [A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices,
         A 0-dimensional polyhedron in ZZ^1 defined as the convex hull of 1 vertex,
         A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices]
        sage: M_lower = M.submodule([M(I11)]); M_lower
        Free module generated by {0} over Rational Field
        sage: M_lower.is_submodule(M)
        True
        sage: x = M(I01) - 2*M(I11) + M(I12)
        sage: M_lower.reduce(x)
        [ conv([0], [1]) ] + [ conv([1], [2]) ]
        sage: M_lower.retract.domain() is M
        True
        sage: y = M_lower.retract(M(I11)); y
        B[0]
        sage: M_lower.lift(y)
        [ {[1]} ]
        sage: M_lower(I11)     # not tested -- does not work
        sage: M_lower(M(I11))  # not tested -- does not work
        sage: M_mod_lower = M.quotient_module(M_lower); M_mod_lower
        Free module generated by {A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices, A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices} over Rational Field
        sage: M_mod_lower(x)   # not tested -- this does not work
        sage: M_mod_lower.retract(x)
        B[A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices] + B[A 1-dimensional polyhedron in ZZ^1 defined as the convex hull of 2 vertices]
        sage: M_mod_lower.retract(M(I01) - 2*M(I11) + M(I12)) ==  M_mod_lower.retract(M(I01) + M(I12))
        True

    """
    @staticmethod
    def __classcall__(cls, base_ring, dimension, basis=None):
        r"""
        Normalize the arguments for caching.

        TESTS::

            sage: FormalPolyhedraModule(QQ, 1) is FormalPolyhedraModule(QQ, 1, ClosedPolyhedra(QQ, 1))
            True
        """
        if basis is None:
            basis = ClosedPolyhedra(QQ, dimension)
        if isinstance(basis, list):
            basis = tuple(basis)
        return super(FormalPolyhedraModule, cls).__classcall__(cls,
                                                               base_ring=base_ring,
                                                               dimension=dimension,
                                                               basis=basis)

    def __init__(self, base_ring, dimension, basis, category=None):
        ## Should add some options that control what is used as a basis...
        ## Closed? Closed and half-open? Only relatively open?
        if category is None:
            category = PolyhedraModules(base_ring) & GradedModulesWithBasis(base_ring)
        # Need backend="normaliz" for triangulations of unbounded polyhedra
        super(FormalPolyhedraModule, self).__init__(base_ring, basis,
           prefix="", category=category)
        self._polyhedra = HalfOpenPolyhedra_base(base_ring, dimension, backend='normaliz')
        self._ambient_dim = dimension

    def set_order(self, order=None):
        """
        EXAMPLES::

            sage: M1 = PolyhedraModule(QQ, 1)
            sage: I01 = closed_interval(0, 1)
            sage: I11 = closed_interval(1, 1)
            sage: I12 = closed_interval(1, 2)
            sage: sorted([M1(I01), M1(I11), M1(I12)], key=M1.get_order_key())
            [[ conv([0], [1]) ], [ {[1]} ], [ conv([1], [2]) ]]

        """
        if self.basis().cardinality() == Infinity:
            from sage.combinat.ranker import on_fly
            # We could consider some more specific orders.
            # such as some refinement of a term order on vertices...
            rank, unrank = on_fly()
            self._rank_basis = rank
        else:
            super(FormalPolyhedraModule, self).set_order(order)

    @cached_method
    def get_order(self):
        if self._order is None:
            if self.basis().cardinality() == Infinity:
                self.set_order()
            else:
                self.set_order(self.basis().keys().list())
        return self._order

    def closed_polyhedra_family(self):
        return LazyFamily(ClosedPolyhedra(self.base_ring(), self._ambient_dim),
                          function=self.monomial,
                          name="generator")

    def lower_dimensional_polyhedra_family(self):
        return LazyFamily(LowerDimensionalPolyhedra(self.base_ring(), self._ambient_dim),
                          function=self.monomial,
                          name="generator")

    def lower_dimensional_polyhedra_submodule(self):
        # unclear if this is correct because a "basis in echelon form" is required
        #return self.submodule(self.lower_dimensional_polyhedra_family())
        return FormalLowerDimensionalPolyhedraModule(self)

    def closure_relation(self, a):
        try:
            c = a.closure()
            return self.term(a) - self.term(c)
        except AttributeError:
            return self.zero()

    def closure_relations_family(self):
        return LazyFamily(self.indices(), function=self.closure_relation)

    def closure_relations_submodule(self):
        """
        EXAMPLES::

            sage: parent = HalfOpenPolyhedra_base(QQ, 1)
            sage: pos = Polyhedron_halfopen_QQ_normaliz(parent, [[[0]], [[1]], []], None, open_rays=[0])
            sage: nonpos = Polyhedron_halfopen_QQ_normaliz(parent, [[[0]], [[-1]], []], None)
            sage: M1_halfopen = PolyhedraModule(QQ, 1, basis=HalfOpenPolyhedra(QQ, 1))
            sage: M1_halfopen.closure_relation(pos)
            [ pos([1]) ] - [ cone([1]) ]
            sage: M1_halfopen.closure_relation(nonpos)
            0
            sage: M1_clos_rel = M1_halfopen.closure_relations_submodule()  ### error

        Finite-dimensional::

            sage: M = PolyhedraModule(QQ, 1, basis=[pos, pos.closure(), nonpos])
            sage: M_clos_rel = M.closure_relations_submodule()
            sage: M_clos_rel.lift(M_clos_rel.basis()[0])
            [ pos([1]) ] - [ cone([1]) ]
            sage: M_clos_rel.retract(M.closure_relation(pos))  ### error
            sage: M_mod_clos_rel = M.quotient_module(M_clos_rel)
            sage: M_mod_clos_rel.retract(M(pos))      ### error

        """
        return self.submodule(self.closure_relations_family(), unitriangular=True)

    def convex_inclusion_exclusion(u, a, b):
        """
        U must be a polyhedron that is the union of a and b (not checked).
        """
        return self.term(u) - (self.term(a) + self.term(b) - self.term(a & b))

    ## def halfspaces_family(self):
    ##     return LazyFamily(

    def halfspace_inclusion_exclusion(u, halfspace):
        raise NotImplementedError()
        ## if len(halfspace.inequalities()) == 1 and not halfspace.equations():

        ##     opposite_halfspace =
        ##     return convex_inclusion_exclusion(u,

    def indicator_relations_family(self):
        return LazyFamily(cartesian_product([self.indices(), self.indices()]),
                          lambda uh: halfspace_inclusion_exclusion(uh[0], uh[1]),
                          "inclusion_exclusion")

    def indicator_relations_submodule(self):
        return IndicatorRelationsSubmodule(ambient=self, category=None)

    def submodule(self, gens, check=True, already_echelonized=False,
                  unitriangular=False, category=None):
        r"""
        The submodule spanned by a finite set of elements.

        INPUT:

        - ``gens`` -- a list or family of elements of ``self``
        """
        if not already_echelonized:
            gens = self.echelon_form(gens, unitriangular)
        return self._submodule_class()(gens, ambient=self, unitriangular=unitriangular,
                                       category=category)

    def _submodule_class(self):
        r"""
        The class of submodules of this module. This is a separate method so it
        can be overridden in derived classes.
        """
        return FormalPolyhedraSubmodule

    ## def echelon_form(self, elements, row_reduced=False):
    ##     r"""
    ##     Return a basis in echelon form of the subspace spanned by
    ##     a finite set of elements.

    ##     INPUT:

    ##     - ``elements`` -- a list or finite iterable of elements of ``self``
    ##     - ``row_reduced`` -- (default: ``False``) whether to compute the
    ##       basis for the row reduced echelon form

    ##     OUTPUT:

    ##     A list of elements of ``self`` whose expressions as
    ##     vectors form a matrix in echelon form.
    ##     """
    ##     # We have to override ModulesWithBasis.ParentMethods.echelon_form
    ##     # because it depends on _vector_, which we cannot define.



    def is_mod_lower_dimensional(self):
        return False

    def is_mod_non_pointed(self):
        return False

    ## def _coerce_map_from_(self, X):
    ##     r"""
    ##     Return whether there is a coercion from ``X``.

    ##     EXAMPLES::

    ##         sage: M2 = FormalPolyhedraModule(QQ, 2)
    ##         sage: M2_mod_lower = FormalPolyhedraModule(QQ, 2, mod_lower_dimensional=True)
    ##         sage: M2.has_coerce_map_from(FormalPolyhedraModule(QQ, 1))
    ##         False
    ##         sage: M2_mod_lower.has_coerce_map_from(M2)
    ##         True
    ##         sage: M2.has_coerce_map_from(M2_mod_lower)
    ##         False
    ##     """
    ##     if not isinstance(X, FormalPolyhedraModule):
    ##         return False
    ##     if not X.indices().ambient_space().is_subspace(self.indices().ambient_space()):
    ##         return False
    ##     if X.is_mod_lower_dimensional() and not self.is_mod_lower_dimensional():
    ##         return False
    ##     if X.is_mod_non_pointed() and not self.is_mod_non_pointed():
    ##         return False
    ##     return True

    def _repr_(self):
        s = "Formal module of {}".format(self.indices())
        if self.is_mod_lower_dimensional():
            s += " modulo lower-dimensional polyhedra"
        if self.is_mod_non_pointed():
            s += " modulo non-pointed polyhedra"
        return s

    def _poly_is_affine_cone(self, m):
        return len(m.vertices()) == 1

    def _poly_is_cone(self, m):
        return self._poly_is_affine_cone(m) and all(x == 0 for x in m.vertices()[0])

    def _poly_is_nontrivial_cone(self, m):
        return self._poly_is_cone(m) and (m.rays() or m.lines())

    def _poly_repr(self, m):
        if m.is_empty():
            return "{}"
        if m.is_universe():
            return m.parent()._repr_ambient_module()
        s = []
        if m.vertices():
            if len(m.vertices()) == 1:
                if not self._poly_is_nontrivial_cone(m):
                    s.append("{" + repr(m.vertices_list()[0]) + "}")
            else:
                s.append("conv(" + ", ".join(repr(x) for x in sorted(m.vertices_list())) + ")")
        if m.rays():
            if hasattr(m, "_open_rays"):
                cone_rays = sorted(list(ray) for ray in m.ray_generator() if ray not in m._open_rays)
                open_rays = sorted(list(ray) for ray in m.ray_generator() if ray in m._open_rays)
            else:
                cone_rays = sorted(m.rays_list())
                open_rays = []
            if cone_rays:
                s.append("cone(" + ", ".join(repr(x) for x in cone_rays) + ")")
            if open_rays:
                s.append("pos(" + ", ".join(repr(x) for x in open_rays) + ")")
        if m.lines():
            s.append("lin(" + ", ".join(repr(x) for x in sorted(m.lines_list())) + ")")
        return " + ".join(s)

    def _repr_generator(self, m):
        return "[ " + self._poly_repr(m) + " ]"

    _repr_term = _repr_generator

    @cached_method
    def dual(self):
        r"""
        EXAMPLES::

            sage: M1 = FormalPolyhedraModule(QQ, 1)
            sage: M1.dual() is M1
            True
            sage: M1_mod_lower = PolyhedraModule(QQ, 1, mod_lower_dimensional=True)
            sage: M1_mod_lower.dual() is PolyhedraModule(QQ, 1, mod_non_pointed=True)
            True
        """
        return PolyhedraModule(self.base_ring(), self.indices().ambient_dim(),
                               mod_lower_dimensional=self.is_mod_non_pointed(),
                               mod_non_pointed=self.is_mod_lower_dimensional())

    def _poly_dual(self, m):
        # FIXME: only for cones. We also want it for affine cones.
        # And later, by Gram-Brianchon, also for arbitrary polyhedra.
        if not self._poly_is_affine_cone(m):
            raise ValueError("dualization is only defined for affine cones")
        #return self(Cone(m).dual().polyhedron())
        if hasattr(m, "_open_facets") and m._open_facets:
            raise NotImplementedError("dualization is not implemented yet for half-open cones")
        apex = m.vertices()[0]
        rays = [ i.A() for i in m.inequalities() ]
        lines = [ e.A() for e in m.equations() ]
        return self(Polyhedron_halfopen_QQ_normaliz(self._polyhedra, [[apex], rays, lines], None))

    @cached_method
    def duality_valuation(self):
        r"""
        EXAMPLES::

            sage: M1 = FormalPolyhedraModule(QQ, 1)
            sage: M1.duality_valuation()
            Generic endomorphism of Formal module of Closed Polyhedra in QQ^1
        """
        return self.module_morphism(on_basis=self._poly_dual, codomain=self.dual())

    ## @cached_method
    ## def indicator_valuation(self):
    ##   ..... the kernel of this map is the polyhedral relations.....
    ##     return self.module_morphism....

    def _poly_construct(self, Vrep, Hrep):
        constructor = self.indices()
        return constructor(Vrep, Hrep)

    def _poly_triang(self, codomain, m):
        """
        TESTS::

            sage: M3 = PolyhedraModule(QQ, 3, mod_lower_dimensional=True)
            sage: egyptian = Polyhedron(rays=[[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], backend='normaliz')
            sage: M3._poly_triang(M3, egyptian)
            [ cone([1, 0, 0], [1, 0, 1], [1, 1, 0]) ] + [ cone([1, 0, 1], [1, 1, 0], [1, 1, 1]) ]
            sage: affine_egyptian = Polyhedron(vertices=[[2, 3, 4]], rays=[[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], backend='normaliz')
            sage: M3._poly_triang(M3, affine_egyptian)
            [ {[2, 3, 4]} + cone([1, 0, 0], [1, 0, 1], [1, 1, 0]) ] + [ {[2, 3, 4]} + cone([1, 0, 1], [1, 1, 0], [1, 1, 1]) ]
        """
        if not codomain.is_mod_lower_dimensional():
            raise NotImplementedError("Triangulations are only implemented modulo lower dimensional polyhedra")
        if not self._poly_is_affine_cone(m) or m.lines():
            raise NotImplementedError("Triangulations are only implemented for pointed cones")
        triangulation = m.triangulate(engine='normaliz')
        rays = m.rays()
        apex = m.vertices()[0]
        terms = ((codomain._poly_construct(([apex], [rays[i] for i in s], []), None), 1) for s in triangulation)
        return codomain.sum_of_terms(terms)

    @cached_method
    def triangulation_morphism(self, codomain=None):
        if codomain is None:
            codomain = self
        return self.module_morphism(on_basis=lambda m: self._poly_triang(codomain, m), codomain=codomain)

from sage.modules.with_basis.subquotient import SubmoduleWithBasis, QuotientModuleWithBasis

class FormalPolyhedraSubmodule(SubmoduleWithBasis):

    @staticmethod
    def __classcall_private__(cls, basis, ambient=None, unitriangular=False,
                              category=None, *args, **opts):
        r"""
        Normalize the input.
        """
        basis = Family(basis)
        if ambient is None:
            ambient = basis.an_element().parent()
        default_category = PolyhedraModules(ambient.category().base_ring()).Subobjects()
        category = default_category.or_subcategory(category, join=True)
        return super(FormalPolyhedraSubmodule, cls).__classcall__(
            cls, basis, ambient, unitriangular, category, *args, **opts)

    def __init__(self, basis, ambient, unitriangular, category):
        SubmoduleWithBasis.__init__(self, basis, ambient, unitriangular, category)

class IndicatorRelationsSubmodule(FormalPolyhedraSubmodule):   ## perhaps easier to implement with Submodule instead of SubmoduleWithBasis!

    @staticmethod
    def __classcall_private__(cls, ambient, category=None, *args, **opts):
        basis = ambient.indicator_relations_family()  # ???
        default_category = PolyhedraModules(ambient.category().base_ring()).Subobjects()
        category = default_category.or_subcategory(category, join=True)
        return super(IndicatorRelationsSubmodule, cls).__classcall__(
            cls, basis, ambient, True, category, *args, **opts)

    ## Try to define only this one:

    #lift.

    ## def lift_on_basis():

    ##     r"""
    ##     EXAMPLES::

    ##         sage: M1 = FormalPolyhedraModule(QQ, 1)
    ##         sage: M1_ind = M1.indicator_relations_submodule()
    ##         sage: I1 = closed_interval(1, 3)
    ##         sage: I2 = closed_interval(2, 4)
    ##         sage: I1_union_I2 = closed_interval(1, 4)
    ##         sage: R = M1(I1_union_I2) + M1(I1&I2) - M1(I1) - M1(I2)
    ##         sage: M1_ind.reduce(R)
    ##         0
    ##     """
    ##     raise NotImplementedError()

M1 = FormalPolyhedraModule(QQ, 1)
M1_ind = M1.indicator_relations_submodule()
#M1_lower = M1.lower_dimensional_polyhedra_submodule()
M1_mod_lower = PolyhedraModule(QQ, 1, mod_lower_dimensional=True)


class FormalLowerDimensionalPolyhedraModule(FormalPolyhedraModule, SubmoduleWithBasis):

    ## Should we also put the relations between half-open polyhedra
    ## and their closures into this module, or a separate one?



    @staticmethod
    def __classcall_private__(cls, ambient, category=None, *args, **opts):
        basis = ambient.lower_dimensional_polyhedra_family()
        default_category = PolyhedraModules(ambient.category().base_ring()).Subobjects()
        category = default_category.or_subcategory(category, join=True)
        return super(FormalLowerDimensionalPolyhedraModule, cls).__classcall__(
            cls, basis, ambient, True, category, *args, **opts)

    ## def lift_on_basis():

    ## pass

    ### diagonal family?  --> no, this needs same indices in both modules.

    ## @lazy_attribute
    ## def lift(self):


## class ClosureRelationsSubmodule(


class FormalPolyhedraQuotientModuloLower(SubmoduleWithBasis):

    pass





def FormalPolyhedraTensorAlgebra(base_ring, dimension):
    """
    Formal tensor algebra generated by polyhedra.

    Multiplication is formal.

    EXAMPLES::

        sage: T1 = FormalPolyhedraTensorAlgebra(QQ, 1); T1
        Tensor Algebra of Formal module of Closed Polyhedra in QQ^1
        sage: T1()
        0
        sage: T1.one()
        1
        sage: X = T1(closed_interval(0, 1)) + T1(closed_interval(3, 5)); X
        [ conv([3], [5]) ] + [ conv([0], [1]) ]
        sage: Y = T1(closed_interval(1, 2)) - T1(closed_interval(4, 6)); Y
        -[ conv([4], [6]) ] + [ conv([1], [2]) ]
        sage: X * Y
        [ conv([3], [5]) ] * [ conv([1], [2]) ] - [ conv([3], [5]) ] * [ conv([4], [6]) ] + [ conv([0], [1]) ] * [ conv([1], [2]) ] - [ conv([0], [1]) ] * [ conv([4], [6]) ]

    """
    M = FormalPolyhedraModule(base_ring, dimension)
    return TensorAlgebra(M, tensor_symbol=' * ')

T1 = FormalPolyhedraTensorAlgebra(QQ, 1)

# Tensors are overkill -- we really just need symmetric tensors (in other
# words, a polynomial ring whose variables are the polyhedra.)
# check https://groups.google.com/d/msg/sage-devel/wAXisWv1iUk/2nQzdLS8AwAJ
# for how to do symmetric tensors!!

class FormalPolyhedraAlgebra(FormalPolyhedraModule):

    """
    Formal algebra generated by polyhedra.

    Multiplication of polyhedra is intersection.

    It is formal because as a module it is free -- it does not know
    about linear relations of polyhedra.

    EXAMPLES::

        sage: A1 = FormalPolyhedraAlgebra(QQ, 1); A1
        Formal algebra of Closed Polyhedra in QQ^1
        sage: A1()
        0
        sage: A1.one()
        [ QQ^1 ]
        sage: X = A1(closed_interval(0, 1)) + A1(closed_interval(3, 5)); X
        [ conv([0], [1]) ] + [ conv([3], [5]) ]
        sage: Y = A1(closed_interval(1, 2)) - A1(closed_interval(4, 6)); Y
        [ conv([1], [2]) ] - [ conv([4], [6]) ]
        sage: X * Y
        [ {[1]} ] - [ conv([4], [5]) ]
    """

    @staticmethod
    def __classcall__(cls, base_ring, dimension, basis=None):
        return super(FormalPolyhedraAlgebra, cls).__classcall__(cls,
                                                                base_ring=base_ring,
                                                                dimension=dimension,
                                                                basis=basis)

    def __init__(self, base_ring, dimension, basis,
                 category=None):
        if category is None:
            category = MonoidAlgebras(base_ring).Commutative() & PolyhedraModules(base_ring)
        super(FormalPolyhedraAlgebra, self).__init__(base_ring, dimension, basis,
                                                     category=category)

    def _repr_(self):
        return "Formal algebra of {}".format(self.indices())

    def product_on_basis(self, g1, g2):
        return self.term(g1 & g2)

    @cached_method
    def one_basis(self):
        return self.indices().universe()

    @cached_method
    def dual(self):
        ## FIXME: Can we avoid this copy-paste by using a Construction Functor?
        r"""
        EXAMPLES::

            sage: M1 = FormalPolyhedraAlgebra(QQ, 1)
            sage: M1.dual() is M1
            True
        """
        return FormalPolyhedraAlgebra(self.base_ring(), self.indices().ambient_dim())


A1 = FormalPolyhedraAlgebra(QQ, 1)


## empty_gen = A1(A1.indices().empty())

## Q1 = A1.quotient(A1.indices().empty())

## empty_ideal = A1.ideal(empty_gen)
#empty_gen in empty_ideal
# Error has no attribute 'divides'

## sage: Q1(A1.indices().empty())
## [ {} ]
###### hm.....
