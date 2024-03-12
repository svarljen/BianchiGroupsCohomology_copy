import sage
from sage.categories.groups import Groups
from sage.groups.matrix_gps.linear import LinearMatrixGroup_generic
from sage.modular.cusps_nf import NFCusp
from sage.rings.infinity import infinity
from sage.rings.number_field.number_field import QuadraticField, CyclotomicField
from sage.all import latex, Integer, Matrix, matrix
from sage.misc.cachefunc import cached_method

from upperhalfspace import ComplexSpaceElement__class
from Bianchimodulargroupelement import BianchiModularGroupElement

import logging

logger = logging.getLogger(__name__)
logger.setLevel(10)


def is_BianchiModularGroup(x) -> bool:
    """
    Return `True` if ``x`` is an instance of a BianchiModularGroup

    INPUT:

    - ``x`` -- something to test if it is a Bianchi modular group or not

    OUTPUT:

    - boolean

    EXAMPLES::

        sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup, is_BianchiModularGroup
        sage: is_BianchiModularGroup(0)
        False
        sage: B = BianchiModularGroup(5)
        sage: is_BianchiModularGroup(B)
        True
    """
    return isinstance(x, BianchiModularGroup_class)


def BianchiModularGroup(number_field):
    r"""
    Create the Bianchi modular group over the ring of integers in the given number field


    INPUT:

    - ``number_field`` (NumberField) -- a quadratic imaginary number field or positive integer.
      If a positive integer D is specified
      then the number field $Q(\sqrt(-D))$ is used.


    EXAMPLES::

        sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
        sage: BianchiModularGroup(5)
        Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
        sage: BianchiModularGroup(QuadraticField(-5))
        Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
        sage: BianchiModularGroup(QuadraticField(-3))
        Bianchi Modular Group PSL(2) over Eisenstein Integers in Number Field in a with defining polynomial x^2 + 3 with a = 1.732050807568878?*I

    """
    if isinstance(number_field, (int, Integer)) and number_field > 0:
        ring = QuadraticField(-number_field).ring_of_integers()
    elif isinstance(number_field, sage.rings.number_field.number_field_base.NumberField) \
            and number_field.is_totally_imaginary() and number_field.degree()==2:
        ring = number_field.ring_of_integers()
    else:
        raise ValueError("The input must be an imaginary quadratic Number Field or a positive integer")
    degree = Integer(2)
    name = f'Bianchi Modular Group PSL({degree}) over {ring}'
    ltx = 'PSL({0}, {1})'.format(degree, latex(ring))
    return BianchiModularGroup_class(base_ring=ring, sage_name=name, latex_string=ltx)


class BianchiModularGroup_class(LinearMatrixGroup_generic):
    r"""
    Class for Hilbert modular groups, here defined as either PSL(2) (default) or  SL(2)
    over rings of integers in totally real number fields.


    """

    Element = BianchiModularGroupElement

    def __init__(self, base_ring, sage_name, latex_string):
        r"""
         Init a Bianchi modular group over the ring of integers in the given number field

        INPUT:

        - ``base_ring`` - ring
        - ``sage_name`` - string
        - ``latex_string`` - string

        EXAMPLES::

            sage: from Bianchimodulargroup import *
            sage: OK = QuadraticField(-5).OK()
            sage: name = f'Bianchi Modular Group PSL(2) over {OK}'
            sage: ltx = f'PSL(2, {latex(OK)})'
            sage: BianchiModularGroup_class(base_ring=OK, sage_name=name, latex_string=ltx)
            Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: B = BianchiModularGroup(5)
            sage: TestSuite(B).run()
            sage: B(1)
            [1 0]
            [0 1]
            sage: B(2)
            TypeError: matrix must have determinant 1
            sage: B([1,1,0,1])
            [1 1]
            [0 1]
            sage: B([1,B.base_ring().gens()[0],0,1])
            [1 1]
            [0 1]
        """
        if not base_ring==base_ring.number_field().ring_of_integers() or not base_ring.number_field().is_totally_imaginary() or not base_ring.number_field().degree()==2:
            raise ValueError("Input (={0}) can not be used to create a Bianchi modular group. " +
                             "Need a ring of integers of an imaginary quadratic number field")
        self._discriminant = base_ring.discriminant()
        # Instance data related to cusps
        self._ncusps = None
        self._cusps = []
        self._ideal_cusp_representatives = []
        self._cusp_normalizing_maps = {}
        self._cusp_normalizing_maps_inverse = {}
        # At the moment we only deal with full level (1)
        self._level = base_ring.fractional_ideal(1)
        super().__init__(degree=Integer(2), base_ring=base_ring,
                         special=True,
                         sage_name=sage_name,
                         latex_string=latex_string,
                         category=Groups().Infinite(),
                         invariant_form=None)

    @cached_method
    def generators(self):
        r"""
        Return a list of generators of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(1).generators()
            [
            [ 0 -1]  [1 1]  [1 a]  [ a  0]
            [ 1  0], [0 1], [0 1], [ 0 -a]
            ]
            sage: BianchiModularGroup(3).generators()
            [
            [ 0 -1]  [          1 1/2*a + 1/2]  [1 a]  [-1/2*a + 1/2            0]
            [ 1  0], [          0           1], [0 1], [           0  1/2*a + 1/2]
            ]
            sage: BianchiModularGroup(5).generators()
            [
            [ 0 -1]  [1 1]  [1 a]
            [ 1  0], [0 1], [0 1]
            ]
            sage: BianchiModularGroup(7).generators()
            [
            [ 0 -1]  [          1 1/2*a + 1/2]  [1 a]
            [ 1  0], [          0           1], [0 1]
            ]


        """
        gens = [self.S()]
        gens.extend(self.T(x) for x in self.base_ring().basis())
        if self._discriminant == -4:
            gens.append(self.E(self.base_ring().number_field().unit_group().gens()[0]))
        elif self._discriminant == -3:
            gens.append(self.E(self.base_ring().number_field().unit_group().gens()[0]))
        return gens

    @cached_method
    def S(self):
        """
        Return the element S = ( 0 & -1 // 1 & 0 ) of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(5).S()
            [ 0 -1]
            [ 1  0]
            sage: BianchiModularGroup(3).S()
            [ 0 -1]
            [ 1  0]

        """
        return self([0, -1, 1, 0])

    @cached_method
    def T(self, a=1):
        """
        Return the element T^a = ( 1 & a // 0 & 1 ) of self.

        INPUT:

        - ``a`` -- integer in number field (default=1)

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: B.T()
            [1 1]
            [0 1]
            sage: a = B.base_ring().gens()[1]
            sage: B.T(a)
            [1 a]
            [0 1]
        """
        return self([1, a, 0, 1])

    @cached_method
    def L(self, a=1):
        """
        Return the element L=( 1 & 0 // a & 1 ) of self.

        INPUT:

        - ``a`` -- integer in number field (default=1)

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: B.L()
            [1 0]
            [1 1]
            sage: a = B.base_ring().gens()[1]
            sage: B.L(a)
            [1 0]
            [a 1]

        """
        return self([1, 0, a, 1])

    @cached_method
    def E(self, u):
        """
        Return the element U=( u & 0 // 0 & u**-1 ) of self.

        INPUT:
        - `u` unit in self.base_ring()

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(5).E(-1)
            [-1  0]
            [ 0 -1]
            sage: BianchiModularGroup(1).E(QuadraticField(-1).unit_group().gens()[0])
            [ a  0]
            [ 0 -a]
            sage: BianchiModularGroup(3).E(QuadraticField(-3).unit_group().gens()[0])
            [-1/2*a + 1/2            0]
            [           0  1/2*a + 1/2]

        """
        return self([u, 0, 0, u ** -1])

    def gens(self):
        r"""
            Return a tuple of generators for this Bianchi modular group.

            The generators need not be minimal. For arguments, see :meth:`~generators`.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(5).gens()
            (
            [ 0 -1]  [1 1]  [1 a]
            [ 1  0], [0 1], [0 1]
            )
            sage: BianchiModularGroup(1).gens()
            (
            [ 0 -1]  [1 1]  [1 a]  [ a  0]
            [ 1  0], [0 1], [0 1], [ 0 -a]
            )


        """
        return tuple(self.generators())

    def ngens(self):
        r"""
            Return the number of generators of self as given by the function 'gens'.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(5).ngens()
            3
            sage: BianchiModularGroup(3).ngens()
            4

        """
        return len(self.generators())

    def gen(self, i):
        r"""
        Return the i-th generator of self, i.e. the i-th element of the
        tuple self.gens().

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(1).gen(3)
            [ a  0]
            [ 0 -a]
            sage: BianchiModularGroup(5).gen(1)
            [1 1]
            [0 1]
        """
        return self.generators()[i]

    def random_element(self, *args, **kwds):
        r"""
        Return a 'random' element of this Bianchi Modular Group.

        INPUT:

        - `args`, `kwds` -- arguments passed to the base ring's random element function
                            and are in turn passed to the random integer function.
                            See the documentation for "ZZ.random_element()" for details.


        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: A = B.random_element()
            sage: A in B
            True
            sage: A #random
            [-61*a + 699     5*a + 2]
            [  -28*a - 1           1]

        """
        a = self.base_ring().random_element(**kwds)
        b = self.base_ring().random_element(**kwds)
        return self.T(a)*self.L(b)

    def level(self):
        """
        Return the level of this Bianchi modular group (currently only (1))

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(5).level()
            Fractional ideal (1)

        """
        return self._level

    def ncusps(self):
        """
        Return number of cusps of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(3).ncusps()
            1
            sage: BianchiModularGroup(5).ncusps()
            2
            sage: BianchiModularGroup(17).ncusps()
            4

        """
        if not self._ncusps:
            self._ncusps = self.base_ring().class_number()
        return self._ncusps

    @cached_method
    def cusps(self):
        """
        A set of cusp representatives of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(3).cusps()
            [Cusp Infinity of Number Field in a with defining polynomial x^2 + 3 with a = 1.732050807568878?*I]
            sage: BianchiModularGroup(5).cusps()
            [Cusp Infinity of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I,
            Cusp [2: a + 1] of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I]
            sage: BianchiModularGroup(17).cusps()
            [Cusp Infinity of Number Field in a with defining polynomial x^2 + 17 with a = 4.123105625617660?*I,
            Cusp [2: a + 1] of Number Field in a with defining polynomial x^2 + 17 with a = 4.123105625617660?*I,
            Cusp [3: a + 1] of Number Field in a with defining polynomial x^2 + 17 with a = 4.123105625617660?*I,
            Cusp [3: a + 2] of Number Field in a with defining polynomial x^2 + 17 with a = 4.123105625617660?*I]

        """
        for a in self.ideal_cusp_representatives():
            logger.debug("Set cusp info for ideal a={0}".format(a))
            if a.is_trivial():
                ca = NFCusp(self.base_ring().number_field(),
                            self.base_ring()(1),
                            self.base_ring()(0),
                            lreps=self.ideal_cusp_representatives())
            else:
                ag = a.gens_reduced()
                ca = NFCusp(self.base_ring().number_field(), ag[0], ag[1],
                            lreps=self.ideal_cusp_representatives())
            self._cusps.append(ca)
            if ca.ideal() != a:
                raise ArithmeticError("Failed to associate a cusp to ideal {0}".format(a))
        return self._cusps

    def ideal_cusp_representatives(self):
        r"""
        Return a list of ideals corresponding to cusp representatives, i.e.
        ideal representatives of ideal classes.

        Note: We choose an ideal of smallest norm in each class.
            If the ideal given by sage is already minimal we return this.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: BianchiModularGroup(3).ideal_cusp_representatives()
            [Fractional ideal (1)]
            sage: BianchiModularGroup(5).ideal_cusp_representatives()
            [Fractional ideal (1), Fractional ideal (2, a + 1)]
            sage: BianchiModularGroup(17).ideal_cusp_representatives()
            [Fractional ideal (1),
            Fractional ideal (2, a + 1),
            Fractional ideal (3, a + 1),
            Fractional ideal (3, a + 2)]

        """
        if not self._ideal_cusp_representatives:
            self._ideal_cusp_representatives = []

            def _find_equivalent_ideal_of_minimal_norm(c):
                for a in self.base_ring().number_field().ideals_of_bdd_norm(c.norm() - 1).items():
                    for ideala in a[1]:
                        if (ideala * c ** -1).is_principal():
                            if c.norm() <= ideala.norm():
                                return c
                            return ideala
                return c

            for ideal_class in self.base_ring().class_group():
                c = ideal_class.ideal().reduce_equiv()
                # NOTE: Even though we use 'reduce_equiv' we are not guaranteed a representative
                #       with minimal **norm**
                #       To make certain we choose a representative of minimal norm explicitly.
                c = _find_equivalent_ideal_of_minimal_norm(c)
                self._ideal_cusp_representatives.append(c)
            # We finally sort all representatives according to norm.
            self._ideal_cusp_representatives.sort(key=lambda x: x.norm())

        return self._ideal_cusp_representatives

    def cusp_representative(self, cusp, return_map=False):
        r"""
        Return a representative cusp and optionally a corresponding map.

        INPUT:

        - ``cusp`` -- cusp
        - ``return_map`` -- bool (default: False)
                            Set to True to also return the map giving the equivalence.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: c1 = NFCusp(B.base_ring().number_field(),0,1)
            sage: B.cusp_representative(c1)
            Cusp Infinity of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: c2 = NFCusp(B.base_ring().number_field(),2,5)
            sage: B.cusp_representative(c2)
            Cusp Infinity of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: c3 = NFCusp(B.base_ring().number_field(),a+2,3)
            sage: B.cusp_representative(c3)
            Cusp [2: a + 1] of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I

        """
        for c in self.cusps():
            if return_map:
                t, B = cusp.is_Gamma0_equivalent(c, self.level(), Transformation=True)
                if t:
                    return c, self(B)
            elif cusp.is_Gamma0_equivalent(c, self.level()):
                return c
        raise ArithmeticError(f"Could not find cusp representative for {cusp}")

    # Functions for working with cusps.

    def cusp_normalizing_map(self, cusp, inverse=False, check=False):
        r"""
        Given a cusp (a:c) Return a matrix A = [[ a ,b ], [c , d]] in SL(2,K) such that
        A(Infinity)=(a:c) and b, d in self.base_ring().ideal(a,c)**-1

        INPUT:

        - ``cusp`` -- Instance of NFCusp
        - ``inverse`` -- bool (default: False) set to True to return the inverse map
        - ``check`` -- bool (default: False) set to True to check the result

        NOTE: The sage function NFCusp.ABmatrix() returns a matrix with determinant which is not
            necessarily equal to 1 even though 1 is a generator of the ideal (1)=(a,c)*(a,c)**-1

        If inverse = True then return A^-1

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B1 = BianchiModularGroup(5)
            sage: B1.cusp_normalizing_map(B1.cusps()[0])
            [1 0]
            [0 1]
            sage: B1.cusp_normalizing_map(B1.cusps()[1])
            [          2 1/2*a - 1/2]
            [      a + 1          -1]
            sage: K = B1.base_ring().number_field()
            sage: a = K.gen()
            sage: B1.cusp_normalizing_map(NFCusp(K,0,1))
            [ 0 -1]
            [ 1  0]
            sage: B1.cusp_normalizing_map(NFCusp(K,1,1))
            [ 1 -1]
            [ 1  0]
            sage: B1.cusp_normalizing_map(NFCusp(K,a,1+a))
            [    a a - 1]
            [a + 1     a]
            sage: B2 = BianchiModularGroup(17)
            sage: B2.cusp_normalizing_map(B2.cusps()[3])
            [          3 1/3*a - 2/3]
            [      a + 2          -2]
            sage: B2.cusp_normalizing_map(B2.cusps()[3],inverse=True)
            [          -2 -1/3*a + 2/3]
            [      -a - 2            3]

        """
        base_nf = self.base_ring().number_field()
        if not isinstance(cusp, NFCusp) or cusp.number_field() != base_nf:
            raise ValueError(f"Input should be a NF cusp defined over {base_nf}!")
        ca, cb = (cusp.numerator(), cusp.denominator())
        if (ca, cb) not in self._cusp_normalizing_maps:
            # First find the equivalent representative
            # crep, B = self.cusp_representative(cusp,return_map=True)
            # crepa,crepb = crep.numerator(),crep.denominator()
            # crep_normalizing_map = self._cusp_normalizing_maps.get((crepa,crepb))
            # if not crep_normalizing_map:
            # Find a normalizing map of the cusp representative
            a, b, c, d = cusp.ABmatrix()
            det = a * d - b * c
            A = Matrix(self.base_ring().number_field(), 2, 2, [a, b / det, c, d / det])
            # A = B.matrix().inverse()*crep_normalizing_map
            if check:
                infinity = NFCusp(self.base_ring().number_field(), 1, 0)
                if infinity.apply(A.list()) != cusp or A.det() != 1:
                    msg = f"Did not get correct normalizing map A={A} to cusp: {cusp}"
                    raise ArithmeticError(msg)
            logger.debug(f"A={0}".format(A))
            logger.debug("A.det()={0}".format(A.det().complex_embeddings()))
            self._cusp_normalizing_maps_inverse[(ca, cb)] = A.inverse()
            self._cusp_normalizing_maps[(ca, cb)] = A
        if inverse:
            return self._cusp_normalizing_maps_inverse[(ca, cb)]
        else:
            return self._cusp_normalizing_maps[(ca, cb)]

    def apply_cusp_normalizing_map(self, cusp, z, inverse=False):
        """
        Apply the cusp normalising map associated with the cusp to an element z

        INPUT:

        - `cusp` - an instance of NFcusp
        - `z` - an element in
                 - the base number field
                 - the set of cusps
                 - ComplexSpaceElement__class
        - `inverse` - set to True if applying the inverse map

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: B = BianchiModularGroup(5)
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],1.0)
            0.500000000000000 - 0.670820393249937*I

            # If we apply a matrix to a an element in K we get back an element in K
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],1)
            -3/10*a + 1/2
            sage: a = B.base_ring().gens()[1]
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],a)
            -29/82*a + 31/82

            # If we apply the matrix to a cusp we return a cusp.
            sage: c = NFCusp(B.base_ring().number_field(),1)
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],c)
            Cusp [-3*a + 5: 10] of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I

            # If we apply the matrix to an element in the complex space we get an element in the complex space 
            sage: z = ComplexSpaceElement([CC(1,0),RR(1)])
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],z)
            [0.409090909090909 - 0.711476174659024*I, 0.0909090909090909]

            # Applying the inverse of a cusp normalising map to the same cusp returns infinity.
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],B.cusps()[1],inverse=True)
            Cusp Infinity of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: c1 = B.cusps()[1]
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],Infinity)
            -1/3*a + 1/3
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],Infinity) == c1
            False
            sage: q = c1.numerator()/c1.denominator()
            sage: B.apply_cusp_normalizing_map(B.cusps()[1],Infinity) == q
            True
            sage: c0, c1 = B.cusps()
            sage: B.apply_cusp_normalizing_map(c1, c0) == c1
            True

        """
        a, b, c, d = self.cusp_normalizing_map(cusp, inverse=inverse).list()
        if z == infinity:
            return a / c
        number_field = self.base_ring().number_field()
        if isinstance(z, NFCusp) and z.number_field() == number_field:
            return z.apply([a, b, c, d])
        if z in number_field:
            return (a * z + b) / (c * z + d)
        if isinstance(z, ComplexSpaceElement__class):
            return z.apply(matrix(2, 2, [a, b, c, d]))
        raise ValueError("Unsupported type for acting with cusp normalizer! (z={0})".format(z))