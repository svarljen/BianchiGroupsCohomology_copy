from sage.structure.element import MultiplicativeGroupElement
#from sage.structure.richcmp cimport richcmp
from sage.rings.all import ZZ
from sage.rings.infinity import infinity
from sage.matrix.matrix_space import MatrixSpace
from sage.misc.cachefunc import cached_method
from sage.structure.richcmp import (op_EQ, op_NE, op_LT, op_LE, op_GT, op_GE)

from upperhalfspace import ComplexSpaceElement__class, UpperHalfSpaceElement__class

class BianchiModularGroupElement(MultiplicativeGroupElement):

    def __init__(self, parent, x, check=True):
        """
        Create an element of a Bianchi Modular Group.

        INPUT:

        - ``parent`` -- an arithmetic subgroup

        - `x` -- data defining a 2x2 matrix over ZZ
                 which lives in parent


        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: from BianchiGroupsCohomology.Bianchimodulargroupelement import BianchiModularGroupElement
            sage: B = BianchiModularGroup(5)
            sage: B.base_ring().gens()
            (1, a)
            sage: a = B.base_ring().gens()[1]
            sage: BianchiModularGroupElement(B,[1,a,0,1])
            [1 a]
            [0 1]
            sage: BianchiModularGroupElement(B,[1,a,0,1]) in B
            True

        """
        if not 'BianchiModularGroup_class' in parent.__class__.__name__:
            raise TypeError("parent (= {0}) must be a Bianchi Modular group".format(parent))
        x = MatrixSpace(parent.base_ring(),2,2)(x, copy=True, coerce=True)
        if x.determinant() != 1:
            raise TypeError("matrix must have determinant 1")
        x.set_immutable()
        MultiplicativeGroupElement.__init__(self, parent)
        self.__x = x

    def __iter__(self):
        """
        Iterate over self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: B.base_ring().gens()
            (1, a)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: list(A)
            [1, a, 0, 1]

        """
        yield self.__x[0, 0]
        yield self.__x[0, 1]
        yield self.__x[1, 0]
        yield self.__x[1, 1]

    def list(self):
        r"""
        List of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: B.base_ring().gens()
            (1, a)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.list()
            [1, a, 0, 1]

        """
        return list(self)

    def __repr__(self):
        r"""
        Return the string representation of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5); B
            Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: B.base_ring().gens()
            (1, a)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1]); A
            [1 a]
            [0 1]

        """
        return repr(self.__x)

    def _latex_(self):
        r"""
        Return the latex representation of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: latex(A)
            \left(\begin{array}{rr}
            1 & \sqrt{-5} \\
            0 & 1
            \end{array}\right)

        """
        return self.__x._latex_()

    def _richcmp_(self, other, op):

        r"""
        Compare self to right, where right is guaranteed to have the same
        parent as self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: S = B([0,-1,1,0])
            sage: T = B([1,1,0,1])
            sage: M = B([3,2,1,1])
            sage: A == S
            False
            sage: A != S
            True
            sage: M >= T
            True
            sage: M < T
            False

        """
        
        if not isinstance(other, BianchiModularGroupElement):
            return NotImplemented
        if op == op_EQ:  # Equal
            return self.__x == other.__x
        elif op == op_NE:  # Not equal
            return self.__x != other.__x
        elif op == op_LT:  # Less than
            return self.__x < other.__x
        elif op == op_LE:  # Less than or equal
            return self.__x <= other.__x
        elif op == op_GT:  # Greater than
            return self.__x > other.__x
        elif op == op_GE:  # Greater than or equal
            return self.__x >= other.__x
        else:
            return NotImplemented

    def __nonzero__(self):
        """
        Return ``True``, since the ``self`` lives in SL(2,Z), which does not
        contain the zero matrix.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: bool(A)
            True
        """
        return True

    def _mul_(self, right):
        """
        Return self * right.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A1 = B([1,a,0,1])
            sage: A2 = B([1,1,0,1])
            sage: C = A1*A2
            sage: C
            [    1 a + 1]
            [    0     1]
            sage: C.parent()
            Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
          
        """
        return self.__class__(self.parent(), self.__x * BianchiModularGroupElement(self.parent(),right).__x, check=False)

    def __invert__(self):
        r"""
        Return the inverse of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.__invert__()
            [ 1 -a]
            [ 0  1]
            sage: S = B([0,-1,1,0])
            sage: S.__invert__()
            [ 0  1]
            [-1  0]
        """
        return self.parent(
                [self.__x[1, 1], -self.__x[0, 1],
                 -self.__x[1, 0], self.__x[0, 0]]
                )

    def matrix(self):
        """
        Return the matrix corresponding to ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.matrix()
            [1 a]
            [0 1]
            sage: type(A.matrix())
            <class 'sage.matrix.matrix_generic_dense.Matrix_generic_dense'>

        """
        return self.__x

    def determinant(self):
        """
        Return the determinant of ``self``, which is always 1.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.determinant()
            1
        """
        return self.base_ring().one()

    def det(self):
        """
        Return the determinant of ``self``, which is always 1.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.det()
            1
        """
        return self.determinant()

    def a(self):
        """
        Return the upper left entry of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.a()
            1
        """
        return self.__x[0, 0]

    def b(self):
        """
        Return the upper right entry of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.b()
            a
        """
        return self.__x[0, 1]

    def c(self):
        """
        Return the lower left entry of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.c()
            0

        """
        return self.__x[1, 0]

    def d(self):
        """
        Return the lower right entry of ``self``.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.d()
            1
        """
        return self.__x[1, 1]
    
    def acton(self, z):
        """
        Return the result of the action of ``self`` on z as a fractional linear
        transformation.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([-1,-a,a+1,a-6])


        An example of A acting on a symbolic variable::

            sage: z = var('z')
            sage: A.acton(z)
            -(z + I*sqrt(5))/(z*(I*sqrt(5) + 1) + I*sqrt(5) - 6)

        An example of A acting on an element of the base field:

            sage: z = a/5 + 1/7
            sage: z in A.base_ring().number_field()
            True
            sage: A.acton(z)
            2063/13729*a - 1734/13729

        An example with complex numbers::

            sage: C.<i> = ComplexField()
            sage: A.acton(i)
            -0.133735296806345 + 0.340367694111036*I

        An example with the cusp infinity::

            sage: A.acton(infinity)
            1/6*a - 1/6

        An example which maps a finite cusp to infinity::

            sage: A.acton(-7/6*a + 1/6)
            +Infinity

        An example acting on the NFCusp elements

            sage: K = A.base_ring().number_field(); K
            Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: c = NFCusp(K,a,1); c
            Cusp [a: 1] of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: A.acton(c)
            22/141*a - 20/141

        another example of acting on a NFCusp element

            sage: c = NFCusp(K,-7/6*a + 1/6); c
            Cusp [-7*a + 1: 6] of Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            sage: A.acton(c)
            +Infinity

        Example acting on points in the upper half-plane

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
            sage: z = UpperHalfSpaceElement([CC(1,1),RR(1)])
            sage: A.acton(z)
            [-0.129914979127367 + 0.352467344170306*I, 0.0113243932635535]
            sage: S = B([0,-1,1,0])
            sage: S.acton(z)
            [-0.333333333333333 + 0.333333333333333*I, 0.333333333333333]

        NOTE: when acting on instances of cusps the return value
        is still an element of the underlying number field or infinity (Note the presence of
        '+', which does not show up for cusp instances)::


        TESTS:

        We cover the remaining case, i.e., infinity mapped to infinity::

            sage: B([1,a,0,1]).acton(infinity)
            +Infinity

        """
        from sage.rings.infinity import is_Infinite, infinity
        if is_Infinite(z):
            if self.c() != 0:
                return self.a() / self.c()
            else:
                return infinity
        if hasattr(z, 'denominator') and hasattr(z, 'numerator'):
            p = z.numerator()
            q = z.denominator()
            P = self.a() * p + self.b() * q
            Q = self.c() * p + self.d() * q
            if not Q and P:
                return infinity
            else:
                return P / Q
        if isinstance(z, ComplexSpaceElement__class):
            return self._acton_complex_space_element(z)
        try:
            return (self.a() * z + self.b()) / (self.c() * z + self.d())
        except:
            raise ValueError(f"Can not apply self to z of type: {type(z)}")

    def _acton_complex_space_element(self, zl: ComplexSpaceElement__class):
        """
        Act on an element of the type ComplexSpaceElement__class
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([-1,-a,a+1,a-6])
            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
            sage: z = UpperHalfSpaceElement([CC(1,1),RR(1)])
            sage: A.acton(z)
            [-0.129914979127367 + 0.352467344170306*I, 0.0113243932635535]
            sage: S = B([0,-1,1,0])
            sage: S.acton(z)
            [-0.333333333333333 + 0.333333333333333*I, 0.333333333333333]
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: w = ComplexSpaceElement([CC(1,1),RR(-1)])
            sage: A.acton(w)
            [-0.129914979127367 + 0.352467344170306*I, -0.0113243932635535]
            sage: S.acton(w)
            [-0.333333333333333 + 0.333333333333333*I, -0.333333333333333]

        """
        bfz = zl[0]
        bfzbar = bfz.conjugate()
        zeta = zl[1]
        cbar = self.c().conjugate()
        dbar = self.d().conjugate()
        denominator = (self.c() * bfz + self.d())*(cbar * bfzbar + dbar) + self.c() * cbar * zeta**2
        if denominator == 0:
            return infinity
        zlist = [ ((self.a()*bfz+self.b())*(cbar*bfzbar+dbar)+self.a()*cbar*zeta**2)/denominator, zeta/denominator]
        return zl.parent()(zlist)

    def __getitem__(self, q):
        r"""
        Fetch entries by direct indexing.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A[1,1]
            1
            sage: A[0,1]
            a
        """
        return self.__x[q]

    def __hash__(self):
        r"""
        Return a hash value.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: hash(A)
            -563626983561368930
        """
        return hash(self.__x)

    def __reduce__(self):
        r"""
        Used for pickling.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.__reduce__()
            (Bianchi Modular Group PSL(2) over Maximal Order in Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I,
            (
            [1 a]
            [0 1]
            ))
            sage: loads(dumps(A)) == A
            True
        """
        return self.parent(), (self.__x,)

    def trace(self):
        r"""
        Return the trace of the trace of the matrix of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: B = BianchiModularGroup(5)
            sage: a = B.base_ring().gens()[1]
            sage: A = B([1,a,0,1])
            sage: A.trace()
            2
            sage: B = B([0,-1,1,0])
            sage: S.trace()
            0
        """
        return self.matrix().trace()