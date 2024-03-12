import sage
from sage.rings.real_mpfr import RealField
from sage.rings.real_mpfr import RealField_class
from sage.rings.real_mpfr import RealNumber
from sage.rings.real_mpfr import is_RealNumber
from sage.structure.element import Element
from sage.rings.all import Integer, CC, RR
from sage.rings.infinity import Infinity
from sage.structure.parent import Parent
from sage.structure.element import parent
from sage.rings.complex_mpfr import ComplexNumber
from sage.rings.complex_mpc import MPComplexNumber, MPComplexField_class
from sage.rings.complex_mpc import MPComplexField
from sage.modules.free_module_element import vector
from sage.categories.map import Map
from sage.structure.richcmp import (op_EQ, op_NE)

def ComplexSpace(**kwds):
    r"""
    Construct a 3d-space CxR.

    INPUT:

    TODO: possibly account for base imaginary quadratic field, to account for cusps

    EXAMPLES:

        sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
        sage: ComplexSpace()
        Complex space

    """
    return ComplexSpace__class(**kwds)

def UpperHalfSpace(**kwds):
    r"""
    Construct a 3d space Cx(R>=0)

    INPUT:

    TODO: possibly account for base imaginary quadratic field, to account for cusps

    EXAMPLES::

        sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace
        sage: UpperHalfSpace()
        Upper half-space

    """
    return UpperHalfSpace__class(**kwds)

def ComplexSpaceElement(zl,**kwds):
    """
    Construct an element in the complex space.

    INPUT:

        - ``zl`` -- input (bfz, zeta) in Cx(R>=0) to construct a point
        - ``kwds`` -- dict.

    OUTPUT:
        - Element of the type ComplexSpaceElement__class

    EXAMPLES::

        sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
        sage: zl=ComplexSpaceElement([CC(1,1),RR(-1)]); zl
        [1.00000000000000 + 1.00000000000000*I, -1.00000000000000]
        sage: ComplexSpaceElement([1+I,-1])
        [1.00000000000000 + 1.00000000000000*I, -1.00000000000000]
        sage: a=QuadraticField(-5).gen(); ComplexSpaceElement([a,RR(1)])
        [2.23606797749979*I, 1.00000000000000]
        sage: ComplexSpaceElement(CC(0,1))
        ValueError: Need a list of a complex and a real number
        sage: ComplexSpaceElement([CC(0,1),CC(0,1)])
        TypeError: unable to convert '1.00000000000000*I' to a real number
        sage: ComplexSpaceElement([RR(1),RR(1)])
        [1.00000000000000, 1.00000000000000]

    """
    if isinstance(zl,ComplexSpaceElement__class):
        parent = kwds.get('parent')
        if parent is zl.parent():
            return zl
        if parent:
            return ComplexSpaceElement__class(list(zl), parent=parent)
        return zl
    # Get precision in the first hand from kwds, second from z and third set default to 53 bits
    prec_CC = kwds.get('prec_CC',getattr(zl,'prec_CC',lambda : 53)())
    prec_RR = kwds.get('prec_RR',getattr(zl,'prec_RR',lambda : 53)())
    if not (isinstance(zl, list)):
        raise ValueError("Need a list of a complex and a real number")
    if isinstance(zl,list) and not (isinstance(zl[0],(ComplexNumber,MPComplexNumber)) and isinstance(zl[1],RealNumber)):
        zl = [MPComplexField(prec_CC)(zl[0]), RealField(prec_RR)(zl[1])]
    return ComplexSpaceElement__class(zl,**kwds)

def UpperHalfSpaceElement(zl, **kwds):
    """
    Construct an element in the upper half space.

    INPUT:

        - ``zl`` -- input (bfz, zeta) in Cx(R>=0) to construct a point
        - ``kwds`` -- dict.

    OUTPUT:
        - Element of the type UpperHalfSpaceElement__class

    EXAMPLES::

        sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
        sage: UpperHalfSpaceElement([1+I,1])
        [1.00000000000000 + 1.00000000000000*I, 1.00000000000000]
        sage: UpperHalfSpaceElement([1+I,-1])
        ValueError: Point [1.00000000000000 + 1.00000000000000*I, -1.00000000000000] not in upper half-space!

    """
    if isinstance(zl,UpperHalfSpaceElement__class):
        parent = kwds.get('parent')
        if parent is zl.parent():
            return zl
        if parent:
            return UpperHalfSpaceElement__class(list(zl), parent=parent)
        return zl
    prec_CC = kwds.get('prec_CC',getattr(zl,'prec_CC',lambda : 53)())
    prec_RR = kwds.get('prec_RR',getattr(zl,'prec_RR',lambda : 53)())
    if not isinstance(zl, list):
        raise ValueError("Need a list of a complex and a real number")
    if isinstance(zl,list) and not (isinstance(zl[0],(ComplexNumber,MPComplexNumber)) or isinstance(zl[1],RealNumber)):
        zl = [MPComplexField(prec_CC)(zl[0]), RealField(prec_RR)(zl[1])]
    if 'parent' not in kwds:
        kwds['parent'] = UpperHalfSpace()
    return UpperHalfSpaceElement__class(zl,**kwds)

class ComplexSpace__class(Parent):

    def __init__(self, **kwds):
        r"""
        Class for a complex space CxR

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace__class
            sage: ComplexSpace__class()
            Complex space

        """
        Parent.__init__(self)
        self._coerce_from_hash = {}

    def __hash__(self):
        """
        Return hash of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: hash(ComplexSpace()) == hash('Complex space')
            True

        """
        return hash(str(self))

    def construction(self):
        r"""
        No functor exists here but this needs to be defined for coercion to work properly.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: ComplexSpace().construction() is None
            True

        """
        return None

    def _coerce_map_from_(self, S):
        r"""
        Coerce maps from S to self.
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: H = ComplexSpace()
            sage: H._coerce_map_from_(ZZ)
            Generic map:
            From: Integer Ring
            To:   Complex space
            sage: H._coerce_map_from_(QuadraticField(-5))
            Generic map:
            From: Number Field in a with defining polynomial x^2 + 5 with a = 2.236067977499790?*I
            To:   Complex space
            sage: H._coerce_map_from_(ZZ)([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: H._coerce_map_from_(ZZ)(1)
            ValueError: Need a list of a complex and a real number

        """
        if self._coerce_from_hash is None:
            self.init_coerce(False)
        if isinstance(S, type(self)):
            from sage.categories.homset import Hom
            morphism = Hom(self, self).identity()
            morphism._is_coercion = True
            self._coerce_from_hash[S] = morphism
            return morphism
        try:
            morphism = AnytoCS(S,self)
            self._coerce_from_hash[S] = morphism
            return morphism
        except:
            pass
        return super(ComplexSpace__class,self)._internal_coerce_map_from(S)
    
    def _an_element_(self):
        r"""
        Create a typical element of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: ComplexSpace()._an_element_()
            [ - 1.00000000000000*I, 1.00000000000000]

        """
        return self._element_constructor_([CC(0,1), RR(1)])

    def __eq__(self, other):
        r"""
        Check if self is equal to other

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: ComplexSpace() == ComplexSpace()
            True

        """
        if not isinstance(other,type(self)):
            return False
        return True

    def __str__(self):
        r"""
        String representation of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace__class
            sage: ComplexSpace__class()
            Complex space

        """
        return f"Complex space"

    def __repr__(self):
        """
        Representation of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace__class
            sage: ComplexSpace__class()
            Complex space

        """
        return str(self)

    def __reduce__(self):
        r"""
        Prepare self for pickling

        TESTS::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: loads(dumps(H)) == H
            True

        """
        return ComplexSpace, ()

    def _element_constructor_(self, zl, **kwds):
        r"""

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: ComplexSpace()._element_constructor_([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: ComplexSpace()._element_constructor_([1+I,-1])
            [1.00000000000000 + 1.00000000000000*I, -1.00000000000000]
        """
        kwds['parent'] = self
        return ComplexSpaceElement(zl, **kwds)

    def coerce(self, x):
        r"""
        Coerce x to an element of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpace
            sage: ComplexSpace().coerce([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: ComplexSpace().coerce([1+I,-1])
            [1.00000000000000 + 1.00000000000000*I, -1.00000000000000]

        """
        return self._element_constructor_(x)
    
class UpperHalfSpace__class(ComplexSpace__class):
    r"""
    Class for elements in a upper half-space including the boundary (i.e. Cx(R>=0)).

    EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace__class
            sage: UpperHalfSpace__class()
            Upper half-space

    """

    def _element_constructor_(self, zl, **kwds):
        r"""
        Construct an element of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace
            sage: UpperHalfSpace()._element_constructor_([1+I,-1])
            ValueError: Point [1.00000000000000 + 1.00000000000000*I, -1.00000000000000] not in upper half-space!
            sage: UpperHalfSpace()._element_constructor_([1+I,1])
            [1.00000000000000 + 1.00000000000000*I, 1.00000000000000]


        """
        kwds['parent'] = self
        return UpperHalfSpaceElement(zl, **kwds)

    def _an_element_(self):
        r"""
        Create a typical element of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace
            sage: UpperHalfSpace()._an_element_()
            [ - 1.00000000000000*I, 1.00000000000000]

        """
        return self._element_constructor_([CC(0,1),RR(1)])

    def __str__(self):
        r"""
        String representation of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace__class
            sage: UpperHalfSpace__class()
            Upper half-space

        """
        return f"Upper half-space"

    def __repr__(self):
        """
        Representation of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace__class
            sage: UpperHalfSpace__class()
            Upper half-space

        """
        return str(self)

    def __reduce__(self):
        r"""
        Prepare self for pickling

        TESTS::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpace
            sage: H = UpperHalfSpace()
            sage: loads(dumps(H)) == H
            True

        """
        return UpperHalfSpace, ()

class ComplexSpaceElement__class(Element):
    r"""
    Class of elements in 3d-space CxR
    with additional ring structure given by:
    - component-wise multiplication and division

    EXAMPLES::

        sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
        sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)]); zl
        [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]
        sage: zl.parent()
        Complex space

    """

    Parent = ComplexSpace__class

    def __init__(self, zl, verbose=0, *argv, **kwds):
        r"""
        Init self from zl=(bfz, zeta) in CxR
        Currently we only work with double (53 bits) precision.

        INPUT:

        - `zl` - list of the form (bfz, zeta) with `bfz` complex number and `zeta` real number

        EXAMPLES:

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class, ComplexSpace, ComplexSpaceElement
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)]); zl
            [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]
            sage: H = ComplexSpace()
            sage: ComplexSpaceElement([1+I,1],parent=H)
            [1.00000000000000 + 1.00000000000000*I, 1.00000000000000]
            sage: ComplexSpaceElement([1,I])
            TypeError: unable to convert '1.00000000000000*I' to a real number

        """
        self._verbose = verbose
        bfz = zl[0]
        zeta = zl[1]
        if verbose>0:
            print("in __init__")
        if not isinstance(zl, list):
            raise ValueError("Need a list to init")
        if not isinstance(bfz,(MPComplexNumber,ComplexNumber)):
            raise ValueError("Need a complex number to init")
        if not is_RealNumber(RR(zeta)):
            raise ValueError("Need a real number to init")   
        parent = kwds.get('parent')
        if not parent:
            parent = ComplexSpace()
        super().__init__(parent)
        self._prec_CC = bfz.prec()
        self._base_ring_CC = MPComplexField(self._prec_CC)
        self._prec_RR = zeta.prec()
        self._base_ring_RR = RealField(self._prec_RR)
        if verbose>0:
            print(bfz, type(bfz), zeta, type(zeta))
        if isinstance(bfz,ComplexNumber) and isinstance(zeta,RealNumber):
            self._z = [self._base_ring_CC(bfz),self._base_ring_RR(zeta)]
        else:
            self._z = zl
        self._x = bfz.real()
        self._y = bfz.imag()
        self._zeta = zeta
        self._bfz = bfz
        if zeta>=0:
            self._is_in_upper_half_space = True
        else:
            self._is_in_upper_half_space = False
        self._imag_norm = 0.0
        self._real_norm = 0.0
        self._norm = 0.0
        self._abs_square_norm = 0.0
        self._imag_norm_set = 0
        self._real_norm_set = 0
        self._norm_set = 0
        self._abs_square_norm_set = 0

    def _cache_key(self):
        """
        Cache key for self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl._cache_key()
            ('ComplexSpaceElement__class',
            (1.00000000000000 + 2.00000000000000*I, 3.00000000000000))

        """
        return (self.__class__.__name__,tuple(self._z))

    def __reduce__(self):
        r"""
        Prepare self for pickling

        TESTS::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: zl = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: loads(dumps(zl)) == zl
            True

        """
        return ComplexSpaceElement, (self.z(),)

    def __hash__(self):
        """
        Hash of self.

        EXAMPLES::

        sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
        sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
        sage: hash(zl)
        -7664962545124311198

        """
        return hash(self._cache_key())

    def z(self):
        r"""
        Return the list [complex number, real number] of this element.
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.z()
            [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]

        """
        return self._z

    def is_in_upper_half_space(self):
        """
        Return True if zeta>=0, False otherwise

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.is_in_upper_half_space()
            True
            sage: zlneg = ComplexSpaceElement__class([CC(1,2),RR(-3)])
            sage: zlneg.is_in_upper_half_space()
            False

        """
        return bool(self._is_in_upper_half_space)

    def as_upper_half_space_element(self):
        r"""
        Return a copy of self with type UpperHalfPlaneProductElement__class

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.as_upper_half_space_element()
            [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]
            sage: zlneg = ComplexSpaceElement__class([CC(1,2),RR(-3)])
            sage: zlneg.as_upper_half_space_element()
            ValueError: Cannot convert self to element in upper half-space.
        """
        if not self.is_in_upper_half_space():
            raise ValueError("Cannot convert self to element in upper half-space.")
        return UpperHalfSpaceElement(self._z)

    def is_zero(self):
        r"""
        Return true if all components of self is zero

        EXAMPLES:

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.is_zero()
            False
            sage: z0 = ComplexSpaceElement__class([CC(0),RR(0)])
            sage: z0.is_zero()
            True
        """

        return all([z==0 for z in self])

    def base_ring_CC(self):
        r"""
        Base ring of self. (bfz)

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.base_ring_CC()
            Complex Field with 53 bits of precision
        """
        return self._base_ring_CC

    def prec_CC(self):
        r"""
        The precision of self. (bfz)


        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.prec_CC()
            53
        """
        return self._prec_CC
    
    def base_ring_RR(self):
        r"""
        Base ring of self. (zeta)

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.base_ring_RR()
            Real Field with 53 bits of precision
        """
        return self._base_ring_RR

    def prec_RR(self):
        r"""
        The precision of self. (zeta)


        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.prec_RR()
            53
        """
        return self._prec_RR

    def real(self):
        """
        Real parts of self (bfz)
        
        EXAMPLES::    
    
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.real()
            1.00000000000000
            
        """
        return self._x

    def imag(self):
        """
        Imaginary parts of self (bfz)


        EXAMPLES::    
    
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.imag()
            2.00000000000000

        """
        return self._y

    def __copy__(self):
        """
        Copy self.
        
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: wl = copy(zl); wl
            [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]
            sage: wl==zl
            False #This shouldn't happen. I think the problem is that the operator == is not defined on this class and needs cython (something about rich comparisons)
        """
        return self.__class__(self._z, verbose=self._verbose)

    def __repr__(self):
        """
        String representation of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: repr(zl)
            '[1.00000000000000 + 2.00000000000000*I, 3.00000000000000]'

        """
        return str(list(self))

    def __getitem__(self,i):
        """
        Get the items of self.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl[0]
            1.00000000000000 + 2.00000000000000*I
            sage: zl[1]
            3.00000000000000
            sage: zl[2]
            IndexError:
        """
        if isinstance(i, (int, Integer)) and 0 <= i < 2:
            return self._z[i]
        else:
            raise IndexError

    def _is_equal(self, other):
        """
        Return 1 if ``self`` is equal to ``other``
        
        INPUT:
        - ``other`` - Element of the type ``ComplexSpaceElement__class``
        
        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: w1 = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: w2 = ComplexSpaceElement__class([CC(0,1),RR(-1)])
            sage: w3 = [CC(0,1),RR(-1)]
            sage: zl._is_equal(zl)
            1
            sage: zl._is_equal(w1)
            1
            sage: zl._is_equal(w2)
            0
            sage: zl._is_equal(w3)
            ValueError: Other must be of the type ComplexSpaceElement__class!

        """
        if not isinstance(other, ComplexSpaceElement__class):
            raise ValueError("Other must be of the type ComplexSpaceElement__class!")
        if self._x != other._x or self._y != other._y or self._zeta != other._zeta:
            return 0
        return 1

    def abs(self):
        """
        Return the element consisting of the absolute value of all elements.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.abs()
            [2.23606797749979, 3.00000000000000]
        """
        return ComplexSpaceElement([abs(self._bfz), abs(self._zeta)])

    def abs_square_norm(self):
        r"""
        Return the norm of |z|^2 
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.abs_square_norm()
            14.0000000000000
            
        """
        if not self._abs_square_norm_set:
            self._abs_square_norm = self.base_ring_CC().base_ring()(1) * (self._x**2 + self._y**2+ self._zeta**2)
            self._abs_square_norm_set = 1
        return self._abs_square_norm

    def norm(self):
        """
        Return the product of all components of self.
        
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.norm()
            3.00000000000000 + 6.00000000000000*I

        """
        if self._norm_set==0:
            self._norm = self.base_ring_CC().base_ring()(1)*self._bfz*self._zeta
            self._norm_set=1
        return self._norm

    def vector(self):
        r"""
        Return self as a vector
        
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.vector()
            (1.00000000000000 + 2.00000000000000*I, 3.00000000000000)

        """
        return vector(self)
    
    def _richcmp_(self, right, op):
        """
        Compare self with other

        INPUT:
        - `right`
        - `op`

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = copy(z)
            sage: w == z
            True
            sage: w != z
            False
            sage: w > z
            NotImplementedError: Ordering of points in H3 is not implemented!
        """
        res = 1
        if op != op_EQ and op != op_NE:
           raise NotImplementedError("Ordering of points in H3 is not implemented!")
        if type(self) != type(right):
            res = 0
        else:
            res = self._is_equal(right)
        if op == op_NE:
            res = 1 - res
        return bool(res)

    def change_ring(self,RComplex,RReal):
        """
        Change the base ring of self.

        INPUT:
        - `RComplex` -- MPComplexField
        - `RReal` -- RealField

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.change_ring(MPComplexField(106),RealField(106)); zl
            [1.000000000000000000000000000000 + 2.000000000000000000000000000000*I, 3.000000000000000000000000000000]
            sage: zl.change_ring(MPComplexField(26),RealField(26)); zl
            [1.000000 + 2.000000*I, 3.000000]

        """
        if not isinstance(RComplex,MPComplexField_class):
            raise ValueError(f"Can not coerce self into {RComplex}")
        if not isinstance(RReal, RealField_class):
            raise ValueError(f"Can not coerce self into {RReal}")
        self.set_prec(RComplex.prec(), RReal.prec())

    def set_prec(self, prec_CC, prec_RR):
        """
        Change the precision of self.
        
        INPUT:
        - `prec_CC` - precision of complex plane
        - `prec_RR` - precision of real line
        
        EXAMPLES::
        
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement__class
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: zl.set_prec(106,106); zl
            [1.000000000000000000000000000000 + 2.000000000000000000000000000000*I, 3.000000000000000000000000000000]
            sage: zl.set_prec(26,26); zl
            [1.000000 + 2.000000*I, 3.000000]

        """
        self._base_ring_CC = MPComplexField(prec_CC)
        self._base_ring_RR = RealField(prec_RR)
        self._bfz = self._base_ring_CC(self._bfz)
        self._x = self._base_ring_CC.base_ring()(self._x)
        self._y = self._base_ring_CC.base_ring()(self._y)
        self._zeta = self._base_ring_RR(self._zeta)
        self._z = [self._base_ring_CC(self._bfz),self._base_ring_RR(self._zeta)]

    @classmethod
    def _extract_left_right_parent(cls,left,right):
        """
        Convert the argument left and right to elements of the type ComplexSpaceElement_class

        INPUT:

        - ``left`` -- object
        - ``right`` -- object

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSaceElement__class, ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = ComplexSpaceElement([CC(-1,1),RR(-1)])
            sage: ComplexSpaceElement__class._extract_left_right_parent(z,w)
            ([1.00000000000000 + 2.00000000000000*I, 3.00000000000000],
            [-1.00000000000000 + 1.00000000000000*I, -1.00000000000000],
            Complex space)
            sage: ComplexSpaceElement__class._extract_left_right_parent([CC(0,1),RR(1)],w)
            ([1.00000000000000*I, 1.00000000000000],
            [-1.00000000000000 + 1.00000000000000*I, -1.00000000000000],
            Complex space)

        """
        if isinstance(left, ComplexSpaceElement__class):
            right = ComplexSpaceElement(right, prec_CC=left.prec_CC(), prec_RR=left.prec_RR())
            parent = left.parent()
        elif isinstance(right, ComplexSpaceElement__class):
            left = ComplexSpaceElement(left, prec_CC=right.prec_CC(), prec_RR=right.prec_RR())
            parent = right.parent()
        else:
            raise ValueError("One of left or right must be of the type ComplexSpaceElement__class!")
        return left,right,parent

    def _add_(self, other):
        """
        Add ``other`` to ``self`` and convert to ``parent``. Used by the ``__add__`` method.
        
        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement, ComplexSpaceElement__class, UpperHalfSpaceElement__class
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = ComplexSpaceElement([CC(0,-1),RR(1)])
            sage: z+w
            [1.00000000000000 + 1.00000000000000*I, 4.00000000000000]

        """
        if self._prec_CC != other[0].prec() or self._prec_RR != other[1].prec():
            raise TypeError
        return self.parent([self._bfz + other[0], self._zeta + other[1]])

    def _neg_(self):
        """
        Negative of self.
        
        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: -z
            [-1.00000000000000 - 2.00000000000000*I, -3.00000000000000]
       
        """
        return self.parent([-self._bfz, -self._zeta])

    def _sub_(self,other):
        """
        Subtract ``other`` from ``self``

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = ComplexSpaceElement([CC(0,-1),RR(1)])
            sage: z-w
            [1.00000000000000 + 3.00000000000000*I, 2.00000000000000]
            sage: w-z
            [-1.00000000000000 - 3.00000000000000*I, -2.00000000000000]

        """
        if self._prec_CC != other[0].prec() or self._prec_RR != other[1].prec():
            raise TypeError
        # Try to make an element of the same class as self and if it doesn't work, coerce to complex plane product element
        try:
            return self.parent([self._bfz - other[0], self._zeta - other[1]])
        except ValueError:
            return ComplexSpaceElement__class([self._bfz - other[0], self._zeta - other[1]])

    def _mul_(self, other):
        r"""
        Multiply ``self`` by ``other`` and convert to ``parent``.
        
        INPUT:
        - `other` - element of product of complex planes
        - `parent` - a parent class 
        
        EXAMPLES::
            
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = ComplexSpaceElement([CC(0,-1),RR(1)])
            sage: z*w
            [2.00000000000000 - 1.00000000000000*I, 3.00000000000000]
            sage: w*z
            [2.00000000000000 - 1.00000000000000*I, 3.00000000000000]
            sage: z*ComplexSpaceElement([CC(3,2),RR(-4)])
            [-1.00000000000000 + 8.00000000000000*I, -12.0000000000000]

        """
        if self._prec_CC != other[0].prec() or self._prec_RR != other[1].prec():
            raise TypeError
        try:
            new_element = [self._bfz*other[0], self._zeta*other[1]]
            return self.parent(new_element)
        except ValueError:
            return ComplexSpaceElement__class(new_element)

    def _div_(self, other):
        r"""
        Divide self by other.
        
        INPUT:
        - `other` - element of product of complex planes
        - `parent` - parent class
        
        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: w = ComplexSpaceElement([CC(0,-1),RR(1)])
            sage: z/w
            [-2.00000000000000 + 1.00000000000000*I, 3.00000000000000]
            sage: w/z
            [-0.400000000000000 - 0.200000000000000*I, 0.333333333333333]
            sage: z/ComplexSpaceElement([CC(0),RR(1)])
            ZeroDivisionError: Can not divide by zero!
            sage: z/ComplexSpaceElement([CC(1),RR(0)])
            ZeroDivisionError: Can not divide by zero!

        """
        if self._prec_CC != other[0].prec() or self._prec_RR != other[1].prec():
            raise TypeError
        if any([z == 0 for z in other]):
            raise ZeroDivisionError("Can not divide by zero!")
        new_element = [self._bfz/other[0], self._zeta/other[1]]
        try:
            return self.parent()(new_element)
        except ValueError:
            return ComplexSpaceElement__class(new_element)

    def __pow__(self, power):
        """
        Self to the power 'power' defined component-wise:
        If `power` is a scalar:
                z^power = (z_1^power,...,z_n^power)
        If `power` is an element of this class:
                z^power = (z_1^power_1,...,z_n^power_n)

        INPUT:

        - `power` -- complex number (will be coerced to the base_ring of self)

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: z**2
            [-3.00000000000000 + 4.00000000000000*I, 9.00000000000000]
            sage: z**(-1)
            [0.200000000000000 - 0.400000000000000*I, 0.333333333333333]
            sage: z**[2,3]
            [-3.00000000000000 + 4.00000000000000*I, 27.0000000000000]
            sage: w = ComplexSpaceElement([CC(0,-1),RR(1)])
            sage: z**w
            [2.09777264942116 - 2.18044142398664*I, 3.00000000000000]
            """
        
        try:
            if not isinstance(power,(list,ComplexSpaceElement__class)):
                power = [power]*2
            power = self.parent()(power)
            if any(power[i].real() < 0 and self[i] == 0 for i in range(2)):
                raise ZeroDivisionError("Can not divide component by 0!")
            new_element = [self._bfz**power[0], self._zeta**power[1]]
            try:
                return self.parent()(new_element)
            except ValueError:
                return ComplexSpaceElement__class(new_element)
        except TypeError as e:
            raise TypeError(f"Can not coerce {power} to self.base_ring(). {e}")

    def apply(self, m):
        r"""
        Apply the matrix m to self. 
    
        INPUT:
        
        - ``m`` -- matrix
        
        EXAMPLES::
            
            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement, UpperHalfSpaceElement
            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: zl = ComplexSpaceElement__class([CC(1,2),RR(3)])
            sage: B5 = BianchiModularGroup(5)
            sage: B5.gens()
            (
            [ 0 -1]  [1 1]  [1 a]
            [ 1  0], [0 1], [0 1]
            )
            sage: zl.apply(B5.gens()[0])
            [-0.0714285714285714 + 0.142857142857143*I, 0.214285714285714]
            sage: zl.apply(B5.gens()[1])
            [2.00000000000000 + 2.00000000000000*I, 3.00000000000000]
            sage: zl.apply(B5.gens()[2])
            [1.00000000000000 + 4.23606797749979*I, 3.00000000000000]
    
        """
        try:
            aa, bb, cc, dd = m.list()
            a = CC(aa)
            b = CC(bb)
            c = CC(cc)
            d = CC(dd)
        except (AttributeError,ValueError):
            raise ValueError("Need a 2 x 2 matrix or object that contains a list of 4 elements to act on self.")
        
        cbar = c.conjugate()
        dbar = d.conjugate()
        denominator = (c*self._bfz+d)*(cbar*self._bfz.conjugate()+dbar) +c*cbar*self._zeta**2
        if denominator == 0:
            return Infinity
        zlist = [ ((a*self._bfz+b)*(cbar*self._bfz.conjugate()+dbar)+a*cbar*self._zeta**2)/denominator, self._zeta*abs(a*d-b*c)/denominator]
        return self.parent()(zlist)

    def as_ComplexSpaceElement(self):
        """
        Convert self to an element in the of complex space.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
            sage: z = UpperHalfSpaceElement([1+I,1])
            sage: z.as_ComplexSpaceElement()
            [1.00000000000000 + 1.00000000000000*I, 1.00000000000000]

        """
        if self.__class__ == ComplexSpaceElement__class:
            return self
        return ComplexSpaceElement(list(self))

    def reflect(self):
        r"""
        Reflection of self in the imaginary axis.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import ComplexSpaceElement, ComplexSpaceElement__class
            sage: z = ComplexSpaceElement([CC(1,2),RR(3)])
            sage: z.reflect()
            [1.00000000000000 - 2.00000000000000*I, 3.00000000000000]

        """
        znew = [CC(self._x, -self._y), self._zeta]
        return self.parent()(znew)
    
class UpperHalfSpaceElement__class(ComplexSpaceElement__class):
    r"""

    Class for elements of the upper half space.

    """

    def __init__(self, zl, verbose=0, *argv, **kwds):
        r"""
        Init self from list of a complex number and a real number.

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement__class, UpperHalfSpace, UpperHalfSpaceElement
            sage: zl = UpperHalfSpaceElement__class([CC(1,2),RR(3)]); zl
            [1.00000000000000 + 2.00000000000000*I, 3.00000000000000]
            sage: H = UpperHalfSpace()
            sage: UpperHalfSpaceElement([1+I,1], parent=H)
            [1.00000000000000 + 1.00000000000000*I, 1.00000000000000]

        """
        super().__init__(zl, verbose=verbose, *argv, **kwds)
        if not self.is_in_upper_half_space():
            raise ValueError("Point {0} not in upper half-space!".format(zl))

    def __reduce__(self):
        r"""
        Prepare self for pickling

        TESTS::

            from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
            sage: z = UpperHalfSpaceElement([1+I,1])
            sage: loads(dumps(zl)) == zl
            True

        """
        return UpperHalfSpaceElement, (self.z(),)

    def apply(self, m):
        r"""
        Apply the matrix m to self. 

        INPUT:

        - ``m`` -- matrix

        EXAMPLES::

            sage: from BianchiGroupsCohomology.upperhalfspace import UpperHalfSpaceElement
            sage: from BianchiGroupsCohomology.Bianchimodulargroup import BianchiModularGroup
            sage: z = UpperHalfSpaceElement([CC(1,1),RR(1)])
            sage: B5 = BianchiModularGroup(5)
            sage: B5.gens()
            (
            [ 0 -1]  [1 1]  [1 a]
            [ 1  0], [0 1], [0 1]
            )
            sage: z.apply(B5.gens()[0])
            [-0.333333333333333 + 0.333333333333333*I, 0.333333333333333]
            sage: z.apply(B5.gens()[1])
            [2.00000000000000 + 1.00000000000000*I, 1.00000000000000]
            sage: z.apply(B5.gens()[2])
            [1.00000000000000 + 3.23606797749979*I, 1.00000000000000]

        """
        new_point = ComplexSpaceElement__class(self._z).apply(m)
        return new_point.as_upper_half_space_element()
    
class AnytoCS(Map):
    """
    Maps from 'anything' into the class ComplexSpace
    """

    def __call__(self, x):
        """
        
        EXAMPLES::
            sage: from BianchiGroupsCohomology.upperhalfspace import AnytoCS, ComplexSpace
            sage: H = ComplexSpace()
            sage: AnytoCS(ZZ,H)
            Generic map:
            From: Integer Ring
            To:   Complex space
            sage: AnytoCS(ZZ,H)(1)
            ValueError: Need a list of a complex and a real number
            sage: AnytoCS(ZZ,H)([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: AnytoCS(RR,H)([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: AnytoCS(CC,H)([1,1])
            [1.00000000000000, 1.00000000000000]
            sage: AnytoCS(str,H)(["1","1"])
            [1.00000000000000, 1.00000000000000]
            sage: AnytoCS(str,H)(["a","b"])
            TypeError: unable to convert 'a' to a MPComplexNumber
            sage: AnytoCS(str,H)(["1","b"])
            TypeError: unable to convert 'b' to a real number
        """
        parent = self.codomain()
        return parent._element_constructor_(x)