from sage.rings.number_field.number_field_element import NumberFieldElement

from upperhalfspace import UpperHalfSpaceElement__class

def distance_to_cusp(alpha: NumberFieldElement, beta: NumberFieldElement, zl: UpperHalfSpaceElement__class):
    r"""
    Return the distance from an element in the upper half-space to a cusp. 
    
    INPUT:
    - ``alpha`` - Number field element, numerator of cusp
    - ``beta`` - Number field element, denominator of cusp
    - ``zl`` - Point in the upper half-space
    
    EXAMPLES::
    
        sage: from distance import distance_to_cusp
        sage: from Bianchimodulargroup import BianchiModularGroup
        sage: from upperhalfspace import UpperHalfSpaceElement
        sage: B = BianchiModularGroup(5)
        sage: K = B.base_ring().number_field()
        sage: a = K.gen()
        sage: z = UpperHalfSpaceElement([CC(1,2),RR(3)])
        sage: distance_to_cusp(K(1),K(0),z)
        0.111111111111111
        sage: distance_to_cusp(K(0),K(1),z)
        21.7777777777778
        sage: distance_to_cusp(K(a+2),K(3),z)
        114.174411486674
        sage: distance_to_cusp(K(-3),K(a),z)
        302.379457184489
 
    """
    ideal_alphabeta = alpha.parent().fractional_ideal(alpha,beta)
    cc = alpha.parent().complex_conjugation()
    d = ((alpha*cc(alpha) - cc(alpha)*beta*zl._bfz - alpha*cc(beta)*zl._bfz.conjugate() + beta*cc(beta)*(zl._bfz*zl._bfz.conjugate() + zl._zeta**2))/(zl._zeta * ideal_alphabeta.norm()))**2
    return d