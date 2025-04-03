import numpy as np

# direct extension along the lines (A1, A1) and (B1, B2)
def line_intersection(A1, A2, B1, B2):
    """
    Compute the intersection point of the infinite lines defined by endpoints A1-A2 and B1-B2.
    Additionally, determine whether the intersection point lies on the segments A1-A2 and B1-B2,
    and compute the distances from the intersection point to each of the endpoints.

    Parameters:
        A1, A2, B1, B2: array-like, shape (2,)
            Endpoints of the two lines. Each should be a 2-element list or NumPy array [x, y].

    Returns:
        A tuple containing:
          - intersection: np.ndarray or None
                The intersection point [x, y] if the lines are not parallel; otherwise, None.
          - on_segment_A: bool
                True if the intersection point lies on the segment A1-A2.
          - on_segment_B: bool
                True if the intersection point lies on the segment B1-B2.
          - dist_A1: float or None
                Distance from the intersection point to A1.
          - dist_A2: float or None
                Distance from the intersection point to A2.
          - dist_B1: float or None
                Distance from the intersection point to B1.
          - dist_B2: float or None
                Distance from the intersection point to B2.
                
    If the lines are parallel or coincident (denom is nearly zero), all outputs related to intersection 
    (point, booleans, distances) will be None or False as appropriate.
    """
    # Unpack endpoints
    x1, y1 = A1
    x2, y2 = A2
    x3, y3 = B1
    x4, y4 = B2

    # Compute the denominator using the determinant method
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # If the denominator is zero (or nearly zero), the lines are parallel or coincident
    if np.isclose(D, 0):
        return None, False, False, None, None, None, None

    # Compute determinants for each line
    detA = x1 * y2 - y1 * x2
    detB = x3 * y4 - y3 * x4

    # Compute the intersection coordinates
    x = (detA * (x3 - x4) - (x1 - x2) * detB) / D
    y = (detA * (y3 - y4) - (y1 - y2) * detB) / D
    intersection = np.array([x, y])
    
    # Check if the intersection is on segment A (from A1 to A2)
    A_vec = np.array(A2) - np.array(A1)
    # Avoid division by zero if A1 and A2 are the same point
    if np.allclose(A_vec, 0):
        tA = 0.0
    else:
        tA = np.dot(intersection - A1, A_vec) / np.dot(A_vec, A_vec)
    on_segment_A = (0 <= tA <= 1)
    
    # Check if the intersection is on segment B (from B1 to B2)
    B_vec = np.array(B2) - np.array(B1)
    if np.allclose(B_vec, 0):
        tB = 0.0
    else:
        tB = np.dot(intersection - B1, B_vec) / np.dot(B_vec, B_vec)
    on_segment_B = (0 <= tB <= 1)
    
    # Compute distances from the intersection to each endpoint
    dist_A1 = np.linalg.norm(intersection - A1)
    dist_A2 = np.linalg.norm(intersection - A2)
    dist_B1 = np.linalg.norm(intersection - B1)
    dist_B2 = np.linalg.norm(intersection - B2)
    
    return intersection, on_segment_A, on_segment_B, dist_A1, dist_A2, dist_B1, dist_B2

# extension intersect of the lines, one cross A1 and perpendicular to (A1, A2), the other one cross B1 and perpendicular to (B1, B2)
def find_intersection_perpendicular(A1, A2, B1, B2):
    # Convert points to numpy arrays (float for precision)
    A1 = np.array(A1, dtype=float)
    A2 = np.array(A2, dtype=float)
    B1 = np.array(B1, dtype=float)
    B2 = np.array(B2, dtype=float)
    
    # Compute the direction vectors for each given line
    dA = A2 - A1
    dB = B2 - B1
    
    # Compute perpendicular direction vectors:
    # For a vector (dx, dy), a perpendicular vector is (-dy, dx)
    vA = np.array([-dA[1], dA[0]])
    vB = np.array([-dB[1], dB[0]])
    
    # The intersection satisfies: A1 + t*vA = B1 + s*vB
    # Rearranged: t*vA - s*vB = B1 - A1
    # Form the 2x2 matrix for the coefficients of t and s:
    M = np.array([vA, -vB]).T  # Each column corresponds to t and s respectively.
    rhs = B1 - A1
    
    # Check if the matrix is invertible (i.e., the lines are not parallel)
    det = np.linalg.det(M)
    if np.abs(det) < 1e-10:
        raise ValueError("The lines are parallel or coincident; no unique intersection.")
    
    # Solve for t and s
    t, s = np.linalg.solve(M, rhs)
    
    # Compute the intersection point using the parametric form of the first line
    intersection = A1 + t * vA
    return intersection

if __name__=="__main__": # Example usage
    A1 = np.array([1, 1])
    A2 = np.array([2, 1])
    B1 = np.array([3, 3])
    B2 = np.array([3, 4])

    result = line_intersection(A1, A2, B1, B2)
    if result[0] is not None:
        intersection, on_A, on_B, dA1, dA2, dB1, dB2 = result
        print("Intersection point:", intersection)
        print("On segment A1-A2:", on_A)
        print("On segment B1-B2:", on_B)
        print("Distance to A1:", dA1)
        print("Distance to A2:", dA2)
        print("Distance to B1:", dB1)
        print("Distance to B2:", dB2)
    else:
        print("The lines are parallel or coincident.")