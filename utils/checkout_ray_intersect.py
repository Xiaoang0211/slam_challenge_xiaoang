import numpy as np

def find_ray_intersection(A1, A2, B1, B2):
    """
    Find the intersection of two rays defined as follows:
    
    - The first ray starts at A1 and its direction is perpendicular to (A1, A2). 
      Among the two possible perpendicular directions, the one is chosen that 
      points toward B1.
      
    - The second ray starts at B1 and its direction is from B2 to B1 (i.e. B2->B1).
    
    Parameters:
      A1, A2, B1, B2: array-like or tuples of (x, y) coordinates.
      
    Returns:
      The intersection point as a NumPy array [x, y] if it exists on both rays,
      or None if the rays do not intersect (or are parallel).
    """
    
    # Convert points to NumPy arrays (float type for precision)
    A1 = np.array(A1, dtype=float)
    A2 = np.array(A2, dtype=float)
    B1 = np.array(B1, dtype=float)
    B2 = np.array(B2, dtype=float)
    
    # Compute the direction vector from A1 to A2.
    dA = A2 - A1
    
    # Compute a candidate perpendicular vector: (-dA_y, dA_x)
    perp_candidate = np.array([-dA[1], dA[0]])
    
    # Choose the perpendicular that points towards B1.
    # We do this by checking the dot product with (B1 - A1).
    if np.dot(perp_candidate, B1 - A1) < 0:
        n = -perp_candidate
    else:
        n = perp_candidate
    
    # For the second ray, its direction is from B2 to B1.
    dB = B1 - B2
    
    # Define the 2D cross product
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    
    # Let v be the vector from A1 to B1.
    v = B1 - A1
    
    # Solve for t and s in: A1 + t*n = B1 + s*dB.
    # Compute denominator (n x dB)
    denom = cross(n, dB)
    if np.abs(denom) < 1e-10:
        # The rays are parallel (or nearly so)
        return None
    
    # Compute parameters using cross products.
    t = cross(v, dB) / denom
    s = cross(v, n) / denom
    
    # Check that the intersection lies on both rays (t and s must be non-negative)
    if t < 0 or s < 0:
        return None
    
    # Intersection point
    intersection = A1 + t * n
    return intersection

if __name__=="__main__": # Example usage
  A1 = (0, 4)
  A2 = (0, 0)
  B1 = (1, 2)
  B2 = (1, 0)

  print("Intersection:", find_ray_intersection(A1, A2, B1, B2))
