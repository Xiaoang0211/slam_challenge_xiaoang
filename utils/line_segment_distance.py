import numpy as np

def point_to_segment_distance(P, A, B):
    """
    Compute the distance from point P to segment AB.
    """
    AB = B - A
    AP = P - A
    ab2 = np.dot(AB, AB)
    
    # If A and B are the same point, return the distance from P to A.
    if ab2 == 0:
        return np.linalg.norm(P - A)
    
    # Projection factor of P onto AB
    t = np.dot(AP, AB) / ab2
    # Clamp t to the range [0, 1]
    t = np.clip(t, 0, 1)
    projection = A + t * AB
    return np.linalg.norm(P - projection)

def orientation(P, Q, R):
    """
    Compute the orientation of the triplet (P, Q, R).
    Returns the scalar cross product which indicates:
      - Positive if counter-clockwise,
      - Negative if clockwise,
      - Zero if collinear.
    """
    return np.cross(Q - P, R - P)

def on_segment(P, Q, R):
    """
    Check if point Q lies on segment PR.
    """
    return (min(P[0], R[0]) <= Q[0] <= max(P[0], R[0]) and
            min(P[1], R[1]) <= Q[1] <= max(P[1], R[1]))

def segments_intersect(A, B, C, D):
    """
    Check if segments AB and CD intersect.
    """
    o1 = orientation(A, B, C)
    o2 = orientation(A, B, D)
    o3 = orientation(C, D, A)
    o4 = orientation(C, D, B)
    
    # General case: segments intersect if orientations differ
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    
    # Special cases: check if any endpoint lies on the other segment
    if np.isclose(o1, 0) and on_segment(A, C, B): return True
    if np.isclose(o2, 0) and on_segment(A, D, B): return True
    if np.isclose(o3, 0) and on_segment(C, A, D): return True
    if np.isclose(o4, 0) and on_segment(C, B, D): return True
    
    return False

def segment_distance(A, B, C, D):
    """
    Compute the closest distance between segments AB and CD.
    If the segments intersect, the distance is 0.
    """
    if segments_intersect(A, B, C, D):
        return 0.0
    
    # Compute distances from endpoints to the opposite segment.
    distances = [
        point_to_segment_distance(A, C, D),
        point_to_segment_distance(B, C, D),
        point_to_segment_distance(C, A, B),
        point_to_segment_distance(D, A, B)
    ]
    return min(distances)

# Test example
if __name__ == "__main__":
    A = np.array([46.3, 288.8])
    B = np.array([96.0, 276.8])
    C = np.array([48.0, 290.8])
    D = np.array([128.6, 264.8])
    
    dist = segment_distance(A, B, C, D)
    print("The closest distance between the segments is:", dist)
