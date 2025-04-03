import numpy as np

def perpendicular_toward_target(d, from_point, target_segment):
    # Get both perpendicular directions
    perp1 = np.array([-d[1], d[0]])
    perp2 = np.array([d[1], -d[0]])
    
    # Vector from current point to midpoint of target
    midpoint = (target_segment[0] + target_segment[1]) / 2
    to_target = midpoint - from_point
    
    # Pick the perpendicular direction with larger dot product (more aligned)
    if np.dot(perp1, to_target) > np.dot(perp2, to_target):
        return perp1
    else:
        return perp2

def perpendicular_ray_lineseg_intersection(A1, A2, B1, B2):
    """
    Parameters:
        - A1: ray starting point, perpendicular to the line segment A
        - A2: the other endpoint of the line segment A
        - B1: endpoint1 of line segment B
        - B2: endpoint2 of line segment B
    """
    A1 = np.array(A1, dtype=float)
    A2 = np.array(A2, dtype=float)
    B1 = np.array(B1, dtype=float)
    B2 = np.array(B2, dtype=float)
    
    d = A1 - A2
    if np.allclose(d, 0):
        raise ValueError("A1 and A2 cannot be the same point.")
    
    direction = perpendicular_toward_target(d, A1, [B1, B2])
    origin = A1
    d_b = B2 - B1
    
    # Solve: origin + t*direction = B1 + u*d_b
    M = np.array([[direction[0], -d_b[0]],
                  [direction[1], -d_b[1]]])
    b = B1 - origin
    
    det = np.linalg.det(M)
    if np.isclose(det, 0):
        return False, None
    
    t, u = np.linalg.solve(M, b)
    
    if t >= 0 and 0 <= u <= 1:
        intersection = origin + t * direction
        return True, intersection
    else:
        return False, None    


if __name__=="__main__": # Example usage

    A1 = [1, 2]
    A2 = [1, 3]
    B1 = [2, 2.5]
    B2 = [2, 0]

    # intersects, point = perpendicular_ray_and_intersection_check(A1, A2, B1, B2)  #(A1, A2) is the ray, (B1, B2) is the line segment. ray starts at A2
    intersects1, point1 = perpendicular_ray_lineseg_intersection(A1, A2, B1, B2)
    intersects2, point2 = perpendicular_ray_lineseg_intersection(B1, B2, A1, A2)
    if intersects1 and intersects2:
        print("✅ Perpendicular ray intersects at:", point1)
        print("✅ Perpendicular ray intersects at:", point2)
    else:
        print("❌ No intersection")
        
    print(intersects1)
    print(intersects2)