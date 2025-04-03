import numpy as np 

class RefineContour:
    def __init__(self):
        pass
    
    def remove_lines(self, line_segments):
        """
        if any line segment's both flags are false, then remove the line segment
        """
        line_segments[:] = [line for line in line_segments if line["p1_paired"] or line["p2_paired"]]


    def refine_contour(self, lines_segments, thresh=50):
        # Loop over all line segments
        for i, line in enumerate(lines_segments):
            # Process both endpoints for each segment
            for ep_key in ('endp1', 'endp2'):
                endpoint = line[ep_key]
                exact_match_found = False

                # Check if the endpoint exactly matches any endpoint in a different segment
                for j, other_line in enumerate(lines_segments):
                    if i == j:
                        continue  # Skip the same segment
                    if np.array_equal(endpoint, other_line['endp1']) or np.array_equal(endpoint, other_line['endp2']):
                        exact_match_found = True
                        break

                # If no exact match was found, find the nearest endpoint from another segment
                if not exact_match_found:
                    min_dist = float('inf')
                    nearest_point = None

                    # Iterate over all endpoints in other segments to find the closest one
                    for j, other_line in enumerate(lines_segments):
                        if i == j:
                            continue  # Skip the same segment
                        for other_ep in [other_line['endp1'], other_line['endp2']]:
                            dist = np.linalg.norm(endpoint - other_ep)
                            if dist < min_dist :
                                min_dist = dist
                                nearest_point = other_ep

                    # Update the endpoint to the nearest point if found
                    if nearest_point is not None and min_dist < thresh:
                        line[ep_key] = nearest_point