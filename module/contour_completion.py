import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

from utils.check_extension_intersect import line_intersection, find_intersection_perpendicular
from utils.check_perpendicular_intersect import perpendicular_ray_lineseg_intersection
from utils.checkout_ray_intersect import find_ray_intersection

class ContourCompletion:
    def __init__(self, config):
        self.horizon = config.get('horizon')
        self.sort_method = config.get('sort')
        self.ascending = config.get('ascending')
    
        self.thresh_parallel = config.get('thresh_parallel')
        self.thresh_perp = config.get('thresh_perp')
        self.thresh_collinear = config.get('thresh_collinear')
        self.thresh_dist = config.get('thresh_dist')
        
        self.min_seg_length = config.get('min_seg_length')
        self.max_closepoint_dist = config.get('max_closepoint_dist')        
        
        self.sorted_lines = []
       
        
    def rotate_point(self, px, py, cx, cy, angle_deg):
        """Rotate point (px, py) around center (cx, cy) by angle_rad radians."""
        angle_rad = np.deg2rad(angle_deg)
        
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # Translate point to origin
        px -= cx
        py -= cy

        # Rotate
        xnew = px * c - py * s
        ynew = px * s + py * c

        # Translate back
        return xnew + cx, ynew + cy

    def enforce_parallel(self, line, target_angle_deg, closest_point_key, endpoint, endpoint_aux):
        
        # Extract line components from the dictionary
        cendp1 = line["endp1"]
        cendp2 = line["endp2"]
        ccentroid = line["centroid"]
        p1_paired = line["p1_paired"]
        p2_paired = line["p2_paired"]
        original_angle_deg = line["angle"]
        
        # Use the provided key to get the closest point
        closest_point = line[closest_point_key]

        # Unpack the coordinate values
        x1, y1 = cendp1
        x2, y2 = cendp2
        xc, yc = ccentroid

        # Compute rotation needed to align with target_angle_degree
        angle_diff = target_angle_deg - original_angle_deg

        # Rotate both points around the center
        x1r, y1r = self.rotate_point(x1, y1, xc, yc, angle_diff)
        x2r, y2r = self.rotate_point(x2, y2, xc, yc, angle_diff)

        parallel_line_segment = {
            "endp1": np.array([x1r, y1r]),
            "endp2": np.array([x2r, y2r]),
            "centroid": np.array([xc, yc]),
            "p1_paired": p1_paired,
            "p2_paired": p2_paired,
            "angle": target_angle_deg
        }
        
        is_collinear = self.is_nearly_collinear(closest_point, endpoint, endpoint_aux)
        
        if is_collinear:
            return self.enforce_collinear(parallel_line_segment, endpoint, endpoint_aux), is_collinear
        else:
            return parallel_line_segment, is_collinear

    def enforce_perp(self, line, baseline_angle_deg):
        """
        Rotate the given line around its center so that it is perpendicular to the baseline.
        `line` is a tuple: (x1, y1, x2, y2, xc, yc, p1_paired, p2_paired, angle_in_deg)
        (endpoint1, endpoint2) is the baseline as tuples (x, y)
        """
        # Extract line components from the dictionary
        cendp1 = line["endp1"]
        cendp2 = line["endp2"]
        ccentroid = line["centroid"]
        p1_paired = line["p1_paired"]
        p2_paired = line["p2_paired"]
        original_angle_deg = line["angle"]

        # Unpack the coordinate values
        x1, y1 = cendp1
        x2, y2 = cendp2
        xc, yc = ccentroid
        
        # Desired angle is perpendicular
        target_angle_deg = baseline_angle_deg + 90

        # Compute rotation angle
        angle_diff = self.normalize_angle(target_angle_deg - original_angle_deg)

        # Rotate both endpoints around the centroid
        x1r, y1r = self.rotate_point(x1, y1, xc, yc, angle_diff)
        x2r, y2r = self.rotate_point(x2, y2, xc, yc, angle_diff)

        perp_line_segment = {
            "endp1": np.array([x1r, y1r]),
            "endp2": np.array([x2r, y2r]),
            "centroid": np.array([xc, yc]),
            "p1_paired": p1_paired,
            "p2_paired": p2_paired,
            "angle": target_angle_deg
        }

        return perp_line_segment



    def is_nearly_collinear(self, pt, endpoint1, endpoint2):
        """
        Check if point `pt` is nearly collinear with the line defined by two points.
        
        Parameters:
            pt: Tuple (x, y)
            line: Tuple of two points ((x1, y1), (x2, y2))
        
        Returns:
            True if point is nearly collinear, False otherwise.
        """
        x0, y0 = pt
        x1, y1 = endpoint1
        x2, y2 = endpoint2

        # Line length squared
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_len_sq == 0:
            return False  # Degenerate line

        # Perpendicular distance formula (point to line)
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denom = np.sqrt(line_len_sq)
        distance = numerator / denom

        return distance < self.thresh_collinear
    
    def enforce_collinear(self, line, fixed_endp1, fixed_endp2):
        """
        Enforce collinearity of the candidate line with a fixed reference line.

        Parameters:
            line: tuple (x1, y1, x2, y2, xc, yc, p1_paired, p2_paired, angle_in_deg)
            fixed_line: tuple ((fx1, fy1), (fx2, fy2)) defining the fixed line direction

        Returns:
            Dictionary with enforced line segment:
                {
                    "endp1": np.array([x, y]),
                    "endp2": np.array([x, y]),
                    "centroid": np.array([x, y]),
                    "p1_paired": bool,
                    "p2_paired": bool,
                    "angle": float (degrees)
                }
        """
        

        # Unpack candidate line
        # Extract line components from the dictionary
        cendp1 = line["endp1"]
        cendp2 = line["endp2"]
        ccentroid = line["centroid"]
        p1_paired = line["p1_paired"]
        p2_paired = line["p2_paired"]
        orig_angle = line["angle"]
        
        # Unpack the coordinate values
        x1, y1 = cendp1
        x2, y2 = cendp2
        xc, yc = ccentroid 

        # Unpack fixed line endpoints
        fx1, fy1 = fixed_endp1
        fx2, fy2 = fixed_endp2

        # Define fixed line direction
        fixed_vec = np.array([fx2 - fx1, fy2 - fy1])
        fixed_norm = np.linalg.norm(fixed_vec)

        # Handle degenerate fixed line
        if fixed_norm == 0:
            return {
                "endp1": np.array([x1, y1]),
                "endp2": np.array([x2, y2]),
                "centroid": np.array([xc, yc]),
                "p1_paired": p1_paired,
                "p2_paired": p2_paired,
                "angle": orig_angle
            }

        fixed_unit_vec = fixed_vec / fixed_norm

        # Projection function onto fixed line
        def project(pt):
            pt_vec = np.array(pt) - np.array([fx1, fy1])
            projected_length = np.dot(pt_vec, fixed_unit_vec)
            return np.array([fx1, fy1]) + projected_length * fixed_unit_vec

        # Project endpoints onto fixed line
        new_endp1 = project([x1, y1])
        new_endp2 = project([x2, y2])
        new_centroid = (new_endp1 + new_endp2) / 2

        # Compute new angle from fixed direction vector
        angle_rad = np.arctan2(fixed_unit_vec[1], fixed_unit_vec[0])
        angle_deg = (np.degrees(angle_rad) + 360) % 180

        return {
            "endp1": new_endp1,
            "endp2": new_endp2,
            "centroid": new_centroid,
            "p1_paired": p1_paired,
            "p2_paired": p2_paired,
            "angle": angle_deg
        }
        
    def get_lineseg_length(self, line):
        return np.linalg.norm(line["endp1"] - line["endp2"])
    
    def normalize_angle(self, angle):
        # First convert to range (-180, 180]
        angle = (angle + 180) % 360 - 180
        
        # Then handle the case where we need to get the perpendicular angle
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        return angle
    
    def complete_line_segments(self, lines):
        # Sort by xc
        if self.sort_method == "xc" and self.ascending:
            self.sorted_lines = sorted(lines, key=lambda seg: seg["centroid"][0])    
        elif self.sort_method == "xc" and not self.ascending:
            self.sorted_lines = sorted(lines, key=lambda seg: seg["centroid"][0], reverse=True)    
        elif self.sort_method == "yc" and self.ascending:
            self.sorted_lines = sorted(lines, key=lambda seg: seg["centroid"][1])    
        elif self.sort_method == "yc" and not self.ascending:
            self.sorted_lines = sorted(lines, key=lambda seg: seg["centroid"][1], reverse=True)    

        # deep copy of sorted lines, new line segments will be appended to it
        completed_lines = self.sorted_lines.copy()
        
        for line_idx in range(len(self.sorted_lines)):
            
            line = completed_lines[line_idx]
            
            # angle of the i-th line segment
            angle_l = self.normalize_angle(line["angle"])
            # angle_l = line["angle"]
            

            if not line["p1_paired"]:
                closest_found_p1 = self.handle_line_endpoint("endp1", angle_l, line_idx, completed_lines)
                
            if not line["p2_paired"]:
                closest_found_p2 = self.handle_line_endpoint("endp2", angle_l, line_idx, completed_lines)
                
            if line["p1_paired"] and line["p2_paired"]:
                continue
            
            length = self.get_lineseg_length(line)
            
            if line["p1_paired"] and not line["p2_paired"]:
                if length < self.min_seg_length: #50
                    line["p1_paired"] = False
                    if closest_found_p1:
                        closest_line_idx = closest_found_p1[0]
                        closest_point_key = closest_found_p1[1]
                        num_new_seg = closest_found_p1[2]
                        flag = "p1_paired" if closest_point_key == "endp1" else "p2_paired"
                        completed_lines[closest_line_idx][flag] = False
                        if num_new_seg > 0:
                            completed_lines = completed_lines[:-num_new_seg]
                        completed_lines[closest_line_idx]["endp1"] = original_line_pose[0]
                        completed_lines[closest_line_idx]["endp2"] = original_line_pose[1]
                        completed_lines[closest_line_idx]["angle"] = original_line_pose[2]
                else:
                    line["p2_paired"] = True
            elif line["p2_paired"] and not line["p1_paired"]:
                if length < self.min_seg_length: #50
                    line["p2_paired"] = False
                    if closest_found_p2:
                        closest_line_idx = closest_found_p2[0]
                        closest_point_key = closest_found_p2[1]
                        num_new_seg = closest_found_p2[2]
                        original_line_pose = closest_found_p2[3]
                        flag = "p1_paired" if closest_point_key == "endp1" else "p2_paired"
                        completed_lines[closest_line_idx][flag] = False
                        if num_new_seg > 0:
                            completed_lines = completed_lines[:-num_new_seg]
                        completed_lines[closest_line_idx]["endp1"] = original_line_pose[0]
                        completed_lines[closest_line_idx]["endp2"] = original_line_pose[1]
                        completed_lines[closest_line_idx]["angle"] = original_line_pose[2]
                else:
                    line["p1_paired"] = True
        
        return completed_lines
            
    def find_closest_neighbor_endpoint(self, query_point, query_line_idx, completed_lines):
        closest_line_idx = None
        closest_line = None
        closest_point_key = None
        min_dist = float("inf")
        found_neighbor_point = False
        
        for local_candidate_idx in range(self.horizon):
            candidate_idx = query_line_idx + local_candidate_idx + 1
            candidate = completed_lines[candidate_idx]
            
            for label, flag in zip(["endp1", "endp2"], ["p1_paired", "p2_paired"]):
                point = candidate[label]
                dist = np.linalg.norm(query_point - point)
                if dist < min_dist and not candidate[flag]:
                    if min_dist - dist >= 10:
                        min_dist = dist
                        closest_line_idx = candidate_idx
                        closest_line = candidate
                        closest_point_key = label
            if min_dist < self.max_closepoint_dist:
                found_neighbor_point = True
        
        return closest_line_idx, closest_line, closest_point_key, found_neighbor_point
        
           
    def handle_line_endpoint(self, query_endp, angle_l, line_idx, completed_lines):
        if query_endp == "endp1":
            endpoint = completed_lines[line_idx]["endp1"]
            endpoint_aux = completed_lines[line_idx]["endp2"]
        elif query_endp == "endp2":
            endpoint = completed_lines[line_idx]["endp2"]
            endpoint_aux = completed_lines[line_idx]["endp1"]
        
        # Find the neighbor line with the closest endpoint and get its angle.
        closest_line_idx, closest_line, closest_point_key, found = self.find_closest_neighbor_endpoint(endpoint, line_idx, completed_lines)

        if found:
            angle_c = self.normalize_angle(closest_line["angle"])
            # angle_c = closest_line["angle"]

            # handler based on the angle difference.
            if np.abs(angle_c - angle_l) < self.thresh_parallel:
                num_new_seg, original_line_pose = self.handle_parallel_case(closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, angle_l, line_idx, query_endp, completed_lines)
            elif np.abs(90 - np.abs(angle_c - angle_l)) < self.thresh_perp:
                num_new_seg, original_line_pose = self.handle_perpendicular_case(closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, angle_l, line_idx, query_endp, completed_lines)
            else:
                num_new_seg, original_line_pose = self.handle_default_case(closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, line_idx, query_endp, completed_lines)
            return (closest_line_idx, closest_point_key, num_new_seg, original_line_pose)
        else:
            return


    def get_line_points(self, line, closest_point_key):
        """Return (closest_point, aux_point) based on the closest point index."""
        if closest_point_key == "endp1":
            return line["endp1"], line["endp2"]
        elif closest_point_key == "endp2":
            return line["endp2"], line["endp1"]
        else:
            raise ValueError("Unexpected closest_point_key: " + str(closest_point_key))


    def set_pairing_flags(self, line_idx, query_endp_key, completed_lines):
        """Helper to update pairing flags in both line representations."""
        flag_name = "p1_paired" if query_endp_key == "endp1" else "p2_paired"
        completed_lines[line_idx][flag_name] = True
        self.sorted_lines[line_idx][flag_name] = True

    def reset_pairing_flags(self, line_segments):
        for segment in line_segments:
            segment["p1_paired"] = False
            segment["p2_paired"] = False

    def create_line_segment(self, p1, p2):
        """Factory function for new line segments with angle and centroid computation using NumPy."""
        p1 = np.array(p1)
        p2 = np.array(p2)

        # Compute centroid
        centroid = (p1 + p2) / 2.0

        # Compute angle in degrees
        delta = p2 - p1
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        # Normalize angle
        angle = self.normalize_angle(angle)

        return {
            "endp1": p1,
            "endp2": p2,
            "centroid": centroid,
            "p1_paired": True,
            "p2_paired": True,
            "angle": angle
    }
    def handle_parallel_case(self, closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, angle_l, line_idx, query_endp_key, completed_lines):
        # Enforce parallelism on the closest neighbor.
        original_line_pose = (closest_line["endp1"], closest_line["endp2"], closest_line["angle"])
        
        updated_line, is_collinear = self.enforce_parallel(closest_line, angle_l, closest_point_key, endpoint, endpoint_aux)
        completed_lines[closest_line_idx] = updated_line
        closest_line = updated_line

        # Get points from the neighbor.
        closest_point, aux_point = self.get_line_points(closest_line, closest_point_key)

        if is_collinear:
            # For collinear line segments, simply create a segment connecting them.
            new_line_segment = self.create_line_segment(endpoint, closest_point)
            
            completed_lines.append(new_line_segment)
            num_new_seg = 1
        else:
            # Check intersections.
            intersect_c2l, point_c2l = perpendicular_ray_lineseg_intersection(closest_point, aux_point, endpoint, endpoint_aux)
            intersect_l2c, point_l2c = perpendicular_ray_lineseg_intersection(endpoint, endpoint_aux, closest_point, aux_point)

            if not (intersect_c2l or intersect_l2c): # checked 
                new_endpoint = find_ray_intersection(closest_point, aux_point, endpoint, endpoint_aux)
                completed_lines[line_idx][query_endp_key] = new_endpoint
                
                # Add new line segment
                new_seg = self.create_line_segment(new_endpoint, closest_point)
                completed_lines.append(new_seg)
                num_new_seg = 1
            elif intersect_c2l and intersect_l2c: # checked
                completed_lines[closest_line_idx][closest_point_key] = point_l2c
                
                new_seg = self.create_line_segment(point_l2c, endpoint)
                completed_lines.append(new_seg)
                num_new_seg = 1
            else:  # Only one of the intersections exists. checked
                if intersect_l2c:
                    new_endpoint = find_ray_intersection(closest_point, aux_point, endpoint, endpoint_aux)
                    completed_lines[line_idx][query_endp_key] = new_endpoint

                    new_line_segment = self.create_line_segment(new_endpoint, closest_point)
                    completed_lines.append(new_line_segment)
                    num_new_seg = 1
                else:
                    new_endpoint = find_ray_intersection(endpoint, endpoint_aux, closest_point, aux_point)
                    completed_lines[closest_line_idx][closest_point_key] = new_endpoint
                    
                    new_line_segment = self.create_line_segment(new_endpoint, endpoint)
                    completed_lines.append(new_line_segment)
                    num_new_seg = 1
            
        self.set_pairing_flags(line_idx, query_endp_key, completed_lines)
        self.set_pairing_flags(closest_line_idx, closest_point_key, completed_lines)
        
        return num_new_seg, original_line_pose

    def handle_perpendicular_case(self, closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, angle_l, line_idx, query_endp_key, completed_lines):
        original_line_pose = (closest_line["endp1"], closest_line["endp2"], closest_line["angle"])
        
        updated_line = self.enforce_perp(closest_line, angle_l)
        completed_lines[closest_line_idx] = updated_line
        closest_line = updated_line
        
        # A: candidate, B: query
        closest_point, aux_point = self.get_line_points(closest_line, closest_point_key)
        intersection, on_A, on_B, dA1, _, dB1, _ = line_intersection(closest_point, aux_point, endpoint, endpoint_aux)

        if not (on_A or on_B): 
            if dA1 < self.thresh_dist or dB1 < self.thresh_dist:
                completed_lines[line_idx][query_endp_key] = intersection
                completed_lines[closest_line_idx][closest_point_key] = intersection
                
                num_new_seg = 0
            else:
                new_endpoint = find_intersection_perpendicular(closest_point, aux_point, endpoint, endpoint_aux)
                new_seg1 = self.create_line_segment(new_endpoint, endpoint)
                new_seg2 = self.create_line_segment(new_endpoint, closest_point)
                completed_lines.extend([new_seg1, new_seg2])
                
                num_new_seg = 2
        else:      # on_A and on_B: 
            completed_lines[line_idx][query_endp_key] = intersection
            completed_lines[closest_line_idx][closest_point_key] = intersection
        
            num_new_seg = 0
        
        self.set_pairing_flags(line_idx, query_endp_key, completed_lines)
        self.set_pairing_flags(closest_line_idx, closest_point_key, completed_lines)
        
        return num_new_seg, original_line_pose

    def handle_default_case(self, closest_line_idx, closest_line, closest_point_key, endpoint, endpoint_aux, line_idx, query_endp_key, completed_lines):
        closest_point, aux_point = self.get_line_points(closest_line, closest_point_key)
        intersection, on_A, on_B, _, _, _, _ = line_intersection(closest_point, aux_point, endpoint, endpoint_aux)
        if not (on_A or on_B):
            new_seg = self.create_line_segment(endpoint, closest_point)
            completed_lines.append(new_seg)
            num_new_seg = 1
        else:
            completed_lines[line_idx][query_endp_key] = intersection
            completed_lines[closest_line_idx][closest_point_key] = intersection
            num_new_seg = 0

        self.set_pairing_flags(line_idx, query_endp_key, completed_lines)
        self.set_pairing_flags(closest_line_idx, closest_point_key, completed_lines)
        original_line_pose = None
        return num_new_seg, original_line_pose

        