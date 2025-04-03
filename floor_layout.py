import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.line_segment_distance import segment_distance
from module.contour_completion import ContourCompletion
from module.contour_refinement import RefineContour

import argparse
import yaml
import ezdxf


class FloorMapFromSLAM:
    def __init__(self, map_path, config_file, output_path):
        self.map_path = map_path
        self.output_path = output_path
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.extract_contour_first = self.config['FloorMapFromSLAM'].get("extract_contour_first")
        self.epsilon = self.config['FloorMapFromSLAM'].get("epsilon")
        self.min_area = self.config['FloorMapFromSLAM'].get("min_area")
        self.kernel_size = self.config['FloorMapFromSLAM'].get("kernel_size")
        self.dilate_iteration = self.config['FloorMapFromSLAM'].get("dilate_iteration")
        
        self.Hough_thresh = self.config['FloorMapFromSLAM'].get("Hough_thresh")
        self.Hough_minLength = self.config['FloorMapFromSLAM'].get("Hough_minLength")
        self.Hough_maxGap = self.config['FloorMapFromSLAM'].get("Hough_maxGap")
        
        self.room_num = self.config['FloorMapFromSLAM'].get("room_num")
        self.output_vector_image = self.config['FloorMapFromSLAM'].get("output_vector_image")
        self.simple_output = self.config['FloorMapFromSLAM'].get("simple_output")
        self.output_format = self.config['FloorMapFromSLAM'].get("output_format")
        
        self.refinement = self.config['FloorMapFromSLAM'].get("refinement")
        self.refine_contour = RefineContour()
        self.output_vector_image = self.config['FloorMapFromSLAM'].get("output_vector_image")
                    
        self.angle_thresh = self.config['remove_redundant_lines'].get("angle_thresh")
        self.distance_thresh = self.config['remove_redundant_lines'].get("distance_thresh")
        self.length_thresh = self.config['remove_redundant_lines'].get("length_thresh")
        
        self.img_height = None
        self.img_width = None
        self.image_exts = [".png", ".jpg", ".jpeg"]
                
        self.contour_completion = ContourCompletion(self.config['contour_completion'])

    def remove_redundant_lines(self, lines):
        """
        Remove lines that are nearly identical in orientation and position.
    
        Input:
        lines: list/array of lines from cv2.HoughLinesP in the shape (N,1,4)
      
        Output:
        unique_lines: list of unique lines, each represented as [x1, y1, x2, y2, xc, yc, p1_paired, p2_paired, angle]
        """
        unique_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle in degrees
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            redundant = False
            
            # Calculate the length of the line segment
            length = np.hypot(x2 - x1, y2 - y1)
            if length < self.length_thresh:
                continue  # Skip lines shorter than the threshold
            
            # Compare with each line already in unique_lines
            for u_line in unique_lines:
                # unpack the dictionary
                endp1 = u_line["endp1"]
                endp2 = u_line["endp2"]
                uangle = u_line["angle"]
                
                # Check if the angle difference is small enough
                if abs(angle - uangle) < self.angle_thresh:
                    # Check if the line segments are close
                    A = np.array([x1, y1])
                    B = np.array([x2, y2])
                    line_seg_dist = segment_distance(A, B, endp1, endp2)
                    if line_seg_dist < self.distance_thresh:
                        redundant = True
                        break
        
            if not redundant:
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                unique_line_segment = {"endp1": np.array([x1, y1]), "endp2": np.array([x2, y2]), "centroid": np.array([xc, yc]), "p1_paired": False, "p2_paired": False, "angle": angle}
                unique_lines.append(unique_line_segment)
            
        return unique_lines
    
    def shift_to_positive(self, line_segments):
        """
        ensure all endpoint coordinates are non-negative for the output image
        """
        min_x = min(min(line["endp1"][0], line["endp2"][0]) for line in line_segments)
        min_y = min(min(line["endp1"][1], line["endp2"][1]) for line in line_segments)
        shift_x = -min_x if min_x < 0 else 0
        shift_y = -min_y if min_y < 0 else 0

        shift_vec = np.array([shift_x, shift_y], dtype=np.int32)
         
        for line in line_segments:
            line["endp1"] += shift_vec
            line["endp2"] += shift_vec
            line["centroid"] += shift_vec

    def flip_y(self, y, height):
        return height - y
            
    def output_image(self, image_color, line_segments, output, visualize=True):
        """
        Outputs images based on the processed line segments.
        
        Parameters:
            line_segments: List of line segment dictionaries. Each dictionary should have:
                "endp1", "endp2", "centroid", "p1_paired", "p2_paired", "angle".
            output: Output file name including extension. For raster images, typical extensions are .png or .jpg.
            vector_image: If True, export a DXF vector file.
                        (Requires the ezdxf package: pip install ezdxf)
        """
        # If a DXF export is requested
        if self.output_vector_image:
            try:
                doc = ezdxf.new()
                msp = doc.modelspace()
                for line in line_segments:
                    p1 = line["endp1"]
                    p2 = line["endp2"]
                    if p1 is None or p2 is None:
                        continue
                    # Add the line to the DXF file (DXF allows negative coordinates)
                    x1, y1 = float(p1[0]), self.flip_y(float(p1[1]), self.img_height)
                    x2, y2 = float(p2[0]), self.flip_y(float(p2[1]), self.img_height)

                    msp.add_line((x1, y1), (x2, y2))
                vector_image_output = self.output_path + ".dxf"
                doc.saveas(vector_image_output)
                print(f"[INFO] Saved DXF to {vector_image_output}")
            except ImportError:
                print("[ERROR] DXF export requires the ezdxf package. Install it with 'pip install ezdxf'.")

        # --- Create image output using matplotlib ---
        
        # Convert background image from BGR to RGB for plotting
        background = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
        
        # ----- Plot 1: Overlay of original .pgm image with extracted lines -----
        fig_overlay, ax_overlay = plt.subplots(figsize=(8, 8))
        ax_overlay.imshow(background)
        for i, line in enumerate(line_segments):
            p1 = line["endp1"]
            p2 = line["endp2"]
            if p1 is None or p2 is None:
                continue
            if i == 0:
                ax_overlay.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=3, label='Detected Walls')
            else:
                ax_overlay.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=3)

        # Adjust axes to match image coordinate system (origin at top-left)
        h, w = background.shape[:2]
        ax_overlay.set_xlim([0, w])
        ax_overlay.set_ylim([h, 0])
        title = "floor plan " + self.room_num
        ax_overlay.set_title(title)
        ax_overlay.legend()
        ax_overlay.axis("off")
        fig_overlay.tight_layout()
        
        # ----- Plot 2: Lines-only on a blank background -----
        # Create a blank white background (same dimensions as background image)
        blank_background = np.ones_like(background) * 255
        fig_lines, ax_lines = plt.subplots(figsize=(8, 8))
        ax_lines.imshow(blank_background)
        for i, line in enumerate(line_segments):
            p1 = line["endp1"]
            p2 = line["endp2"]
            if p1 is None or p2 is None:
                continue

            # Label only the first line to avoid duplicate legend entries
            if i == 0:
                ax_lines.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=2, label='Detected Walls')
            else:
                ax_lines.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=2)

        ax_lines.set_xlim([-20, blank_background.shape[1] + 20])
        ax_lines.set_ylim([blank_background.shape[0] + 20, -20])
        ax_lines.legend()
        ax_lines.axis("off")
        ax_lines.set_title(title)
        fig_lines.tight_layout()
        
        # Save image file.
        if output in self.image_exts:
            lines_filename = self.output_path + output
        else:
            raise ValueError(f"[ERROR] Unsupported image format: {output}")
        fig_lines.savefig(lines_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig_lines)
        print(f"[INFO] Saved image to {lines_filename}")
        
        # Optionally, display the plots
        if visualize:
            plt.show()

    
    def run(self, output=".png", visualize=True):
        # Load the grayscale PGM map
        img = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]
        
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # for room1 we need to extract contour first
        if self.extract_contour_first:
            # Step 1: Threshold to isolate occupied (black) regions
            occupied_mask = cv2.inRange(img, 0, 50)  # adjust 50 if needed

            # Step 2: Find contours
            contours, _ = cv2.findContours(occupied_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Step 3: Filter out small noisy contours
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

            # Step 4: Approximate contours with straight lines
            binary = np.zeros_like(img)
        
            for contour in filtered_contours:
                epsilon = self.epsilon * cv2.arcLength(contour, True)  # smaller -> more accurate
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.polylines(binary, [approx], isClosed=True, color=255, thickness=2)  # blue lines
                
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=self.dilate_iteration)
            
            if self.simple_output:
                if output in self.image_exts:
                    output_file_name = self.output_path + output
                else:
                    raise ValueError(f"[ERROR] Unsupported image format: {output}")
                
                output_img = np.copy(binary)
                cv2.putText(output_img, "Detected Boundaries", (60, 736), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(225, 225, 225), thickness=2, lineType=cv2.LINE_AA)
                cv2.imwrite(output_file_name, output_img)
                
                if visualize:
                    img_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    contour_mask = binary > 0
                    img_overlay[contour_mask] = [255, 0, 0]  # Blue lines
                    img_overlay = cv2.resize(img_overlay, (600, 800), interpolation=cv2.INTER_AREA)
                    cv2.putText(img_overlay, "Detected Boundaries", (60, 736), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(225, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                    cv2.imshow("Alignment Check for Boundaries", img_overlay)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return 


        # ----- extract line segments -------------------------------------------#
        if not self.extract_contour_first:
            # Threshold to binary (occupied = black)
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Remove small noisy components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                    binary[labels == i] = 0

        # necessary for cv2.HoughLinesP() function
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Hough Line Transform to reinforce linear walls
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=self.Hough_thresh, minLineLength=self.Hough_minLength, maxLineGap=self.Hough_maxGap)
        
        if lines is not None:
            unique_lines = self.remove_redundant_lines(lines)
            completed_lines = self.contour_completion.complete_line_segments(unique_lines)
            if self.refinement:
                # remove remaining noise line segments
                self.refine_contour.remove_lines(completed_lines)
                
                # ensure each endpoint is connected to another endpoint from a neighbor line segment
                self.refine_contour.refine_contour(completed_lines)
                            
            # ensure all endpoint coordinates are positive for valid image output
            self.shift_to_positive(completed_lines)
            
            # store the output images
            self.output_image(img_color, completed_lines, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate floor layout based on 2D SLAM map')
    parser.add_argument('-m','--map', help='Path where 2D SLAM maps are stored in .pgm format', required=True)
    parser.add_argument('-c','--config', help='Path to yaml config file', required=True)
    parser.add_argument('-o','--output', help='Path to output path with the room name, e.g. /room2', required=True)
    args = parser.parse_args()
    
    floor_map_generator = FloorMapFromSLAM(map_path=args.map, config_file=args.config, output_path=args.output)
    floor_map_generator.run(floor_map_generator.output_format, visualize=True) # default in .png format
    
