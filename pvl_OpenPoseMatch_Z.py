import math
import numpy as np
import cv2
import torch

class PVL_OpenPoseMatch_Z:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_keypoints": ("POSE_KEYPOINT",),
                "target_keypoints": ("POSE_KEYPOINT",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "legs": ("BOOLEAN", {"default": True}),
                "torso": ("BOOLEAN", {"default": True}),
                "head": ("BOOLEAN", {"default": True}),
                "hands": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer_proportions"
    CATEGORY = "OpenPose"

    def transfer_proportions(self, source_keypoints, target_keypoints, width, height, legs=True, torso=True, head=True, hands=True):
        # Extract canvas dimensions if available
        canvas_width = source_keypoints[0].get('canvas_width', width)
        canvas_height = source_keypoints[0].get('canvas_height', height)
        
        # Extract the first person's keypoints from each input
        try:
            # Get the first person from source
            source_person = source_keypoints[0]['people'][0]
            source_pts = self._extract_keypoints(source_person, canvas_width, canvas_height)
            
            # Get the first person from target
            target_person = target_keypoints[0]['people'][0]
            target_pts = self._extract_keypoints(target_person, canvas_width, canvas_height)
        except (KeyError, IndexError):
            # If no people detected, return blank image
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image_tensor = torch.from_numpy(image).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            return (image_tensor,)
        
        # Define chains and options
        chains = self._build_chains(source_pts, target_pts)
        options = {"legs": legs, "torso": torso, "head": head, "hands": hands}
        
        # Process chains
        new_target_pts = self._process_chains(chains, options, target_pts)
        
        # Scale keypoints to output dimensions
        scaled_pts = self._scale_keypoints_to_output(new_target_pts, canvas_width, canvas_height, width, height)
        
        # Render OpenPose image
        image = self._render_openpose(scaled_pts, width, height)
        
        # Convert to tensor in BHWC format (batch, height, width, channels)
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)

    def _extract_keypoints(self, person_data, canvas_width, canvas_height):
        """Extract (x,y) keypoints from person dictionary and convert to pixel coordinates"""
        try:
            keypoints = person_data['pose_keypoints_2d']
        except KeyError:
            # Fallback for different key names
            if 'pose_keypoints' in person_data:
                keypoints = person_data['pose_keypoints']
            else:
                # If no keypoints found, return zeros
                return [(0, 0)] * 18  # 18 COCO keypoints
        
        points = []
        for i in range(0, len(keypoints), 3):
            if i+1 < len(keypoints):
                x, y = keypoints[i], keypoints[i+1]
                # Convert normalized coordinates to pixel coordinates
                px = x * canvas_width
                py = y * canvas_height
                points.append((px, py) if x != 0 or y != 0 else (0, 0))
            else:
                points.append((0, 0))
        return points

    def _scale_keypoints_to_output(self, keypoints, canvas_width, canvas_height, output_width, output_height):
        """Scale keypoints from canvas dimensions to output dimensions"""
        scaled_points = []
        for x, y in keypoints:
            if x == 0 and y == 0:
                scaled_points.append((0, 0))
            else:
                # Scale from canvas to output dimensions
                scaled_x = (x / canvas_width) * output_width
                scaled_y = (y / canvas_height) * output_height
                scaled_points.append((scaled_x, scaled_y))
        return scaled_points

    def _build_chains(self, source_pts, target_pts):
        """Build skeletal chains with source/target keypoints"""
        # Calculate midpoints
        hip_mid = lambda pts: ((pts[11][0] + pts[12][0])/2, (pts[11][1] + pts[12][1])/2)
        shoulder_mid = lambda pts: ((pts[5][0] + pts[6][0])/2, (pts[5][1] + pts[6][1])/2)
        
        return {
            "right_leg": {
                "source": [source_pts[16], source_pts[14], source_pts[12]],
                "target": [target_pts[16], target_pts[14], target_pts[12]]
            },
            "left_leg": {
                "source": [source_pts[15], source_pts[13], source_pts[11]],
                "target": [target_pts[15], target_pts[13], target_pts[11]]
            },
            "torso": {
                "source": [hip_mid(source_pts), shoulder_mid(source_pts)],
                "target": [hip_mid(target_pts), shoulder_mid(target_pts)]
            },
            "head": {
                "source": [shoulder_mid(source_pts), source_pts[0]],
                "target": [shoulder_mid(target_pts), target_pts[0]]
            },
            "right_arm": {
                "source": [source_pts[6], source_pts[8], source_pts[10]],
                "target": [target_pts[6], target_pts[8], target_pts[10]]
            },
            "left_arm": {
                "source": [source_pts[5], source_pts[7], source_pts[9]],
                "target": [target_pts[5], target_pts[7], target_pts[9]]
            }
        }

    def _process_chains(self, chains, options, target_pts):
        """Process chains based on options and return new keypoints"""
        # Initialize with all target keypoints
        new_pts = list(target_pts)  # Make a copy of all 18 keypoints
        
        # Process legs
        if options["legs"]:
            self._scale_chain(chains["right_leg"], new_pts, [16, 14, 12])
            self._scale_chain(chains["left_leg"], new_pts, [15, 13, 11])
        
        # Update hip midpoint
        hip_mid = ((new_pts[11][0] + new_pts[12][0])/2, (new_pts[11][1] + new_pts[12][1])/2)
        chains["torso"]["target"][0] = hip_mid
        
        # Process torso
        if options["torso"]:
            self._scale_chain(chains["torso"], new_pts, [])
            # Update shoulder positions
            shoulder_offset = [
                (new_pts[5][0] - chains["torso"]["target"][1][0], new_pts[5][1] - chains["torso"]["target"][1][1]),
                (new_pts[6][0] - chains["torso"]["target"][1][0], new_pts[6][1] - chains["torso"]["target"][1][1])
            ]
            new_shoulder_mid = chains["torso"]["target"][1]
            new_pts[5] = (new_shoulder_mid[0] + shoulder_offset[0][0], new_shoulder_mid[1] + shoulder_offset[0][1])
            new_pts[6] = (new_shoulder_mid[0] + shoulder_offset[1][0], new_shoulder_mid[1] + shoulder_offset[1][1])
        
        # Process head
        if options["head"]:
            chains["head"]["target"][0] = chains["torso"]["target"][1]
            self._scale_chain(chains["head"], new_pts, [0])
        
        # Process arms
        if options["hands"]:
            self._scale_chain(chains["right_arm"], new_pts, [6, 8, 10])
            self._scale_chain(chains["left_arm"], new_pts, [5, 7, 9])
        
        return new_pts

    def _scale_chain(self, chain, new_pts, indices):
        """Scale a single chain and update keypoints"""
        src_chain = chain["source"]
        tgt_chain = chain["target"]
        
        # Calculate segment lengths
        src_lengths = [self._distance(src_chain[i], src_chain[i+1]) for i in range(len(src_chain)-1)]
        tgt_lengths = [self._distance(tgt_chain[i], tgt_chain[i+1]) for i in range(len(tgt_chain)-1)]
        
        # Skip if invalid chain
        if sum(src_lengths) == 0 or sum(tgt_lengths) == 0:
            return
        
        # Calculate scaling factors
        total_src = sum(src_lengths)
        total_tgt = sum(tgt_lengths)
        scale_factors = [(src_lengths[i] / total_src) * total_tgt / tgt_lengths[i] for i in range(len(tgt_lengths))]
        
        # Update keypoints
        if indices:  # Direct indices provided
            for i in range(len(indices)-1):
                idx1, idx2 = indices[i], indices[i+1]
                new_pts[idx2] = self._scale_point(new_pts[idx1], tgt_chain[i], tgt_chain[i+1], scale_factors[i])
        else:  # Update midpoint (torso)
            new_shoulder = self._scale_point(
                tgt_chain[0], tgt_chain[0], tgt_chain[1], scale_factors[0]
            )
            chain["target"][1] = new_shoulder

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _scale_point(self, anchor, p1, p2, scale):
        """Scale a point relative to anchor"""
        if p1 == p2:
            return p2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return (anchor[0] + dx * scale, anchor[1] + dy * scale)

    def _render_openpose(self, keypoints, width, height):
        """Render OpenPose skeleton to image with standard OpenPose visualization"""
        # Create a black background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Standard OpenPose COCO connections (18-point model)
        # Format: (point_index_1, point_index_2, color_r, color_g, color_b)
        connections = [
            # Face
            (0, 1, 255, 0, 0),      # Nose-Neck (red)
            (1, 2, 0, 255, 0),      # Neck-RShoulder (green)
            (1, 5, 0, 255, 0),      # Neck-LShoulder (green)
            (2, 3, 0, 255, 0),      # RShoulder-RElbow (green)
            (3, 4, 0, 255, 0),      # RElbow-RWrist (green)
            (5, 6, 0, 255, 0),      # LShoulder-LElbow (green)
            (6, 7, 0, 255, 0),      # LElbow-LWrist (green)
            (1, 8, 255, 255, 0),    # Neck-MidHip (yellow)
            (8, 9, 0, 0, 255),      # MidHip-RHip (blue)
            (9, 10, 0, 0, 255),     # RHip-RKnee (blue)
            (10, 11, 0, 0, 255),    # RKnee-RAnkle (blue)
            (8, 12, 0, 0, 255),     # MidHip-LHip (blue)
            (12, 13, 0, 0, 255),    # LHip-LKnee (blue)
            (13, 14, 0, 0, 255),    # LKnee-LAnkle (blue)
            # Eyes and ears
            (0, 15, 255, 0, 255),   # Nose-REye (magenta)
            (15, 17, 255, 0, 255),  # REye-REar (magenta)
            (0, 16, 255, 0, 255),   # Nose-LEye (magenta)
            (16, 18, 255, 0, 255),  # LEye-LEar (magenta)
            # Additional connections for stability
            (2, 5, 255, 165, 0),    # RShoulder-LShoulder (orange)
            (2, 8, 255, 165, 0),    # RShoulder-MidHip (orange)
            (5, 8, 255, 165, 0),    # LShoulder-MidHip (orange)
            (9, 12, 255, 165, 0)    # RHip-LHip (orange)
        ]
        
        # Draw connections
        for conn in connections:
            pt1_idx, pt2_idx, r, g, b = conn
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                # Skip if either point is (0,0) (not detected)
                if (0,0) in (pt1, pt2):
                    continue
                
                # Convert to integer coordinates
                x1 = int(pt1[0])
                y1 = int(pt1[1])
                x2 = int(pt2[0])
                y2 = int(pt2[1])
                
                # Ensure points are within image bounds
                x1 = max(0, min(width-1, x1))
                y1 = max(0, min(height-1, y1))
                x2 = max(0, min(width-1, x2))
                y2 = max(0, min(height-1, y2))
                
                # Draw the line with appropriate color
                cv2.line(image, (x1, y1), (x2, y2), (b, g, r), 3)  # OpenCV uses BGR format
        
        # Draw keypoints with appropriate colors
        for i, pt in enumerate(keypoints):
            if pt != (0, 0):
                # Convert to integer coordinates
                x = int(pt[0])
                y = int(pt[1])
                
                # Ensure point is within image bounds
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                # Different colors for different body parts
                if i == 0:  # Nose
                    color = (255, 0, 0)  # Red
                elif i in [15, 16]:  # Eyes
                    color = (255, 0, 255)  # Magenta
                elif i in [17, 18]:  # Ears
                    color = (255, 0, 255)  # Magenta
                elif i in [1, 8]:  # Neck and MidHip
                    color = (255, 255, 0)  # Yellow
                elif i in [2, 5]:  # Shoulders
                    color = (0, 255, 0)  # Green
                elif i in [3, 6]:  # Elbows
                    color = (0, 255, 0)  # Green
                elif i in [4, 7]:  # Wrists
                    color = (0, 255, 0)  # Green
                elif i in [9, 12]:  # Hips
                    color = (0, 0, 255)  # Blue
                elif i in [10, 13]:  # Knees
                    color = (0, 0, 255)  # Blue
                elif i in [11, 14]:  # Ankles
                    color = (0, 0, 255)  # Blue
                else:
                    color = (255, 255, 255)  # White for any other points
                
                # Draw the keypoint
                cv2.circle(image, (x, y), 5, color, -1)  # Filled circle
                cv2.circle(image, (x, y), 6, (255, 255, 255), 1)  # White outline
        
        return image