import cv2
import numpy as np
import os

class RuneTagDetector:
    def __init__(self, codebook_path=None, hamming_dist=4):
        self.codebook = {}
        self.codes_matrix = None
        self.ids_vector = None
        self.exact_codebook = {}
        self.hamming_dist = hamming_dist
        
        # Official RUNEtag constants (from first-party C++ source)
        self.NUM_SLOTS = 43
        self.NUM_LAYERS = 3
        # Radii ratios: r = (n + layer + 1) / (2n) where n=3
        self.RADII = [4/6, 5/6, 6/6] # [0.666, 0.833, 1.0]
        
        if codebook_path:
            self.load_codebook(codebook_path)

    def load_codebook(self, path):
        """Loads the codebook as Z7 symbols (0-6)."""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                start = 1 if lines[0].strip().isdigit() else 0
                temp_codes = []
                temp_ids = []
                for line in lines[start:]:
                    parts = [int(p) for p in line.strip().replace(',', ' ').split()]
                    if len(parts) >= 44:
                        temp_ids.append(parts[0])
                        temp_codes.append(parts[2:45]) # The 43 Z7 symbols
                
                self.codes_matrix = np.array(temp_codes, dtype=np.int8)
                self.ids_vector = np.array(temp_ids, dtype=np.int32)
                self.exact_codebook = {tuple(c): i for c, i in zip(temp_codes, temp_ids)}
            print(f"RuneTag-CV (Official Port): Loaded {len(self.ids_vector)} codes.")
        except Exception as e:
            print(f"RuneTag-CV Error: {e}")

    def detect(self, gray, invert=False, min_score=0.3, detect_scale=1.0):
        results = []
        # If user inverted (e.g. phone screen), we flip it back to black-on-white 
        # so our "look for dark dots" logic always works.
        if invert: gray = 255 - gray
            
        # Official C++ uses BINARY_INV to turn BLACK dots into WHITE blobs for findContours
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 3)
        
        # 2. Official Dot Extraction
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            if len(cnt) < 10: continue
            area = cv2.contourArea(cnt)
            # Official area range [10.0, 10000.0]
            if 10 < area < 10000:
                # Check Roundness (4*pi*A/P^2) - Official uses 0.3
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ = 4 * np.pi * area / (peri * peri)
                if circ > 0.3:
                    ellipse = cv2.fitEllipse(cnt)
                    dots.append(ellipse) # Store the ellipse objects (dots)

        if len(dots) < 10: return results
        
        # 3. Geometric Consensus Grouping
        # We group dots that appear to be concentric
        # (Simplified for Python performance, using centroid clustering first)
        centroids = np.array([d[0] for d in dots])
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=60*detect_scale, min_samples=max(10, int(43 * min_score))).fit(centroids)
        
        for label in set(clustering.labels_):
            if label == -1: continue
            cluster_indices = np.where(clustering.labels_ == label)[0]
            cluster_dots = [dots[i] for i in cluster_indices]
            
            # Fit a master ellipse to all dot centroids in the cluster
            points = np.array([d[0] for d in cluster_dots], dtype=np.float32)
            master_ellipse = cv2.fitEllipse(points)
            
            tag_data = self._decode_official(gray, master_ellipse, cluster_dots)
            if tag_data:
                results.append(tag_data)
        
        return results

    def _decode_official(self, gray, master_ellipse, cluster_dots):
        (cx, cy), (ma, Ma), angle = master_ellipse
        num_slots = self.NUM_SLOTS
        
        # Create a polar map of the dots relative to the master ellipse
        # We project each dot into (slot_index, ring_index)
        
        # Transformation matrix to "unwarp" the ellipse into a unit circle
        # (Simplified version of the official projective unwarping)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        
        slot_data = np.zeros((num_slots, 3), dtype=int)
        
        for dot in cluster_dots:
            dx, dy = dot[0][0] - cx, dot[0][1] - cy
            # Rotate back
            rx = dx * cos_a + dy * sin_a
            ry = -dx * sin_a + dy * cos_a
            
            # Scale to unit circle
            nx, ny = rx / (ma/2), ry / (Ma/2)
            dist = np.sqrt(nx*nx + ny*ny)
            
            # Identify Ring (Widened for robustness)
            # Official radii: [0.66, 0.83, 1.0]
            ring_idx = -1
            if 0.50 < dist < 0.75: ring_idx = 0 # Inner
            elif 0.75 < dist < 0.90: ring_idx = 1 # Middle
            elif 0.90 < dist < 1.25: ring_idx = 2 # Outer
            
            if ring_idx != -1:
                # Identify Slot
                angle_deg = np.degrees(np.arctan2(ny, nx)) % 360
                slot_idx = int((angle_deg / 360.0) * num_slots + 0.5) % num_slots
                slot_data[slot_idx, ring_idx] = 1

        # Convert 3-bit rings to Z7 symbol (Inner, Middle, Outer)
        detected_symbols = []
        for s in range(num_slots):
            # Z7 symbol mapping based on codebook analysis:
            # 1 (Outer) -> 0, 2 (Mid) -> 1, 3 (Mid+Out) -> 2, 4 (In) -> 3...
            val = 4*slot_data[s,0] + 2*slot_data[s,1] + 1*slot_data[s,2]
            
            # Map val (1-7) to symbol (0-6). 0 (Empty) is treated as -1 (Mismatch)
            symbol = val - 1 if val > 0 else -1
            detected_symbols.append(symbol)
            
        valid_dots = np.sum(slot_data)
        if valid_dots < 10: return None
        
        # DEBUG: Print the raw pattern found
        print(f"  [DEBUG] Found {int(valid_dots)} dots. Pattern (0-6): {' '.join(str(s) for s in detected_symbols[:15])}...")
        
        current_pattern = np.array(detected_symbols, dtype=np.int8)
        
        # Cyclic Matching (Official Logic)
        for shift in range(num_slots):
            shifted = np.roll(current_pattern, -shift)
            shifted_tuple = tuple(shifted.tolist())
            
            if shifted_tuple in self.exact_codebook:
                return {
                    'id': self.exact_codebook[shifted_tuple],
                    'center': (int(cx), int(cy)),
                    'corners': self._get_ellipse_corners(master_ellipse)
                }
                
            if self.hamming_dist > 0:
                mismatches = np.sum(self.codes_matrix != shifted, axis=1)
                best_idx = np.argmin(mismatches)
                if mismatches[best_idx] <= self.hamming_dist:
                    return {
                        'id': int(self.ids_vector[best_idx]),
                        'center': (int(cx), int(cy)),
                        'corners': self._get_ellipse_corners(master_ellipse)
                    }
        return None

    def _get_ellipse_corners(self, ellipse):
        (cx, cy), (ma, Ma), angle = ellipse
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        pts = []
        for dx, dy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
            rx, ry = dx * (ma/2), dy * (Ma/2)
            px = cx + rx * cos_a - ry * sin_a
            py = cy + rx * sin_a + ry * cos_a
            pts.append([int(px), int(py)])
        return pts
