import cv2
import numpy as np
import os

class RuneTagDetector:
    def __init__(self, codebook_path=None, hamming_dist=4):
        self.hamming_dist = hamming_dist
        self.codebook = {} # {tuple_of_bits: tag_id}
        if codebook_path and os.path.exists(codebook_path):
            self.load_codebook(codebook_path)
            
    def load_codebook(self, path):
        """Loads the standard RuneTag-43 codebook."""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                # Skip first line if it's the count
                start = 1 if lines[0].strip().isdigit() else 0
                for line in lines[start:]:
                    parts = [int(p) for p in line.strip().replace(',', ' ').split()]
                    if len(parts) >= 44:
                        tag_id = parts[0]
                        # The codebook in deeptag seems to have 0..6 values?
                        # Standard RuneTag is binary. Let's convert > 0 to 1
                        bits = tuple(1 if b > 0 else 0 for b in parts[2:45])
                        self.codebook[bits] = tag_id
            print(f"RuneTag-CV: Loaded {len(self.codebook)} codes from {path}")
        except Exception as e:
            print(f"RuneTag-CV: Error loading codebook: {e}")

    def detect(self, gray, invert=False):
        """
        Detects RuneTags in a grayscale image.
        Returns a list of decoded tags: [{'id': 1, 'center': (x,y), 'corners': [...]}]
        """
        results = []
        
        # 1. Pre-process
        if invert:
            gray = 255 - gray
            
        # Adaptive threshold to find the rings
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 51, 10)
        
        # 2. Find Ellipses
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) < 5: continue
            
            area = cv2.contourArea(cnt)
            if area < 400: continue # Too small
            
            # Filter for circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.5: continue
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            
            # 3. Unwarp and Sample
            # For simplicity in this lightweight version, we'll sample the rings directly
            # using the elliptical geometry
            tag_data = self._decode_ellipse(gray, ellipse)
            if tag_data:
                results.append(tag_data)
                
        return results

    def _decode_ellipse(self, gray, ellipse):
        (cx, cy), (ma, Ma), angle = ellipse
        # RuneTag-43 has 3 rings: 9, 13, 21 dots
        rings = [
            {'count': 9,  'radius': 0.38},
            {'count': 13, 'radius': 0.60},
            {'count': 21, 'radius': 0.82}
        ]
        
        # Step 1: Get local intensity range to handle glare/brightness
        # Sample center of marker and outer edge
        center_val = gray[int(cy), int(cx)] if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1] else 128
        # Use an adaptive threshold based on the local neighborhood
        
        detected_bits = []
        
        # Step 2: Sample each ring with multi-point averaging
        for ring in rings:
            n = ring['count']
            rx = (ma/2) * ring['radius']
            ry = (Ma/2) * ring['radius']
            cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            
            for i in range(n):
                theta = (2 * np.pi * i / n)
                # Sample 5 points around the dot center for robustness
                samples = []
                for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                    px = cx + (rx+dx) * np.cos(theta) * cos_a - (ry+dy) * np.sin(theta) * sin_a
                    py = cy + (rx+dx) * np.cos(theta) * sin_a + (ry+dy) * np.sin(theta) * cos_a
                    
                    if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                        samples.append(gray[int(py), int(px)])
                
                if not samples:
                    detected_bits.append(0)
                    continue
                    
                avg_val = sum(samples) / len(samples)
                # If avg_val is significantly different from the local background, it's a dot
                # For light dots on dark (Invert=False), we look for high values
                detected_bits.append(1 if avg_val > 160 else 0)
                    
        if len(detected_bits) != 43:
            return None
            
        # Step 3: Pattern Matching with Hamming Distance
        bits_tuple = tuple(detected_bits)
        best_id = -1
        best_dist = self.hamming_dist + 1
        
        # Fast path: exact match
        for shift in range(43):
            shifted = bits_tuple[shift:] + bits_tuple[:shift]
            if shifted in self.codebook:
                return {
                    'id': self.codebook[shifted],
                    'center': (int(cx), int(cy)),
                    'corners': self._get_ellipse_corners(ellipse)
                }
        
        # Slow path: Hamming distance (only if no exact match and hamming_dist > 0)
        if self.hamming_dist > 0:
            for shift in range(43):
                shifted = np.array(bits_tuple[shift:] + bits_tuple[:shift])
                for code_bits, tid in self.codebook.items():
                    dist = np.count_nonzero(shifted != code_bits)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid
                        if dist == 0: break
                if best_dist == 0: break
                        
        if best_id != -1 and best_dist <= self.hamming_dist:
            return {
                'id': best_id,
                'center': (int(cx), int(cy)),
                'corners': self._get_ellipse_corners(ellipse)
            }
                
        return None

    def _get_ellipse_corners(self, ellipse):
        # Approximates 4 corners from the ellipse for visualization
        (cx, cy), (ma, Ma), angle = ellipse
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        half_ma, half_Ma = ma/2, Ma/2
        
        corners = [
            (cx - half_ma * cos_a + half_Ma * sin_a, cy - half_ma * sin_a - half_Ma * cos_a),
            (cx + half_ma * cos_a + half_Ma * sin_a, cy + half_ma * sin_a - half_Ma * cos_a),
            (cx + half_ma * cos_a - half_Ma * sin_a, cy + half_ma * sin_a + half_Ma * cos_a),
            (cx - half_ma * cos_a - half_Ma * sin_a, cy - half_ma * sin_a + half_Ma * cos_a)
        ]
        return [[int(c[0]), int(c[1])] for c in corners]
