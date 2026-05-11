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
        # Radii ratios (approx): R1=0.4, R2=0.6, R3=0.8
        rings = [
            {'count': 9,  'radius': 0.35},
            {'count': 13, 'radius': 0.58},
            {'count': 21, 'radius': 0.82}
        ]
        
        detected_bits = []
        
        # Sample each ring
        for ring in rings:
            n = ring['count']
            r_scale = ring['radius'] * (Ma / 2) # Use semi-major axis
            # We sample n points around the ellipse
            for i in range(n):
                theta = (2 * np.pi * i / n)
                # Ellipse point formula (rotated)
                cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
                # Simple circle-to-ellipse mapping
                rx = (ma/2) * ring['radius']
                ry = (Ma/2) * ring['radius']
                
                px = cx + rx * np.cos(theta) * cos_a - ry * np.sin(theta) * sin_a
                py = cy + rx * np.cos(theta) * sin_a + ry * np.sin(theta) * cos_a
                
                # Check bounds
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    val = gray[int(py), int(px)]
                    detected_bits.append(1 if val > 128 else 0) # Assumes light dots on dark background
                else:
                    detected_bits.append(0)
                    
        if len(detected_bits) != 43:
            return None
            
        # 4. Pattern Matching (Cyclic)
        # We need to try all 43 rotations for EACH ring? No, the whole sequence rotates together.
        # But wait, RuneTag rings rotate at different speeds? No, it's a single rigid marker.
        # However, the 43-bit sequence is usually flattened.
        
        best_match = -1
        min_dist = 99
        
        bits_tuple = tuple(detected_bits)
        
        # Try all cyclic shifts
        for shift in range(43):
            shifted = bits_tuple[shift:] + bits_tuple[:shift]
            if shifted in self.codebook:
                return {
                    'id': self.codebook[shifted],
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
