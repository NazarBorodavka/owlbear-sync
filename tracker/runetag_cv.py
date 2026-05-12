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
        """Loads the standard RuneTag-43 codebook with Z7 (0-6) values."""
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
                        temp_codes.append(parts[2:45])
                
                self.codes_matrix = np.array(temp_codes, dtype=np.int8)
                self.ids_vector = np.array(temp_ids, dtype=np.int32)
                # For fast exact lookup
                self.exact_codebook = {tuple(c): i for c, i in zip(temp_codes, temp_ids)}
                
            print(f"RuneTag-CV: Loaded {len(self.ids_vector)} codes (Vectorized) from {path}")
        except Exception as e:
            print(f"RuneTag-CV: Error loading codebook: {e}")

    def detect(self, gray, invert=False):
        results = []
        if invert: gray = 255 - gray
            
        # Use Canny + Morphological closing to join dots into a solid marker shape
        # This is critical for markers without a solid outer ring
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process
        for cnt in contours:
            if len(cnt) < 10: continue
            area = cv2.contourArea(cnt)
            if area < 400: continue 
            
            # Fit ellipse to the merged blob
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, Ma), angle = ellipse
            
            # Rough filter to avoid noise
            if Ma == 0 or ma/Ma < 0.4: continue
            
            tag_data = self._decode_ellipse(gray, ellipse)
            if tag_data:
                results.append(tag_data)
            elif area > 1000: 
                # Return a 'debug' entry if it's a likely marker but failed to decode
                # This ensures the user sees a yellow box in 'Show ROI Boxes' mode
                results.append({
                    'id': -1, 
                    'center': (int(cx), int(cy)), 
                    'corners': self._get_ellipse_corners(ellipse)
                })
                
        return results

    def _decode_ellipse(self, gray, ellipse):
        (cx, cy), (ma, Ma), angle = ellipse
        rings = [0.38, 0.60, 0.82] 
        num_sectors = 43
        sectors_symbols = []
        
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        
        # Step 1: Calculate local adaptive threshold for this specific marker
        # Sample the central area and the outer area to find the "White" and "Black" levels
        try:
            # Create a small mask for the marker area to get intensity statistics
            # This is faster than masking the whole image
            rect = cv2.boundingRect(np.array([
                [cx-Ma/2, cy-Ma/2], [cx+Ma/2, cy-Ma/2], 
                [cx+Ma/2, cy+Ma/2], [cx-Ma/2, cy+Ma/2]
            ], dtype=np.int32))
            x, y, w, h = rect
            # Bounds check
            x, y = max(0, x), max(0, y)
            w, h = min(gray.shape[1]-x, w), min(gray.shape[0]-y, h)
            
            roi = gray[y:y+h, x:x+w]
            if roi.size < 100: return None
            
            # Use percentiles to find the local black/white levels (ignoring outliers)
            local_low = np.percentile(roi, 15)
            local_high = np.percentile(roi, 85)
            if local_high - local_low < 30: return None # Low contrast marker
            
            # The threshold is the midpoint
            local_thresh = (local_low + local_high) / 2
        except:
            local_thresh = 127 # Fallback
            
        # Step 2: Sample all dots
        for s in range(num_sectors):
            theta = (2 * np.pi * s / num_sectors)
            val = 0
            for i, r_scale in enumerate(rings):
                rx, ry = (ma/2) * r_scale, (Ma/2) * r_scale
                local_x, local_y = rx * np.cos(theta), ry * np.sin(theta)
                px, py = int(cx + local_x * cos_a - local_y * sin_a), int(cy + local_x * sin_a + local_y * cos_a)
                
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    # Sample a 3x3 neighborhood for robustness
                    patch = gray[max(0, py-1):py+2, max(0, px-1):px+2]
                    if np.mean(patch) > local_thresh:
                        val += (2**i)
            
            sectors_symbols.append(val - 1 if val > 0 else -1)
        
        if len([s for s in sectors_symbols if s != -1]) < 8: return None
            
        current_pattern = np.array(sectors_symbols, dtype=np.int8)
        
        # Fast Vectorized Matching
        for shift in range(num_sectors):
            shifted = np.roll(current_pattern, -shift)
            
            # Exact match (Dict lookup is O(1))
            shifted_tuple = tuple(shifted.tolist())
            if shifted_tuple in self.exact_codebook:
                return {
                    'id': self.exact_codebook[shifted_tuple],
                    'center': (int(cx), int(cy)),
                    'corners': self._get_ellipse_corners(ellipse)
                }
            
            # Fuzzy match (Vectorized NumPy is O(N))
            if self.hamming_dist > 0:
                # Hamming distance: count where elements are NOT equal
                # Only check if shifted has no -1s for speed, or handle -1s
                mismatches = np.sum(self.codes_matrix != shifted, axis=1)
                best_match_idx = np.argmin(mismatches)
                if mismatches[best_match_idx] <= self.hamming_dist:
                    return {
                        'id': int(self.ids_vector[best_match_idx]),
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
