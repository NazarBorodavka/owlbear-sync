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

    def detect(self, gray, invert=False, min_score=0.3, detect_scale=1.0):
        results = []
        if invert: gray = 255 - gray
            
        # 1. Find potential dots (blobs)
        # detect_scale affects the blockSize of adaptive threshold
        block_size = int(21 * detect_scale)
        if block_size % 2 == 0: block_size += 1
        block_size = max(3, block_size)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, block_size, 5)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Dot size scales with detection scale
            if 3 < area < (500 * detect_scale): 
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ = 4 * np.pi * area / (peri * peri)
                if circ > 0.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        dots.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        if len(dots) < 5: return results
        
        # 2. Cluster dots into potential markers
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        dots_np = np.array(dots)
        # min_samples is derived from min_score (lower score = fewer dots needed)
        # RUNE-129 has 43 slots. If score=0.3, we need ~13 dots.
        min_dots = max(5, int(43 * min_score))
        
        # eps is the max distance between dots. Scales with detect_scale.
        eps = 40 * detect_scale
        
        clustering = DBSCAN(eps=eps, min_samples=min_dots).fit(dots_np)
        
        for label in set(clustering.labels_):
            if label == -1: continue 
            
            cluster_dots = dots_np[clustering.labels_ == label]
            if len(cluster_dots) >= 5:
                # 3. Fit ellipse
                ellipse = cv2.fitEllipse(cluster_dots.astype(np.float32))
                tag_data = self._decode_ellipse(gray, ellipse)
                if tag_data:
                    results.append(tag_data)
                else:
                    results.append({
                        'id': -1,
                        'center': (int(ellipse[0][0]), int(ellipse[0][1])),
                        'corners': self._get_ellipse_corners(ellipse)
                    })
                
        return results

    def _decode_ellipse(self, gray, ellipse):
        (cx, cy), (ma, Ma), angle = ellipse
        rings = [0.38, 0.60, 0.82] 
        num_sectors = 43
        sectors_symbols = []
        
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        
        local_thresh = 127 # Standard midpoint for raw camera feed
            
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
