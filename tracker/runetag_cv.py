import cv2
import numpy as np
import os
from skimage.feature import blob_log

class RuneTagDetector:
    def __init__(self, codebook_path=None, hamming_dist=4):
        self.codebook = {}
        self.codes_matrix = None
        self.ids_vector = None
        self.exact_codebook = {}
        self.hamming_dist = hamming_dist
        
        if codebook_path:
            self.load_codebook(codebook_path)

    def load_codebook(self, path):
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
                self.exact_codebook = {tuple(c): i for c, i in zip(temp_codes, temp_ids)}
            print(f"RuneTag-CV (Blob-Pro): Loaded {len(self.ids_vector)} codes.")
        except Exception as e:
            print(f"RuneTag-CV Error: {e}")

    def detect(self, gray, invert=False, min_score=0.3, detect_scale=1.0):
        results = []
        if invert: gray = 255 - gray
            
        # 1. Professional Blob Detection (Laplacian of Gaussian)
        # This is very robust to glare and handles different dot sizes
        # min_sigma/max_sigma are dot sizes in pixels
        blobs = blob_log(gray, min_sigma=1*detect_scale, max_sigma=5*detect_scale, num_sigma=5, threshold=0.1)
        
        if len(blobs) < 10: return results
        
        # blobs is (y, x, sigma)
        dots = blobs[:, :2][:, ::-1] # Convert to (x, y)
        
        # 2. Cluster and Fit
        # We look for clusters of dots that form an ellipse
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=50*detect_scale, min_samples=max(8, int(43 * min_score))).fit(dots)
        
        for label in set(clustering.labels_):
            if label == -1: continue
            cluster_dots = dots[clustering.labels_ == label]
            
            if len(cluster_dots) >= 10:
                # Fit ellipse using standard library fitEllipse
                ellipse = cv2.fitEllipse(cluster_dots.astype(np.float32))
                
                tag_data = self._decode_ellipse(gray, ellipse)
                if tag_data:
                    results.append(tag_data)
                elif min_score < 0.2:
                    # Debug box for potential markers
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
        
        # Local threshold for the marker area
        try:
            x, y, w, h = cv2.boundingRect(np.array(self._get_ellipse_corners(ellipse), dtype=np.int32))
            roi = gray[max(0,y):y+h, max(0,x):x+w]
            local_thresh = np.mean(roi) if roi.size > 0 else 127
        except:
            local_thresh = 127
            
        for s in range(num_sectors):
            theta = (2 * np.pi * s / num_sectors)
            val = 0
            for i, r_scale in enumerate(rings):
                rx, ry = (ma/2) * r_scale, (Ma/2) * r_scale
                local_x, local_y = rx * np.cos(theta), ry * np.sin(theta)
                px, py = int(cx + local_x * cos_a - local_y * sin_a), int(cy + local_x * sin_a + local_y * cos_a)
                
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    # Sub-pixel sampling using 3x3 patch
                    patch = gray[max(0, py-1):py+2, max(0, px-1):px+2]
                    if np.mean(patch) > local_thresh:
                        val += (2**i)
            
            sectors_symbols.append(val - 1 if val > 0 else -1)
        
        if len([s for s in sectors_symbols if s != -1]) < 8: return None
        
        current_pattern = np.array(sectors_symbols, dtype=np.int8)
        
        # Fast Vectorized Matching
        for shift in range(num_sectors):
            shifted = np.roll(current_pattern, -shift)
            shifted_tuple = tuple(shifted.tolist())
            if shifted_tuple in self.exact_codebook:
                return {
                    'id': self.exact_codebook[shifted_tuple],
                    'center': (int(cx), int(cy)),
                    'corners': self._get_ellipse_corners(ellipse)
                }
            
            if self.hamming_dist > 0:
                mismatches = np.sum(self.codes_matrix != shifted, axis=1)
                best_idx = np.argmin(mismatches)
                if mismatches[best_idx] <= self.hamming_dist:
                    return {
                        'id': int(self.ids_vector[best_idx]),
                        'center': (int(cx), int(cy)),
                        'corners': self._get_ellipse_corners(ellipse)
                    }
        return None

    def _get_ellipse_corners(self, ellipse):
        (cx, cy), (ma, Ma), angle = ellipse
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        # Approximate 4 corners of the bounding box of the ellipse
        pts = []
        for dx, dy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
            rx, ry = dx * (ma/2), dy * (Ma/2)
            px = cx + rx * cos_a - ry * sin_a
            py = cy + rx * sin_a + ry * cos_a
            pts.append([int(px), int(py)])
        return pts
