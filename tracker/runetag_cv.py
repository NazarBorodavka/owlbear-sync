import cv2
import numpy as np
import os
import stag

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
            print(f"RuneTag-CV (STag-Hybrid): Loaded {len(self.ids_vector)} codes.")
        except Exception as e:
            print(f"RuneTag-CV Error: {e}")

    def detect(self, gray, invert=False, min_score=0.3, detect_scale=1.0):
        results = []
        try:
            # Use STag (Library 17 is for circular HD markers)
            # This is extremely robust to glare and distortion
            detected_markers, _ = stag.detectMarkers(gray, 17)
            
            for m in detected_markers:
                # corners are 4 points [top-left, top-right, bottom-right, bottom-left]
                corners = m.corners
                
                # Unwarp marker to a perfect 128x128 square
                size = 128
                dst_pts = np.array([[0,0], [size-1,0], [size-1,size-1], [0,size-1]], dtype=np.float32)
                H, _ = cv2.findHomography(corners, dst_pts)
                warped = cv2.warpPerspective(gray, H, (size, size))
                
                if invert: warped = 255 - warped
                
                tag_data = self._decode_warped(warped)
                if tag_data:
                    # Map back to original image
                    center = np.mean(corners, axis=0)
                    tag_data['center'] = (int(center[0]), int(center[1]))
                    tag_data['corners'] = corners.tolist()
                    results.append(tag_data)
                elif min_score < 0.5:
                    # Show diagnostic box for unrecognized STag candidates
                    center = np.mean(corners, axis=0)
                    results.append({
                        'id': -1,
                        'center': (int(center[0]), int(center[1])),
                        'corners': corners.tolist()
                    })
        except Exception as e:
            print(f"STag-Hybrid Error: {e}")
            
        return results

    def _decode_warped(self, warped):
        # On a 128x128 square, sampling is deterministic and 100% accurate
        cx, cy = 64, 64
        # Original DeepTag/RUNE radii scaled to 128px
        rings = [24, 38, 52] 
        num_sectors = 43
        sectors_symbols = []
        
        # Local threshold for the unwarped patch
        local_thresh = np.mean(warped)
        
        for s in range(num_sectors):
            theta = (2 * np.pi * s / num_sectors)
            val = 0
            for i, r in enumerate(rings):
                px = int(cx + r * np.cos(theta))
                py = int(cy + r * np.sin(theta))
                
                # Safe bounds and 3x3 sample
                if 1 <= px < 127 and 1 <= py < 127:
                    patch = warped[py-1:py+2, px-1:px+2]
                    if np.mean(patch) > local_thresh:
                        val += (2**i)
            
            sectors_symbols.append(val - 1 if val > 0 else -1)
        
        if len([s for s in sectors_symbols if s != -1]) < 10: return None
        
        current_pattern = np.array(sectors_symbols, dtype=np.int8)
        
        # Fast Vectorized Matching
        for shift in range(num_sectors):
            shifted = np.roll(current_pattern, -shift)
            shifted_tuple = tuple(shifted.tolist())
            if shifted_tuple in self.exact_codebook:
                return {'id': self.exact_codebook[shifted_tuple]}
            
            if self.hamming_dist > 0:
                mismatches = np.sum(self.codes_matrix != shifted, axis=1)
                best_idx = np.argmin(mismatches)
                if mismatches[best_idx] <= self.hamming_dist:
                    return {'id': int(self.ids_vector[best_idx])}
        return None
