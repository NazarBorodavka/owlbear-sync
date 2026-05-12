import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from runetag_coding import RuneTagCoding

class RuneTagDetector:
    def __init__(self, codebook_path=None, hamming_dist=4):
        self.coder = RuneTagCoding()
        self.hamming_dist = hamming_dist
        self.radii_normalized = [0.65, 0.82, 1.00]
        self.num_slots = 43

    def detect(self, img, invert=False, min_score=0.3, detect_scale=1.0, 
               adaptive_block=31, adaptive_C=3, min_dots=15):
        """
        Detects multiple RuneTags in an image.
        """
        results = []
        if img is None: return results
        
        # 1. Resize for performance if requested
        if detect_scale != 1.0:
            h, w = img.shape[:2]
            proc_img = cv2.resize(img, (int(w * detect_scale), int(h * detect_scale)))
        else:
            proc_img = img
            
        if len(proc_img.shape) == 3:
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = proc_img
            
        if invert:
            gray = 255 - gray
            
        # 2. Adaptive Thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C)
        
        # 3. Contour detection and filtering
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            if len(cnt) < 5: continue
            area = cv2.contourArea(cnt)
            if area < 5: continue
            
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (ma, mi), angle = ellipse
            if ma == 0: continue
            ratio = mi / ma
            if ratio < 0.3: continue # Allow for perspective tilt
            
            dots.append((x, y))
            
        if len(dots) < min_dots:
            return results
        
        # 4. Clustering (DBSCAN to group dots into markers)
        dots_pos = np.array(dots)
        # eps should scale with dot size/density. Approx 150 pixels for 1024 frame.
        clustering = DBSCAN(eps=80, min_samples=min_dots).fit(dots_pos)
        
        for label in set(clustering.labels_):
            if label == -1: continue # Noise
            
            cluster_indices = np.where(clustering.labels_ == label)[0]
            cluster_dots = dots_pos[cluster_indices]
            
            if len(cluster_dots) < min_dots: continue
            
            # 5. Process each cluster
            tag_data = self._process_cluster(gray, cluster_dots)
            if tag_data:
                # Scale coordinates back to original size
                if detect_scale != 1.0:
                    tag_data['center'] = (tag_data['center'][0] / detect_scale, tag_data['center'][1] / detect_scale)
                    tag_data['corners'] = [[c[0] / detect_scale, c[1] / detect_scale] for c in tag_data['corners']]
                results.append(tag_data)
                
        return results

    def _process_cluster(self, gray, cluster_dots):
        center = np.mean(cluster_dots, axis=0)
        dists = np.linalg.norm(cluster_dots - center, axis=1)
        
        # Fit outer ellipse for initial homography
        max_dist = np.max(dists)
        outer_dots = cluster_dots[dists > 0.8 * max_dist]
        if len(outer_dots) < 5: return None
        
        try:
            outer_ellipse = cv2.fitEllipse(outer_dots.astype(np.float32))
        except: return None
        
        # Homography to unit circle
        def get_ellipse_point(ellipse, ang_deg):
            (x, y), (ma, mi), ang = ellipse
            a, b = ma / 2.0, mi / 2.0
            rad = np.radians(ang_deg)
            px, py = a * np.cos(rad), b * np.sin(rad)
            s, c = np.sin(np.radians(ang)), np.cos(np.radians(ang))
            return [x + px*c - py*s, y + px*s + py*c]

        src_pts = np.array([get_ellipse_point(outer_ellipse, a) for a in [0, 90, 180, 270, 45]], dtype=np.float32)
        dst_pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0.707, 0.707]], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        
        # Transform cluster dots to normalized space
        dots_homo = np.hstack([cluster_dots, np.ones((len(cluster_dots), 1))])
        dots_norm = (H @ dots_homo.T).T
        dots_norm = dots_norm[:, :2] / dots_norm[:, 2:3]
        
        dots_r = np.linalg.norm(dots_norm, axis=1)
        dots_theta = np.arctan2(dots_norm[:, 1], dots_norm[:, 0]) % (2 * np.pi)
        
        # Try rotations to find valid code
        for rotation_offset in np.linspace(0, 2 * np.pi / 43, 8):
            bit_grid = np.zeros((43, 3), dtype=int)
            for r, theta in zip(dots_r, dots_theta):
                slot_idx = int(((theta - rotation_offset) % (2 * np.pi)) / (2 * np.pi / 43))
                ring_idx = np.argmin([abs(r - dr) for dr in self.radii_normalized])
                if abs(r - self.radii_normalized[ring_idx]) < 0.15:
                    bit_grid[slot_idx % 43, ring_idx] = 1
            
            bitcode = bit_grid.flatten().tolist()
            try:
                code = self.coder.pack(bitcode)
                # Note: We use the algorithmic decoder which is much better than a codebook
                if self.coder.decode(code) == 0:
                    aligned_code, tid, rotation = self.coder.align(code)
                    if tid >= 0:
                        # Success!
                        # We don't return full debug drawing here, just tag info
                        return {
                            'id': tid,
                            'center': (int(center[0]), int(center[1])),
                            'corners': self._get_corners(outer_ellipse)
                        }
            except: pass
        return None

    def _get_corners(self, ellipse):
        # Return a bounding box representing the tag
        (cx, cy), (ma, Ma), angle = ellipse
        pts = cv2.boxPoints(ellipse)
        return pts.astype(int).tolist()
