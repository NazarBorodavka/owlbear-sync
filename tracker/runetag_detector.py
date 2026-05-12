import numpy as np
import cv2
from runetag_coding import RuneTagCoding

class RuneTagDetector:
    def __init__(self):
        self.coder = RuneTagCoding()
        self.radii_normalized = [0.65, 0.82, 1.00]
        self.num_slots = 43

    def detect(self, img, debug_path=None):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            if len(cnt) < 5: continue
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (ma, mi), angle = ellipse
            if ma == 0: continue
            ratio = mi / ma
            if ratio < 0.4: continue
            area = cv2.contourArea(cnt)
            if area < 5: continue
            dots.append((x, y))
            
        if len(dots) < 10:
            return None
        
        dots_pos = np.array(dots)
        center = np.mean(dots_pos, axis=0)
        dists = np.linalg.norm(dots_pos - center, axis=1)
        
        # Fit outer ellipse to get homography
        max_dist = np.max(dists)
        outer_dots = dots_pos[dists > 0.8 * max_dist]
        if len(outer_dots) < 5: return None
        outer_ellipse = cv2.fitEllipse(outer_dots.astype(np.float32))
        (ex, ey), (ema, emi), eangle = outer_ellipse
        
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
        
        # Transform dots to normalized space
        dots_homo = np.hstack([dots_pos, np.ones((len(dots_pos), 1))])
        dots_norm = (H @ dots_homo.T).T
        dots_norm = dots_norm[:, :2] / dots_norm[:, 2:3]
        
        # Convert to polar
        dots_r = np.linalg.norm(dots_norm, axis=1)
        dots_theta = np.arctan2(dots_norm[:, 1], dots_norm[:, 0]) % (2 * np.pi)
        
        # Binning
        for rotation_offset in np.linspace(0, 2 * np.pi / 43, 10):
            bit_grid = np.zeros((43, 3), dtype=int)
            for r, theta in zip(dots_r, dots_theta):
                # Slot index
                slot_idx = int(((theta - rotation_offset) % (2 * np.pi)) / (2 * np.pi / 43))
                # Ring index
                # Radii: 0.49, 0.70, 1.00
                # Find closest ring
                ring_idx = np.argmin([abs(r - dr) for dr in self.radii_normalized])
                if abs(r - self.radii_normalized[ring_idx]) < 0.15:
                    bit_grid[slot_idx % 43, ring_idx] = 1
            
            bitcode = bit_grid.flatten().tolist()
            try:
                code = self.coder.pack(bitcode)
                if self.coder.decode(code) == 0:
                    aligned_code, tid, rotation = self.coder.align(code)
                    if tid >= 0:
                        # Refine Homography using ALL dots for perfect centering
                        src_pts_refine = []
                        dst_pts_refine = []
                        # Slot 0 in image corresponds to slot 'rotation' in aligned_code?
                        # Actually, let's just re-bin to find the mapping
                        for r, theta, i in zip(dots_r, dots_theta, range(len(dots_r))):
                            s_idx = int(((theta - rotation_offset) % (2 * np.pi)) / (2 * np.pi / 43))
                            r_idx = np.argmin([abs(r - dr) for dr in self.radii_normalized])
                            if abs(r - self.radii_normalized[r_idx]) < 0.15:
                                # This dot (dots_pos[i]) belongs to s_idx, r_idx
                                ideal_angle = rotation_offset + (s_idx + 0.5) * (2 * np.pi / 43)
                                ideal_r = self.radii_normalized[r_idx]
                                src_pts_refine.append(dots_pos[i])
                                dst_pts_refine.append([ideal_r * np.cos(ideal_angle), ideal_r * np.sin(ideal_angle)])
                        
                        if len(src_pts_refine) >= 4:
                            H_refined, _ = cv2.findHomography(np.array(src_pts_refine), np.array(dst_pts_refine))
                            H_inv = np.linalg.inv(H_refined)
                        else:
                            H_inv = np.linalg.inv(H)

                        if debug_path:
                            dbg_img = img.copy()
                            cv2.putText(dbg_img, f"Detected ID: {tid}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Draw rings
                            for ring_r in self.radii_normalized:
                                pts = []
                                for a in np.linspace(0, 2*np.pi, 200):
                                    p = H_inv @ [ring_r*np.cos(a), ring_r*np.sin(a), 1]
                                    pts.append([int(p[0]/p[2]), int(p[1]/p[2])])
                                cv2.polylines(dbg_img, [np.array(pts)], True, (255, 255, 0), 1, cv2.LINE_AA)
                            
                            # Draw lines for slots (half-way between slots)
                            for s in range(43):
                                a = rotation_offset + s * (2 * np.pi / 43)
                                p1 = H_inv @ [0, 0, 1]
                                p2 = H_inv @ [1.1 * np.cos(a), 1.1 * np.sin(a), 1]
                                cv2.line(dbg_img, (int(p1[0]/p1[2]), int(p1[1]/p1[2])), (int(p2[0]/p2[2]), int(p2[1]/p2[2])), (0, 0, 255), 1, cv2.LINE_AA)
                            
                            cv2.imwrite(debug_path, dbg_img)
                        return tid
            except:
                pass
        
        return None
