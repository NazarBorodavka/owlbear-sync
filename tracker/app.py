import cv2
import numpy as np
import time
import concurrent.futures
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import threading
import time
import os
from flask_httpauth import HTTPBasicAuth
import json
import collections

# CCTag Support
CCTAG_AVAILABLE = False
cctag_detector = None
try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
    try:
        from cctag_ext import FastCCTagDetector
        CCTAG_AVAILABLE = True
        cctag_detector = None
        cctag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        print("[OK] CCTag Native Backend Initialized.")
    except Exception as e:
        print("CCTag init error:", e)
        CCTAG_AVAILABLE = False
except Exception as e:
    print("CCTag init error:", e)
    CCTAG_AVAILABLE = False

class IPCameraCapture:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        # Use deque with maxlen=1 for atomic "latest frame" access
        self.frame_buffer = collections.deque(maxlen=1)
        self.is_running = True
        
        # Tapo/RTSP specific optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            self.frame_buffer.append(frame)

    def read(self):
        if not self.frame_buffer:
            return False, None
        return True, self.frame_buffer[0]

    def isOpened(self):
        return self.cap.isOpened()

    def grab(self):
        return True

    def retrieve(self):
        return self.read()

    def release(self):
        self.is_running = False
        self.cap.release()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder=None)
app.config['JSON_SORT_KEYS'] = False

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Simple HTTP Basic Auth (used by route decorators)
auth = HTTPBasicAuth()
# Default user store (can be overridden by config.json on disk)
USER_DATA = {"admin": "admin"}


@auth.verify_password
def verify_password(username, password):
    """Verify username/password against USER_DATA.

    Returns True when credentials match, False otherwise.
    The stored passwords are plain-text here for simplicity; consider
    switching to hashed passwords for production.
    """
    if not username or not password:
        return False
    expected = USER_DATA.get(username)
    return expected is not None and expected == password

# State variables
camera_url = "" # IP camera URL
cap = None
is_running = False
# Store config at the container bind-mount path used by docker-compose/CasaOS.
CONFIG_FILE = os.environ.get('TRACKER_CONFIG_FILE', '/app/config.json')
LEGACY_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')


def load_config_from_disk():
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure
    global hough_param1, hough_param2, hough_min_radius, hough_max_radius
    global auto_blank
    global cctag_min_area, cctag_min_id, cctag_max_id
    global token_aliases, manual_blank
    global camera_matrix, dist_coeffs, calibration_model, settings_dirty, undistort_map1, undistort_map2
    global src_pts, corner_idx, homography_matrix
    for config_path in (CONFIG_FILE, LEGACY_CONFIG_FILE):
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                c = json.load(f)
                cctag_min_area = float(c.get('cctag_min_area', 100.0))
                cctag_min_id = int(c.get('cctag_min_id', 0))
                cctag_max_id = int(c.get('cctag_max_id', 9))
                cctag_min_ident_proba = float(c.get('cctag_min_ident_proba', 1e-6))
                if 'password' in c:
                    USER_DATA["admin"] = c['password']
                distortion_k1 = c.get('distortion_k1', 0.0)
                zoom_level = c.get('zoom_level', 1.0)
                offset_x = c.get('offset_x', 0.0)
                offset_y = c.get('offset_y', 0.0)
                rotation = c.get('rotation', 0.0)
                brightness = c.get('brightness', 0.0)
                contrast = c.get('contrast', 1.0)
                exposure = c.get('exposure', 1.0)
                hough_param1 = c.get('hough_param1', 40)
                hough_param2 = c.get('hough_param2', 45)
                hough_min_radius = c.get('hough_min_radius', 30)
                hough_max_radius = c.get('hough_max_radius', 40)
                auto_blank = c.get('auto_blank', False)
                token_aliases = c.get('token_aliases', {})
                print(f"Loaded config from disk: {config_path}")
                # Load camera calibration if present
                cm = c.get('camera_matrix')
                dd = c.get('dist_coeffs')
                if cm is not None and dd is not None:
                    try:
                        import numpy as _np
                        camera_matrix = _np.array(cm, dtype=_np.float64)
                        dist_coeffs = _np.array(dd, dtype=_np.float64)
                        calibration_model = c.get('calibration_model') or ('fisheye' if _np.array(dd).size == 4 else 'standard')
                        settings_dirty = True
                        undistort_map1 = None
                        undistort_map2 = None
                        print(f"Loaded camera calibration from config ({calibration_model}).")
                    except Exception:
                        camera_matrix = None
                        dist_coeffs = None
                        calibration_model = None
                
                # Load corner calibration
                if 'src_pts' in c and 'corner_idx' in c:
                    loaded_idx = c['corner_idx']
                    if loaded_idx > 0 and 'src_pts' in c:
                        import numpy as _np
                        import cv2 as _cv2
                        loaded_pts = c['src_pts']
                        for i in range(min(4, loaded_idx, len(loaded_pts))):
                            src_pts[i] = [float(loaded_pts[i][0]), float(loaded_pts[i][1])]
                        corner_idx = loaded_idx
                        
                        if corner_idx == 4:
                            dst_pts = _np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=_np.float32)
                            homography_matrix, _ = _cv2.findHomography(src_pts, dst_pts)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            continue
        break

def save_config_to_disk():
    global calibration_model
    c = {
        'distortion_k1': distortion_k1, 'zoom_level': zoom_level, 'offset_x': offset_x, 'offset_y': offset_y,
        'rotation': rotation, 'brightness': brightness, 'contrast': contrast, 'exposure': exposure,
        'hough_param1': hough_param1, 'hough_param2': hough_param2,
        'hough_min_radius': hough_min_radius, 'hough_max_radius': hough_max_radius,
        'auto_blank': auto_blank,
        'cctag_min_area': cctag_min_area,
        'cctag_min_id': cctag_min_id,
        'cctag_max_id': cctag_max_id,
        'cctag_min_ident_proba': cctag_min_ident_proba,
        'token_aliases': token_aliases,
        'password': USER_DATA.get("admin", "admin"),
        'calibration_model': calibration_model,
        'corner_idx': corner_idx,
        'src_pts': src_pts.tolist()[:corner_idx] if corner_idx > 0 else []
    }
    # Save camera calibration if available
    if camera_matrix is not None and dist_coeffs is not None:
        try:
            c['camera_matrix'] = camera_matrix.tolist()
            c['dist_coeffs'] = dist_coeffs.tolist()
        except Exception:
            pass
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    except Exception:
        pass
    for path in (CONFIG_FILE, LEGACY_CONFIG_FILE if LEGACY_CONFIG_FILE != CONFIG_FILE else None):
        if not path:
            continue
        try:
            with open(path, 'w') as f:
                json.dump(c, f)
            print(f"Saved config to {path}")
            return
        except Exception as e:
            print(f"Error saving config to {path}: {e}")
    print("Error saving config: all save targets failed")

# Preprocessing parameters
distortion_k1 = 0.0
zoom_level = 1.0
offset_x = 0.0
offset_y = 0.0
rotation = 0.0
brightness = 0.0
contrast = 1.0
exposure = 1.0
show_overlay = True
token_aliases = {}
ignored_tokens = set()

# Disk Detection (Hough Circles)
hough_dp = 1.2
hough_min_dist = 20
hough_param1 = 40
hough_param2 = 30
hough_min_radius = 30
hough_max_radius = 40

auto_blank = False # Toggle for anti-reflection mode
flip_x = False
flip_y = False

cctag_min_area = 100.0
cctag_min_id = 0
cctag_max_id = 9
cctag_min_ident_proba = 1e-6
manual_blank = False

# Performance optimization: Cache for software exposure table
exposure_table = None
last_exposure = -1.0

# Calibration corners
src_pts = np.zeros((4, 2), dtype=np.float32)
corner_idx = 0
homography_matrix = None

# Add locks for thread-safe frame reading
frame_lock = threading.Lock()
camera_lock = threading.Lock()
current_frame = None

undistort_map1 = None
undistort_map2 = None
undistort_model = None
settings_dirty = True

# For diagnostic overlay: keep a raw and undistorted copy of the latest frame
raw_frame_for_stream = None
undistorted_frame_for_stream = None

# Camera calibration state (fisheye model)
camera_matrix = None
dist_coeffs = None
calibration_model = None
calib_mode = False
calib_objpoints = []
calib_imgpoints = []
# Default chessboard pattern (cols, rows) internal corners - change if you use a different board
chessboard_size = (9, 6)

load_config_from_disk()


def get_video_stream():
    global cap, is_running, current_frame, camera_url, undistort_map1, undistort_map2, settings_dirty, undistort_model
    # Expose diagnostic copies of the latest frame
    global raw_frame_for_stream, undistorted_frame_for_stream
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure, show_overlay
    global hough_dp, hough_min_dist, hough_param1, hough_param2, hough_min_radius, hough_max_radius
    global CCTAG_AVAILABLE, cctag_detector
    global src_pts, corner_idx, homography_matrix, auto_blank, manual_blank
    global cctag_min_area, cctag_min_id, cctag_max_id
    global camera_matrix, dist_coeffs, calibration_model

    fail_count = 0
    
    # 20 FPS target
    frame_interval = 0.1 
    last_marker_ids = set()
    
    while is_running:
        loop_start = time.time()
        
        success = False
        frame = None
        
        with camera_lock:
            if cap is not None and cap.isOpened():
                # To prevent OpenCV from buffering and causing slow-motion lag, 
                # we grab frames rapidly to clear the buffer, then retrieve the latest one.
                # Since OpenCV doesn't easily let us bypass the TCP buffer, grabbing 2-3 extra frames catches us up.
                for _ in range(4):
                    cap.grab()
                success, frame = cap.retrieve()
            
        if not success or frame is None:
            fail_count += 1
            if fail_count % 30 == 0:
                print(f"Waiting for stream... (attempt {fail_count}/150)", flush=True)
            if fail_count > 150: # Wait up to 15 seconds for stream to start
                print(f"Stream connection timed out (url: {camera_url}), attempting full reconnect...", flush=True)
                with camera_lock:
                    if cap is not None:
                        cap.release()
                    
                    try:
                        source = int(camera_url)
                    except (ValueError, TypeError):
                        source = camera_url
                    
                    if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp') or source.startswith('rtmp')):
                        cap = IPCameraCapture(source)
                    else:
                        cap = cv2.VideoCapture(source)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                fail_count = 0
                time.sleep(1)
            else:
                time.sleep(0.1)
            continue
            
        fail_count = 0

        # Preprocessing: Apply Distortion Correction, Zoom, Pan, Rotation, Colors
        h, w = frame.shape[:2]
        # Store raw frame for diagnostic streaming (before any processing)
        try:
            raw_frame_for_stream = frame.copy()
        except Exception:
            raw_frame_for_stream = None
        
        # 1. Distortion Correction via precomputed maps.
        # Prefer a full camera calibration when available. Fall back to single-k1 model.
        if camera_matrix is not None and dist_coeffs is not None:
            model = calibration_model or ('fisheye' if getattr(dist_coeffs, 'size', 0) == 4 else 'standard')
            if settings_dirty or undistort_map1 is None or undistort_model != model:
                try:
                    if model == 'standard':
                        dist = np.array(dist_coeffs, dtype=np.float64).reshape(-1, 1)
                        cam = np.array(camera_matrix, dtype=np.float64)
                        new_cam, _ = cv2.getOptimalNewCameraMatrix(cam, dist, (w, h), 1.0, (w, h))
                        undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(cam, dist, None, new_cam, (w, h), cv2.CV_32FC1)
                    else:
                        cam = np.array(camera_matrix, dtype=np.float64)
                        dist = np.array(dist_coeffs, dtype=np.float64)
                        new_cam = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                            cam, dist, (w, h), np.eye(3), balance=1.0)
                        undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                            cam, dist, np.eye(3), new_cam, (w, h), cv2.CV_32FC1)
                    undistort_model = model
                    settings_dirty = False
                except Exception:
                    # If fisheye module or functions are not available, skip calibration
                    undistort_map1 = None
                    undistort_map2 = None
            if undistort_map1 is not None:
                frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)
                try:
                    undistorted_frame_for_stream = frame.copy()
                except Exception:
                    undistorted_frame_for_stream = None
        elif distortion_k1 != 0.0:
            # Legacy single-coefficient radial distortion correction
            if settings_dirty or undistort_map1 is None:
                fx, fy = w, h
                cx, cy = w / 2, h / 2
                tmp_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist = np.array([distortion_k1, 0, 0, 0, 0], dtype=np.float32)
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(tmp_cam, dist, (w, h), 1.0)
                undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(tmp_cam, dist, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
                settings_dirty = False
            frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)
            try:
                undistorted_frame_for_stream = frame.copy()
            except Exception:
                undistorted_frame_for_stream = None
            
        # 2. Optimized Zoom, Pan, Rotation (Merged into one warp)
        if zoom_level != 1.0 or offset_x != 0.0 or offset_y != 0.0 or rotation != 0.0:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, zoom_level)
            M[0, 2] += offset_x * w
            M[1, 2] += offset_y * h
            frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        # 3. Brightness, Contrast, Exposure
        # Apply Brightness and Contrast
        if brightness != 0.0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
        # Apply software Exposure (Gamma correction)
        if exposure != 1.0 and exposure > 0:
            global exposure_table, last_exposure
            if exposure != last_exposure or exposure_table is None:
                invGamma = 1.0 / exposure
                exposure_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                last_exposure = exposure
            frame = cv2.LUT(frame, exposure_table)

        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Apply Calibration Mask ---
        # If the user has set the 4 calibration corners, ignore anything outside that area
        if corner_idx == 4:
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [np.int32(src_pts)], 255)
            gray = cv2.bitwise_and(gray, mask)
        
        # --- Optimized Circle Detection (Downsampled) ---
        # Always run disk detection so AprilTag mode can be constrained to disks.
        circles = None
        # Resize to 50% for circle search - much faster and reduces noise
        small_gray = cv2.resize(gray, (w // 2, h // 2))
        blurred = cv2.GaussianBlur(small_gray, (5, 5), 1.5)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_min_dist // 2,
                                param1=hough_param1, param2=hough_param2,
                                minRadius=hough_min_radius // 2, maxRadius=hough_max_radius // 2)
                                    
        detected_circles = []
        if circles is not None:
            circles = circles[0, :]
            # Sort by radius descending
            sorted_circles = sorted(circles, key=lambda x: x[2], reverse=True)
            
            for (sx, sy, sr) in sorted_circles:
                # Scale coordinates back up
                x, y, r = int(sx * 2), int(sy * 2), int(sr * 2)
                # Check if this circle is inside any already accepted circle
                is_inner = False
                for (ax, ay, ar) in detected_circles:
                    dist = np.sqrt((x - ax)**2 + (y - ay)**2)
                    if dist < ar * 1.5: # Center is close to an existing larger circle (nested ring)
                        is_inner = True
                        break
                if not is_inner:
                    detected_circles.append((x, y, r))
                    # Draw the circle in faint yellow
                    if show_overlay:
                        cv2.circle(frame, (x, y), r, (0, 255, 255), 2)

        # Collect CCTag detections per disk
        cctag_candidates = {}
        markers = {}
        
        if CCTAG_AVAILABLE and cctag_detector is None:
            cctag_detector = FastCCTagDetector(3) # 3 rings
            
        def _detect_single_roi(circ, frame_gray=gray, fw=w, fh=h):
            circ_x, circ_y, circ_r = circ
            pad = int(circ_r * 0.2) + 10
            y1, y2 = max(0, circ_y - circ_r - pad), min(fh, circ_y + circ_r + pad)
            x1, x2 = max(0, circ_x - circ_r - pad), min(fw, circ_x + circ_r + pad)

            roi = frame_gray[y1:y2, x1:x2]
            if roi.size < 100:
                return []

            roi_enhanced = cv2.equalizeHist(roi)
            img = np.ascontiguousarray(roi_enhanced, dtype=np.uint8)
            
            try:
                raw_results = cctag_detector.detect(img, min_ident_proba=cctag_min_ident_proba, cx=img.shape[1]/2.0, cy=img.shape[0]/2.0, fx=800.0, fy=800.0, offset_x=float(x1), offset_y=float(y1))
                if raw_results:
                    best_result = max(raw_results, key=lambda x: x.get('decision_margin', 0))
                    return [best_result]
                return []
            except Exception as e:
                print(f"CCTag detection error: {e}")
                return []

        # 1. Check if previous background CCTag analysis is complete
        if hasattr(get_video_stream, "cctag_futures") and get_video_stream.cctag_futures:
            if all(f.done() for f in get_video_stream.cctag_futures):
                for future in get_video_stream.cctag_futures:
                    try:
                        raw_results = future.result()
                        for r in raw_results:
                            rid = int(r.get('idx', -1))
                            if rid < cctag_min_id or rid > cctag_max_id:
                                continue
                            
                            center = (float(r.get('x', 0)), float(r.get('y', 0)))
                            cctag_candidates.setdefault(rid, []).append({
                                "center": center,
                                "decision_margin": float(r.get('decision_margin', 1.0))
                            })
                    except Exception as e:
                        pass
                get_video_stream.cctag_futures = None

        # 2. If free, submit current frame for async analysis
        if CCTAG_AVAILABLE and detected_circles and not getattr(get_video_stream, "cctag_futures", None):
            get_video_stream.cctag_futures = [cctag_executor.submit(_detect_single_roi, circ, gray, w, h) for circ in detected_circles]

        if cctag_candidates:
            if not hasattr(get_video_stream, "tracked_tokens"):
                get_video_stream.tracked_tokens = {}

            for rid, candidates in cctag_candidates.items():
                prev = get_video_stream.tracked_tokens.get(f"Marker_{rid}")

                def _cand_score(c):
                    dm = c.get('decision_margin', -1.0)
                    if prev is None:
                        return dm
                    cx, cy = c['center']
                    dist = float(np.hypot(prev["x"] - cx, prev["y"] - cy))
                    # Prefer confidence, but bias toward prior position to avoid jitter when multiple disks are nearby.
                    return dm - (dist * 0.01)

                best = max(candidates, key=_cand_score)
                markers[rid] = best['center']

        # --- Temporal token fusion ---
        detected_tokens = []
        if not hasattr(get_video_stream, "tracked_tokens"):
            get_video_stream.tracked_tokens = {}
        matched_ids = set()
        used_circles = set()

        # 1. Process all detected markers (primary source of truth)
        for m_id, (m_x, m_y) in markers.items():
            token_id = f"Marker_{m_id}"
            
            # Find the best circle that encloses this marker for precision
            best_circ = None
            best_dist = float('inf')
            for (cx, cy, cr) in detected_circles:
                if (cx, cy, cr) in used_circles: continue
                d = np.sqrt((cx - m_x)**2 + (cy - m_y)**2)
                if d < cr and d < best_dist:
                    best_dist = d
                    best_circ = (cx, cy, cr)
            
            if best_circ:
                cx, cy, cr = best_circ
                used_circles.add(best_circ)
                if token_id in get_video_stream.tracked_tokens:
                    t = get_video_stream.tracked_tokens[token_id]
                    # Dynamic smoothing: if distance is large (moving fast), snap instantly. If small (stationary/jitter), smooth heavily.
                    dist = np.hypot(t["x"] - cx, t["y"] - cy)
                    alpha = 1.0 if dist > 15 else 0.2
                    t["x"] = t["x"] * (1 - alpha) + cx * alpha
                    t["y"] = t["y"] * (1 - alpha) + cy * alpha
                    t["r"] = t["r"] * 0.8 + cr * 0.2 # Smooth radius heavily to avoid outline jitter
                    t["missed"] = 0
                else:
                    get_video_stream.tracked_tokens[token_id] = {
                        "x": cx, "y": cy, "r": cr, "missed": 0, "marker_id": m_id
                    }
            else:
                # No disk found? Use marker center as fallback
                if token_id in get_video_stream.tracked_tokens:
                    t = get_video_stream.tracked_tokens[token_id]
                    dist = np.hypot(t["x"] - m_x, t["y"] - m_y)
                    alpha = 1.0 if dist > 15 else 0.2
                    t["x"] = t["x"] * (1 - alpha) + m_x * alpha
                    t["y"] = t["y"] * (1 - alpha) + m_y * alpha
                    t["missed"] = 0
                else:
                    get_video_stream.tracked_tokens[token_id] = {
                        "x": m_x, "y": m_y, "r": 25, "missed": 0, "marker_id": m_id
                    }
            matched_ids.add(token_id)
            
        # --- Appear/Disappear Logging ---
        current_marker_ids = set(markers.keys())
        if not hasattr(get_video_stream, "last_marker_ids"):
            get_video_stream.last_marker_ids = set()
            
        new_ids = current_marker_ids - get_video_stream.last_marker_ids
        lost_ids = get_video_stream.last_marker_ids - current_marker_ids
        
        for nid in new_ids:
            print(f"[TRACKER] Token Detected: ID {nid}", flush=True)
        for lid in lost_ids:
            print(f"[TRACKER] Token Lost: ID {lid}", flush=True)
            
        get_video_stream.last_marker_ids = current_marker_ids

        # 2. Temporal Fallback: Match remaining circles to "missed" tokens
        for (cx, cy, cr) in detected_circles:
            if (cx, cy, cr) in used_circles: continue

            best_id = None
            best_dist = 60 # Search radius for moving markers
            for t_id, t_data in get_video_stream.tracked_tokens.items():
                if t_id in matched_ids: continue
                dist = np.sqrt((cx - t_data["x"])**2 + (cy - t_data["y"])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = t_id
            
            if best_id:
                token = get_video_stream.tracked_tokens[best_id]
                token = get_video_stream.tracked_tokens[best_id]
                # Update position based on circle, maintain ID with dynamic smoothing
                dist_moved = np.hypot(token["x"] - cx, token["y"] - cy)
                alpha = 1.0 if dist_moved > 15 else 0.2
                token["x"] = token["x"] * (1 - alpha) + cx * alpha
                token["y"] = token["y"] * (1 - alpha) + cy * alpha
                token["r"] = token["r"] * 0.8 + cr * 0.2
                token["missed"] = 0
                matched_ids.add(best_id)
                used_circles.add((cx, cy, cr))

        # Increment missed frames and delete old tokens
        for token_id in list(get_video_stream.tracked_tokens.keys()):
            if token_id not in matched_ids:
                get_video_stream.tracked_tokens[token_id]["missed"] += 1
                if get_video_stream.tracked_tokens[token_id]["missed"] > 10: # ~0.5 second ghosting
                    del get_video_stream.tracked_tokens[token_id]
                    
        # Render and prepare payloads
        for token_id, t_data in get_video_stream.tracked_tokens.items():
            if token_id in ignored_tokens:
                continue

            display_name = token_aliases.get(token_id, token_id)
            
            if show_overlay:
                cv2.circle(frame, (int(t_data["x"]), int(t_data["y"])), int(t_data["r"]), (0, 255, 0), 2)
                cv2.circle(frame, (int(t_data["x"]), int(t_data["y"])), 5, (0, 255, 0), -1)
                cv2.putText(frame, display_name, (int(t_data["x"]) + 10, int(t_data["y"]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
            grid_x, grid_y = t_data["x"], t_data["y"]
            if homography_matrix is not None:
                pts = np.array([[[t_data["x"], t_data["y"]]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(pts, homography_matrix)
                grid_x, grid_y = dst[0][0]
            else:
                h, w = frame.shape[:2]
                grid_x = t_data["x"] / w
                grid_y = t_data["y"] / h
            
            # Apply Flips and Clamp to 0.0-1.0
            if flip_x: grid_x = 1.0 - grid_x
            if flip_y: grid_y = 1.0 - grid_y
            
            grid_x = max(0.0, min(1.0, grid_x))
            grid_y = max(0.0, min(1.0, grid_y))
                
            detected_tokens.append({
                "id": token_id,
                "alias": token_aliases.get(token_id),
                "x": float(grid_x),
                "y": float(grid_y),
                "has_base": True
            })
                
        # Clean up tokens that have been missing for too long (e.g. 5 seconds)
        to_remove = []
        for t_id, t_data in get_video_stream.tracked_tokens.items():
            if t_data["missed"] > 100: # ~5 seconds at 20fps
                to_remove.append(t_id)
        for t_id in to_remove:
            del get_video_stream.tracked_tokens[t_id]

        # Check if any confirmed token is currently "missed" for more than 3 frames
        any_missed = False
        if auto_blank:
            for t_id, t_data in get_video_stream.tracked_tokens.items():
                if t_data["missed"] > 2: # 2-frame buffer
                    any_missed = True
                    break
        
        if any_missed:
            missed_ids = [tid for tid, td in get_video_stream.tracked_tokens.items() if td["missed"] > 2]
            print(f"DEBUG: Blackout active! Missing tokens: {missed_ids}")

        # --- Final Blackout Decision ---
        # Blackout if: Manual override is ON OR (Auto-Blank is ON and tokens are missing)
        blackout_active = manual_blank or (auto_blank and any_missed)

        # Send data to websocket clients
        socketio.emit('tokens_update', {
            "tokens": detected_tokens,
            "blank_screen": blackout_active
        })
            
        # Draw radius guide and calibration corners
        if show_overlay:
            # Radius Guide (Top-Left)
            gx, gy = 70, 70
            cv2.circle(frame, (gx, gy), int(hough_max_radius), (150, 150, 150), 1)
            cv2.circle(frame, (gx, gy), int(hough_min_radius), (255, 255, 255), 1)
            cv2.putText(frame, f"Size Guide: {int(hough_min_radius)}-{int(hough_max_radius)}", (gx - 50, gy + int(hough_max_radius) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            for i in range(corner_idx):
                cv2.circle(frame, (int(src_pts[i][0]), int(src_pts[i][1])), 8, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(frame, (int(src_pts[i-1][0]), int(src_pts[i-1][1])), (int(src_pts[i][0]), int(src_pts[i][1])), (255, 0, 0), 2)
                if corner_idx == 4 and i == 3:
                    cv2.line(frame, (int(src_pts[3][0]), int(src_pts[3][1])), (int(src_pts[0][0]), int(src_pts[0][1])), (255, 0, 0), 2)

        with frame_lock:
            current_frame = frame.copy()
            
        # Throttle loop to maintain ~5 FPS target
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                # Need to yield *something* so the stream keeps connection alive
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Limit framerate to browser to save bandwidth

def generate_raw_frames():
    global raw_frame_for_stream
    while True:
        try:
            if raw_frame_for_stream is None:
                time.sleep(0.1)
                continue
            with frame_lock:
                rf = raw_frame_for_stream.copy() if raw_frame_for_stream is not None else None
            if rf is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', rf)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            time.sleep(0.1)


def generate_undistorted_frames():
    global undistorted_frame_for_stream
    while True:
        try:
            if undistorted_frame_for_stream is None:
                time.sleep(0.1)
                continue
            with frame_lock:
                uf = undistorted_frame_for_stream.copy() if undistorted_frame_for_stream is not None else None
            if uf is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', uf)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            time.sleep(0.1)

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@auth.login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/diag_raw_feed')
def diag_raw_feed():
    return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/diag_undistort_feed')
def diag_undistort_feed():
    return Response(generate_undistorted_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/diagnostic')
def diagnostic_page():
    # Simple side-by-side viewer for raw vs undistorted frames
    html = '''<!doctype html>
<html><head><title>Camera Diagnostic</title></head><body>
<h2>Camera Diagnostic: Raw (left) vs Undistorted (right)</h2>
<div style="display:flex;gap:10px;">
  <div><h3>Raw</h3><img id="raw" src="/diag_raw_feed" style="max-width:45vw;" /></div>
  <div><h3>Undistorted</h3><img id="und" src="/diag_undistort_feed" style="max-width:45vw;"/></div>
</div>
<p>Use this page to visually verify undistortion. Reload after calibration finishes.</p>
</body></html>'''
    return html

@app.route('/api/connect', methods=['POST'])
def connect_camera():
    global camera_url, cap, is_running
    data = request.json
    camera_url = data.get('url', '')
    
    with camera_lock:
        if cap is not None:
            cap.release()
            
        # Use 0 for local webcam if url is empty or '0'
        try:
            source = int(camera_url)
        except (ValueError, TypeError):
            source = camera_url
            
        if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp') or source.startswith('rtmp')):
            cap = IPCameraCapture(source)
        else:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Could not open camera"})
            
    if not is_running:
        is_running = True
        threading.Thread(target=get_video_stream, daemon=True).start()
        
    return jsonify({"success": True})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global src_pts, corner_idx, homography_matrix
    data = request.json
    action = data.get('action')
    
    if action == 'add_point':
        x = data.get('x')
        y = data.get('y')
        if corner_idx < 4:
            src_pts[corner_idx] = [x, y]
            corner_idx += 1
            
        if corner_idx == 4:
            # Map to a standard square 0.0 to 1.0 space
            dst_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            save_config_to_disk()
            
        return jsonify({"success": True, "corners": corner_idx})
        
    elif action == 'reset':
        corner_idx = 0
        src_pts = np.zeros((4, 2), dtype=np.float32)
        homography_matrix = None
        save_config_to_disk()
        return jsonify({"success": True})


@app.route('/api/camera_calibration/start', methods=['POST'])
@auth.login_required
def camera_calibration_start():
    """Begin a new camera calibration session (fisheye model)."""
    global calib_mode, calib_objpoints, calib_imgpoints, calibration_model
    calib_mode = True
    calib_objpoints = []
    calib_imgpoints = []
    calibration_model = None
    return jsonify({"success": True, "message": "Calibration started; submit frames using /api/camera_calibration/add_frame"})


@app.route('/api/camera_calibration/add_frame', methods=['POST'])
@auth.login_required
def camera_calibration_add_frame():
    """Capture current frame and try to detect chessboard corners. Returns how many valid frames collected."""
    global calib_mode, calib_objpoints, calib_imgpoints, chessboard_size
    if not calib_mode:
        return jsonify({"success": False, "error": "Calibration not started"}), 400

    with frame_lock:
        # Prefer the raw captured frame for calibration (before processing)
        src_img = raw_frame_for_stream if raw_frame_for_stream is not None else current_frame
        if src_img is None:
            return jsonify({"success": False, "error": "No frame available"}), 400
        img = src_img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = (int(chessboard_size[0]), int(chessboard_size[1]))
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return jsonify({"success": False, "found": False, "message": "Chessboard not detected in frame"}), 200

    # refine corners
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # prepare object points for this view
    objp = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # reshape corners to expected format
    imgp = corners_sub.reshape(-1, 2).astype(np.float64)

    # store as required by fisheye.calibrate: (N,1,points,3)/(N,1,points,2)
    calib_objpoints.append(objp.reshape(1, -1, 3).copy())
    calib_imgpoints.append(imgp.reshape(1, -1, 2).copy())

    return jsonify({"success": True, "found": True, "frames_collected": len(calib_imgpoints)})


@app.route('/api/camera_calibration/finish', methods=['POST'])
@auth.login_required
def camera_calibration_finish():
    """Run calibration using collected frames and store the camera matrix + distortion coeffs."""
    global calib_mode, calib_objpoints, calib_imgpoints, camera_matrix, dist_coeffs, settings_dirty, calibration_model, undistort_map1, undistort_map2, undistort_model
    if not calib_mode or len(calib_imgpoints) == 0:
        return jsonify({"success": False, "error": "No calibration frames collected"}), 400

    # Validate minimum frames
    if len(calib_imgpoints) < 3:
        return jsonify({"success": False, "error": f"Need at least 3 frames, only have {len(calib_imgpoints)}. Collect more snapshots from different angles."}), 400

    # Build proper lists
    objpoints = [op for op in calib_objpoints]
    imgpoints = [ip for ip in calib_imgpoints]

    # image size from the last collected frame
    with frame_lock:
        if current_frame is None:
            return jsonify({"success": False, "error": "No frame to determine image size"}), 400
        h, w = current_frame.shape[:2]

    # Try fisheye calibration first (better for strong fisheye lenses), fallback to classical calibrateCamera
    try:
        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        objp_list = [op.astype(np.float64) for op in objpoints]
        imgp_list = [ip.astype(np.float64) for ip in imgpoints]

        # Criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objp_list, imgp_list, (w, h), K, D, flags=flags, criteria=criteria)

        camera_matrix = K
        dist_coeffs = D
        calibration_model = 'fisheye'
        settings_dirty = True
        undistort_map1 = None
        undistort_map2 = None
        undistort_model = None
        calib_mode = False

        # Save calibration to disk
        save_config_to_disk()

        return jsonify({"success": True, "model": "fisheye", "rms": float(rms), "camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist()})
    except Exception as e_fisheye:
        # Fallback to classical camera calibration
        try:
            # Prepare standard lists: reshape each (1, n, dim) to (n, dim) and ensure float32
            objp_std = [op.reshape(-1, 3).astype(np.float32) for op in objpoints]
            imgp_std = [ip.reshape(-1, 2).astype(np.float32) for ip in imgpoints]

            ret, Kc, dc, rvecs, tvecs = cv2.calibrateCamera(
                objp_std, imgp_std, (w, h), None, None,
                flags=cv2.CALIB_FIX_K3)

            camera_matrix = Kc
            # dist_coeffs from calibrateCamera is already (1, 4) or (4, 1), ensure (4, 1) format
            dist_coeffs = np.array(dc, dtype=np.float64)
            if dist_coeffs.ndim == 1:
                dist_coeffs = dist_coeffs.reshape(-1, 1)
            calibration_model = 'standard'
            settings_dirty = True
            undistort_map1 = None
            undistort_map2 = None
            undistort_model = None
            calib_mode = False
            save_config_to_disk()
            return jsonify({"success": True, "model": "standard", "rms": float(ret), "camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist()})
        except Exception as e_std:
            # Both calibration attempts failed - provide diagnostic info
            err_msg = (
                f"Calibration failed. "
                f"Frames: {len(objpoints)}. "
                f"Issues: Fisheye={str(e_fisheye)[:80]}... | Standard={str(e_std)[:80]}... "
                f"Try: (1) Collect 10-20+ frames, (2) Move camera to different angles/distances, "
                f"(3) Ensure good lighting and chessboard contrast."
            )
            return jsonify({"success": False, "error": err_msg}), 500


@app.route('/api/camera_calibration/reset', methods=['POST'])
@auth.login_required
def camera_calibration_reset():
    global calib_mode, calib_objpoints, calib_imgpoints, calibration_model, undistort_map1, undistort_map2, undistort_model
    calib_mode = False
    calib_objpoints = []
    calib_imgpoints = []
    calibration_model = None
    undistort_map1 = None
    undistort_map2 = None
    undistort_model = None
    return jsonify({"success": True})


@app.route('/api/camera_calibration/status', methods=['GET'])
@auth.login_required
def camera_calibration_status():
    global calib_mode, calib_objpoints, calib_imgpoints, camera_matrix, dist_coeffs, calibration_model
    return jsonify({
        "calib_mode": bool(calib_mode),
        "frames_collected": len(calib_imgpoints),
        "calibrated": camera_matrix is not None and dist_coeffs is not None,
        "model": calibration_model
    })

@app.route('/api/settings', methods=['POST'])
@auth.login_required
def update_settings():
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure, show_overlay
    global hough_dp, hough_min_dist, hough_param1, hough_param2, hough_min_radius, hough_max_radius
    global auto_blank, token_aliases
    global camera_url, manual_blank, flip_x, flip_y
    global CCTAG_AVAILABLE, cctag_min_area, cctag_min_id, cctag_max_id, cctag_min_ident_proba

    data = request.json
    if 'camera_url' in data:
        camera_url = data['camera_url']
    if 'cctag_min_id' in data:
        try:
            cctag_min_id = int(data['cctag_min_id'])
        except Exception:
            pass
    if 'cctag_max_id' in data:
        try:
            cctag_max_id = int(data['cctag_max_id'])
        except Exception:
            pass
    if 'cctag_min_ident_proba' in data:
        try:
            cctag_min_ident_proba = float(data['cctag_min_ident_proba'])
        except Exception:
            pass
    if 'distortion_k1' in data: distortion_k1 = float(data['distortion_k1'])
    if 'zoom' in data: zoom_level = float(data['zoom'])
    if 'offset_x' in data: offset_x = float(data['offset_x'])
    if 'offset_y' in data: offset_y = float(data['offset_y'])
    if 'rotation' in data: rotation = float(data['rotation'])
    if 'brightness' in data: brightness = float(data['brightness'])
    if 'contrast' in data: contrast = float(data['contrast'])
    if 'exposure' in data: exposure = float(data['exposure'])
    if 'show_overlay' in data: show_overlay = bool(data['show_overlay'])
    if 'auto_blank' in data: auto_blank = bool(data['auto_blank'])
    if 'manual_blank' in data: manual_blank = bool(data['manual_blank'])
    if 'flip_x' in data: flip_x = bool(data['flip_x'])
    if 'flip_y' in data: flip_y = bool(data['flip_y'])
    
    # Hough Params
    if 'hough_dp' in data: hough_dp = float(data['hough_dp'])
    if 'hough_min_dist' in data: hough_min_dist = int(data['hough_min_dist'])
    if 'hough_param1' in data: hough_param1 = float(data['hough_param1'])
    if 'hough_param2' in data: hough_param2 = float(data['hough_param2'])
    if 'hough_min_radius' in data: hough_min_radius = int(data['hough_min_radius'])
    if 'hough_max_radius' in data: hough_max_radius = int(data['hough_max_radius'])
    
    if 'auto_blank' in data: auto_blank = bool(data['auto_blank'])

    # Re-evaluate all active tracked tokens against the new ID limits
    if hasattr(get_video_stream, 'tracked_tokens'):
        to_delete = []
        for tid, tdata in get_video_stream.tracked_tokens.items():
            rid = tdata.get('marker_id', -1)
            if rid != -1 and (rid < cctag_min_id or rid > cctag_max_id):
                to_delete.append(tid)
        for tid in to_delete:
            del get_video_stream.tracked_tokens[tid]
            if hasattr(get_video_stream, 'last_marker_ids') and rid in get_video_stream.last_marker_ids:
                get_video_stream.last_marker_ids.remove(rid)


    save_config_to_disk()
    global settings_dirty
    settings_dirty = True
    return jsonify({"success": True})


@app.route('/api/token/alias', methods=['POST'])
def set_token_alias():
    data = request.json
    token_id = data.get('id')
    alias = data.get('alias')
    if token_id:
        token_aliases[token_id] = alias
        save_config_to_disk()
    return jsonify({"success": True})

@app.route('/api/token/delete', methods=['POST'])
def delete_token():
    data = request.json
    token_id = data.get('id')
    if token_id:
        ignored_tokens.add(token_id)
        if token_id in get_video_stream.tracked_tokens:
            del get_video_stream.tracked_tokens[token_id]
    return jsonify({"success": True})

@app.route('/api/token/reset', methods=['POST'])
def reset_tokens():
    global ignored_tokens, token_aliases
    ignored_tokens = set()
    token_aliases = {}
    return jsonify({"success": True})

@app.route('/extension/<path:filename>', methods=['GET', 'OPTIONS'])
def serve_extension(filename):
    if request.method == 'OPTIONS':
        return '', 200
    ext_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extension')
    return send_from_directory(ext_dir, filename)

if __name__ == '__main__':
    print("Starting server on port 5000...")
    # Eventlet is the async mode recommended for SocketIO in production
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
