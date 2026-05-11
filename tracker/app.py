import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import logging
import os
import requests
from flask_httpauth import HTTPBasicAuth
import json
import queue
import collections
try:
    import stag
    STAG_AVAILABLE = hasattr(stag, 'detectMarkers')
except ImportError:
    STAG_AVAILABLE = False

# AprilTag Support
try:
    import apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    try:
        from pupil_apriltags import Detector as AprilTagDetector
        APRILTAG_AVAILABLE = True
    except ImportError:
        APRILTAG_AVAILABLE = False

# RuneTag Support (Lightweight Custom CV Implementation)
try:
    from runetag_cv import RuneTagDetector
    RUNETAG_AVAILABLE = True
except ImportError:
    RUNETAG_AVAILABLE = False

print(f"--- STag Detection Support: {'ENABLED' if STAG_AVAILABLE else 'DISABLED'} ---")
print(f"--- RuneTag-CV Support: {'ENABLED' if RUNETAG_AVAILABLE else 'DISABLED'} ---")

# FFMPEG timeout, force TCP, and disable buffering for lowest latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000|rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
# Suppress FFMPEG decoding spam
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

app = Flask(__name__)
# Suppress werkzeug logging for cleaner terminal output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

auth = HTTPBasicAuth()

# Default credentials (can be changed in UI/config.json)
USER_DATA = {
    "admin": "admin"
}

@auth.verify_password
def verify_password(username, password):
    if username in USER_DATA and USER_DATA[username] == password:
        return username
    return None

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

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# State variables
camera_url = "" # IP camera URL
cap = None
is_running = False
runetag_engine = None
CONFIG_FILE = "config.json"
runetag_min_score = 0.3
runetag_detect_scale = 1.0
runetag_invert = False
runetag_precision = False
runetag_show_rois = False
last_runetag_count = -1

def load_config_from_disk():
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure
    global detection_mode, stag_error_correction, stag_roi_padding, runetag_hamming_dist, apriltag_family
    global runetag_min_score, runetag_detect_scale, runetag_invert, runetag_precision, runetag_show_rois
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                c = json.load(f)
                runetag_hamming_dist = int(c.get('runetag_hamming_dist', 4))
                runetag_min_score = float(c.get('runetag_min_score', 0.3))
                runetag_detect_scale = float(c.get('runetag_detect_scale', 0.5))
                runetag_invert = c.get('runetag_invert', False)
                runetag_precision = c.get('runetag_precision', False)
                runetag_show_rois = c.get('runetag_show_rois', False)
                apriltag_family = c.get('apriltag_family', 'tag36h11')
                last_runetag_hamming_dist = runetag_hamming_dist
                apriltag_decision_margin = float(c.get('apriltag_decision_margin', 30.0))
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
                aruco_min_perimeter = c.get('aruco_min_perimeter', 0.01)
                aruco_adaptive_thresh_min = c.get('aruco_adaptive_thresh_min', 3)
                auto_blank = c.get('auto_blank', False)
                token_aliases = c.get('token_aliases', {})
                print("Loaded config from disk.")
        except Exception as e:
            print(f"Error loading config: {e}")

def save_config_to_disk():
    global runetag_min_score, runetag_detect_scale, runetag_invert, runetag_precision, runetag_show_rois
    c = {
        'distortion_k1': distortion_k1, 'zoom_level': zoom_level, 'offset_x': offset_x, 'offset_y': offset_y,
        'rotation': rotation, 'brightness': brightness, 'contrast': contrast, 'exposure': exposure,
        'hough_param1': hough_param1, 'hough_param2': hough_param2,
        'hough_min_radius': hough_min_radius, 'hough_max_radius': hough_max_radius,
        'aruco_min_perimeter': aruco_min_perimeter, 'aruco_adaptive_thresh_min': aruco_adaptive_thresh_min,
        'auto_blank': auto_blank,
        'detection_mode': detection_mode,
        'stag_error_correction': stag_error_correction,
        'stag_roi_padding': stag_roi_padding,
        'runetag_hamming_dist': runetag_hamming_dist,
        'runetag_min_score': runetag_min_score,
        'runetag_detect_scale': runetag_detect_scale,
        'runetag_invert': runetag_invert,
        'runetag_precision': runetag_precision,
        'runetag_show_rois': runetag_show_rois,
        'apriltag_family': apriltag_family,
        'apriltag_decision_margin': apriltag_decision_margin,
        'token_aliases': token_aliases,
        'password': USER_DATA.get("admin", "admin")
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(c, f)
    except Exception as e:
        print(f"Error saving config: {e}")

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
hough_param2 = 45
hough_min_radius = 30
hough_max_radius = 40

# ArUco Detection
aruco_min_perimeter = 0.01
aruco_adaptive_thresh_min = 3
aruco_poly_approx = 0.05
auto_blank = False # Toggle for anti-reflection mode
flip_x = False
flip_y = False
detection_mode = 'aruco' # 'aruco', 'stag', or 'runetag'
stag_error_correction = 3
stag_roi_padding = 20
runetag_hamming_dist = 4
apriltag_family = 'tag36h11'
manual_blank = False

# Global Detection Engines
runetag_engine = None
runetag_load_failed_time = 0
apriltag_detector = None

load_config_from_disk()

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
settings_dirty = True

def get_video_stream():
    global cap, is_running, current_frame, camera_url, undistort_map1, undistort_map2, settings_dirty
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure, show_overlay
    global hough_dp, hough_min_dist, hough_param1, hough_param2, hough_min_radius, hough_max_radius
    global aruco_min_perimeter, aruco_adaptive_thresh_min, aruco_poly_approx, detection_mode
    global DEEPTAG_AVAILABLE, STAG_AVAILABLE, APRILTAG_AVAILABLE, runetag_invert, runetag_precision
    global src_pts, corner_idx, homography_matrix, auto_blank, stag_error_correction, stag_roi_padding, manual_blank
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
        
        # 1. Faster Distortion Correction via Pre-computed Maps
        if distortion_k1 != 0.0:
            if settings_dirty or undistort_map1 is None:
                fx, fy = w, h
                cx, cy = w / 2, h / 2
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.array([distortion_k1, 0, 0, 0, 0], dtype=np.float32)
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0)
                undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
                settings_dirty = False
            frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)
            
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
        # Skip HoughCircles for global detection modes to save significant CPU
        circles = None
        if detection_mode not in ['runetag', 'apriltag', 'stag']:
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
                    if dist < ar: # Center is inside an existing circle
                        is_inner = True
                        break
                if not is_inner:
                    detected_circles.append((x, y, r))
                    # Draw the circle in faint yellow
                    if show_overlay:
                        cv2.circle(frame, (x, y), r, (0, 255, 255), 2)

        # --- ArUco setup ---
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = int(aruco_adaptive_thresh_min)
        aruco_params.minMarkerPerimeterRate = float(aruco_min_perimeter)
        
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        markers = {}
        # We now look for markers ONLY inside detected circles to avoid map noise
        for (circ_x, circ_y, circ_r) in detected_circles:
            # Crop a small area around the disk
            # STag/ArUco context padding
            pad = stag_roi_padding if detection_mode == 'stag' else 10
            y1, y2 = max(0, circ_y - circ_r - pad), min(h, circ_y + circ_r + pad)
            x1, x2 = max(0, circ_x - circ_r - pad), min(w, circ_x + circ_r + pad)
            
            roi = gray[y1:y2, x1:x2]
            if roi.size < 100: continue # Too small to process
            
            # Use enhanced ROI for all modes for consistency
            roi_enhanced = cv2.equalizeHist(roi)
            
            try:
                # Use original ROI for first pass, enhanced for second
                if detection_mode == 'stag' and STAG_AVAILABLE:
                    # STag is now handled globally below for better reliability
                    pass
                elif detection_mode == 'runetag' and DEEPTAG_AVAILABLE:
                    # RuneTag is now handled globally below for better reliability
                    pass
                elif detection_mode == 'stag' and not STAG_AVAILABLE:
                    corners, ids, _ = detector.detectMarkers(roi_enhanced)
                else:
                    corners, ids, _ = detector.detectMarkers(roi_enhanced)
                
                if ids is not None and len(ids) > 0:
                    for i, m_id in enumerate(ids.flatten()):
                        mid_int = int(m_id)
                        markers[mid_int] = (circ_x, circ_y)
                        
                        if show_overlay and corners is not None and i < len(corners):
                            try:
                                c = corners[i]
                                if isinstance(c, np.ndarray) and c.size > 0:
                                    if c.ndim == 3: c = c[0]
                                    if c.ndim == 2 and c.shape[0] == 4:
                                        c = c.copy()
                                        c[:, 0] += x1
                                        c[:, 1] += y1
                                        cv2.polylines(frame, [np.int32(c)], True, (255, 0, 255), 2)
                            except: pass 
                        break 
            except Exception as e:
                continue

        # --- Global Detection Passes (for modes that don't rely on Hough Circles) ---
        if detection_mode == 'stag' and STAG_AVAILABLE:
            try:
                # STag is natively circular and very fast, so we can run it on the whole frame
                (corners, ids, rejected) = stag.detectMarkers(gray, 11, stag_error_correction)
                if ids is not None and len(ids) > 0:
                    for i, m_id in enumerate(ids.flatten()):
                        mid = int(m_id)
                        c = corners[i]
                        if c.ndim == 3: c = c[0]
                        center_x = int(np.mean(c[:, 0]))
                        center_y = int(np.mean(c[:, 1]))
                        markers[mid] = (center_x, center_y)
                        if show_overlay:
                            cv2.polylines(frame, [np.int32(c)], True, (0, 255, 255), 2)
                            cv2.putText(frame, f"ID: {mid}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception as e:
                print(f"STag Detection Error: {e}")

        elif detection_mode == 'runetag' and RUNETAG_AVAILABLE:
            global runetag_engine
            if runetag_engine is None:
                try:
                    codebook_path = os.path.join(os.path.dirname(__file__), 'codebooks', 'runetag_codebook.txt')
                    runetag_engine = RuneTagDetector(codebook_path, hamming_dist=runetag_hamming_dist)
                except Exception as e:
                    print(f"RuneTag-CV Init Error: {e}")
                    RUNETAG_AVAILABLE = False
            
            if runetag_engine:
                try:
                    # Run lightweight CV detection
                    tags = runetag_engine.detect(gray, invert=runetag_invert)
                    
                    if runetag_show_rois:
                        for tag in tags:
                            pts = np.array(tag['corners'], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

                    for tag in tags:
                        token_id = int(tag['id'])
                        center_x, center_y = tag['center']
                        markers[token_id] = (center_x, center_y)
                        
                        if show_overlay:
                            cv2.putText(frame, f"ID: {token_id}", (center_x, center_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                except Exception as e:
                    print(f"RuneTag-CV Detection Error: {e}")

        elif detection_mode == 'apriltag' and APRILTAG_AVAILABLE:
            global apriltag_detector
            if apriltag_detector is None:
                try:
                    # Try pupil_apriltags first as it's faster
                    apriltag_detector = AprilTagDetector(families=apriltag_family)
                except:
                    try:
                        apriltag_detector = apriltag.Detector(apriltag.DetectorOptions(families=apriltag_family))
                    except:
                        APRILTAG_AVAILABLE = False
            
            if apriltag_detector:
                try:
                    results = []
                    if hasattr(apriltag_detector, 'detect'): # pupil_apriltags
                        results = apriltag_detector.detect(gray)
                    else: # basic apriltag
                        results = apriltag_detector.detect(gray)
                    
                    for r in results:
                        # Filter by decision margin to reduce false positives
                        if hasattr(r, 'decision_margin') and r.decision_margin < apriltag_decision_margin:
                            continue
                        
                        markers[r.tag_id] = (int(r.center[0]), int(r.center[1]))
                        if show_overlay:
                            cv2.polylines(frame, [np.int32(r.corners)], True, (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {r.tag_id}", (int(r.center[0]), int(r.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(f"AprilTag Detection Error: {e}")

        # --- Temporal & ArUco Fusion ---
        detected_tokens = []
        if not hasattr(get_video_stream, "tracked_tokens"):
            get_video_stream.tracked_tokens = {}
        matched_ids = set()

        # 1. Process all detected ArUco markers (Primary source of truth)
        for m_id, (m_x, m_y) in markers.items():
            token_id = f"Marker_{m_id}"
            
            # Find the best circle that encloses this marker for precision
            best_circ = None
            best_dist = float('inf')
            for (cx, cy, cr) in detected_circles:
                d = np.sqrt((cx - m_x)**2 + (cy - m_y)**2)
                if d < cr and d < best_dist:
                    best_dist = d
                    best_circ = (cx, cy, cr)
            
            if best_circ:
                cx, cy, cr = best_circ
                if token_id in get_video_stream.tracked_tokens:
                    t = get_video_stream.tracked_tokens[token_id]
                    t["x"] = t["x"] * 0.3 + cx * 0.7
                    t["y"] = t["y"] * 0.3 + cy * 0.7
                    t["r"] = t["r"] * 0.3 + cr * 0.7
                    t["missed"] = 0
                else:
                    get_video_stream.tracked_tokens[token_id] = {
                        "x": cx, "y": cy, "r": cr, "missed": 0, "marker_id": m_id
                    }
            else:
                # No disk found? Use marker center as fallback
                if token_id in get_video_stream.tracked_tokens:
                    t = get_video_stream.tracked_tokens[token_id]
                    t["x"] = t["x"] * 0.3 + m_x * 0.7
                    t["y"] = t["y"] * 0.3 + m_y * 0.7
                    t["missed"] = 0
                else:
                    get_video_stream.tracked_tokens[token_id] = {
                        "x": m_x, "y": m_y, "r": 25, "missed": 0, "marker_id": m_id
                    }
            matched_ids.add(token_id)
            
        # --- Appear/Disappear Logging ---
        current_marker_ids = set(markers.keys())
        new_ids = current_marker_ids - last_marker_ids
        lost_ids = last_marker_ids - current_marker_ids
        
        for nid in new_ids:
            print(f"[TRACKER] Token Detected: ID {nid}", flush=True)
        for lid in lost_ids:
            print(f"[TRACKER] Token Lost: ID {lid}", flush=True)
            
        last_marker_ids = current_marker_ids

        # 2. Temporal Fallback: Match remaining circles to "missed" tokens
        for (cx, cy, cr) in detected_circles:
            is_used = False
            for t_id in matched_ids:
                t = get_video_stream.tracked_tokens[t_id]
                if abs(t["x"]-cx) < 5 and abs(t["y"]-cy) < 5:
                    is_used = True; break
            if is_used: continue

            best_id = None
            best_dist = 200 # Increased search radius for lost markers
        # --- Instant Matching Logic ---
        matched_ids = set()
        for (cx, cy, cr) in detected_circles:
            best_id = None
            best_dist = 40 
            for t_id, t_data in get_video_stream.tracked_tokens.items():
                if t_id in matched_ids: continue
                dist = np.sqrt((cx - t_data["x"])**2 + (cy - t_data["y"])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = t_id
            
            if best_id:
                token = get_video_stream.tracked_tokens[best_id]
                # Direct update for zero latency
                token["x"] = cx
                token["y"] = cy
                token["r"] = cr
                token["missed"] = 0
                matched_ids.add(best_id)

        # Increment missed frames and delete old tokens
        for token_id in list(get_video_stream.tracked_tokens.keys()):
            if token_id not in matched_ids:
                get_video_stream.tracked_tokens[token_id]["missed"] += 1
                if get_video_stream.tracked_tokens[token_id]["missed"] > 45: # 3 second ghosting
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
                
            if detection_mode == 'runetag' and runetag_show_rois and DEEPTAG_AVAILABLE and runetag_engine is not None:
                try:
                    rois_info = getattr(runetag_engine, 'rois_info', [])
                    for roi_info in rois_info:
                        pts = np.array(roi_info['ordered_corners'], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                except:
                    pass

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

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')

@app.route('/video_feed')
@auth.login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
            
        return jsonify({"success": True, "corners": corner_idx})
        
    elif action == 'reset':
        corner_idx = 0
        src_pts = np.zeros((4, 2), dtype=np.float32)
        homography_matrix = None
        return jsonify({"success": True})

@app.route('/api/settings', methods=['POST'])
@auth.login_required
def update_settings():
    global distortion_k1, zoom_level, offset_x, offset_y, rotation, brightness, contrast, exposure, show_overlay
    global hough_dp, hough_min_dist, hough_param1, hough_param2, hough_min_radius, hough_max_radius
    global aruco_min_perimeter, aruco_adaptive_thresh_min, aruco_poly_approx, auto_blank, token_aliases
    global camera_url, detection_mode, stag_error_correction, stag_roi_padding, manual_blank, runetag_hamming_dist
    global RUNETAG_AVAILABLE, STAG_AVAILABLE, APRILTAG_AVAILABLE, apriltag_family, apriltag_decision_margin
    global runetag_min_score, runetag_detect_scale, runetag_invert, runetag_precision, runetag_show_rois
    global apriltag_detector
    
    data = request.json
    if 'camera_url' in data:
        camera_url = data['camera_url']
    if 'detection_mode' in data:
        detection_mode = data['detection_mode']
    if 'apriltag_decision_margin' in data:
        apriltag_decision_margin = float(data['apriltag_decision_margin'])
    if 'runetag_min_score' in data:
        runetag_min_score = float(data['runetag_min_score'])
    if 'runetag_detect_scale' in data:
        runetag_detect_scale = float(data['runetag_detect_scale'])
    if 'runetag_invert' in data:
        runetag_invert = bool(data['runetag_invert'])
    if 'runetag_precision' in data:
        runetag_precision = bool(data['runetag_precision'])
    if 'runetag_show_rois' in data:
        runetag_show_rois = bool(data['runetag_show_rois'])
    if 'apriltag_family' in data and data['apriltag_family'] != apriltag_family:
        apriltag_family = data['apriltag_family']
        apriltag_detector = None # Reset so it re-initializes with new family
    if 'stag_error_correction' in data:
        stag_error_correction = int(data['stag_error_correction'])
    if 'stag_roi_padding' in data:
        stag_roi_padding = int(data['stag_roi_padding'])
    if 'runetag_hamming_dist' in data:
        new_hd = int(data['runetag_hamming_dist'])
        if new_hd != runetag_hamming_dist:
            runetag_hamming_dist = new_hd
            runetag_engine = None # Force re-init with new hamming dist
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
    if 'runetag_invert' in data: runetag_invert = bool(data['runetag_invert'])
    if 'runetag_precision' in data: runetag_precision = bool(data['runetag_precision'])
    if 'runetag_show_rois' in data: runetag_show_rois = bool(data['runetag_show_rois'])
    if 'runetag_min_score' in data: runetag_min_score = float(data['runetag_min_score'])
    if 'runetag_detect_scale' in data: runetag_detect_scale = float(data['runetag_detect_scale'])
    if 'flip_x' in data: flip_x = bool(data['flip_x'])
    if 'flip_y' in data: flip_y = bool(data['flip_y'])
    
    # Hough Params
    if 'hough_dp' in data: hough_dp = float(data['hough_dp'])
    if 'hough_min_dist' in data: hough_min_dist = int(data['hough_min_dist'])
    if 'hough_param1' in data: hough_param1 = float(data['hough_param1'])
    if 'hough_param2' in data: hough_param2 = float(data['hough_param2'])
    if 'hough_min_radius' in data: hough_min_radius = int(data['hough_min_radius'])
    if 'hough_max_radius' in data: hough_max_radius = int(data['hough_max_radius'])
    
    # ArUco Params
    if 'aruco_min_perimeter' in data: aruco_min_perimeter = float(data['aruco_min_perimeter'])
    if 'aruco_adaptive_thresh_min' in data: aruco_adaptive_thresh_min = int(data['aruco_adaptive_thresh_min'])
    if 'aruco_poly_approx' in data: aruco_poly_approx = float(data['aruco_poly_approx'])
    if 'auto_blank' in data: auto_blank = bool(data['auto_blank'])
    
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

if __name__ == '__main__':
    print("Starting server on port 5000...")
    # Eventlet is the async mode recommended for SocketIO in production
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
