"""
CCTag Diagnostic Script
Run this INSIDE the container to identify root causes:
  docker exec -it owlbear-tracker python /app/tracker/diagnose_cctag.py

Tests:
  1. Basic CCTag init
  2. Detection on clean tag image (ground truth)
  3. Detection on washed-out/low-contrast version (simulating camera)
  4. Detection on normalized version
  5. Streaming detection timing
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.append('/app/python')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

print("=" * 60)
print("CCTag Diagnostic Script")
print("=" * 60)

# ---- Test 1: Import ----
print("\n[TEST 1] Import...")
try:
    from cctag_ext import FastCCTagDetector
    det = FastCCTagDetector(3)
    print("  OK: CCTag detector created (3 crowns)")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ---- Test 2: Ground truth tag image ----
print("\n[TEST 2] Detection on clean tag image (0000.png)...")
tag_path = '/app/tags/cctag/0000.png'
if not os.path.exists(tag_path):
    tag_path = os.path.join(os.path.dirname(__file__), '../tags/cctag/0000.png')

if os.path.exists(tag_path):
    img = cv2.imread(tag_path, cv2.IMREAD_GRAYSCALE)
    print(f"  Tag size: {img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    # Use a focal length proportional to the image (standard choice: width)
    fx, fy = float(w), float(w)
    
    t0 = time.time()
    results = det.detect(img, min_ident_proba=1e-6, cx=cx, cy=cy, fx=fx, fy=fy)
    dt = time.time() - t0
    print(f"  Detection time: {dt*1000:.1f}ms")
    if results:
        print(f"  OK: Detected {len(results)} marker(s): {results}")
    else:
        print("  FAIL: No markers detected on clean image - CCTag algo or param issue!")
else:
    print("  SKIP: Tag file not found at", tag_path)

# ---- Test 3: Washed-out simulation ----
print("\n[TEST 3] Detection on washed-out version (gamma 0.4, simulating bright projection)...")
if os.path.exists(tag_path):
    img_raw = cv2.imread(tag_path, cv2.IMREAD_GRAYSCALE)
    # Gamma 0.4 = very washed out, bright
    gamma = 0.4
    lut = np.array([(i/255.0)**gamma * 255 for i in range(256)], dtype=np.uint8)
    img_washed = cv2.LUT(img_raw, lut)
    print(f"  Washed size: {img_washed.shape}, min={img_washed.min()}, max={img_washed.max()}, mean={img_washed.mean():.1f}")
    
    h, w = img_washed.shape
    cx, cy = w / 2.0, h / 2.0
    fx, fy = float(w), float(w)
    
    results = det.detect(img_washed, min_ident_proba=1e-6, cx=cx, cy=cy, fx=fx, fy=fy)
    if results:
        print(f"  OK: Detected on washed: {results}")
    else:
        print("  NOTE: No markers on washed image (expected - this is what happens from camera)")

# ---- Test 4: Normalized version ----
print("\n[TEST 4] Detection after cv2.normalize (current approach)...")
if os.path.exists(tag_path):
    img_washed = cv2.LUT(cv2.imread(tag_path, cv2.IMREAD_GRAYSCALE), lut)
    img_norm = cv2.normalize(img_washed, None, 0, 255, cv2.NORM_MINMAX)
    print(f"  Normalized: min={img_norm.min()}, max={img_norm.max()}, mean={img_norm.mean():.1f}")

    h, w = img_norm.shape
    cx, cy = w / 2.0, h / 2.0
    fx, fy = float(w), float(w)

    results = det.detect(img_norm, min_ident_proba=1e-6, cx=cx, cy=cy, fx=fx, fy=fy)
    if results:
        print(f"  OK: Detected after normalize: {results}")
    else:
        print("  FAIL: Still no markers after normalize")

# ---- Test 5: EqualizeHist ----
print("\n[TEST 5] Detection after equalizeHist...")
if os.path.exists(tag_path):
    img_washed = cv2.LUT(cv2.imread(tag_path, cv2.IMREAD_GRAYSCALE), lut)
    img_eq = cv2.equalizeHist(img_washed)
    print(f"  EqualizeHist: min={img_eq.min()}, max={img_eq.max()}, mean={img_eq.mean():.1f}")
    
    h, w = img_eq.shape
    cx, cy = w / 2.0, h / 2.0
    fx, fy = float(w), float(w)
    
    results = det.detect(img_eq, min_ident_proba=1e-6, cx=cx, cy=cy, fx=fx, fy=fy)
    if results:
        print(f"  OK: Detected after equalizeHist: {results}")
    else:
        print("  FAIL: Still no markers after equalizeHist")

# ---- Test 6: Small crop (simulating Hough ROI) ----
print("\n[TEST 6] Detection on small cropped ROI (like we feed from Hough)...")
if os.path.exists(tag_path):
    img_full = cv2.imread(tag_path, cv2.IMREAD_GRAYSCALE)
    # Simulate a 120x120 crop of the center tag
    pad = 20
    cy_c, cx_c = img_full.shape[0]//2, img_full.shape[1]//2
    roi = img_full[cy_c - 60 - pad : cy_c + 60 + pad, cx_c - 60 - pad : cx_c + 60 + pad]
    print(f"  ROI size: {roi.shape}, min={roi.min()}, max={roi.max()}, mean={roi.mean():.1f}")

    h, w = roi.shape
    cx, cy = w / 2.0, h / 2.0
    # CRITICAL: fx/fy must match the full image scale, not the crop scale!
    # CCTag's ellipse fitting uses focal length to determine expected ellipse shapes.
    # If we pass fx=width_of_crop (e.g. 160), but the markers were designed for a
    # 1920-wide camera, the math will be wrong and it will reject every ellipse!
    fx_full = 800.0  # Standard approximation
    fy_full = 800.0
    
    roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
    results = det.detect(roi_norm, min_ident_proba=1e-6, cx=cx, cy=cy, fx=fx_full, fy=fy_full)
    if results:
        print(f"  OK: Detected in crop: {results}")
    else:
        print("  FAIL: No markers in crop - fx/fy may be wrong for crop size!")
        # Try with fx=crop_width
        results2 = det.detect(roi_norm, min_ident_proba=1e-6, cx=cx, cy=cy, fx=float(w), fy=float(h))
        if results2:
            print(f"  OK with fx=crop_width: {results2} -> Use fx=frame_width, not crop_width!")
        else:
            print("  Still failed with fx=crop_width")

# ---- Test 7: RTSP frame (live) ----
print("\n[TEST 7] Live RTSP frame test (if camera available)...")
rtsp_url = os.environ.get("DIAG_RTSP_URL", "")
if not rtsp_url:
    print("  SKIP: Set DIAG_RTSP_URL env var to test live camera (e.g. rtsp://user:pass@IP:port/stream)")

if rtsp_url:
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret: break
            time.sleep(0.2)
        
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            print(f"  Frame: {w}x{h}, min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
            
            # Full-frame detection (slow but definitive)
            print("  Running full-frame CCTag detection (may take a few seconds)...")
            t0 = time.time()
            gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            results = det.detect(gray_norm, min_ident_proba=1e-6, cx=w/2.0, cy=h/2.0, fx=float(w), fy=float(w))
            dt = time.time() - t0
            print(f"  Full-frame detection time: {dt*1000:.0f}ms")
            if results:
                print(f"  OK: Full-frame found: {results}")
            else:
                print("  NOTE: Nothing found full-frame")
            
            # Save a frame for inspection
            cv2.imwrite('/tmp/live_frame.jpg', frame)
            print("  Saved frame to /tmp/live_frame.jpg")
        else:
            print("  SKIP: Could not connect to RTSP camera (network may not be accessible from container)")
        cap.release()
    except Exception as e:
        print(f"  SKIP: {e}")


print("\n" + "=" * 60)
print("Diagnostics complete.")
print("=" * 60)
