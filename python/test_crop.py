import sys
import cv2
sys.path.append('python')
import runetag_ext

img = cv2.imread("../try/RUNEtag/result.jpg")
if img is None:
    print("Image not found")
    sys.exit(1)

# Full image is 800x600 or something. Let's pretend it's 1920x1080.
h, w = img.shape[:2]
print(f"Full image: {w}x{h}")

# Create engine
engine = runetag_ext.FastRuneTagDetector(["tags/tag_24.txt"], w/2.0, h/2.0)

# Try full image first
res_full = engine.detect(img, 10, 100000, 0.1, w/2.0, h/2.0, 0, 0)
print("Full image results:", res_full)

# Try cropped image (middle 400x400)
x1, y1 = int(w/2 - 200), int(h/2 - 200)
x2, y2 = x1 + 400, y1 + 400
crop = img[y1:y2, x1:x2]
cx_crop = (w/2.0) - x1
cy_crop = (h/2.0) - y1

res_crop = engine.detect(crop, 10, 100000, 0.1, cx_crop, cy_crop, x1, y1)
print("Crop results:", res_crop)
