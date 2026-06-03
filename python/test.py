import sys
import cv2
sys.path.append('python')
import runetag_ext

print("Imported runetag_ext")
try:
    img = cv2.imread("../try/RUNEtag/result.jpg")
    if img is None:
        print("Image not found")
        sys.exit(1)
        
    engine = runetag_ext.FastRuneTagDetector(["tags/tag_24.txt"], 400.0, 300.0)
    print("Engine created")
    
    res = engine.detect(img, 10, 10000, 0.1)
    print("Results:", res)
except Exception as e:
    print("Error:", e)
