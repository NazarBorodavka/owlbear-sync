import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('python'))
from cctag_ext import FastCCTagDetector

print("Initializing detector...")
det = FastCCTagDetector(3)
print("Detector initialized.")

img = np.zeros((100, 100), dtype=np.uint8)
print("Running detection...")
try:
    res = det.detect(img, minarea=10, maxarea=1000, minroundness=0.3)
    print("Result:", res)
except Exception as e:
    print("Exception:", e)
