import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('python'))
from cctag_ext import FastCCTagDetector

det = FastCCTagDetector(3)
img = np.zeros((10, 10), dtype=np.uint8)
res = det.detect(img)
print("Result 10x10:", res)

img2 = np.zeros((20, 20), dtype=np.uint8)
res2 = det.detect(img2)
print("Result 20x20:", res2)
