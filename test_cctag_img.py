import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('python'))
from cctag_ext import FastCCTagDetector

det = FastCCTagDetector(3)
img = cv2.imread('tags/cctag/0000.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
res = det.detect(img, minarea=10, maxarea=100000, minroundness=0.3)
print("Result:", res)
