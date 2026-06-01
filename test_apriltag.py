import numpy as np
import cv2
from pupil_apriltags import Detector

d = Detector(families="tag16h5")
img = np.zeros((100, 100), dtype=np.uint8)
res = d.detect(img)
print(res)
