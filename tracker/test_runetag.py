import cv2
import numpy as np
import os
from runetag_cv import RuneTagDetector

def test_static_image(image_path):
    print(f"--- Testing RuneTag-CV on {image_path} ---")
    
    # Initialize detector
    codebook_path = "codebooks/runetag_codebook.txt"
    detector = RuneTagDetector(codebook_path=codebook_path, hamming_dist=8)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run detection
    # Test both normal and inverted
    for inv in [False, True]:
        print(f"\n[Mode: {'Inverted' if inv else 'Normal'}]")
        results = detector.detect(gray, invert=inv, min_score=0.1)
        
        if not results:
            print("No markers found.")
        else:
            for res in results:
                print(f"MATCH FOUND: ID {res['id']} at {res['center']}")
                if res['id'] == -1:
                    print(" (Marker found but ID not decoded/recognized)")

if __name__ == "__main__":
    # Path relative to tracker/
    test_static_image("../R129-0-new.bmp")
