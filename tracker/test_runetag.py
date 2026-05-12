import cv2
import os
from runetag_generator import generate_runetag
from runetag_detector import RuneTagDetector

def run_test():
    image_path = "R129-0-new.bmp"
    debug_path = "runetag_debug.jpg"
    
    # Check if image exists, if not generate it for ID 0
    if not os.path.exists(image_path):
        print(f"{image_path} not found. Generating a sample for ID 0.")
        generate_runetag(0, image_path)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return
        
    detector = RuneTagDetector()
    detected_id = detector.detect(img, debug_path=debug_path)
    
    print("-" * 30)
    if detected_id is not None:
        print(f"SUCCESS: RuneTag detected!")
        print(f"DETECTED ID: {detected_id}")
    else:
        print("FAILURE: No RuneTag detected.")
    print("-" * 30)
    
    if os.path.exists(debug_path):
        print(f"Debug image saved to {debug_path}")

if __name__ == "__main__":
    run_test()
