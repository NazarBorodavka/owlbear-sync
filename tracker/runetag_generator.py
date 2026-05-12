import numpy as np
import cv2
import sys
import os
from runetag_coding import RuneTagCoding

def generate_runetag(tag_id, output_path, img_size=1024):
    coder = RuneTagCoding()
    code, canonical_idx = coder.generate(tag_id)
    bitcode = coder.unpack(code)
    
    # Create white image
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    center = (img_size // 2, img_size // 2)
    max_radius = img_size * 0.4  # Leave some margin
    
    # Radii from paper
    # Condensed radii for a tighter design
    radii_normalized = [0.65, 0.82, 1.00]
    radii = [r * max_radius for r in radii_normalized]
    
    # Maximize dot sizes while preventing overlap (approx 40% of slot width)
    # R0: 0.65, R1: 0.82, R2: 1.00
    ring_dot_radii = [
        int(max_radius * (2 * np.pi * 0.65 / 43) * 0.4), # Inner (~15-16px)
        int(max_radius * (2 * np.pi * 0.82 / 43) * 0.4), # Middle (~19-20px)
        int(max_radius * (2 * np.pi * 1.00 / 43) * 0.4)  # Outer (~24-25px)
    ]
    
    num_slots = 43
    for slot_idx in range(num_slots):
        angle = 2 * np.pi * slot_idx / num_slots
        
        # Bits for this slot: b0, b1, b2 (Inner, Middle, Outer)
        b0 = bitcode[slot_idx * 3]
        b1 = bitcode[slot_idx * 3 + 1]
        b2 = bitcode[slot_idx * 3 + 2]
        
        bits = [b0, b1, b2]
        for ring_idx in range(3):
            if bits[ring_idx]:
                r = radii[ring_idx]
                dr = ring_dot_radii[ring_idx]
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                cv2.circle(img, (x, y), dr, (0, 0, 0), -1, cv2.LINE_AA)
                
    cv2.imwrite(output_path, img)
    print(f"Generated RuneTag ID {tag_id} (Canonical ID {canonical_idx}) at {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runetag_generator.py <ID> [output.png]")
    else:
        tid = int(sys.argv[1])
        out = sys.argv[2] if len(sys.argv) > 2 else f"runetag_{tid}.png"
        generate_runetag(tid, out)
