import numpy as np

def generate_official_rune129(index):
    # Official generator array from coding.cpp
    generator = [1, 1, 6, 4, 6, 0, 3, 1, 5, 3, 5, 4, 0, 4, 6, 3, 4, 6, 3, 6, 4, 3, 6, 4, 0, 4, 5, 3, 5, 1, 3, 0, 6, 4, 6, 1, 1, 0, 0, 0, 0, 0, 0]
    code_length = 43
    
    code = [0] * code_length
    temp_idx = index
    start = 0
    while temp_idx > 0:
        val = temp_idx % 7
        if val > 0:
            for i in range(code_length):
                code[(start + i) % code_length] = (code[(start + i) % code_length] + val * generator[i]) % 7
        temp_idx //= 7
        start += 1
        
    return code

if __name__ == "__main__":
    # The user's codebook starts at ID 8. 
    # Let's generate 0-7.
    for i in range(10): # Generate a few extra just to see
        pattern = generate_official_rune129(i)
        pattern_str = " ".join(map(str, pattern))
        print(f"{i} 43 {pattern_str}")
