import numpy as np

def get_index(code):
    # Official get_index logic from coding.cpp
    # index += (5*code_vec[0]+2*code_vec[3]+6*code_vec[4]+code_vec[5])%7;
    # ... and so on
    val5 = (5*code[0] + 2*code[3] + 6*code[4] + code[5]) % 7
    val4 = (2*code[2] + 6*code[3] + code[4]) % 7
    val3 = (2*code[1] + 6*code[2] + code[3]) % 7
    val2 = (2*code[0] + 6*code[1] + code[2]) % 7
    val1 = (6*code[0] + code[1]) % 7
    val0 = code[0]
    
    idx = val5
    idx = idx * 7 + val4
    idx = idx * 7 + val3
    idx = idx * 7 + val2
    idx = idx * 7 + val1
    idx = idx * 7 + val0
    return idx

def generate_official_rune129(target_index):
    generator = [1, 1, 6, 4, 6, 0, 3, 1, 5, 3, 5, 4, 0, 4, 6, 3, 4, 6, 3, 6, 4, 3, 6, 4, 0, 4, 5, 3, 5, 1, 3, 0, 6, 4, 6, 1, 1, 0, 0, 0, 0, 0, 0]
    code_length = 43
    
    # 1. Generate the raw codeword for the target index
    raw_code = [0] * code_length
    temp_idx = target_index
    start = 0
    while temp_idx > 0:
        val = temp_idx % 7
        if val > 0:
            for i in range(code_length):
                raw_code[(start + i) % code_length] = (raw_code[(start + i) % code_length] + val * generator[i]) % 7
        temp_idx //= 7
        start += 1
    
    # 2. Alignment (Search for the rotation that returns the correct index)
    # The official align() uses SFT/ISFT but for a codebook we can just brute-force 43 rotations.
    for r in range(code_length):
        rotated = np.roll(raw_code, -r)
        if get_index(rotated) == target_index:
            return rotated.tolist()
            
    return raw_code # Fallback

if __name__ == "__main__":
    # We need IDs 0 through 7
    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        pattern = generate_official_rune129(i)
        pattern_str = " ".join(map(str, pattern))
        print(f"{i} 43 {pattern_str}")
