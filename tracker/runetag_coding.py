import numpy as np

class RuneTagCoding:
    def __init__(self):
        self.code_length = 43
        self.num_words = 117649  # 7^6
        self.start_range = 8
        self.end_range = 36
        self.align_k = 4
        self.align_p = 173
        self.align_root = 2
        self.align_logn1 = 89
        
        # Generator polynomial coefficients (g0 to g36)
        self.generator = [1, 1, 6, 4, 6, 0, 3, 1, 5, 3, 5, 4, 0, 4, 6, 3, 4, 6, 3, 6, 4, 3, 6, 4, 0, 4, 5, 3, 5, 1, 3, 0, 6, 4, 6, 1, 1] + [0]*6
        
        # GF(7) tables
        self.p = 7
        
        # GF(7^6) with P(x) = x^6 + 6x^5 + 2x^3 + 6x + 1
        # P(x) coeff: [1, 6, 0, 2, 0, 6, 1]
        self.P = np.array([1, 6, 0, 2, 0, 6, 1], dtype=int)
        
        # Generator root w = x^5 + 4x^4 + 5x^2 + 6x
        self.w_poly = np.array([0, 6, 5, 0, 4, 1], dtype=int)
        
        # Precompute alpha powers for GF(7^6)
        self.alpha = [self._poly_to_int(np.array([1, 0, 0, 0, 0, 0]))]
        w = self.w_poly
        curr = np.array([1, 0, 0, 0, 0, 0], dtype=int)
        for i in range(self.code_length):
            curr = self._poly_mul(curr, w)
            self.alpha.append(self._poly_to_int(curr))
            
        # Align tables (GF(173))
        self.align_pow = [1] * (self.align_p - 1)
        self.align_log = [0] * self.align_p
        a = 1
        for i in range(self.align_p - 1):
            self.align_pow[i] = a
            self.align_log[a] = i
            a = (a * self.align_root) % self.align_p

    def _poly_to_int(self, poly):
        res = 0
        p_pow = 1
        for c in poly:
            res += int(c) * p_pow
            p_pow *= 7
        return res

    def _int_to_poly(self, n):
        poly = np.zeros(6, dtype=int)
        for i in range(6):
            poly[i] = n % 7
            n //= 7
        return poly

    def _poly_mul(self, a_poly, b_poly):
        res = np.zeros(11, dtype=int)
        for i, ca in enumerate(a_poly):
            for j, cb in enumerate(b_poly):
                res[i+j] = (res[i+j] + ca * cb) % 7
        
        # Reduce mod P(x) = x^6 + 6x^5 + 2x^3 + 6x + 1
        # x^6 = x^5 + 5x^3 + x + 6
        for i in range(10, 5, -1):
            coeff = res[i]
            if coeff == 0: continue
            # x^i = x^{i-6} * x^6
            res[i-6] = (res[i-6] + coeff * 6) % 7
            res[i-5] = (res[i-5] + coeff * 1) % 7
            res[i-4] = (res[i-4] + coeff * 0) % 7
            res[i-3] = (res[i-3] + coeff * 5) % 7
            res[i-2] = (res[i-2] + coeff * 0) % 7
            res[i-1] = (res[i-1] + coeff * 1) % 7
            res[i] = 0
        return res[:6]

    def _poly_add(self, a_poly, b_poly):
        return (a_poly + b_poly) % 7

    def _gf_mul(self, a_int, b_int):
        return self._poly_to_int(self._poly_mul(self._int_to_poly(a_int), self._int_to_poly(b_int)))

    def _gf_add(self, a_int, b_int):
        return self._poly_to_int(self._poly_add(self._int_to_poly(a_int), self._int_to_poly(b_int)))

    def _gf_sub(self, a_int, b_int):
        a_p = self._int_to_poly(a_int)
        b_p = self._int_to_poly(b_int)
        return self._poly_to_int((a_p - b_p) % 7)

    def _gf_inv(self, a_int):
        if a_int == 0: raise ZeroDivisionError()
        # Fermat's Little Theorem or Extended Euclidean
        # In GF(7^6), a^(7^6-1) = 1. So a^-1 = a^(7^6-2).
        # 7^6-2 = 117647
        res = self._poly_to_int(np.array([1,0,0,0,0,0]))
        base = a_int
        exp = 117647
        while exp > 0:
            if exp % 2 == 1:
                res = self._gf_mul(res, base)
            base = self._gf_mul(base, base)
            exp //= 2
        return res

    def sft(self, code_vec):
        out = [0] * self.code_length
        for i in range(self.code_length):
            strobe = (self.align_k * i) % (self.align_p - 1)
            val = 0
            for j in range(self.code_length):
                psn = (strobe * j) % (self.align_p - 1)
                val = (val + self.align_pow[psn] * code_vec[j]) % self.align_p
            out[i] = val
        return out

    def isft(self, in_vec):
        out = [0] * self.code_length
        for i in range(self.code_length):
            strobe = (self.align_k * i) % (self.align_p - 1)
            val = 0
            for j in range(self.code_length):
                psn = (strobe * j) % (self.align_p - 1)
                idx = (self.align_logn1 + self.align_p - 2 - psn) % (self.align_p - 1)
                val = (val + self.align_pow[idx] * in_vec[j]) % self.align_p
            out[i] = val
        return out

    def get_index(self, code_vec):
        s5 = (5*code_vec[0]+2*code_vec[3]+6*code_vec[4]+code_vec[5])%7
        s4 = (2*code_vec[2]+6*code_vec[3]+code_vec[4])%7
        s3 = (2*code_vec[1]+6*code_vec[2]+code_vec[3])%7
        s2 = (2*code_vec[0]+6*code_vec[1]+code_vec[2])%7
        s1 = (6*code_vec[0]+code_vec[1])%7
        s0 = code_vec[0]
        
        index = s5
        index = index * 7 + s4
        index = index * 7 + s3
        index = index * 7 + s2
        index = index * 7 + s1
        index = index * 7 + s0
        return index

    def align(self, code_vec):
        ft = self.sft(code_vec)
        if ft[1] == 0:
            raise ValueError("periodic code")
        
        rotation = self.align_log[ft[1]] // self.align_k
        rot_idx = self.align_p - 1 - self.align_k * rotation
        
        for i in range(1, self.code_length):
            psn = (rot_idx * i) % (self.align_p - 1)
            ft[i] = (ft[i] * self.align_pow[psn]) % self.align_p
            
        aligned_code = self.isft(ft)
        # Normalize to 0-6
        aligned_code = [v % 7 for v in aligned_code]
        index = self.get_index(aligned_code)
        return aligned_code, index, rotation

    def generate(self, index):
        index %= self.num_words
        code = [0] * self.code_length
        start = 0
        temp_index = index
        while temp_index > 0:
            val = temp_index % 7
            for i in range(self.code_length):
                pos = (start + i) % self.code_length
                code[pos] = (code[pos] + val * self.generator[i]) % 7
            temp_index //= 7
            start += 1
            
        aligned_code, canonical_index, rotation = self.align(code)
        return aligned_code, canonical_index

    def unpack(self, code):
        bitcode = []
        for val in code:
            c = (val + 1) % 8
            bitcode.append((c >> 2) & 1)
            bitcode.append((c >> 1) & 1)
            bitcode.append(c & 1)
        return bitcode

    def pack(self, bitcode):
        if len(bitcode) % 3 != 0:
            raise ValueError("Wrong code length")
        code = []
        for i in range(0, len(bitcode), 3):
            val = bitcode[i]*4 + bitcode[i+1]*2 + bitcode[i+2]
            code.append((val - 1) % 7)
        return code

    def decode(self, code_vec):
        # Simplified decoding: check if syndromes are zero
        # If not zero, we'd need full BCH. 
        # But for this task, let's just use the alignment and check syndromes.
        syndromes = []
        for i in range(self.start_range, self.end_range):
            s = 0
            # Evaluate code poly at alpha[i]
            # alpha[i] is in GF(7^6)
            # code_vec[j] are in GF(7)
            # res = sum code_vec[j] * (alpha[i])^j
            a_i = self.alpha[i]
            res = 0
            curr_a = self._poly_to_int(np.array([1, 0, 0, 0, 0, 0]))
            for val in code_vec:
                term = self._gf_mul(val, curr_a) # val is in GF(7), promoted to GF(7^6)
                res = self._gf_add(res, term)
                curr_a = self._gf_mul(curr_a, a_i)
            syndromes.append(res)
            
        if any(s != 0 for s in syndromes):
            # Error detected. We could implement Euclidean here.
            # But let's see if we can find the closest valid rotation first.
            return 1 # Error
        return 0 # Success
