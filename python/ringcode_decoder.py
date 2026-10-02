"""Reference decoder for the custom 10-ID "ring code" marker (see
tags/ringcode/*.svg). Pure Python/OpenCV — no native library, no build step,
unlike CCTag. Designed for exactly 10 IDs at much lower per-tag resolution
than CCTag needs, by trading CCTag's fine sub-pixel radial-profile reading
for a coarse 16-slot present/absent read, which tolerates far smaller/blurrier
markers and is cheap enough to run on every frame instead of a background
worker.

Marker geometry (as a fraction of the token's own outer radius, matching the
tags/ringcode/*.svg generator):
    0.00 - 0.46  blank center  (mini mounting keep-out zone)
    0.50 - 0.80  code ring     (16 slots, 22.5 deg each, marked/unmarked)
    0.82 - 0.90  solid black ring (reuses the EXISTING Hough circle
                 detection this project already runs every frame — no new
                 detection step needed to locate the token itself)

Three robustness properties below were found necessary by direct testing,
not assumed — each one failed in an earlier version before being added:

1. Sampling is INSET well away from the code ring's own printed edges
   (0.56-0.74 of the token radius, not the full 0.50-0.80 span). Sampling
   exactly at a printed boundary catches antialiasing/discretization noise
   at that edge, which silently corrupted readings even with a correct mean
   threshold.
2. Each patch's classification checks internal CONSISTENCY (spread), not
   just its mean. A patch straddling an edge — a printed-wedge boundary at
   the wrong rotation, or a partial occluder clipping into just the inner
   portion of a slot — reads as a mix of two different intensities. Blindly
   averaging that mix can land anywhere, including confidently (and
   wrongly) on the other side of the threshold. Rejecting high-spread
   patches as "unknown" instead is what makes this erasure-tolerant rather
   than silently-wrong-tolerant.
3. Sampling tries several candidate PHASE offsets and keeps whichever one
   actually lines up with the print, rather than sampling at one fixed set
   of absolute angles and hoping the physical disk's rotation happens to
   agree with it. Without this, a token rotated to land near the exact
   midpoint between two sample phases degrades EVERY slot simultaneously
   (confirmed by direct test: single-phase sampling read all 16 slots as a
   uniform, confidently-wrong "all blank" at the worst-case rotation,
   because every sample patch landed in the gap between two wedges at once)
   — a periodic blind spot recurring every 22.5 degrees of rotation. With
   phase search, the same worst-case rotation decodes with 0 unknown bits.

Known remaining limitation (also verified directly, not assumed): a
CONCENTRIC occluder (centered on the token, e.g. a mini mounted dead center)
degrades gracefully only up to the point where its radius reaches the
sampling band's inner edge (~56% of token radius) — past that it blanks
all 16 slots simultaneously, since every slot's sample band is covered at
once. This is consistent with the project's established mounting rule
(keep any occlusion source off-center): an OFF-CENTER occluder — even one
sized larger than intended, reaching deep into the ring on one side — was
verified to decode correctly, because it only ever affects a limited
angular range of slots, leaving the rest to decode from. This decoder does
not and cannot fix a dead-center occluder; that must be solved by mounting,
same as for CCTag.

Also verified: moderate camera tilt/parallax (the marker projecting as an
ellipse rather than a circle, since this decoder only uses Hough's single
circular radius, not a true ellipse fit) degrades gracefully up to roughly
25-30 degrees of tilt — handled by the same erasure tolerance as occlusion,
no special perspective correction needed. Beyond roughly 35 degrees it
fails safely (refuses to decode) rather than misidentifying. If your camera
mount produces viewing angles beyond that at some table positions, replace
the circular Hough fit with cv2.fitEllipse() on the outer ring's contour and
sample along the fitted ellipse instead of a circle — not implemented here
since it wasn't yet needed to pass testing.
"""
import numpy as np
import cv2

N_SLOTS = 16
SLOT_ANGLE_DEG = 360.0 / N_SLOTS

# Sampling band deliberately inset from the printed code ring's true span
# (0.50-0.80 of token radius) to stay clear of antialiasing/discretization
# noise at its edges (see module docstring, point 1).
R_SAMPLE_INNER_FRAC = 0.56
R_SAMPLE_OUTER_FRAC = 0.74

# Each patch samples a narrower angular slice than the full 22.5-degree slot
# (the printed wedge itself only fills ~65% of its slot, see the generator),
# so a well-aligned sample has margin on both sides before touching a
# printed edge.
SAMPLE_ANGLE_DEG = SLOT_ANGLE_DEG * 0.35

# How many candidate phase offsets to try per decode (point 3). 8 candidates
# spans the 22.5-degree slot period in 2.8125-degree steps — small enough
# that the true best alignment is always within half a step of one of them.
N_PHASE_CANDIDATES = 8

# Generated by a combinatorial search (minimum pairwise separation across
# all relative rotations = 4 bits) -- 10 necklaces, 7-of-16 marks each.
# Index = ID.
CODEWORDS = [
    [4, 5, 7, 8, 10, 13, 15],
    [4, 5, 6, 7, 11, 13, 15],
    [6, 8, 9, 11, 12, 14, 15],
    [7, 8, 10, 11, 12, 14, 15],
    [3, 6, 7, 9, 10, 14, 15],
    [3, 6, 8, 10, 11, 13, 15],
    [4, 5, 6, 9, 10, 13, 15],
    [4, 6, 7, 9, 13, 14, 15],
    [3, 6, 9, 10, 11, 12, 15],
    [4, 5, 8, 9, 10, 11, 15],
]
_CODEWORD_BITS = []
for _marks in CODEWORDS:
    _bits = [0] * N_SLOTS
    for _m in _marks:
        _bits[_m] = 1
    _CODEWORD_BITS.append(tuple(_bits))

# Classification thresholds on mean grayscale intensity (0-255) within a
# slot's sample patch. Tune against your own lighting.
BLACK_MAX = 90    # mean intensity below this -> confidently a printed mark
WHITE_MIN = 165   # mean intensity above this -> confidently blank
# A patch whose own samples span more than this range is internally
# inconsistent (straddling an edge, or partially occluded) -- rejected as
# unknown regardless of what its mean happens to be (see point 2).
MAX_SPREAD = 60


def _sample_patch(gray, cx, cy, r, slot_idx, phase_deg):
    """Raw sample values within one slot's patch, at a given sampling phase
    offset (not the token's physical rotation — see read_slots_best_phase)."""
    center_deg = slot_idx * SLOT_ANGLE_DEG - 90 + SLOT_ANGLE_DEG / 2 + phase_deg
    a0 = np.radians(center_deg - SAMPLE_ANGLE_DEG / 2)
    a1 = np.radians(center_deg + SAMPLE_ANGLE_DEG / 2)
    r0 = r * R_SAMPLE_INNER_FRAC
    r1 = r * R_SAMPLE_OUTER_FRAC
    vals = []
    h, w = gray.shape[:2]
    for a in np.linspace(a0, a1, 5):
        for rad in np.linspace(r0, r1, 5):
            x = int(round(cx + rad * np.cos(a)))
            y = int(round(cy + rad * np.sin(a)))
            if 0 <= x < w and 0 <= y < h:
                vals.append(gray[y, x])
    return vals


def _classify(vals, black_max=BLACK_MAX, white_min=WHITE_MIN):
    """1 (mark), 0 (blank), or None (unknown/occluded/inconsistent).
    black_max/white_min are parameters (not just the module defaults above)
    so the host app can make them live-tunable from a dashboard without
    needing a restart — lighting varies per setup, everything else about
    this decoder (sampling geometry, phase search, spread rejection) is
    structural and shouldn't need tuning."""
    if not vals:
        return None
    vals = np.asarray(vals, dtype=np.float32)
    spread = float(vals.max() - vals.min())
    if spread > MAX_SPREAD:
        return None
    mean = float(vals.mean())
    if mean <= black_max:
        return 1
    if mean >= white_min:
        return 0
    return None


def read_slots(gray, cx, cy, r, phase_deg=0.0, black_max=BLACK_MAX, white_min=WHITE_MIN):
    """Length-16 list of 1/0/None at a single, fixed sampling phase."""
    return [
        _classify(_sample_patch(gray, cx, cy, r, s, phase_deg), black_max, white_min)
        for s in range(N_SLOTS)
    ]


def read_slots_best_phase(gray, cx, cy, r, n_candidates=N_PHASE_CANDIDATES,
                           black_max=BLACK_MAX, white_min=WHITE_MIN):
    """Tries several candidate sampling phase offsets spanning one slot
    width and keeps whichever gives the most confidently-classified (non-
    unknown) slots. Necessary because the physical disk's rotation is
    arbitrary and unrelated to this sampling grid — without this, a
    rotation landing near the midpoint between two phases degrades every
    slot at once (see module docstring, point 3)."""
    best_bits, best_known = None, -1
    for k in range(n_candidates):
        phase = k * (SLOT_ANGLE_DEG / n_candidates)
        bits = read_slots(gray, cx, cy, r, phase, black_max, white_min)
        n_known = sum(1 for b in bits if b is not None)
        if n_known > best_known:
            best_known, best_bits = n_known, bits
    return best_bits


def decode(gray, cx, cy, r, min_margin=2, black_max=BLACK_MAX, white_min=WHITE_MIN,
           return_debug=False):
    """Identify which of the 10 IDs this token is, tolerant of unknown/
    occluded slots and of arbitrary physical rotation. Tries every rotation
    of every codeword, scores by how many KNOWN (non-None) bits match, and
    only accepts a result if the best (id, rotation) pair beats the
    second-best by at least min_margin matching bits -- same hysteresis
    principle used elsewhere in this project (see cctag_id_switch_margin)
    to avoid accepting a noisy near-tie.

    Returns (id, confidence), or (id, confidence, debug_dict) if
    return_debug=True -- the debug dict carries the raw bits read, how many
    were confidently classified, and the best/second-best match scores, for
    logging exactly what the decoder saw when a token sits unidentified
    (same purpose as CCTag's "tracked but unidentified" diagnostic log).
    Returns (None, 0.0[, debug]) if no confident match.
    """
    bits = read_slots_best_phase(gray, cx, cy, r, black_max=black_max, white_min=white_min)
    known_mask = [b is not None for b in bits]
    n_known = sum(known_mask)
    if n_known < N_SLOTS // 2:
        # Too much of the ring is occluded/unreadable to trust any match.
        debug = {"bits": bits, "n_known": n_known, "best_matches": None,
                  "second_best": None, "reason": "too few known bits"}
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    scores = []  # (matches, id, rotation)
    for id_idx, codeword in enumerate(_CODEWORD_BITS):
        for rot in range(N_SLOTS):
            rotated = codeword[rot:] + codeword[:rot]
            matches = sum(
                1 for i in range(N_SLOTS)
                if known_mask[i] and bits[i] == rotated[i]
            )
            scores.append((matches, id_idx, rot))

    scores.sort(key=lambda s: -s[0])
    best_matches, best_id, _ = scores[0]
    second_best = next((m for m, i, _ in scores[1:] if i != best_id), 0)
    debug = {"bits": bits, "n_known": n_known, "best_matches": best_matches,
              "second_best": second_best, "best_id": best_id, "reason": None}

    if best_matches - second_best < min_margin:
        debug["reason"] = "margin too small"
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    confidence = best_matches / n_known
    return (best_id, confidence, debug) if return_debug else (best_id, confidence)
