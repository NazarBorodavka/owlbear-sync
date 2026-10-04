"""Reference decoder for the custom 10-ID "ring code" marker (see
tags/ringcode/*.svg). Pure Python/numpy — no native library, no build step,
unlike CCTag; doesn't even use OpenCV's own helper functions (see below for
why), despite running inside a project that otherwise uses cv2 throughout.
Designed for exactly 10 IDs at much lower per-tag resolution than CCTag
needs, and for a large blank center so a miniature's base/peg has somewhere
safe to sit.

This is the fourth design iteration. Each of the previous three failed on
real camera footage for a specific, found-by-testing reason, and each fix
is still load-bearing here:

v1: compared each slot's mean brightness to FIXED absolute thresholds.
    Failed under any lighting different from where the thresholds were
    tuned -- including a phone screen displaying the marker directly.
v2: replaced the fixed thresholds with one ADAPTIVE threshold computed once
    across all 16 slots (largest gap in their sorted means). Passed every
    uniform-brightness synthetic test, still failed on real footage,
    because real lighting is rarely uniform ACROSS a whole marker -- a mild
    directional light or projector produces a GRADIENT, and under a
    gradient there is no single threshold (fixed or adaptive) that
    correctly separates ink from paper everywhere on the marker at once.
    This is a documented problem: AprilTag's own detector specifically uses
    a spatially-varying threshold (each region compared only to its own
    local neighborhood) for exactly this reason.
v3: switched to round dots (RUNE-Tag precedent) classified by comparing
    each dot ONLY to its two immediately-adjacent blank gaps -- genuinely
    local, fixed the gradient problem (verified against a +/-120 gray-level
    synthetic gradient). But v3 still SAMPLED AT A FIXED, COMPUTED
    position for each dot -- it trusted Hough's reported (cx, cy, r) to be
    accurate and never checked. Direct testing found it tolerates only
    about 5% error in Hough's radius estimate before refusing to decode,
    and real Hough circle fitting on real camera footage isn't that
    precise -- this is what "detecting, but lighting/noise makes the disk
    detector imprecise enough to stop working" in practice looks like.

v4 (this version) fixes that by actually SEARCHING for each dot instead of
assuming its position: for each of the 16 expected dot positions, crop a
generously-sized local window around it, run Otsu's method on just that
window to find its own local black/white split (this is what makes it
still lighting-gradient-robust -- the exact same principle as v3, just
computed over a small 2D neighborhood instead of two point samples), then
check whether a plausible, roughly dot-sized, roughly dot-shaped dark blob
exists near the window's center (not just whether SOME dark pixels are
present somewhere in it). This directly targets the v3 failure mode: a dot
can now be meaningfully off from its computed position -- not just
blurred/lit differently -- and still be found, because the search window is
larger than the position error it needs to absorb.

Two bugs were found and fixed while verifying that fix against real radius
error (not just assumed from the larger window):
  - A window that genuinely has no dot in it (flat, uniform background) has
    no bimodal content for Otsu to split at all -- the first cut of this
    logic treated "no clean split found" as ambiguous regardless of why,
    which meant every blank slot read as unknown instead of confidently
    blank. Fixed by checking for flatness first: a flat-and-bright window
    is confidently blank; a flat-and-suspiciously-dark one (shadow,
    occlusion) stays ambiguous. This also sidesteps the Otsu histogram
    entirely for most slots, since most slots in any codeword are blank.
  - Once Hough's radius estimate is far enough off, a dot's search window
    (sized and positioned from that same wrong radius) can drift outward
    enough that a corner of it overlaps the marker's solid outer ring --
    contaminating an otherwise-blank slot's window with genuine black-ring
    pixels. Fixed by masking out any window pixel farther from the token
    center than RING_SAFE_CUTOFF_FRAC before doing anything else with it.
Verified directly against synthetic Hough-radius-error renders (see
python/test fixtures used during development): tolerates up to ~10% radius
error with every slot read correctly, and fails SAFELY (refuses to decode
rather than returning a wrong ID) beyond that, up to the point where the
ring-safe cutoff itself starts to cross the ring's true position. Also
verified: zero wrong-ID decodes across all 10 IDs x 10 radius-error levels
each (100 trials) -- errors either decode correctly or refuse, never
confidently wrong.

Marker geometry (as a fraction of the token's own outer radius, matching
the tags/ringcode/*.svg generator). The dot ring was pushed outward and
the dots drawn bigger -- more pixels per dot, easier to detect at low
resolution or under blur -- funded by shrinking the solid outer ring,
which only needs to stay thick enough for reliable Hough circle fitting,
not as thick as v1-v3 had it:
    0.00 - 0.60  blank center  (mini mounting keep-out zone, unchanged)
    0.73         radius of the ring of 16 dot positions (7 printed per ID)
                 (was 0.72; each dot's own radius is 0.10, was 0.08)
    0.89 - 1.00  solid black ring, at the token's true outer edge (reuses
                 the EXISTING Hough circle detection this project already
                 runs every frame) -- thickness 0.11, was 0.14

Properties carried over from earlier versions, still true here:
  - Sampling tries several candidate PHASE offsets and keeps whichever
    lines up best with the print, since the physical disk's ROTATION is
    arbitrary and unrelated to the sampling grid -- this is a different
    kind of uncertainty than the per-dot POSITION search added in v4 (phase
    handles "which way is the marker turned", the window search handles
    "is Hough's size/position estimate for the whole disk exactly right"),
    and both are needed together.
  - Decoding tries every rotation of every known codeword and only accepts
    a result that clearly beats the second-best candidate.

Known limitation, unchanged from v3: a CONCENTRIC occluder (centered on the
token) degrades gracefully only until its radius reaches the dot ring --
past that it covers every dot and its surrounding window simultaneously and
there is nothing left to search. This cannot be fixed by better detection;
it is what the large blank center exists to prevent in the first place
(keep any mounting peg/base off-center, same rule as for CCTag).
"""
import numpy as np

N_SLOTS = 16
SLOT_ANGLE_DEG = 360.0 / N_SLOTS

# Marker geometry, as fractions of the token's own outer radius (must match
# the tags/ringcode/*.svg generator).
DOT_RING_R_FRAC = 0.73
DOT_RADIUS_FRAC = 0.10

# Search window half-size around each dot's COMPUTED position, as a
# fraction of the token radius r. Scaled up from v4's original 0.11 in the
# same proportion as DOT_RADIUS_FRAC grew (0.10/0.08), to keep the same
# dot-to-window area ratio (and therefore the same MIN/MAX_BLOB_AREA_FRAC
# margins) as the original, smaller-dot design. Tried going bigger during
# development on the assumption it would buy more radius-error tolerance
# too -- direct testing found the opposite: a bigger window picks up more
# ambiguous/contaminating content than it gains in reach, so this stays at
# the proportional value rather than the larger one that was tried.
SEARCH_WINDOW_FRAC = 0.1375

N_PHASE_CANDIDATES = 16

# A found dark blob only counts as "this dot" if its pixel count falls in
# this range relative to the window's own area -- too few pixels is noise,
# too many means the whole window got thresholded dark (uniform shadow,
# severe occlusion, or no real bimodal content here at all) rather than
# showing an isolated dot.
MIN_BLOB_AREA_FRAC = 0.03
MAX_BLOB_AREA_FRAC = 0.55

# A found blob's centroid must land within this fraction of the window's
# half-size from the window's own center (i.e. from the dot's computed
# position) to count as genuinely being that dot, rather than a stray dark
# feature or a neighboring dot bleeding in at the window's edge.
MAX_CENTROID_OFFSET_FRAC = 0.6

# Otsu's own between-class separation (see _otsu_threshold) must clear this
# to trust the split at all -- guards against a washed-out/no-contrast
# window producing a meaningless "threshold" with nothing real on either
# side of it.
MIN_OTSU_VARIANCE = 25.0

# A window with essentially no brightness variation at all (very low
# standard deviation) cannot contain a dot -- a real dot always creates
# local dark/light structure against its background. In this marker's
# design that leaves exactly two possibilities for a flat window: it is
# genuinely blank background (confidently 0), or something external has
# flattened it at a suspiciously dark level -- a shadow, an occluder, or
# Hough's radius/center estimate being far enough off that the window
# landed partly outside the printed marker. That second case is ambiguous,
# not a confident blank, so it must stay None rather than being read as 0.
# This mean check is a coarse sanity floor, not a calibrated split value --
# it only needs to tell "flat and plausibly paper-colored" apart from
# "flat and suspiciously dark", which holds under any lighting short of
# near-black exposure.
FLAT_STD_MAX = 12.0
FLAT_MIN_MEAN_FOR_BLANK = 60.0

# When Hough's radius estimate is large enough, a dot's window (computed
# from that same wrong radius) can drift far enough outward that a corner
# of it reaches the marker's solid outer ring (RING_INNER_FRAC=0.89 and
# up) -- found directly during testing: at 8% radius error, blank slots
# started reading as ambiguous instead of confidently blank because their
# search windows' corners were picking up genuine black-ring pixels. Any
# window pixel farther from the token center than this cutoff is excluded
# from the search entirely rather than trusted. Found empirically (swept
# directly against the radius-error and minimum-size tests, not derived
# geometrically): a cutoff much closer to the dot ring itself (0.73) than
# to the ring's inner edge (0.89) gives noticeably better radius-error
# tolerance -- excluding more of the window's own outer pixels costs
# little (there's margin before MIN_BLOB_AREA_FRAC), while any exposure to
# ring contamination costs a lot. Going even tighter than this stops
# working on small/low-resolution markers instead, where the window has
# too few pixels left after masking -- 0.78 is the balance of both found
# by that sweep.
RING_SAFE_CUTOFF_FRAC = 0.78

# Shared by every _otsu_threshold call -- profiling found np.histogram's own
# bin-edge setup (recomputed from scratch every call) was a significant
# fraction of total decode() time at the per-slot-per-phase call volume this
# runs at, so the levels array is hoisted out and bincount (fixed-width
# integer bins, no edge computation at all) replaces histogram entirely.
_HIST_LEVELS = np.arange(256, dtype=np.float64)


def _otsu_threshold(values):
    """Standard Otsu's method on a 1D array of grayscale values: finds the
    threshold that maximizes between-class variance. Implemented directly
    in numpy (bincount-based histogram) rather than calling
    cv2.threshold(..., THRESH_OTSU) -- it's a plain histogram computation,
    there's no real benefit to the OpenCV version at this scale, and doing
    it this way keeps the whole module genuinely free of any native-library
    dependency (matching its own design goal) with no risk of behaving
    differently from what's actually tested here.

    Returns (threshold, between_class_variance) or (None, 0.0) if there
    aren't enough values to compute anything meaningful. The variance is
    returned as a confidence signal -- near zero means no real separation
    exists in this data at all, as opposed to a clean two-cluster split.
    """
    values = values[~np.isnan(values)]
    if values.size < 8:
        return None, 0.0
    clipped = np.clip(values, 0, 255).astype(np.uint8)
    hist = np.bincount(clipped, minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return None, 0.0

    levels = _HIST_LEVELS
    sum_total = np.sum(hist * levels)
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * levels)
    # Avoid div-by-zero for thresholds where one side is empty.
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_bg = np.where(w_bg > 0, sum_bg / np.where(w_bg > 0, w_bg, 1), 0)
        mean_fg = np.where(w_fg > 0, (sum_total - sum_bg) / np.where(w_fg > 0, w_fg, 1), 0)
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2

    valid = (w_bg > 0) & (w_fg > 0)
    if not np.any(valid):
        return None, 0.0
    between_var = np.where(valid, between_var, -1)
    best_t = int(np.argmax(between_var))
    return best_t, float(between_var[best_t]) / (total * total)


def _find_dot(gray, cx, cy, r, ex, ey, window_half):
    """Search a square window centered on the expected position (ex, ey)
    for a plausible dark dot. Returns one of:
        1   -- a dot was found near the window's center
        0   -- confidently no dot here (window is mostly uniform/light)
        None -- ambiguous (no clean local split, or something dark was
                found but not where/how a real dot should look)

    (cx, cy, r) are the token's own circle, needed to mask out any part of
    the window that has drifted past RING_SAFE_CUTOFF_FRAC -- see that
    constant's comment for why.
    """
    h, w = gray.shape[:2]
    x0, x1 = int(round(ex - window_half)), int(round(ex + window_half))
    y0, y1 = int(round(ey - window_half)), int(round(ey + window_half))
    x0c, x1c = max(0, x0), min(w, x1)
    y0c, y1c = max(0, y0), min(h, y1)
    if x1c - x0c < 4 or y1c - y0c < 4:
        return None

    window = gray[y0c:y1c, x0c:x1c].astype(np.float64)

    # Broadcasting two 1D index arrays through hypot (rather than
    # building full 2D coordinate grids via np.mgrid) avoids allocating
    # two full-size intermediate arrays per call -- this runs for every
    # slot at every phase candidate, so it matters.
    xs_idx = x0c + np.arange(x1c - x0c, dtype=np.float64)
    ys_idx = y0c + np.arange(y1c - y0c, dtype=np.float64)
    valid = np.hypot(xs_idx[None, :] - cx, ys_idx[:, None] - cy) <= RING_SAFE_CUTOFF_FRAC * r
    n_valid = int(np.sum(valid))
    if n_valid < 0.5 * window.size:
        return None  # window mostly clipped by the ring cutoff -- not safe to judge

    vals = window[valid]

    # Cheap pre-check: most slots in any given codeword are blank, and a
    # flat window needs no histogram at all to classify (see FLAT_STD_MAX
    # above) -- this also avoids paying for a full Otsu histogram on the
    # majority of slots every call.
    if float(vals.std()) < FLAT_STD_MAX:
        return 0 if float(vals.mean()) >= FLAT_MIN_MEAN_FOR_BLANK else None

    thresh, variance = _otsu_threshold(vals)
    if thresh is None or variance < MIN_OTSU_VARIANCE:
        return None

    dark_mask = (window <= thresh) & valid
    area_frac = float(np.sum(dark_mask)) / n_valid

    if area_frac < MIN_BLOB_AREA_FRAC:
        return 0  # essentially nothing dark in this window -- confidently blank
    if area_frac > MAX_BLOB_AREA_FRAC:
        return None  # too much of the window is dark to be a clean isolated dot

    # Weighted centroid of the dark pixels, relative to the window's own
    # center (where the dot is expected to be if present). A simple
    # intensity-moment centroid rather than full connected-component
    # labeling -- the window is small and sized to exclude neighboring
    # dots, so "where is the dark mass concentrated" is enough to tell a
    # genuine centered dot apart from a stray dark feature or a
    # neighboring dot bleeding in at the window's edge.
    ys, xs = np.nonzero(dark_mask)
    cy_blob = ys.mean()
    cx_blob = xs.mean()
    win_cy = (y1c - y0c) / 2.0
    win_cx = (x1c - x0c) / 2.0
    offset = np.hypot(cx_blob - win_cx, cy_blob - win_cy)
    if offset > MAX_CENTROID_OFFSET_FRAC * window_half:
        return None

    return 1


def read_slots(gray, cx, cy, r, phase_deg=0.0):
    """Length-16 list of 1 (dot found), 0 (confidently blank), or None
    (unknown/ambiguous) at a single, fixed sampling phase."""
    window_half = SEARCH_WINDOW_FRAC * r
    dot_ring_r = DOT_RING_R_FRAC * r
    bits = []
    for slot in range(N_SLOTS):
        deg = slot * SLOT_ANGLE_DEG - 90 + phase_deg
        rad = np.radians(deg)
        ex = cx + dot_ring_r * np.cos(rad)
        ey = cy + dot_ring_r * np.sin(rad)
        bits.append(_find_dot(gray, cx, cy, r, ex, ey, window_half))
    return bits


def read_slots_best_phase(gray, cx, cy, r, n_candidates=N_PHASE_CANDIDATES):
    """Tries several candidate sampling phase offsets spanning one slot
    width and keeps whichever gives the most confidently-classified (non-
    unknown) slots. Necessary because the physical disk's rotation is
    arbitrary and unrelated to this sampling grid."""
    best_bits, best_known = None, -1
    for k in range(n_candidates):
        phase = k * (SLOT_ANGLE_DEG / n_candidates)
        bits = read_slots(gray, cx, cy, r, phase)
        n_known = sum(1 for b in bits if b is not None)
        if n_known > best_known:
            best_known, best_bits = n_known, bits
    return best_bits


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


def decode(gray, cx, cy, r, min_margin=2, return_debug=False):
    """Identify which of the 10 IDs this token is, tolerant of unknown/
    occluded slots, of arbitrary physical rotation, and (new in v4) of
    meaningful imprecision in Hough's own (cx, cy, r) estimate. Tries every
    rotation of every codeword, scores by how many KNOWN (non-None) bits
    match, and only accepts a result if the best (id, rotation) pair beats
    the second-best by at least min_margin matching bits.

    Returns (id, confidence), or (id, confidence, debug_dict) if
    return_debug=True. Returns (None, 0.0[, debug]) if no confident match.
    """
    bits = read_slots_best_phase(gray, cx, cy, r)
    known_mask = [b is not None for b in bits]
    n_known = sum(known_mask)
    if n_known < N_SLOTS // 2:
        debug = {"bits": bits, "n_known": n_known,
                  "best_matches": None, "second_best": None, "reason": "too few known bits"}
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
