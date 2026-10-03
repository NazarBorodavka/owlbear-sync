"""Reference decoder for the custom 10-ID "ring code" marker (see
tags/ringcode/*.svg). Pure Python/OpenCV — no native library, no build step,
unlike CCTag. Designed for exactly 10 IDs at much lower per-tag resolution
than CCTag needs, and for a large blank center so a miniature's base/peg has
somewhere safe to sit.

This is the third design iteration, and the change from v2 to v3 is the
important one — it's a response to a real failure mode found in actual
camera footage, not a synthetic test:

v1/v2 both classified each of the 16 slots by comparing its mean brightness
against either a fixed absolute threshold (v1) or a single SPLIT POINT
computed once across all 16 slots' means (v2, "find the largest gap"). Both
versions passed every synthetic lighting test thrown at them (uniform
brightness/gain shifts) and both still failed on real camera footage, even
on large, unoccluded markers. The reason: real lighting is rarely uniform
across a whole marker — a mild directional light or projector produces a
GRADIENT, where one side of the marker is measurably brighter than the
other. Under a gradient, "white" on the dim side can read darker than
"black" on the bright side, which breaks any method that computes ONE
threshold (fixed or adaptive) for the WHOLE marker — there's no single
number that correctly separates ink from paper everywhere at once.

This is a known, documented problem in the fiducial marker literature, not
a one-off: AprilTag's own detector specifically uses a "spatially-varying"
adaptive threshold (each pixel compared to its own local neighborhood, not
a global or per-tag value) for exactly this reason. RUNE-Tag, a dot-pattern
marker designed for strong occlusion/illumination robustness, reports
"performance remains largely unaffected [by illumination gradients] until
the gradient becomes very steep" — because each dot is evaluated using
information local to that dot, not a whole-marker statistic.

v3 applies the same principle here: markers use round DOTS (not angular
wedges) arranged in a ring, and each dot is classified by comparing it
ONLY to the two blank gaps immediately beside it (geometrically a few
degrees away, i.e. under near-identical local lighting), using a RATIO
(not a difference) so the comparison is also robust to the whole scene's
exposure/gain level, not just gradients. No comparison ever spans more than
~22 degrees of the marker, and no fixed or marker-wide brightness value
appears anywhere in the classification. Verified by direct testing (see
the test scenarios run during development) to tolerate a lighting gradient
across the marker that breaks the old global-split approach outright, in
addition to all the uniform-lighting/rotation/occlusion robustness already
established for the wedge design.

Marker geometry (as a fraction of the token's own outer radius, matching
the tags/ringcode/*.svg generator):
    0.00 - 0.60  blank center  (mini mounting keep-out zone)
    0.72         radius of the ring of 16 dot positions (7 printed per ID)
    0.86 - 1.00  solid black ring, at the token's true outer edge (reuses
                 the EXISTING Hough circle detection this project already
                 runs every frame — no new detection step needed to locate
                 the token itself)

Properties carried over from the wedge design, still true here for the
same reasons (see git history for the original derivations):
  - Sampling tries several candidate PHASE offsets and keeps whichever
    lines up best with the print, since the physical disk's rotation is
    arbitrary and unrelated to the sampling grid.
  - A patch (dot or gap) whose own samples are internally inconsistent
    (straddling a printed edge, or partially occluded) is rejected as
    unknown rather than trusted.
  - Decoding tries every rotation of every known codeword and only accepts
    a result that clearly beats the second-best candidate.

Known limitation, unchanged from v2: a CONCENTRIC occluder (centered on the
token) degrades gracefully only until its radius reaches the dot ring —
past that it covers every dot and gap simultaneously and there is nothing
left to compare. This cannot be fixed by sampling/classification; it is
what the large blank center exists to prevent in the first place (keep any
mounting peg/base off-center, same rule as for CCTag).
"""
import numpy as np
import cv2

N_SLOTS = 16
SLOT_ANGLE_DEG = 360.0 / N_SLOTS

# Marker geometry, as fractions of the token's own outer radius (must match
# the tags/ringcode/*.svg generator).
DOT_RING_R_FRAC = 0.72
DOT_RADIUS_FRAC = 0.08

# Sample points are inset within each dot/gap's own footprint rather than
# spanning its full extent, to stay clear of antialiasing/discretization
# noise at a printed edge (the same lesson learned and documented in the
# wedge design). 0.5 means samples stay within the inner 50% of the dot's
# own radius.
_SAMPLE_INSET = 0.5
_SAMPLE_RADIAL_HALF_FRAC = DOT_RADIUS_FRAC * _SAMPLE_INSET
_SAMPLE_ANGULAR_HALF_DEG = np.degrees(_SAMPLE_RADIAL_HALF_FRAC / DOT_RING_R_FRAC)

N_PHASE_CANDIDATES = 16

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

# A patch's own samples spanning more than this many gray levels marks it
# internally inconsistent (straddling a printed edge, or partially
# occluded) -- generous and rarely needs tuning, since it only needs to
# catch gross inconsistency, not perform precision classification (that's
# what the ratio comparison below does).
MAX_SPREAD = 60.0

# A dot counts as "ink present" only if it's at most this fraction as
# bright as its own two neighboring (guaranteed-blank) gaps -- a RATIO, not
# a difference, so it's unaffected by the scene's overall exposure/gain,
# and computed from geometrically adjacent samples, so it's unaffected by
# lighting gradients across the marker. 0.75 means the dot must read at
# least 25% darker than its immediate local background.
DOT_RATIO_THRESHOLD = 0.75
# ...and a dot counts as "confidently blank" only if it's at least this
# close to its local reference (not just "not dark enough to be a mark") --
# anything between the two ratios is genuinely ambiguous and left unknown.
BLANK_RATIO_THRESHOLD = 0.92


def _sample_ring_points(gray, cx, cy, r, center_degs):
    """Vectorized sampling of N patches (dots or gaps), each a small
    (angle, radius) grid centered on its own nominal position. Returns a
    (len(center_degs), 25) array; out-of-frame samples are NaN."""
    h, w = gray.shape[:2]
    n = len(center_degs)
    angle_offsets = np.linspace(-_SAMPLE_ANGULAR_HALF_DEG, _SAMPLE_ANGULAR_HALF_DEG, 5)
    radius_offsets = np.linspace(-_SAMPLE_RADIAL_HALF_FRAC, _SAMPLE_RADIAL_HALF_FRAC, 5)

    angles_deg = np.asarray(center_degs)[:, None] + angle_offsets[None, :]  # (n, 5)
    angles_rad = np.radians(angles_deg)
    cos_a = np.cos(angles_rad)[:, :, None]  # (n, 5, 1)
    sin_a = np.sin(angles_rad)[:, :, None]
    radii = (r * (DOT_RING_R_FRAC + radius_offsets))[None, None, :]  # (1, 1, 5)

    xs = np.round(cx + radii * cos_a).astype(np.int32)  # (n, 5, 5)
    ys = np.round(cy + radii * sin_a).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vals = gray[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)].astype(np.float32)
    vals[~valid] = np.nan
    return vals.reshape(n, 25)


def _patch_stats(vals_2d):
    """Per-patch (mean, spread, n_valid) from a (N, 25) sample array."""
    valid_counts = np.sum(~np.isnan(vals_2d), axis=1)
    means = np.nanmean(vals_2d, axis=1)
    maxs = np.nanmax(np.where(np.isnan(vals_2d), -np.inf, vals_2d), axis=1)
    mins = np.nanmin(np.where(np.isnan(vals_2d), np.inf, vals_2d), axis=1)
    spreads = np.where(valid_counts > 0, maxs - mins, np.inf)
    return means, spreads, valid_counts


def read_slots(gray, cx, cy, r, phase_deg=0.0):
    """(bits, min_ratio_margin) at a single, fixed sampling phase. bits is a
    length-16 list of 1 (dot present), 0 (blank), or None (unknown/
    occluded/inconsistent). min_ratio_margin is purely diagnostic: how far
    the least-confident classified slot sat from its own decision boundary."""
    dot_centers_deg = np.arange(N_SLOTS) * SLOT_ANGLE_DEG - 90 + phase_deg
    gap_centers_deg = dot_centers_deg + SLOT_ANGLE_DEG / 2.0

    dot_means, dot_spreads, dot_valid = _patch_stats(_sample_ring_points(gray, cx, cy, r, dot_centers_deg))
    gap_means, gap_spreads, gap_valid = _patch_stats(_sample_ring_points(gray, cx, cy, r, gap_centers_deg))

    bits = []
    margins = []
    for i in range(N_SLOTS):
        gap_before = gap_means[i - 1]  # gap just before dot i (wraps for i=0)
        gap_after = gap_means[i]       # gap just after dot i
        # Use whichever flanking gap(s) are internally consistent and
        # actually sampled; average if both are usable, for a little
        # redundancy against one of them being clipped by an occluder.
        candidates = []
        if gap_valid[i - 1] > 0 and gap_spreads[i - 1] <= MAX_SPREAD:
            candidates.append(gap_before)
        if gap_valid[i] > 0 and gap_spreads[i] <= MAX_SPREAD:
            candidates.append(gap_after)

        if dot_valid[i] == 0 or dot_spreads[i] > MAX_SPREAD or not candidates:
            bits.append(None)
            margins.append(0.0)
            continue

        local_ref = float(np.mean(candidates))
        if local_ref <= 1.0:
            # Degenerate (near-black local background) -- can't form a
            # meaningful ratio.
            bits.append(None)
            margins.append(0.0)
            continue

        ratio = dot_means[i] / local_ref
        if ratio <= DOT_RATIO_THRESHOLD:
            bits.append(1)
            margins.append(DOT_RATIO_THRESHOLD - ratio)
        elif ratio >= BLANK_RATIO_THRESHOLD:
            bits.append(0)
            margins.append(ratio - BLANK_RATIO_THRESHOLD)
        else:
            bits.append(None)
            margins.append(0.0)

    return bits, (min(margins) if margins else 0.0)


def read_slots_best_phase(gray, cx, cy, r, n_candidates=N_PHASE_CANDIDATES):
    """Tries several candidate sampling phase offsets spanning one slot
    width and keeps whichever gives the most confidently-classified (non-
    unknown) slots, breaking ties by the HIGHEST margin (how far the least-
    confident classified bit sat from its own decision boundary).

    The margin tiebreak matters more than it looks: with small/marginal
    dots, two phases can produce the same count of "confident" bits while
    one of them is confident-and-CORRECT and the other is confident-and-
    WRONG (found by direct testing — a dot's own sample patch can land just
    outside a tiny dot at a slightly-off phase and read a falsely confident
    blank instead of the true mark, rather than correctly landing in the
    ambiguous zone and being flagged unknown). A higher margin means the
    classification is further from that knife's edge, so preferring it
    among equally-confident phases is a real, if imperfect, defense against
    quietly misreading a bit instead of just failing to decode it.

    Returns (bits, margin)."""
    best_bits, best_margin, best_known = None, -1.0, -1
    for k in range(n_candidates):
        phase = k * (SLOT_ANGLE_DEG / n_candidates)
        bits, margin = read_slots(gray, cx, cy, r, phase)
        n_known = sum(1 for b in bits if b is not None)
        if (n_known, margin) > (best_known, best_margin):
            best_known, best_bits, best_margin = n_known, bits, margin
    return best_bits, best_margin


def decode(gray, cx, cy, r, min_margin=2, return_debug=False):
    """Identify which of the 10 IDs this token is, tolerant of unknown/
    occluded slots and of arbitrary physical rotation. Tries every rotation
    of every codeword, scores by how many KNOWN (non-None) bits match, and
    only accepts a result if the best (id, rotation) pair beats the
    second-best by at least min_margin matching bits -- same hysteresis
    principle used elsewhere in this project (see cctag_id_switch_margin)
    to avoid accepting a noisy near-tie.

    Returns (id, confidence), or (id, confidence, debug_dict) if
    return_debug=True. Returns (None, 0.0[, debug]) if no confident match.
    """
    bits, ratio_margin = read_slots_best_phase(gray, cx, cy, r)
    known_mask = [b is not None for b in bits]
    n_known = sum(known_mask)
    if n_known < N_SLOTS // 2:
        debug = {"bits": bits, "n_known": n_known, "ratio_margin": ratio_margin,
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
    debug = {"bits": bits, "n_known": n_known, "ratio_margin": ratio_margin,
              "best_matches": best_matches, "second_best": second_best,
              "best_id": best_id, "reason": None}

    if best_matches - second_best < min_margin:
        debug["reason"] = "margin too small"
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    confidence = best_matches / n_known
    return (best_id, confidence, debug) if return_debug else (best_id, confidence)
