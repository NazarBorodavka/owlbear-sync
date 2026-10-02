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

# --- Classification is RELATIVE, not against fixed absolute brightness
# values. An earlier version compared each patch's mean against fixed
# BLACK_MAX/WHITE_MIN constants (90/165) — this failed even on a phone
# screen displaying the marker directly (about as close to ideal, high-
# contrast input as exists), because absolute brightness depends on
# exposure/backlight/ambient light in a way that has nothing to do with
# which parts of the marker are actually ink vs. background. CCTag never
# has this problem because it never compares against an absolute brightness
# value either — it finds transitions relative to each marker's own local
# signal. This decoder now does the same thing: every marker carries
# exactly 7 marked / 9 unmarked slots (by construction — see CODEWORDS
# above), so a clean reading of any one of the 10 IDs should split its own
# 16 sampled means into two clusters. Finding that split from the data
# itself (the largest gap between sorted means) instead of comparing to a
# fixed number is what makes this adapt automatically to whatever exposure/
# lighting is active, the same way CCTag does.
#
# The two constants below are deliberately generous sanity floors, not
# precision thresholds — they only reject genuinely degenerate input (a
# blank/no-contrast view, or a patch that's internally split down the
# middle), and shouldn't need retuning per setup the way the old absolute
# values did.
MIN_GAP = 15.0          # refuse to decode if no real bimodal split exists at all
SPREAD_FRACTION = 0.6   # a patch's own internal spread beyond this fraction of
                        # the frame's observed range marks it inconsistent
MIN_SPREAD_FLOOR = 40.0  # ...with this floor so a very low-contrast but still
                         # genuinely bimodal frame doesn't make the fraction
                         # above reject everything


# Sample grid, precomputed once at import time (not per call): 5 angular
# offsets within each slot's sample window, 5 radial fractions within the
# sample band. Reused identically for every slot/phase/token — only the
# slot center angle and token (cx, cy, r) actually vary per call.
_ANGLE_OFFSETS_DEG = np.linspace(-SAMPLE_ANGLE_DEG / 2, SAMPLE_ANGLE_DEG / 2, 5)
_RADIUS_FRACS = np.linspace(R_SAMPLE_INNER_FRAC, R_SAMPLE_OUTER_FRAC, 5)
_SAMPLES_PER_SLOT = len(_ANGLE_OFFSETS_DEG) * len(_RADIUS_FRACS)


def _sample_all_slots(gray, cx, cy, r, phase_deg):
    """Vectorized equivalent of calling a per-slot sampler 16 times with
    nested Python loops inside each — that original version cost ~3200
    nested-loop iterations per token per phase (16 slots x 25 samples x 8
    phases), confirmed by direct measurement to be the dominant per-frame
    cost once this backend was wired into the main loop (~137ms/frame vs.
    an expected few ms). This computes all 16 slots' sample coordinates and
    gathers their pixel values in a handful of numpy array ops instead.

    Returns a (16, 25) array of raw pixel values; out-of-frame samples are
    NaN rather than silently dropped (handled by the nan-aware reductions in
    _classify_batch, equivalent to the original's "skip and average over
    whatever was left" behavior)."""
    h, w = gray.shape[:2]
    slot_centers_deg = (
        np.arange(N_SLOTS) * SLOT_ANGLE_DEG - 90 + SLOT_ANGLE_DEG / 2 + phase_deg
    )  # (16,)
    angles_deg = slot_centers_deg[:, None] + _ANGLE_OFFSETS_DEG[None, :]  # (16, 5)
    angles_rad = np.radians(angles_deg)
    cos_a = np.cos(angles_rad)[:, :, None]  # (16, 5, 1)
    sin_a = np.sin(angles_rad)[:, :, None]
    radii = (r * _RADIUS_FRACS)[None, None, :]  # (1, 1, 5)

    xs = np.round(cx + radii * cos_a).astype(np.int32)  # (16, 5, 5)
    ys = np.round(cy + radii * sin_a).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vals = gray[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)].astype(np.float32)
    vals[~valid] = np.nan
    return vals.reshape(N_SLOTS, _SAMPLES_PER_SLOT)


def _classify_batch(vals_2d):
    """Length-16 list of 1/0/None from a (16, 25) sample array, classified
    RELATIVE to this specific reading's own measured brightness distribution
    rather than fixed absolute thresholds (see the constants block above for
    why). Also returns the gap size found, purely for diagnostic logging —
    a small gap is a direct, human-readable signal of "this frame currently
    has too little contrast to trust," independent of absolute exposure.
    Returns (bits, gap_size)."""
    valid_counts = np.sum(~np.isnan(vals_2d), axis=1)
    means = np.nanmean(vals_2d, axis=1)
    maxs = np.nanmax(np.where(np.isnan(vals_2d), -np.inf, vals_2d), axis=1)
    mins = np.nanmin(np.where(np.isnan(vals_2d), np.inf, vals_2d), axis=1)
    spreads = maxs - mins

    valid = valid_counts > 0
    if not np.any(valid):
        return [None] * N_SLOTS, 0.0

    # Stage 1: filter out patches whose own samples are internally
    # inconsistent (straddling a printed edge, or partially occluded) before
    # they can pollute the stage-2 split below. The limit scales with the
    # observed range this frame actually has, not a fixed brightness value.
    observed_range = float(np.max(means[valid]) - np.min(means[valid]))
    spread_limit = max(MIN_SPREAD_FLOOR, observed_range * SPREAD_FRACTION)
    consistent = valid & (spreads <= spread_limit)

    if np.sum(consistent) < 2:
        return [None] * N_SLOTS, 0.0

    # Stage 2: find the natural black/white split among the trustworthy
    # patches' means — the largest gap in their sorted values. All 10 IDs
    # have exactly 7 marked / 9 unmarked slots by construction, so a clean
    # reading should show a genuine two-cluster split here regardless of
    # the absolute brightness level it happens to sit at.
    trusted_means = np.sort(means[consistent])
    gaps = np.diff(trusted_means)
    split_idx = int(np.argmax(gaps))
    gap_size = float(gaps[split_idx])
    threshold = float((trusted_means[split_idx] + trusted_means[split_idx + 1]) / 2.0)

    if gap_size < MIN_GAP:
        # No genuine bimodal separation -- e.g. a blank/no-contrast view.
        return [None] * N_SLOTS, gap_size

    bits = []
    for i in range(N_SLOTS):
        if not consistent[i]:
            bits.append(None)
        elif means[i] < threshold:
            bits.append(1)
        else:
            bits.append(0)
    return bits, gap_size


def read_slots(gray, cx, cy, r, phase_deg=0.0):
    """(bits, gap_size) at a single, fixed sampling phase. bits is a
    length-16 list of 1/0/None; gap_size is how well-separated the black/
    white clusters were this reading (purely diagnostic)."""
    return _classify_batch(_sample_all_slots(gray, cx, cy, r, phase_deg))


def read_slots_best_phase(gray, cx, cy, r, n_candidates=N_PHASE_CANDIDATES):
    """Tries several candidate sampling phase offsets spanning one slot
    width and keeps whichever gives the most confidently-classified (non-
    unknown) slots. Necessary because the physical disk's rotation is
    arbitrary and unrelated to this sampling grid — without this, a
    rotation landing near the midpoint between two phases degrades every
    slot at once (see module docstring, point 3). Returns (bits, gap_size)."""
    best_bits, best_gap, best_known = None, 0.0, -1
    for k in range(n_candidates):
        phase = k * (SLOT_ANGLE_DEG / n_candidates)
        bits, gap_size = read_slots(gray, cx, cy, r, phase)
        n_known = sum(1 for b in bits if b is not None)
        if n_known > best_known:
            best_known, best_bits, best_gap = n_known, bits, gap_size
    return best_bits, best_gap


def decode(gray, cx, cy, r, min_margin=2, return_debug=False):
    """Identify which of the 10 IDs this token is, tolerant of unknown/
    occluded slots and of arbitrary physical rotation. Tries every rotation
    of every codeword, scores by how many KNOWN (non-None) bits match, and
    only accepts a result if the best (id, rotation) pair beats the
    second-best by at least min_margin matching bits -- same hysteresis
    principle used elsewhere in this project (see cctag_id_switch_margin)
    to avoid accepting a noisy near-tie.

    Returns (id, confidence), or (id, confidence, debug_dict) if
    return_debug=True -- the debug dict carries the raw bits read, how many
    were confidently classified, the measured black/white gap size, and the
    best/second-best match scores, for logging exactly what the decoder saw
    when a token sits unidentified (same purpose as CCTag's "tracked but
    unidentified" diagnostic log). Returns (None, 0.0[, debug]) if no
    confident match.
    """
    bits, gap_size = read_slots_best_phase(gray, cx, cy, r)
    known_mask = [b is not None for b in bits]
    n_known = sum(known_mask)
    if n_known < N_SLOTS // 2:
        # Too much of the ring is occluded/unreadable, or there's no real
        # black/white contrast in this reading at all (gap_size near 0).
        debug = {"bits": bits, "n_known": n_known, "gap_size": gap_size,
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
    debug = {"bits": bits, "n_known": n_known, "gap_size": gap_size, "best_matches": best_matches,
              "second_best": second_best, "best_id": best_id, "reason": None}

    if best_matches - second_best < min_margin:
        debug["reason"] = "margin too small"
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    confidence = best_matches / n_known
    return (best_id, confidence, debug) if return_debug else (best_id, confidence)
