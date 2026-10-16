"""Reference decoder for the custom 6-ID "ring code" marker (see
tags/ringcode/*.svg). Pure Python/numpy — no native library, no build step,
unlike CCTag; doesn't even use OpenCV's own helper functions (see below for
why), despite running inside a project that otherwise uses cv2 throughout.
Designed for up to 6 IDs (a full D&D party) at much lower per-tag
resolution than CCTag needs, and for a large blank center so a miniature's
base/peg has somewhere safe to sit.

This is the fifth design iteration. v5 dropped from 10 IDs to 6 (this rig
never needs to track more than a party's worth of tokens at once) and used
that slack to add a dedicated orientation sync word, nearly doubling the
guaranteed error-correction distance -- see SYNC_WORD below for the full
reasoning. Everything about how dots are physically found and classified
(v4's windowed per-dot search) is unchanged; only how the 16 read bits are
turned into an ID changed. Each of the earlier iterations failed on real
camera footage for a specific, found-by-testing reason, and each fix is
still load-bearing here:

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
# fraction of the token radius r.
#
# This used to be 0.1375 (scaled up from v4's 0.11 in proportion to
# DOT_RADIUS_FRAC growing from 0.08 to 0.10) -- found, from real printed-
# marker captures (not synthetic testing), to be a genuine bug: at 0.1375,
# the straight-line margin between two ADJACENT dots' search windows is
# only ~3.5% of their center-to-center spacing -- under 1px at the actual
# deployed marker sizes (r=63-70px). Confirmed directly: a synthetic
# render at that exact radius decoded perfectly at zero error, but as
# little as 2-3px of Hough center offset -- well within normal measurement
# noise, nowhere near an extreme case -- was enough to make a real dot
# get attributed to the WRONG adjacent slot, confidently (not an erasure,
# a wrong-but-plausible-looking bit), which is exactly the "16/16 known,
# but ties between two IDs that differ by one adjacent-slot swap" failure
# found in real logs.
#
# 0.10 was found by direct sweep (not the first value tried -- 0.1375 was
# the original, inherited value, not re-validated against this specific
# margin until this bug was found): smaller windows free up real margin
# between neighbors, and once MAX_BLOB_AREA_FRAC below is scaled to match
# (a smaller window makes a FIXED-size dot occupy more of it, which the
# old 0.55 ceiling wasn't sized for), tested to raise Hough radius-error
# tolerance from ~4% back to its originally-intended ~14-16% AND eliminate
# every wrong-but-confident read across a dense sweep of center-offset
# jitter at the real deployed size (0 wrong out of 132+ trials, vs several
# at 0.1375) -- strictly better on every axis tested, not a trade-off.
SEARCH_WINDOW_FRAC = 0.10

N_PHASE_CANDIDATES = 16

# A found dark blob only counts as "this dot" if its pixel count falls in
# this range relative to the window's own area -- too few pixels is noise,
# too many means the whole window got thresholded dark (uniform shadow,
# severe occlusion, or no real bimodal content here at all) rather than
# showing an isolated dot.
MIN_BLOB_AREA_FRAC = 0.03
# Raised alongside SEARCH_WINDOW_FRAC's reduction above: a smaller window
# makes the SAME physical dot occupy a bigger fraction of it (nominally
# (pi/4)*(DOT_RADIUS_FRAC/SEARCH_WINDOW_FRAC)^2 ~= 0.79 at the current
# values), so the old 0.55 ceiling -- sized for the previous, larger
# window -- would reject genuine dots outright, not just contamination.
# 0.9 leaves real area to work with for true occlusion/shadow cases (which
# tend to swallow the ENTIRE window, area_frac close to 1.0) while no
# longer clipping legitimate dots.
MAX_BLOB_AREA_FRAC = 0.9

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

# --- Structural pre-filter (what IS this circle at all?) ---
# Every check above assumes the Hough circle being decoded is actually a
# ring-code marker and only asks "which one". That assumption doesn't hold
# in a cluttered real scene: cv2.HoughCircles finds circular blobs on all
# kinds of ordinary objects (can lids, bottle caps, patterned fabric), and
# nothing before this point ever checked whether the content inside one
# even LOOKS like a marker before trying to read dots out of it. With
# min_margin pushed down to 1 (needed in practice -- see decode()'s
# docstring), that's enough bits of chance agreement for essentially
# arbitrary sampled content to "win" some codeword over the others,
# producing a confident wrong ID on a completely unrelated object. Found
# directly from a real camera frame: multiple non-marker objects (a can, a
# bottle cap) were being confidently identified this way.
#
# The fix is a cheap structural check run BEFORE any per-dot search even
# starts. It CANNOT assume the center is blank/light the way the first cut
# of this did: the whole reason the center is a large blank zone in the
# first place is so a miniature's base/peg can sit there -- in normal use
# it's covered. And it can't assume the cover is even a single uniform
# color: these minis are PAINTED (multiple colors, washes, texture on the
# base), not molded single-color plastic -- so the center being internally
# uniform is the UNCOMMON case, not something to fall back on only rarely.
# It also can't assume any particular absolute brightness level for any
# region, since this rig projects imagery onto the play surface and the
# same marker can sit under noticeably different projected light from one
# spot to the next.
#
# What's actually reliable regardless of any of that: the background
# immediately OUTSIDE the token's own edge. It isn't covered by anything
# mounted on the token (that's on top of the token, not beside it), and
# cv2.HoughCircles finding a circle boundary there at all already implies
# *some* real contrast exists right at that edge -- this check just
# verifies it's the specific dark-ring-against-background contrast a
# marker has, with its own independent thresholds. So the primary
# reference is a thin band just outside the ring (EXTERIOR_SAMPLE_R_FRAC);
# the center is kept only as a secondary, bonus check for the cases where
# it DOES happen to be usable (no mini mounted, e.g. during setup/testing).
# If neither reference is usable (busy/patterned background right at the
# edge, AND a painted, non-uniform center) it falls back to the ring alone
# against a generous absolute ceiling, same as before.
# A single sample radius, not several -- tried averaging over multiple
# radii within the band (same idea as CENTER_SAMPLE_R_FRACS) on the theory
# that it would average out small-size antialiasing noise, but direct
# testing found it strictly worse: any radius close to the band's outer
# edge (1.00) loses almost all tolerance for radius OVERestimation (a
# sample at 0.99*r_wrong overshoots the true ring the moment r_wrong is
# even slightly too large), since the band itself is only 0.11 wide -- not
# enough room to average across while also staying clear of both edges
# under realistic Hough radius error. A single radius at the band's
# midpoint, biased slightly toward its own center rather than hedging
# across the whole band, measured better on both radius-error tolerance
# and small-marker-size tests.
RING_SAMPLE_R_FRAC = 0.95          # inset from the ring's [0.89, 1.00] band
EXTERIOR_SAMPLE_R_FRAC = 1.10      # just past the token's own edge
CENTER_SAMPLE_R_FRACS = (0.20, 0.45)  # two radii inside the [0, 0.60] blank zone
N_STRUCTURE_SAMPLES_PER_RING = 24

# "Internally uniform" bar for the ring/exterior/center regions -- looser
# than the per-dot FLAT_STD_MAX since these are bigger, coarser samples
# (print antialiasing, background texture, a mini's painted detail), not
# single small windows.
STRUCTURE_FLAT_STD_MAX = 25.0

# Generous, deliberately uncalibrated ceiling used ONLY when NEITHER the
# exterior nor the center is usable as a reference (see above) -- same
# "coarse sanity floor, not a precise threshold" precedent as
# FLAT_MIN_MEAN_FOR_BLANK elsewhere in this module. Printed black ink
# should read well under this even under moderately bright/colored
# projected light; it only exists to catch "there's clearly no dark ring
# here at all".
RING_FALLBACK_MAX_MEAN = 170.0

# Otsu's between-class variance (see _otsu_threshold) over a combined
# ring+reference sample set must clear this -- a real marker's contrast
# against either reference is stark, so this can be a stricter bar than
# the per-dot MIN_OTSU_VARIANCE.
MIN_STRUCTURE_VARIANCE = 40.0

# Ring and whichever reference is being checked must fall cleanly on
# OPPOSITE sides of the split -- direction doesn't matter (a dark mini
# base against the ring is as valid as a light one), but at least this
# fraction of each region's samples must land on its own side, or the
# split -- however confident-looking overall -- isn't actually separating
# the two regions the way a real marker would. Swept empirically against
# both the minimum-size and non-marker-rejection tests: a stricter value
# cost real markers at small sizes (the thin printed ring band is noisier
# there, see RING_SAMPLE_R_FRAC) with no measurable gain in false-positive
# rejection in that same sweep (those fail earlier -- no usable reference
# at all -- so this threshold barely matters to them). 0.65 is a
# deliberately held margin above a 0.50 (coin-flip/random) baseline, not
# the most permissive value tested.
MIN_STRUCTURE_PURITY = 0.65

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


def _confident_contrast(vals_a, vals_b):
    """Otsu-based, direction-agnostic check for whether two groups of
    samples genuinely sit on opposite sides of a confident split (used for
    ring-vs-exterior and ring-vs-center alike). Returns (ok, purity,
    variance)."""
    combined = np.concatenate([vals_a, vals_b])
    thresh, variance = _otsu_threshold(combined)
    if thresh is None or variance < MIN_STRUCTURE_VARIANCE:
        return False, 0.0, variance
    a_below = float(np.mean(vals_a <= thresh))
    b_below = float(np.mean(vals_b <= thresh))
    # Best of the two possible directions: a-dark/b-light, or a-light/b-dark.
    purity = max(min(a_below, 1 - b_below), min(1 - a_below, b_below))
    return purity >= MIN_STRUCTURE_PURITY, purity, variance


def _looks_like_marker(gray, cx, cy, r):
    """Cheap structural gate, run before any per-dot search: does this
    circle even have a ring-code marker's basic signature, or is it some
    other circular object Hough happened to find? See the comment above
    RING_SAMPLE_R_FRAC for the full reasoning -- in short: the ring must be
    a uniform band, and it must contrast with EITHER the background just
    outside the token (primary reference -- available no matter what's
    mounted on the token) OR the center (secondary, bonus check -- only
    usable when nothing's covering it, or when what's covering it happens
    to be uniform). If neither reference is usable, falls back to the ring
    alone against a generous absolute ceiling.

    Returns (ok, debug_dict).
    """
    angles = np.linspace(0, 2 * np.pi, N_STRUCTURE_SAMPLES_PER_RING, endpoint=False)

    h, w = gray.shape[:2]
    def _sample_at(frac):
        xs = cx + frac * r * np.cos(angles)
        ys = cy + frac * r * np.sin(angles)
        xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
        yi = np.clip(np.round(ys).astype(np.int64), 0, h - 1)
        return gray[yi, xi]

    ring_vals = _sample_at(RING_SAMPLE_R_FRAC)
    exterior_vals = _sample_at(EXTERIOR_SAMPLE_R_FRAC)
    center_vals = np.concatenate([_sample_at(frac) for frac in CENTER_SAMPLE_R_FRACS])

    debug = {"ring_mean": float(ring_vals.mean()), "ring_std": float(ring_vals.std()),
              "exterior_mean": float(exterior_vals.mean()), "exterior_std": float(exterior_vals.std()),
              "center_mean": float(center_vals.mean()), "center_std": float(center_vals.std()),
              "reason": None}

    # No separate ring-flatness pre-check: at small marker sizes or under
    # real Hough radius error, antialiasing alone can push ring_std above
    # any threshold strict enough to be useful, which would reject real
    # markers before the purity check below even runs (found directly --
    # this exact check used to be here and broke both cases). The purity
    # check already self-polices ring consistency: a genuinely
    # inconsistent ring can't agree with *either* side of a split, so a
    # separate flatness gate on the ring specifically is redundant.
    exterior_usable = exterior_vals.std() < STRUCTURE_FLAT_STD_MAX
    center_usable = center_vals.std() < STRUCTURE_FLAT_STD_MAX
    debug["exterior_usable"] = exterior_usable
    debug["center_usable"] = center_usable

    if exterior_usable:
        ok, purity, variance = _confident_contrast(ring_vals, exterior_vals)
        debug["exterior_purity"] = purity
        if ok:
            debug["reference"] = "exterior"
            return True, debug

    if center_usable:
        ok, purity, variance = _confident_contrast(ring_vals, center_vals)
        debug["center_purity"] = purity
        if ok:
            debug["reference"] = "center"
            return True, debug

    if not exterior_usable and not center_usable:
        # Neither reference is internally consistent enough to trust at
        # all (busy/patterned background right at the token's edge, AND a
        # painted, non-uniform center) -- fall back to the ring alone.
        if ring_vals.mean() > RING_FALLBACK_MAX_MEAN:
            debug["reason"] = "ring too bright to plausibly be printed ink (no usable reference)"
            return False, debug
        debug["reference"] = "ring_only"
        return True, debug

    debug["reason"] = "no confident contrast against any usable reference"
    return False, debug


# --- Orientation sync word + ID codewords (v5) ---
# Every rotation-invariant necklace code of length 16 was found to cap out
# at minimum pairwise distance 4, REGARDLESS of how few codewords are
# needed (verified directly: exhaustive-ish local search found the same
# ceiling for 6, 8, and 10 codewords alike) -- confirmed as the actual
# cause of a real stuck-token failure, where two IDs only 4 bits apart
# were genuinely, persistently ambiguous under normal read noise. The only
# way to beat that ceiling is to stop requiring codewords to be
# distinguishable from every rotation of each other -- i.e. resolve the
# marker's rotation through some OTHER, ID-independent means first, then
# compare only at that one, now-known orientation.
#
# SYNC_WORD is that other means: a short, FIXED bit pattern (same on every
# ID) placed at slots 0..SYNC_K-1. To find orientation, each of the 16
# candidate rotations is scored by how many of ITS slots 0..SYNC_K-1 match
# SYNC_WORD; the best-scoring rotation wins (margin-gated, same
# min-confident-lead principle as everything else here). Once that's
# resolved, the REMAINING 16-SYNC_K slots carry the actual ID, compared
# directly against ID_CODEWORDS with no further rotation search needed at
# all -- which is what unlocks a real distance improvement: with only 6
# IDs needed (this rig only ever has up to 6 players' tokens), the
# remaining 12 bits support codewords with minimum pairwise distance 6
# (verified: guarantees recovering ANY 2 actual bit-flip errors, up from
# 1, and tolerates up to 5 erased/unknown bits, up from 3).
#
# SYNC_WORD and ID_CODEWORDS were both found by direct combinatorial
# search (not hand-picked), jointly verified to have ZERO sync collisions:
# for every one of the 6 real marker patterns, at every one of the 15
# WRONG rotations, the sync slots never once read back as SYNC_WORD. The
# margin between the true rotation's sync score and the best wrong
# rotation's sync score is exactly 1 for all 6 IDs (also verified directly
# -- this appears to be close to the real ceiling for this construction,
# not an under-searched result), which is why SYNC_MIN_MARGIN is fixed at
# 1 rather than exposed as a tunable setting: there's little real room to
# raise it, and the per-frame temporal voting already built on top of this
# (see app.py's ringcode_votes) is what absorbs occasional single-frame
# sync misses, the same way it already does for ID-margin misses.
SYNC_K = 4
_SYNC_WORD_DESIGN = (0, 1, 1, 1)
SYNC_MIN_MARGIN = 1

_ID_CODEWORDS_DESIGN = [
    (0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1),
    (1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0),
    (1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0),
    (1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0),
    (0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0),
    (1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1),
]
ID_LEN = N_SLOTS - SYNC_K

# Set to True if the physical printed sheet is mirrored relative to
# tags/ringcode/*.svg -- found directly from a real deployment: a whole
# print run came out flipped (easy to do by accident -- printing on the
# wrong side of film/sticker stock, or a "mirror" option left on in a
# print driver), and the owner didn't want to reprint. A full left-right
# mirror of the token reverses the direction dots are read in, which is a
# DIFFERENT transform than noise or rotation -- it took reading real
# captured pixel data off several "stuck" markers to tell these apart.
#
# The fix only needs reversing SYNC_WORD and each ID_CODEWORDS entry
# independently (NOT reversing the combined 16-bit pattern as one block,
# which would scramble which bits land in the sync vs ID region) --
# derived by choosing the mirror axis to fall exactly on the sync/ID
# boundary, verified directly against real captures from 3 different
# physical markers: each decodes to a perfect 12/12 id-bit match (not just
# a best-fit) once mirrored, with a massive margin over every other
# candidate, versus the unresolved ties seen before. Since reversal is an
# isometry, it preserves every pairwise Hamming distance EXACTLY -- the
# codebook's distance-6 guarantee and SYNC_MIN_MARGIN=1 ceiling both carry
# over unchanged, verified by re-running the full test suite, not assumed.
#
# This is a property of this specific print run, not the software -- if
# these markers are ever reprinted correctly-oriented from the current
# tags/ringcode/*.svg files, set this back to False.
MARKERS_PRINTED_MIRRORED = True

if MARKERS_PRINTED_MIRRORED:
    SYNC_WORD = tuple(reversed(_SYNC_WORD_DESIGN))
    ID_CODEWORDS = [tuple(reversed(cw)) for cw in _ID_CODEWORDS_DESIGN]
else:
    SYNC_WORD = _SYNC_WORD_DESIGN
    ID_CODEWORDS = _ID_CODEWORDS_DESIGN


def _resolve_orientation(bits):
    """Stage 1: find which of the 16 rotations aligns slots 0..SYNC_K-1
    with SYNC_WORD. Returns (best_rotation, best_score, second_score).
    Erasure-tolerant the same way everything else here is -- only KNOWN
    (non-None) sync slots count toward a rotation's score."""
    scores = []  # (score, rotation)
    for rot in range(N_SLOTS):
        rotated = bits[rot:] + bits[:rot]
        sync_slots = rotated[:SYNC_K]
        score = sum(
            1 for observed, expected in zip(sync_slots, SYNC_WORD)
            if observed is not None and observed == expected
        )
        scores.append((score, rot))
    scores.sort(key=lambda s: -s[0])
    best_score, best_rot = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0
    return best_rot, best_score, second_score


def decode(gray, cx, cy, r, min_margin=2, return_debug=False):
    """Identify which of the 6 IDs this token is, tolerant of unknown/
    occluded slots, of arbitrary physical rotation, and of meaningful
    imprecision in Hough's own (cx, cy, r) estimate.

    Two stages (see SYNC_WORD above for why): first locates the marker's
    orientation via SYNC_WORD, independent of which ID is printed, then
    reads the ID bits at that one resolved rotation and compares them
    directly against ID_CODEWORDS -- no rotation search needed on the ID
    side at all. min_margin gates the ID stage (how many more matching
    bits the best ID candidate needs over the second-best); the
    orientation stage has its own fixed, pre-validated margin
    (SYNC_MIN_MARGIN) since there's little room to tune it usefully.

    Before any of that: rejects circles that don't even structurally look
    like a ring-code marker (see _looks_like_marker) -- this runs first
    specifically so a random circular object never reaches the per-dot
    search and risks a chance codeword match at all, not just so the
    eventual margin checks catch it.

    Returns (id, confidence), or (id, confidence, debug_dict) if
    return_debug=True. Returns (None, 0.0[, debug]) if no confident match.
    """
    structure_ok, structure_debug = _looks_like_marker(gray, cx, cy, r)
    if not structure_ok:
        debug = {"bits": None, "n_known": 0, "best_matches": None,
                  "second_best": None, "reason": structure_debug["reason"],
                  "structure": structure_debug}
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    bits = read_slots_best_phase(gray, cx, cy, r)
    known_mask = [b is not None for b in bits]
    n_known = sum(known_mask)
    if n_known < N_SLOTS // 2:
        debug = {"bits": bits, "n_known": n_known,
                  "best_matches": None, "second_best": None, "reason": "too few known bits"}
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    rot, sync_best, sync_second = _resolve_orientation(bits)
    # Requires a PERFECT sync read (all SYNC_K bits known and correct), not
    # merely a margin over the runner-up -- found directly that margin
    # alone isn't enough here: the true rotation's margin over the best
    # wrong rotation is always exactly SYNC_MIN_MARGIN even in a clean,
    # noise-free read (an inherent ceiling of this construction, not a
    # noise artifact -- see SYNC_WORD above), so margin can't distinguish
    # a solid lock from a risky one. Under heavy noise, a marginal lock
    # (e.g. 3-of-4 vs 2-of-4) can occasionally win onto the WRONG
    # rotation, and reading ID bits at the wrong rotation produces
    # effectively-random bits that can coincidentally look confident
    # against some ID codeword -- a stress test combining this with
    # temporal voting surfaced several such wrong, confident locks. Only
    # trusting a perfect sync read closes that off: wrong rotations are
    # verified (see SYNC_WORD) to never reach a perfect match for any of
    # the 6 real IDs, so this is a hard, not just probabilistic, guard.
    #
    # Tried relaxing this for VOTING purposes only (accept a 3-of-4 lock
    # as weak voting evidence, reasoning that a wrong rotation is unlikely
    # to ALSO produce a decisively clear ID match) after real printed-
    # marker footage showed tokens stuck unidentified for 100+ seconds.
    # Reverted: checked against that same footage's actual logged reads
    # and found the relaxation wouldn't have helped them anyway (their
    # near-miss frames were exact ties, 3-vs-3, not clear 3-vs-2 reads --
    # the real bottleneck there is ID-stage ties even with PERFECT sync,
    # a bit-level read-noise problem this gate doesn't touch) while
    # measurably increasing wrong-answer risk under heavy synthetic noise
    # (roughly doubled at the most extreme level tested). Not a good
    # trade for a fix that doesn't address the actual observed failure.
    if sync_best < SYNC_K or sync_best - sync_second < SYNC_MIN_MARGIN:
        debug = {"bits": bits, "n_known": n_known, "best_matches": None,
                  "second_best": None, "reason": "orientation ambiguous",
                  "sync_best": sync_best, "sync_second": sync_second}
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    rotated = bits[rot:] + bits[:rot]
    id_bits = rotated[SYNC_K:]
    id_known_mask = [b is not None for b in id_bits]
    id_n_known = sum(id_known_mask)

    scores = []  # (matches, id_idx)
    for id_idx, codeword in enumerate(ID_CODEWORDS):
        matches = sum(
            1 for i in range(ID_LEN)
            if id_known_mask[i] and id_bits[i] == codeword[i]
        )
        scores.append((matches, id_idx))
    scores.sort(key=lambda s: -s[0])
    best_matches, best_id = scores[0]
    second_best = scores[1][0] if len(scores) > 1 else 0

    # Per-ID scores at this one resolved rotation, exposed on every
    # attempt (not just failures) so a caller can accumulate evidence
    # across frames (temporal voting) even from frames that don't
    # individually clear min_margin -- same interface/rationale as before,
    # just simpler now since there's no rotation search left to collapse
    # on the ID side. "rot" is included only for debugging/logging; voting
    # itself doesn't use it.
    per_id_scores = [(id_idx, m, rot) for m, id_idx in scores]

    debug = {"bits": bits, "n_known": n_known, "best_matches": best_matches,
              "second_best": second_best, "best_id": best_id, "reason": None,
              "per_id_scores": per_id_scores, "resolved_rotation": rot,
              "sync_best": sync_best, "sync_second": sync_second}

    if best_matches - second_best < min_margin:
        debug["reason"] = "margin too small"
        return (None, 0.0, debug) if return_debug else (None, 0.0)

    confidence = best_matches / id_n_known if id_n_known else 0.0
    return (best_id, confidence, debug) if return_debug else (best_id, confidence)
