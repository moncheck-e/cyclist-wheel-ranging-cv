"""
Cyclist Wheel Detection
EECE 5639 Computer Vision

CV Wheel Detection Process Outline:
    - Orientation-aware undistortion (handles portrait vs landscape mismatch)
    - Grayscale conversion
    - Border mask generation
    - Gaussian blur
    - Canny edge detection + dilation (edge thickening)
    - Border mask applied to suppress image-boundary edge artifacts
    - Contour tracing
    - Corner-based contour splitting
    - Per-segment ellipse fitting (fitEllipseAMS)
    - Candidate filtering (size, axis ratio, bounds, arc coverage, reprojection error)
    - Overlap grouping via Union-Find + refit from merged point clouds
    - Size-consistency filter (discard refits far smaller than largest detection)
    - Range + lateral position estimation (pinhole model)
    - Plot reasonably ranged finds using z limits
    - 2x2 debug grid output (blurred, contours, ellipse estimation, cumulative wheel-position scatter plot)
"""

import cv2
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path
import math

# Efficiency Parameters
SCALE = 0.5 # reduce image scale for faster processing (applied in video mode only)

# Project Folder Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Blur Parameters
GAUSSIAN_KERNEL_SIZE = 9 # must be odd
GAUSSIAN_SIGMA = 3

# Edge Detect Parameters
CANNY_LOW = 40 # lower hysteresis threshold
CANNY_HIGH = 110 # upper hysteresis threshold
DILATE = 3 # "thickens" edges

# Corner Removal Parameters
CORNER_ANGLE_THRESH = 50 # degrees - turning angle that triggers a split

# Contour Selection Parameters
MIN_CONTOUR_PTS = 8 # discard contours with fewer points
SEMI_MAJOR_MIN = 30 # smallest circle major axis in pixels
SEMI_MAJOR_MAX = 600 # largest circle major axis in pixels
AXIS_RATIO_MIN = 0.4 # b/a - exclude near-edge-on views
ARC_COVERAGE_MIN = 0.05 # fraction of ellipse perimeter covered by contour
REPROJ_ERR_MAX = 15.0 # Sampson distance threshold (approx. pixel distance)
WHEEL_SIZE_RATIO = 0.55  # keep refits within 55% of largest semi-major axis this frame

# Contour Averaging Parameters
OVERSPACE = 2

# Physical Constants
WHEEL_RADIUS_M = 0.350  # 700c outer radius (m)
Z_MIN_M = 0.5   # closer than this is physically implausible (meters)
Z_MAX_M = 10.0   # farther than this is outside useful passing-distance range (meters)

# Infer the image dimensions the calibration was captured at from cx/cy.
# ASSUMPTION: the principal point (cx, cy) is at the image centre (cx = W/2, cy = H/2).
def infer_calib_dims(camera_matrix: np.ndarray) -> tuple[int, int]:
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return int(round(cy * 2)), int(round(cx * 2)) # (height, width)

# Correct calibration matrix depending on on image orientation
def adapt_camera_matrix(camera_matrix: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    calib_h, calib_w = infer_calib_dims(camera_matrix)
    K = camera_matrix.copy()

    calib_is_portrait = calib_h > calib_w
    img_is_portrait = img_h > img_w

    if calib_is_portrait != img_is_portrait:
        # Orientation mismatch - swap axes
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        K[0, 0], K[1, 1] = fy, fx
        K[0, 2], K[1, 2] = cy, cx
        eff_calib_w, eff_calib_h = calib_h, calib_w
    else:
        eff_calib_w, eff_calib_h = calib_w, calib_h

    # Scale to match actual image resolution
    K[0, 0] *= img_w / eff_calib_w
    K[0, 2] *= img_w / eff_calib_w
    K[1, 1] *= img_h / eff_calib_h
    K[1, 2] *= img_h / eff_calib_h

    return K

# Load correct calibration files (cam matrix + distortion) depending on camera type
def load_calibration(cam_type: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cam_type == "0.5x":
        camera_matrix = np.load(DATA_DIR / "camera_matrix_0x5.npy")
        dist_coeffs   = np.load(DATA_DIR / "dist_coeffs_0x5.npy")
    elif cam_type == "1x":
        camera_matrix = np.load(DATA_DIR / "camera_matrix_1x.npy")
        dist_coeffs   = np.load(DATA_DIR / "dist_coeffs_1x.npy")
    else:
        camera_matrix = None
        dist_coeffs   = None
    return camera_matrix, dist_coeffs

def load_images(image_dir: Path) -> tuple[list, list]:
    images, names = [], []

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Error: Could not load {img_path.name}")
            continue
        images.append(img)
        names.append(img_path.name)

    print(f"  Successfully loaded {len(images)} images\n")
    return images, names

# computationally efficient method at approximating contour point mean dist from ellipse (uses Sampson dist.)
def reprojection_error(pts, cx, cy, semi_a, semi_b, angle_deg) -> float:
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy

    # rotate points into ellipse-aligned frame
    x_rot = cos_a * dx + sin_a * dy
    y_rot = -sin_a * dx + cos_a * dy

    # algebraic residual: f(x,y) = (x/a)^2 + (y/b)^2 - 1
    f = (x_rot / semi_a)**2 + (y_rot / semi_b)**2 - 1

    # gradient magnitude of f at each point
    gx = 2 * x_rot / semi_a**2
    gy = 2 * y_rot / semi_b**2
    grad_mag = np.sqrt(gx**2 + gy**2)

    # mean distance estimation across all contour points from ellipse
    return float(np.mean(np.abs(f) / (grad_mag + 1e-8)))

# Split a contour wherever the local turning angle exceeds angle_thresh.
def split_contour_at_corners(contour: np.ndarray, angle_thresh: float = CORNER_ANGLE_THRESH) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # separate contour into pts
    pts = contour.reshape(-1, 2)
    window = max(8, len(pts) // 25)
    n = len(pts)
    if n < 2 * window + 1:
        return [contour], []

    split_at = []

    # window provides buffer against false triggers due to noise or single pixel outliers
    # edited to avoid O(n) cost from for loop implementation
    idx = np.arange(n)
    v1  = pts[idx] - pts[(idx - window) % n]
    v2  = pts[(idx + window) % n] - pts[idx]
    n1  = np.linalg.norm(v1, axis=1)
    n2  = np.linalg.norm(v2, axis=1)
    valid  = (n1 > 1e-6) & (n2 > 1e-6)
    dot    = np.einsum('ij,ij->i', v1, v2)
    cos_a  = np.where(valid, np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0), 0.0)
    split_at = list(np.where(np.degrees(np.arccos(cos_a)) > angle_thresh)[0])

    if not split_at:
        return [contour], []

    split_pts = [pts[i] for i in split_at]

    segments = []

    # Middle arcs: between each consecutive pair of split indices.
    for k in range(len(split_at) - 1):
        seg = pts[split_at[k]:split_at[k + 1]]
        if len(seg) >= MIN_CONTOUR_PTS:
            segments.append(seg.reshape(-1, 1, 2).astype(np.int32))

    # Wrap-around arc: the original code produced two separate pieces here -
    # pts[:split_at[0]] (head) and pts[split_at[-1]:] (tail) - because the contour
    # is a closed loop stored as a linear array.  Both pieces belong to the same
    # smooth arc, so stitch them back together (tail first, then head).
    head = pts[:split_at[0]] # may be empty if split_at[0] == 0
    tail = pts[split_at[-1]:] # may be empty if split_at[-1] == n-1
    if len(tail) > 0 and len(head) > 0:
        stitched = np.vstack([tail, head])
    elif len(tail) > 0:
        stitched = tail
    elif len(head) > 0:
        stitched = head
    else:
        stitched = np.empty((0, 2), dtype=np.int32)
    if len(stitched) >= MIN_CONTOUR_PTS:
        segments.append(stitched.reshape(-1, 1, 2).astype(np.int32))

    # segments - list of sub-contour arrays in cv2 shape (-1, 1, 2)
    # split_pts - list of (x, y) pixel coords where splits occurred
    return (segments if segments else [contour]), split_pts

# reject immediately if the two projected sizes differ by more than 2:1
# arc fits from the same wheel must have similar semi-major axes
def ellipses_overlap(e1: tuple, e2: tuple) -> bool:
    (cx1, cy1), (w1, h1), _ = e1
    (cx2, cy2), (w2, h2), _ = e2
    sa1 = max(w1, h1) / 2
    sa2 = max(w2, h2) / 2

    if max(sa1, sa2) > 0 and min(sa1, sa2) / max(sa1, sa2) < 0.5:
        return False

    dist = np.hypot(cx2 - cx1, cy2 - cy1)

    # Condition 1: precise point-in-ellipse test for both pairings
    for outer, inner in [(e1, e2), (e2, e1)]:
        (cx, cy), (w, h), angle = outer
        sa = max(w, h) / 2
        sb = min(w, h) / 2
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        px, py = inner[0]
        dx, dy = px - cx, py - cy
        xr =  cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        if (xr / sa) ** 2 + (yr / sb) ** 2 <= 1.0:
            return True

    # Condition 2: centres are very close relative to the smaller wheel radius
    if dist < 0.5 * min(sa1, sa2):
        return True

    return False


# Pool all raw contour points from a candidate group and refit a single ellipse.
# This avoids the systematic under-sizing caused by averaging short-arc ellipse params -
# fitting to the full point cloud lets the solver see a wider angular span of the wheel.
def refit_ellipse_from_group(group_indices: list[int], candidate_contours: dict) -> tuple | None:
    all_pts = np.vstack([candidate_contours[i].reshape(-1, 2) for i in group_indices]).astype(np.float32)
    if len(all_pts) < 6:
        return None
    try:
        return cv2.fitEllipseAMS(all_pts)
    except cv2.error:
        return None

def est_range(cam_matrix: np.ndarray, major_axis: float, ellipse_angle_deg: float, major_is_height: bool) -> float:
    """Estimate distance to wheel using the pinhole model: Z = f_eff * R_real / r_px.

    Args:
        cam_matrix:         Camera intrinsic matrix, already adapted to the
                            current image resolution (do NOT pass the raw calib
                            matrix - the adapted one is computed once per frame
                            in detect_wheels and should be reused here).
        major_axis:         Full major-axis length in pixels (diameter, not semi).
        ellipse_angle_deg:  OpenCV ellipse angle in degrees (rotation of the
                            first / 'w' axis from the image x-axis, clockwise).
        major_is_height:    True when the fitted ellipse has h > w, i.e. the
                            major axis is the height axis.

    Returns:
        float: Estimated range in metres.
    """
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    semi_a_px = major_axis / 2.0

    # Angle of the major axis from the image x-axis (clockwise, OpenCV convention)
    # When h > w the OpenCV angle describes the minor axis, add 90° to get the major axis angle
    major_angle_rad = np.deg2rad(ellipse_angle_deg + (90.0 if major_is_height else 0.0))
    cos_a = np.cos(major_angle_rad)
    sin_a = np.sin(major_angle_rad)

    # Effective focal length along the major-axis direction:
    # f_eff = || [fx*cos θ, fy*sin θ] ||
    f_eff = np.hypot(fx * cos_a, fy * sin_a)

    return f_eff * WHEEL_RADIUS_M / semi_a_px

def est_pos(cx: float, cy: float, range_m: float, cam_matrix: np.ndarray) -> tuple[float, float, float]:
    """Backproject ellipse centre to 3-D camera-frame coordinates.

    Uses the standard pinhole model:
        X = (u - cx) * Z / fx
        Y = -1*(v - cy) * Z / fy
        Z = range_m  (from est_range)

    Returns:
        (X, Y, Z) in metres relative to the camera optical centre.
        X positive right, Y positive up, Z positive forward.
    """
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    ppx, ppy = cam_matrix[0, 2], cam_matrix[1, 2]
    Z = range_m
    X = (cx - ppx) * Z / fx
    Y = -1*(cy - ppy) * Z / fy
    return (X, Y, Z)

def draw_scatter_panel(world_points: list[tuple[float, float, float]], h: int, w: int) -> np.ndarray:
    """Render a 2-D scatter of detected wheel positions onto a white canvas.

    Horizontal axis : world X (left/right, feet)
    Vertical axis : world Z (depth along optical axis, feet - larger = farther)
    Point label : world Y (height, feet) printed next to each dot
    """
    M_TO_FT = 3.28084

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    ML, MR, MT, MB = 110, 24, 50, 70 # margins: left, right, top, bottom (px)
    plot_w = w - ML - MR
    plot_h = h - MT - MB

    BLACK = ( 0, 0, 0)
    LGRAY = (160, 160, 160)
    DGRAY = (200, 200, 200)

    # Title + axis labels
    cv2.putText(canvas, "Wheel Position", (ML, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, BLACK, 2, cv2.LINE_AA)
    cv2.putText(canvas, "X (ft)", (ML + plot_w // 2 - 30, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Z (ft)", (4, MT + plot_h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)

    if not world_points:
        cv2.putText(canvas, "No detections yet", (ML + 10, MT + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, LGRAY, 2, cv2.LINE_AA)
        return canvas

    # Convert metres -> feet
    xs = [p[0] * M_TO_FT for p in world_points]
    ys = [p[1] * M_TO_FT for p in world_points] # height - label only
    zs = [p[2] * M_TO_FT for p in world_points] # depth - vertical axis

    def _padded_range(vals: list[float], min_span: float = 3.0):
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, min_span)
        pad  = span * 0.18
        return lo - pad, hi + pad

    x_min, x_max = _padded_range(xs)
    z_min, z_max = _padded_range(zs)

    def to_px(xw: float, zw: float) -> tuple[int, int]:
        px = int(ML + (xw - x_min) / (x_max - x_min) * plot_w)
        py = int(MT + (1.0 - (zw - z_min) / (z_max - z_min)) * plot_h)
        return px, py

    def nice_ticks(lo: float, hi: float, max_ticks: int = 7) -> list[float]:
        """Return tick values that land on 'nice' multiples (.25, .5, 1, 2, 5, …)."""
        span = hi - lo
        if span == 0:
            return [lo]
        raw_step = span / max_ticks
        magnitude = 10 ** math.floor(math.log10(raw_step))
        normalized = raw_step / magnitude
        nice_steps = [0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
        step = min(nice_steps, key=lambda s: abs(s - normalized)) * magnitude
        start = math.ceil(lo / step) * step
        ticks = []
        v = start
        while v <= hi + step * 1e-6:
            ticks.append(round(v / step) * step) # snap to exact multiple
            v += step
        return ticks

    # Grid lines + tick labels
    for xw in nice_ticks(x_min, x_max):
        px_tick = int(ML + (xw - x_min) / (x_max - x_min) * plot_w)
        cv2.line(canvas, (px_tick, MT), (px_tick, MT + plot_h), DGRAY, 1)
        cv2.putText(canvas, f"{xw:g}", (px_tick - 18, h - MB + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.70, BLACK, 2, cv2.LINE_AA)

    for zw in nice_ticks(z_min, z_max):
        py_tick = int(MT + (1.0 - (zw - z_min) / (z_max - z_min)) * plot_h)
        cv2.line(canvas, (ML, py_tick), (ML + plot_w, py_tick), DGRAY, 1)
        cv2.putText(canvas, f"{zw:g}", (2, py_tick + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.70, BLACK, 2, cv2.LINE_AA)

    # Plot border
    cv2.rectangle(canvas, (ML, MT), (ML + plot_w, MT + plot_h), LGRAY, 1)

    n_pts = len(world_points)

    def frame_color(i: int) -> tuple[int, int, int]:
        """Cool-to-warm ramp: blue (frame 0) -> cyan -> green -> yellow -> red (latest)."""
        t = i / max(n_pts - 1, 1) # 0.0 ... 1.0
        if t < 0.25: # blue -> cyan
            s = t / 0.25
            return (255, int(255 * s), 0)
        elif t < 0.5: # cyan -> green
            s = (t - 0.25) / 0.25
            return (int(255 * (1 - s)), 255, 0)
        elif t < 0.75: # green -> yellow
            s = (t - 0.5) / 0.25
            return (0, 255, int(255 * s))
        else: # yellow -> red
            s = (t - 0.75) / 0.25
            return (0, int(255 * (1 - s)), 255)

    # Pre-compute pixel coords (already in feet)
    px_coords = [to_px(xw, zw) for xw, zw in zip(xs, zs)]

    # Dots, frame numbers, and Y-height labels
    for i, ((px, py), yw) in enumerate(zip(px_coords, ys)):
        color = frame_color(i)
        cv2.circle(canvas, (px, py), 9, color, -1, cv2.LINE_AA)
        # Frame index inside the dot
        label = str(i)
        lw = 9 * len(label)
        cv2.putText(canvas, label, (px - lw // 2, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        # Y-height annotation offset to avoid overlap with the next point
        cv2.putText(canvas, f"Y={yw:+.1f}ft", (px + 14, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Colorbar legend (early -> latest) at bottom-right of plot area
    bar_x0 = ML + plot_w - 110
    bar_y  = MT + plot_h + 30
    bar_len = 100
    cv2.putText(canvas, "early", (bar_x0 - 2, bar_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, frame_color(0), 2, cv2.LINE_AA)
    for bi in range(bar_len):
        c = frame_color(int(bi / bar_len * (n_pts - 1))) if n_pts > 1 else frame_color(0)
        cv2.line(canvas, (bar_x0 + 42 + bi, bar_y), (bar_x0 + 42 + bi, bar_y - 7), c, 1)
    cv2.putText(canvas, "latest", (bar_x0 + 144, bar_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, frame_color(n_pts - 1), 2, cv2.LINE_AA)

    return canvas

# image processing repackaged for simplified use regardless if data is images or video
def process_frame(img, cam_matrix, remap1, remap2, world_points, scale: float = 1.0):
    # cam_matrix: already adapted + scaled for this frame size (or None)
    # remap1/remap2: precomputed undistortion maps (or None if no calibration)
    # scale: SCALE factor applied before this call, used to adjust pixel thresholds

    h, w = img.shape[:2]

    if remap1 is not None:
        img = cv2.remap(img, remap1, remap2, cv2.INTER_LINEAR)

    # grayscale convert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edges made from img borders after calibration found
    mask = (gray > 0).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)

    # blur + canny edge map + thicken lines
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), GAUSSIAN_SIGMA)

    # move dilate size to tunable parameters
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((DILATE, DILATE), np.uint8), iterations=1)

    # img border edges removed
    edges = cv2.bitwise_and(edges, mask)

    # contour tracing
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # split contours at sharp corners
    segments = []
    all_split_pts = []
    for c in contours:
        segs, spts = split_contour_at_corners(c)
        segments.extend(segs)
        all_split_pts.extend(spts)

    # ellipse fitting + filtering
    candidates = []
    candidate_contours: dict[int, np.ndarray] = {}
    h_img, w_img = img.shape[:2]

    # Scale pixel thresholds to match downsampled resolution
    semi_major_min = SEMI_MAJOR_MIN * scale
    semi_major_max = SEMI_MAJOR_MAX * scale

    # debug panels
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    edges_bgr   = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    ellipse_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # draw split points on edge panel (cyan) before contours so green overlays on top
    for sp in all_split_pts:
        cv2.circle(edges_bgr, (int(sp[0]), int(sp[1])), 4, (255, 255, 0), -1)

    for contour in segments:
        if len(contour) < MIN_CONTOUR_PTS:
            continue

        pts = contour.reshape(-1, 2).astype(np.float32)

        try:
            ellipse_tuple = cv2.fitEllipseAMS(pts)
        except cv2.error:
            continue

        (cx, cy), (w, h), angle = ellipse_tuple
        semi_a = max(w, h) / 2
        semi_b = min(w, h) / 2
        fit_angle = angle + 90 if h > w else angle

        if not (semi_major_min <= semi_a <= semi_major_max):
            continue
        if semi_b / semi_a < AXIS_RATIO_MIN:
            continue
        if not (0 < cx < w_img and 0 < cy < h_img):
            continue

        ellipse_perim = np.pi * (3*(semi_a + semi_b) - np.sqrt((3*semi_a + semi_b)*(semi_a + 3*semi_b)))
        if len(pts) / ellipse_perim < ARC_COVERAGE_MIN:
            continue

        err = reprojection_error(pts, cx, cy, semi_a, semi_b, fit_angle)
        if err > REPROJ_ERR_MAX:
            continue

        cv2.drawContours(edges_bgr, [contour], -1, (0, 255, 0), 1)
        candidate_contours[len(candidates)] = contour
        candidates.append(ellipse_tuple)

    # draw individual candidate ellipses in red on ellipse panel
    for ellipse_tuple in candidates:
        cv2.ellipse(ellipse_bgr, ellipse_tuple, (0, 0, 255), 3)
        cx, cy = ellipse_tuple[0]
        cv2.circle(ellipse_bgr, (int(cx), int(cy)), 6, (0, 0, 255), -1)

    # Union-Find over candidates; merge any pair whose ellipses overlap
    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if ellipses_overlap(candidates[i], candidates[j]):
                union(i, j)

    groups: dict = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    overlap_groups = [g for g in groups.values() if len(g) >= OVERSPACE]

    # Pass 1 — collect all valid refits for this frame
    BRIGHT_GREEN = (0, 255, 0)
    frame_detections = []  # list of (refit, sa_r, group_idxs)

    for group_idxs in overlap_groups:
        refit = refit_ellipse_from_group(group_idxs, candidate_contours)
        if refit is None:
            continue

        (cx_r, cy_r), (w_r, h_r), _ = refit
        sa_r = max(w_r, h_r) / 2
        sb_r = min(w_r, h_r) / 2
        if not (semi_major_min <= sa_r <= semi_major_max):
            continue
        if sb_r / sa_r < AXIS_RATIO_MIN:
            continue
        if not (0 < cx_r < w_img and 0 < cy_r < h_img):
            continue

        frame_detections.append((refit, sa_r, group_idxs))

    # Pass 2 — keep only refits within WHEEL_SIZE_RATIO of the largest this frame
    if frame_detections:
        max_sa = max(sa for _, sa, _ in frame_detections)

        for refit, sa_r, group_idxs in frame_detections:
            if sa_r / max_sa < WHEEL_SIZE_RATIO:
                continue  # too small relative to largest detection, discard

            (cx_r, cy_r), (w_r, h_r), angle_r = refit

            if cam_matrix is not None:
                range_m = est_range(cam_matrix, max(w_r, h_r), angle_r, h_r > w_r)
                if not (Z_MIN_M <= range_m <= Z_MAX_M):
                    continue   # physically implausible — skip scatter but still draw ellipse
                pos_xyz = est_pos(cx_r, cy_r, range_m, cam_matrix)
                world_points.append(pos_xyz)
                cv2.putText(ellipse_bgr, f"Z={range_m:.2f}m  X={pos_xyz[0]:+.2f}m",(int(cx_r) + 10, int(cy_r) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRIGHT_GREEN, 2)

            cv2.ellipse(ellipse_bgr, refit, BRIGHT_GREEN, 3)
            cv2.circle(ellipse_bgr, (int(cx_r), int(cy_r)), 8, BRIGHT_GREEN, -1)
            cv2.putText(ellipse_bgr, f"refit n={len(group_idxs)}", (int(cx_r) + 10, int(cy_r) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRIGHT_GREEN, 2)

    # debug view: 2x2 grid
    # TL: blurred | TR: edges + split pts (cyan) + passing contours (green)
    # BL: grayscale + ellipses (red) + refits (green) | BR: wheel position scatter
    bike_path = draw_scatter_panel(world_points, h_img, w_img)
    top = cv2.hconcat([blurred_bgr, edges_bgr])
    bot = cv2.hconcat([ellipse_bgr, bike_path])
    debug = cv2.vconcat([top, bot])

    return debug

# detect wheels in images
def detect_wheels(image_path: str, cam: str):
    image_folder = Path(image_path)
    images, names = load_images(image_folder)

    # results subfolder mirrors input folder name
    results_dir = Path("results") / image_folder.name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load camera matrix once; adapt per-image inside the loop
    cam_matrix_raw, dist_coeffs = load_calibration(cam)

    # Accumulate detected wheel world positions across frames for the scatter panel
    world_points: list[tuple[float, float, float]] = []

    for img, name in zip(images, names):
        h_img, w_img = img.shape[:2]
        if cam_matrix_raw is not None:
            cam_matrix = adapt_camera_matrix(cam_matrix_raw, h_img, w_img)
            map1, map2 = cv2.initUndistortRectifyMap(cam_matrix, dist_coeffs, None, cam_matrix, (w_img, h_img), cv2.CV_16SC2) # type: ignore
        else:
            cam_matrix, map1, map2 = None, None, None
        debug = process_frame(img, cam_matrix, map1, map2, world_points, scale=1.0) # process image contains workflow from old detect_wheels function
        cv2.imwrite(str(results_dir / name), debug)
        print(f"  Saved {name}")

    print(f"  Results written to {results_dir}\n")
    cv2.destroyAllWindows()

# detect wheels from video data
def detect_wheels_video(video_path: str, cam: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Opened: {fw}x{fh} @ {fps:.1f}fps")

    # Writer output matches the resized debug grid (fw x fh)
    out_w, out_h = fw, fh

    results_dir = Path("results") / Path(video_path).stem
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(results_dir / "output.avi")
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Error: Could not open VideoWriter for {out_path}")
        cap.release()
        return

    cam_matrix_raw, dist_coeffs = load_calibration(cam)

    # Precompute undistortion remap at scaled resolution (computed once, reused every frame)
    sh, sw = int(fh * SCALE), int(fw * SCALE)
    if cam_matrix_raw is not None:
        cam_matrix = adapt_camera_matrix(cam_matrix_raw, sh, sw)
        map1, map2 = cv2.initUndistortRectifyMap(cam_matrix, dist_coeffs, None, cam_matrix, (sw, sh), cv2.CV_16SC2) # type: ignore
    else:
        cam_matrix, map1, map2 = None, None, None

    world_points: list[tuple[float, float, float]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame, (sw, sh))
        debug = process_frame(frame_small, cam_matrix, map1, map2, world_points, scale=SCALE)

        debug_out = cv2.resize(debug, (out_w, out_h))
        writer.write(debug_out)

        print(f"  Frame {frame_idx}", end="\r")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n  Saved {frame_idx} frames to {out_path}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wheel_ranger.py <path> [cam: 0.5x / 1x]")
        print("Included img folder dirs examples: data/1x_burst_2 data/0.5x_tire data/IMG_7537.MOV")
        sys.exit(1)

    path = sys.argv[1]
    cam  = sys.argv[2] if len(sys.argv) > 2 else "none"

    if Path(path).suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}:
        detect_wheels_video(path, cam)
    else:
        detect_wheels(path, cam)