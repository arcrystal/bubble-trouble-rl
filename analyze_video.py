#!/usr/bin/env python3
"""Extract ball physics from a Bubble Struggle 2: Rebubbled gameplay video.

Detects play area, tracks balls frame-by-frame, measures bounce heights/periods,
pop impulse physics, and level features. Outputs video_physics.json.

Usage:
    python analyze_video.py /path/to/video.mp4 --output video_physics.json
    python analyze_video.py /path/to/video.mp4 --debug  # save debug frames
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment


# ─── HSV color ranges for each ball level ─────────────────────────────────────
# (H_low, S_low, V_low, H_high, S_high, V_high)
# OpenCV uses H: 0-179, S: 0-255, V: 0-255
BALL_HSV_RANGES = {
    1: [(20, 130, 180, 40, 255, 255)],         # Yellow
    2: [(35, 80, 100, 85, 255, 255)],           # Green
    3: [(95, 80, 100, 130, 255, 255)],          # Blue
    4: [(5, 140, 170, 22, 255, 255)],           # Orange
    5: [(0, 130, 130, 8, 255, 255),             # Red (wraps around H=0)
        (170, 130, 130, 179, 255, 255)],
    6: [(0, 70, 50, 10, 200, 150)],             # Dark red
}

# Minimum contour area (in pixels) to consider as a ball
MIN_BALL_AREA = 30
# Maximum distance for track assignment between frames (pixels)
MAX_TRACK_DISTANCE = 60
# Minimum track length to consider valid (frames)
MIN_TRACK_LENGTH = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Extract physics from gameplay video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="video_physics.json",
                        help="Output JSON file path")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug frames with overlay")
    parser.add_argument("--debug-dir", type=str, default="debug_frames",
                        help="Directory for debug frames")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process only first N frames (0 = all)")
    return parser.parse_args()


# ─── Play Area Detection ──────────────────────────────────────────────────────

def detect_play_area(cap, num_frames=200):
    """Detect the play area boundaries from the first gameplay frames.

    The game has a stone border. We detect the transition from stone/HUD to
    the play area background color.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Collect frames, skip very first ones (might be loading/title screen)
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        raise RuntimeError("Could not read any frames")

    # Use a frame in the middle of the sample range - likely gameplay
    # Try multiple frames and find consensus
    candidates = []
    for test_idx in range(min(50, len(frames)), len(frames), 10):
        frame = frames[test_idx]
        h, w = frame.shape[:2]

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # The play area is bounded by stone-colored borders and a HUD bar
        # Scan horizontally at multiple y-positions for left/right edges
        # Scan vertically at multiple x-positions for top/bottom edges

        # Detect left boundary: scan rightward from left edge at y=h//2
        # Looking for transition from dark/stone to play area color
        mid_y = h // 2
        row = frame[mid_y]

        # Stone border is roughly RGB(90,75,55) or similar brownish
        # Play area is a colored background (blue, red, green, etc.)
        # Use saturation/brightness jump to find transition
        hsv_row = cv2.cvtColor(row.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]

        # Left boundary: first x where saturation rises significantly
        left = _find_boundary_left(hsv_row, frame[mid_y])
        right = _find_boundary_right(hsv_row, frame[mid_y], w)

        # Top boundary: scan downward from top at center x
        center_x = w // 2
        col = frame[:, center_x]
        hsv_col = cv2.cvtColor(col.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0]
        top = _find_boundary_top(hsv_col, col)

        # Bottom boundary: scan upward from bottom
        bottom = _find_boundary_bottom(hsv_col, col, h)

        if left < right and top < bottom:
            candidates.append((left, right, top, bottom))

    if not candidates:
        # Fallback: use typical values for 640x360 video
        print("WARNING: Could not auto-detect play area, using defaults for 640x360")
        return {"left": 60, "right": 580, "top": 30, "bottom": 285,
                "width": 520, "height": 255}

    # Take median of all candidates
    candidates = np.array(candidates)
    left = int(np.median(candidates[:, 0]))
    right = int(np.median(candidates[:, 1]))
    top = int(np.median(candidates[:, 2]))
    bottom = int(np.median(candidates[:, 3]))

    return {
        "left": left, "right": right, "top": top, "bottom": bottom,
        "width": right - left, "height": bottom - top
    }


def _find_boundary_left(hsv_row, bgr_row):
    """Find leftmost x where we transition into the play area."""
    w = len(hsv_row)
    # Scan from left, look for first sustained non-stone region
    for x in range(5, w // 4):
        # Check a small window for consistent color (not stone)
        window = bgr_row[x:x+5]
        if len(window) < 5:
            continue
        mean_b, mean_g, mean_r = window.mean(axis=0)
        # Stone is brownish/gray with low saturation relative to play backgrounds
        # Play area backgrounds tend to be more saturated
        hsv_window = hsv_row[x:x+5]
        mean_sat = hsv_window[:, 1].mean()
        mean_val = hsv_window[:, 2].mean()

        if mean_sat > 40 and mean_val > 40:
            return x
    return 60  # fallback


def _find_boundary_right(hsv_row, bgr_row, w):
    """Find rightmost x where play area ends."""
    for x in range(w - 5, w * 3 // 4, -1):
        window = bgr_row[x-5:x]
        if len(window) < 5:
            continue
        hsv_window = hsv_row[x-5:x]
        mean_sat = hsv_window[:, 1].mean()
        mean_val = hsv_window[:, 2].mean()

        if mean_sat > 40 and mean_val > 40:
            return x
    return w - 60  # fallback


def _find_boundary_top(hsv_col, bgr_col):
    """Find top y where play area begins."""
    h = len(hsv_col)
    for y in range(5, h // 4):
        window = bgr_col[y:y+5]
        if len(window) < 5:
            continue
        hsv_window = hsv_col[y:y+5]
        mean_sat = hsv_window[:, 1].mean()
        mean_val = hsv_window[:, 2].mean()

        if mean_sat > 30 and mean_val > 40:
            return y
    return 30  # fallback


def _find_boundary_bottom(hsv_col, bgr_col, h):
    """Find bottom y where play area ends (before HUD)."""
    for y in range(h - 5, h * 3 // 4, -1):
        window = bgr_col[y-5:y]
        if len(window) < 5:
            continue
        hsv_window = hsv_col[y-5:y]
        mean_sat = hsv_window[:, 1].mean()
        mean_val = hsv_window[:, 2].mean()

        if mean_sat > 30 and mean_val > 40:
            return y
    return h - 75  # fallback


# ─── Level Transition Detection ──────────────────────────────────────────────

def detect_level_transitions(cap, play_area, total_frames, fps):
    """Detect level transitions by monitoring background color changes and ball count."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pa = play_area
    # Sample background color from a small region in the play area (away from balls)
    sample_x = pa["left"] + 10
    sample_y = pa["top"] + 10
    sample_w = 30
    sample_h = 20

    bg_colors = []
    frame_idx = 0

    # Sample every 3 frames for speed
    sample_interval = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            region = frame[sample_y:sample_y+sample_h, sample_x:sample_x+sample_w]
            mean_color = region.mean(axis=(0, 1))
            bg_colors.append((frame_idx, mean_color))

        frame_idx += 1
        if frame_idx >= total_frames:
            break

    if not bg_colors:
        return {}

    # Detect color transitions: when mean background color changes significantly
    transitions = []
    prev_color = bg_colors[0][1]

    for i in range(1, len(bg_colors)):
        frame_num, color = bg_colors[i]
        diff = np.linalg.norm(color - prev_color)
        if diff > 30:  # Significant color change
            transitions.append(frame_num)
            prev_color = color
        else:
            # Slow update for gradual changes
            prev_color = 0.95 * prev_color + 0.05 * color

    # Group transitions into level segments
    # First transition might be title screen -> level 1
    levels = {}
    level_num = 1

    if transitions:
        # The first gameplay frame
        start = transitions[0] if transitions[0] > fps * 2 else 0

        for i, t in enumerate(transitions):
            end = transitions[i + 1] if i + 1 < len(transitions) else total_frames
            # Only register if the segment is long enough (> 2 seconds)
            if end - t > fps * 2:
                levels[level_num] = {"frame_start": t, "frame_end": end}
                level_num += 1

    if not levels:
        # Fallback: treat whole video as level 1
        levels[1] = {"frame_start": 0, "frame_end": total_frames}

    return levels


# ─── Background Subtraction & Color Sampling ─────────────────────────────────

def sample_background_color(cap, play_area, frame_num):
    """Sample the background color at a specific frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return np.array([0, 0, 0])

    pa = play_area
    # Sample from corners of play area (less likely to have balls)
    regions = [
        frame[pa["top"]+5:pa["top"]+15, pa["left"]+5:pa["left"]+15],
        frame[pa["top"]+5:pa["top"]+15, pa["right"]-15:pa["right"]-5],
    ]

    colors = [r.mean(axis=(0, 1)) for r in regions if r.size > 0]
    if colors:
        return np.mean(colors, axis=0)
    return np.array([0, 0, 0])


# ─── Ball Detection ──────────────────────────────────────────────────────────

def detect_balls_in_frame(frame, play_area, bg_color=None):
    """Detect balls in a single frame using HSV color segmentation.

    Returns list of (center_x, center_y, radius, ball_level).
    """
    pa = play_area
    # Crop to play area
    roi = frame[pa["top"]:pa["bottom"], pa["left"]:pa["right"]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    detections = []

    for level, ranges in BALL_HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in ranges:
            lower = np.array([h_lo, s_lo, v_lo])
            upper = np.array([h_hi, s_hi, v_hi])
            mask |= cv2.inRange(hsv, lower, upper)

        # If we have a background color, suppress it
        if bg_color is not None:
            bg_hsv = cv2.cvtColor(bg_color.reshape(1, 1, 3).astype(np.uint8),
                                   cv2.COLOR_BGR2HSV)[0, 0]
            # If background hue is close to this ball's hue range, be more strict
            for (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in ranges:
                if h_lo <= bg_hsv[0] <= h_hi and bg_hsv[1] > 50:
                    # Background matches this color — apply morphological filtering
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.erode(mask, kernel, iterations=2)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    break

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA:
                continue

            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Allow somewhat non-circular (motion blur)
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # Convert from ROI coordinates to full frame
            cx += pa["left"]
            cy += pa["top"]

            detections.append((float(cx), float(cy), float(radius), level))

    # Remove duplicate detections (same ball detected by multiple color ranges)
    detections = _remove_duplicate_detections(detections)

    return detections


def _remove_duplicate_detections(detections, dist_threshold=15):
    """Remove duplicate ball detections that are too close together."""
    if len(detections) <= 1:
        return detections

    keep = []
    used = set()

    # Sort by area (larger first) to prefer more confident detections
    detections.sort(key=lambda d: -d[2])

    for i, (x1, y1, r1, l1) in enumerate(detections):
        if i in used:
            continue
        keep.append((x1, y1, r1, l1))
        for j in range(i + 1, len(detections)):
            if j in used:
                continue
            x2, y2, r2, l2 = detections[j]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < dist_threshold:
                used.add(j)

    return keep


# ─── Ball Tracking ────────────────────────────────────────────────────────────

class BallTracker:
    """Track balls across frames using Hungarian algorithm assignment."""

    def __init__(self):
        self.tracks = {}  # track_id -> list of (frame, x, y, radius, level)
        self.next_id = 0
        self.active_tracks = {}  # track_id -> (last_x, last_y, last_vx, last_vy, level)

    def update(self, frame_num, detections):
        """Assign detections to existing tracks or create new ones.

        detections: list of (cx, cy, radius, level)
        """
        if not detections and not self.active_tracks:
            return

        if not self.active_tracks:
            # No active tracks — create new ones for all detections
            for (cx, cy, r, lvl) in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = [(frame_num, cx, cy, r, lvl)]
                self.active_tracks[tid] = (cx, cy, 0.0, 0.0, lvl)
            return

        if not detections:
            # No detections — check which tracks to deactivate
            to_remove = []
            for tid, (lx, ly, vx, vy, lvl) in self.active_tracks.items():
                last_frame = self.tracks[tid][-1][0]
                if frame_num - last_frame > 10:  # Lost for 10 frames
                    to_remove.append(tid)
            for tid in to_remove:
                del self.active_tracks[tid]
            return

        # Hungarian algorithm assignment
        track_ids = list(self.active_tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Build cost matrix
        cost = np.full((n_tracks, n_dets), 1e6)
        for i, tid in enumerate(track_ids):
            lx, ly, vx, vy, t_lvl = self.active_tracks[tid]
            # Predict position based on velocity
            last_frame = self.tracks[tid][-1][0]
            dt = frame_num - last_frame
            pred_x = lx + vx * dt
            pred_y = ly + vy * dt

            for j, (cx, cy, r, d_lvl) in enumerate(detections):
                dist = np.sqrt((pred_x - cx) ** 2 + (pred_y - cy) ** 2)
                # Penalize level mismatches
                level_penalty = 0 if t_lvl == d_lvl else 50
                cost[i, j] = dist + level_penalty

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < MAX_TRACK_DISTANCE + 50:  # +50 for level penalty buffer
                tid = track_ids[r]
                cx, cy, radius, lvl = detections[c]

                # Update track
                self.tracks[tid].append((frame_num, cx, cy, radius, lvl))

                # Update velocity estimate
                prev = self.tracks[tid][-2]
                dt = frame_num - prev[0]
                if dt > 0:
                    vx = (cx - prev[1]) / dt
                    vy = (cy - prev[2]) / dt
                else:
                    vx, vy = 0.0, 0.0

                self.active_tracks[tid] = (cx, cy, vx, vy, lvl)
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Unmatched tracks — deactivate if lost too long
        for tid in track_ids:
            if tid not in matched_tracks:
                last_frame = self.tracks[tid][-1][0]
                if frame_num - last_frame > 10:
                    del self.active_tracks[tid]

        # Unmatched detections — create new tracks
        for j, (cx, cy, r, lvl) in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = [(frame_num, cx, cy, r, lvl)]
                self.active_tracks[tid] = (cx, cy, 0.0, 0.0, lvl)


# ─── Physics Extraction ──────────────────────────────────────────────────────

def extract_bounce_physics(tracks, play_area, fps):
    """Extract bounce height, period, and horizontal speed per ball level."""
    pa = play_area
    floor_y = pa["bottom"]

    level_bounces = defaultdict(list)  # level -> list of (height_px, period_s)
    level_xspeeds = defaultdict(list)
    level_radii = defaultdict(list)

    for tid, track_data in tracks.items():
        if len(track_data) < MIN_TRACK_LENGTH:
            continue

        # Determine ball level from most common level in track
        levels = [d[4] for d in track_data]
        level = max(set(levels), key=levels.count)

        frames = np.array([d[0] for d in track_data])
        xs = np.array([d[1] for d in track_data])
        ys = np.array([d[2] for d in track_data])
        radii = np.array([d[3] for d in track_data])

        # Collect radius measurement
        level_radii[level].append(float(np.median(radii)))

        # Find floor contacts (local maxima in y, close to floor)
        # Ball bottom = y + radius, floor contact when y + radius ≈ floor_y
        ball_bottoms = ys + radii

        # Find local maxima in y (floor bounces)
        floor_contacts = []
        for i in range(1, len(ys) - 1):
            if ys[i] > ys[i-1] and ys[i] > ys[i+1]:
                # Check if near floor
                if ball_bottoms[i] > floor_y - 30:
                    floor_contacts.append(i)

        # Find local minima in y (bounce peaks)
        peaks = []
        for i in range(1, len(ys) - 1):
            if ys[i] < ys[i-1] and ys[i] < ys[i+1]:
                peaks.append(i)

        # Compute bounce heights (floor contact to nearest peak)
        for fc_idx in floor_contacts:
            # Find nearest peak before this floor contact
            prev_peaks = [p for p in peaks if p < fc_idx]
            # Find nearest peak after this floor contact
            next_peaks = [p for p in peaks if p > fc_idx]

            for peak_list in [prev_peaks, next_peaks]:
                if peak_list:
                    if peak_list is prev_peaks:
                        nearest_peak = peak_list[-1]
                    else:
                        nearest_peak = peak_list[0]

                    height_px = ys[fc_idx] - ys[nearest_peak]
                    if height_px > 10:  # Minimum meaningful bounce
                        level_bounces[level].append(height_px)

        # Compute bounce periods (time between consecutive floor contacts)
        for i in range(1, len(floor_contacts)):
            fc1 = floor_contacts[i - 1]
            fc2 = floor_contacts[i]
            period_frames = frames[fc2] - frames[fc1]
            period_s = period_frames / fps
            if 0.3 < period_s < 4.0:  # Reasonable range
                level_bounces[level].append((-1, period_s))  # Use -1 as height sentinel

        # Compute horizontal speed (median |dx/dt| excluding wall-bounce frames)
        if len(frames) > 1:
            dframes = np.diff(frames)
            dxs = np.diff(xs)
            # Exclude frames where direction changes (wall bounces)
            valid = dframes > 0
            if np.any(valid):
                speeds = np.abs(dxs[valid]) / dframes[valid] * fps
                # Filter out outliers (wall bounces cause spikes)
                speeds = speeds[(speeds > 5) & (speeds < 300)]
                if len(speeds) > 3:
                    level_xspeeds[level].append(float(np.median(speeds)))

    # Aggregate per level
    results = {}
    for level in sorted(set(list(level_bounces.keys()) + list(level_radii.keys()))):
        # Separate bounce heights from periods
        heights = []
        periods = []
        for item in level_bounces.get(level, []):
            if isinstance(item, tuple):
                if item[0] == -1:
                    periods.append(item[1])
            else:
                heights.append(item)

        median_radius = np.median(level_radii.get(level, [0]))
        median_height = np.median(heights) if heights else 0
        median_period = np.median(periods) if periods else 0
        median_xspeed = np.median(level_xspeeds.get(level, [0]))
        n_samples = len(level_radii.get(level, []))

        pa_w = play_area["width"]
        pa_h = play_area["height"]

        results[str(level)] = {
            "radius_px": round(float(median_radius), 1),
            "radius_ratio": round(float(median_radius) / pa_w, 4) if pa_w > 0 else 0,
            "bounce_height_px": round(float(median_height), 1),
            "bounce_height_ratio": round(float(median_height) / pa_h, 4) if pa_h > 0 else 0,
            "bounce_period_s": round(float(median_period), 3),
            "xspeed_px_per_s": round(float(median_xspeed), 1),
            "xspeed_ratio_per_s": round(float(median_xspeed) / pa_w, 4) if pa_w > 0 else 0,
            "samples": n_samples,
        }

    return results


def extract_pop_events(tracks, play_area, fps):
    """Detect ball split events from track endings/beginnings.

    A split is: one track of level L ends, and within 3 frames + nearby position,
    two tracks of level L-1 begin.
    """
    # Build timeline of track starts and ends
    track_starts = {}  # track_id -> (frame, x, y, level)
    track_ends = {}    # track_id -> (frame, x, y, level, last_yspeed)

    for tid, track_data in tracks.items():
        if len(track_data) < 3:
            continue

        level = max(set(d[4] for d in track_data), key=lambda l: sum(1 for d in track_data if d[4] == l))

        first = track_data[0]
        track_starts[tid] = (first[0], first[1], first[2], level)

        last = track_data[-1]
        # Estimate yspeed from last few frames
        if len(track_data) >= 3:
            recent = track_data[-5:] if len(track_data) >= 5 else track_data[-3:]
            ys = [d[2] for d in recent]
            fs = [d[0] for d in recent]
            if fs[-1] > fs[0]:
                yspeed = (ys[-1] - ys[0]) / (fs[-1] - fs[0]) * fps
            else:
                yspeed = 0.0
        else:
            yspeed = 0.0

        track_ends[tid] = (last[0], last[1], last[2], level, yspeed)

    pop_events = []
    ceiling_pop_events = []

    for end_tid, (end_frame, end_x, end_y, end_level, end_yspeed) in track_ends.items():
        if end_level <= 1:
            # Level 1 balls just disappear (pop), no children
            # Check if it's a ceiling pop
            if end_y < play_area["top"] + 20:
                ceiling_pop_events.append({
                    "frame": int(end_frame),
                    "ball_level": int(end_level),
                    "position": [round(end_x, 1), round(end_y, 1)],
                    "is_ceiling_destruction": True,
                })
            continue

        child_level = end_level - 1

        # Find new tracks starting within 5 frames of this end, nearby position
        children = []
        for start_tid, (start_frame, start_x, start_y, start_level) in track_starts.items():
            if start_tid == end_tid:
                continue
            if start_level != child_level:
                continue
            frame_diff = start_frame - end_frame
            if -2 <= frame_diff <= 5:
                dist = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)
                if dist < 80:  # Reasonable proximity
                    children.append(start_tid)

        if len(children) >= 2:
            # This is a split event
            child_yspeeds = []
            for ctid in children[:2]:
                ctrack = tracks[ctid]
                if len(ctrack) >= 3:
                    ys = [d[2] for d in ctrack[:5]]
                    fs = [d[0] for d in ctrack[:5]]
                    if fs[-1] > fs[0]:
                        cyspeed = (ys[-1] - ys[0]) / (fs[-1] - fs[0]) * fps
                    else:
                        cyspeed = 0.0
                    child_yspeeds.append(cyspeed)

            mean_child_yspeed = np.mean(child_yspeeds) if child_yspeeds else 0.0

            pop_events.append({
                "frame": int(end_frame),
                "parent_level": int(end_level),
                "parent_pos": [round(end_x, 1), round(end_y, 1)],
                "parent_yspeed": round(float(end_yspeed), 1),
                "child_level": int(child_level),
                "child_yspeed": round(float(mean_child_yspeed), 1),
                "is_ceiling_pop": False,
            })

        # Check if this was a ceiling destruction (ball destroyed by ceiling)
        if end_y < play_area["top"] + 20:
            ceiling_pop_events.append({
                "frame": int(end_frame),
                "ball_level": int(end_level),
                "position": [round(end_x, 1), round(end_y, 1)],
                "is_ceiling_destruction": True,
            })

    return pop_events, ceiling_pop_events


def fit_pop_physics_model(pop_events, play_area):
    """Fit parametric and per-level pop physics models from observed events."""
    if not pop_events:
        return {
            "inherit_factor": 0.0,
            "per_level_impulse": {},
            "parametric_fit": {"base": 0.0, "exponent": 0.0, "r_squared": 0.0},
        }

    # Collect (parent_yspeed, child_level, child_yspeed) tuples
    data = []
    for event in pop_events:
        parent_yspeed = event["parent_yspeed"]
        child_level = event["child_level"]
        child_yspeed = event["child_yspeed"]
        data.append((parent_yspeed, child_level, child_yspeed))

    if len(data) < 3:
        return {
            "inherit_factor": 0.0,
            "per_level_impulse": {},
            "parametric_fit": {"base": 0.0, "exponent": 0.0, "r_squared": 0.0},
        }

    data = np.array(data)
    parent_yspeeds = data[:, 0]
    child_levels = data[:, 1]
    child_yspeeds = data[:, 2]

    pa_h = play_area["height"]

    # --- Per-level impulse model ---
    # child_yspeed = inherit * parent_yspeed + impulse(child_level)
    # Linear regression per level group
    per_level_impulse = {}
    inherit_estimates = []

    for lvl in sorted(set(child_levels.astype(int))):
        mask = child_levels == lvl
        if mask.sum() < 2:
            continue
        pvs = parent_yspeeds[mask]
        cvs = child_yspeeds[mask]

        # Simple linear fit: cvs = inherit * pvs + impulse
        if len(pvs) >= 2 and np.std(pvs) > 0.1:
            try:
                coeffs = np.polyfit(pvs, cvs, 1)
                inherit_estimates.append(coeffs[0])
                per_level_impulse[str(lvl)] = round(float(coeffs[1]), 1)
            except np.linalg.LinAlgError:
                per_level_impulse[str(lvl)] = round(float(np.mean(cvs)), 1)
        else:
            per_level_impulse[str(lvl)] = round(float(np.mean(cvs)), 1)

    inherit_factor = float(np.median(inherit_estimates)) if inherit_estimates else 0.0

    # --- Parametric model ---
    # impulse = -base / child_level^exp
    # child_yspeed ≈ inherit * parent_yspeed - base / child_level^exp
    residuals = child_yspeeds - inherit_factor * parent_yspeeds

    try:
        def parametric(child_lvl, base, exp):
            return -base / (child_lvl ** exp)

        popt, _ = curve_fit(parametric, child_levels, residuals,
                           p0=[100.0, 0.3], maxfev=5000)
        base, exponent = popt

        predicted = inherit_factor * parent_yspeeds + parametric(child_levels, base, exponent)
        ss_res = np.sum((child_yspeeds - predicted) ** 2)
        ss_tot = np.sum((child_yspeeds - np.mean(child_yspeeds)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except (RuntimeError, ValueError):
        base, exponent, r_squared = 0.0, 0.0, 0.0

    return {
        "inherit_factor": round(float(inherit_factor), 4),
        "per_level_impulse": per_level_impulse,
        "parametric_fit": {
            "base": round(float(base), 2),
            "exponent": round(float(exponent), 4),
            "r_squared": round(float(r_squared), 4),
        }
    }


# ─── Laser & Agent Detection ─────────────────────────────────────────────────

def measure_laser_speed(cap, play_area, level_segments, fps):
    """Measure laser growth rate from video frames."""
    pa = play_area
    laser_lengths = []  # (frame, length_px)

    # Sample a few level segments
    segments_to_check = list(level_segments.values())[:3]

    for seg in segments_to_check:
        start = seg["frame_start"]
        end = min(seg["frame_end"], start + int(fps * 30))  # First 30 seconds

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        prev_laser = None
        laser_start_frame = None

        for fi in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect laser: thin vertical bright line in the play area
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[pa["top"]:pa["bottom"], pa["left"]:pa["right"]]

            # Laser is a bright vertical line — look for very bright, narrow columns
            bright_mask = roi > 200
            col_sums = bright_mask.sum(axis=0)

            # Find columns with enough bright pixels to be a laser (at least 10px tall)
            laser_cols = np.where(col_sums > 10)[0]

            if len(laser_cols) > 0:
                # Find the tallest continuous bright column
                for col in laser_cols:
                    bright_rows = np.where(bright_mask[:, col])[0]
                    if len(bright_rows) < 5:
                        continue
                    length = bright_rows[-1] - bright_rows[0]

                    if prev_laser is None:
                        laser_start_frame = fi
                        prev_laser = length
                    else:
                        laser_lengths.append((fi - laser_start_frame, length))
                    break
            else:
                if prev_laser is not None:
                    prev_laser = None
                    laser_start_frame = None

    if not laser_lengths:
        return {"growth_rate_px_per_s": 0, "growth_rate_ratio_per_s": 0}

    # Compute growth rate: pixels per second
    frame_diffs = [l[0] for l in laser_lengths if l[0] > 0]
    length_diffs = [l[1] for l in laser_lengths if l[0] > 0]

    if frame_diffs:
        # Simple: total length / total frames * fps
        rates = [l / f * fps for f, l in zip(frame_diffs, length_diffs) if f > 0]
        if rates:
            median_rate = float(np.median(rates))
            return {
                "growth_rate_px_per_s": round(median_rate, 1),
                "growth_rate_ratio_per_s": round(median_rate / pa["height"], 4),
            }

    return {"growth_rate_px_per_s": 0, "growth_rate_ratio_per_s": 0}


def measure_agent_speed(tracks_agent, play_area, fps):
    """Measure agent horizontal speed from agent position tracking."""
    pa = play_area
    # For now, return a placeholder — agent tracking would need additional work
    return {
        "speed_px_per_s": 0,
        "speed_ratio_per_s": 0,
        "width_px": 0,
        "height_px": 0,
    }


# ─── Level Feature Detection ────────────────────────────────────────────────

def detect_obstacles(cap, play_area, level_segments, fps):
    """Detect static obstacles (walls, platforms) in each level."""
    pa = play_area
    level_obstacles = {}

    for level_num, seg in level_segments.items():
        # Sample a few frames early in the level (after "get ready")
        sample_frame = seg["frame_start"] + int(fps * 3)  # 3 seconds in
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)

        ret, frame = cap.read()
        if not ret:
            continue

        roi = frame[pa["top"]:pa["bottom"], pa["left"]:pa["right"]]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Obstacles are stone-colored: low saturation, medium value
        # HSV: H=20-30, S=20-80, V=50-120 (brownish gray)
        stone_lower = np.array([10, 15, 40])
        stone_upper = np.array([35, 100, 140])
        stone_mask = cv2.inRange(hsv, stone_lower, stone_upper)

        # Also check for gray stone: very low saturation
        gray_lower = np.array([0, 0, 60])
        gray_upper = np.array([180, 30, 140])
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)

        combined_mask = stone_mask | gray_mask

        # Morphological cleanup — obstacles are large rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Minimum obstacle area
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Convert to ratios relative to play area
            obstacles.append({
                "x_ratio": round(x / pa["width"], 4),
                "y_ratio": round(y / pa["height"], 4),
                "w_ratio": round(w / pa["width"], 4),
                "h_ratio": round(h / pa["height"], 4),
                "x_px": x + pa["left"],
                "y_px": y + pa["top"],
                "w_px": w,
                "h_px": h,
            })

        if obstacles:
            level_obstacles[level_num] = obstacles

    return level_obstacles


def detect_level_initial_balls(cap, play_area, level_segments, fps):
    """Detect the initial ball configuration for each level.

    Sample a frame right after the "get ready" period but before any pops.
    """
    pa = play_area
    level_balls = {}

    for level_num, seg in level_segments.items():
        # Sample 4-5 seconds into the level (after get ready, before most pops)
        sample_frame = seg["frame_start"] + int(fps * 4)
        if sample_frame >= seg["frame_end"]:
            sample_frame = seg["frame_start"] + int(fps * 2)

        bg_color = sample_background_color(cap, play_area, seg["frame_start"] + int(fps * 2))

        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detect_balls_in_frame(frame, play_area, bg_color)

        balls = []
        for (cx, cy, r, lvl) in detections:
            # Convert to ratios relative to play area
            x_ratio = (cx - pa["left"]) / pa["width"]
            y_ratio = (cy - pa["top"]) / pa["height"]
            balls.append({
                "level": int(lvl),
                "x_ratio": round(float(x_ratio), 3),
                "y_ratio": round(float(y_ratio), 3),
                "radius_px": round(float(r), 1),
            })

        level_balls[level_num] = balls

    return level_balls


# ─── Gravity Fitting ─────────────────────────────────────────────────────────

def fit_gravity_from_tracks(tracks, play_area, fps):
    """Fit gravity (yacc) for each ball level using parabolic segments."""
    level_gravity = defaultdict(list)

    for tid, track_data in tracks.items():
        if len(track_data) < 10:
            continue

        level = max(set(d[4] for d in track_data), key=lambda l: sum(1 for d in track_data if d[4] == l))

        frames = np.array([d[0] for d in track_data])
        ys = np.array([d[2] for d in track_data])

        # Find free-flight segments (between floor bounces)
        # A free-flight segment: y decreases then increases (parabolic)
        # Find floor contacts
        ball_bottoms = ys + np.array([d[3] for d in track_data])
        floor_y = play_area["bottom"]

        near_floor = ball_bottoms > floor_y - 15
        floor_indices = np.where(near_floor)[0]

        if len(floor_indices) < 2:
            continue

        # Extract segments between floor contacts
        for i in range(len(floor_indices) - 1):
            start_idx = floor_indices[i]
            end_idx = floor_indices[i + 1]

            if end_idx - start_idx < 5:
                continue

            seg_frames = frames[start_idx:end_idx+1] - frames[start_idx]
            seg_ys = ys[start_idx:end_idx+1]
            seg_ts = seg_frames / fps

            # Fit parabola: y(t) = y0 + v0*t + 0.5*g*t²
            if len(seg_ts) < 4:
                continue

            try:
                coeffs = np.polyfit(seg_ts, seg_ys, 2)
                gravity = 2 * coeffs[0]  # px/s² (positive = downward in screen coords)

                if 100 < gravity < 5000:  # Reasonable gravity range in px/s²
                    level_gravity[level].append(gravity)
            except (np.linalg.LinAlgError, ValueError):
                continue

    results = {}
    for level in sorted(level_gravity.keys()):
        gravities = level_gravity[level]
        if gravities:
            results[str(level)] = {
                "gravity_px_per_s2": round(float(np.median(gravities)), 1),
                "gravity_ratio_per_s2": round(float(np.median(gravities)) / play_area["height"], 4),
                "samples": len(gravities),
            }

    return results


# ─── Debug Visualization ─────────────────────────────────────────────────────

def save_debug_frame(frame, detections, play_area, frame_num, output_dir):
    """Save a frame with ball detection overlay."""
    debug = frame.copy()
    pa = play_area

    # Draw play area boundary
    cv2.rectangle(debug, (pa["left"], pa["top"]), (pa["right"], pa["bottom"]),
                  (0, 255, 0), 1)

    # Draw detected balls
    colors_bgr = {
        1: (0, 255, 255),   # Yellow
        2: (0, 200, 0),     # Green
        3: (255, 180, 100), # Blue
        4: (0, 141, 237),   # Orange
        5: (0, 0, 255),     # Red
        6: (0, 0, 180),     # Dark red
    }

    for (cx, cy, r, lvl) in detections:
        color = colors_bgr.get(lvl, (255, 255, 255))
        cv2.circle(debug, (int(cx), int(cy)), int(r), color, 2)
        cv2.putText(debug, f"L{lvl}", (int(cx) - 8, int(cy) - int(r) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(debug, f"Frame {frame_num}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    Path(output_dir).mkdir(exist_ok=True)
    cv2.imwrite(f"{output_dir}/frame_{frame_num:06d}.png", debug)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.1f}s")
    print()

    # Step 1A: Detect play area
    print("Step 1: Detecting play area...")
    play_area = detect_play_area(cap, num_frames=300)
    print(f"  Play area: left={play_area['left']}, right={play_area['right']}, "
          f"top={play_area['top']}, bottom={play_area['bottom']}")
    print(f"  Size: {play_area['width']}x{play_area['height']} pixels")
    print()

    # Step 1B: Detect level transitions
    print("Step 2: Detecting level transitions...")
    level_segments = detect_level_transitions(cap, play_area, total_frames, fps)
    print(f"  Found {len(level_segments)} levels")
    for lnum, seg in sorted(level_segments.items()):
        duration = (seg["frame_end"] - seg["frame_start"]) / fps
        print(f"    Level {lnum}: frames {seg['frame_start']}-{seg['frame_end']} ({duration:.1f}s)")
    print()

    # Step 1D: Ball detection & tracking (full video scan)
    print("Step 3: Tracking balls across all frames...")
    tracker = BallTracker()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get background colors per level for better detection
    level_bg_colors = {}
    for lnum, seg in level_segments.items():
        bg = sample_background_color(cap, play_area, seg["frame_start"] + int(fps * 2))
        level_bg_colors[lnum] = bg

    # Determine which level each frame belongs to
    frame_to_level = {}
    for lnum, seg in level_segments.items():
        for f in range(seg["frame_start"], seg["frame_end"]):
            frame_to_level[f] = lnum

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    process_interval = 1  # Process every frame for accuracy
    debug_interval = 300   # Save debug frame every N frames

    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if fi % process_interval != 0:
            continue

        # Get background color for current level
        current_level = frame_to_level.get(fi, 1)
        bg_color = level_bg_colors.get(current_level)

        detections = detect_balls_in_frame(frame, play_area, bg_color)
        tracker.update(fi, detections)

        # Debug output
        if args.debug and fi % debug_interval == 0:
            save_debug_frame(frame, detections, play_area, fi, args.debug_dir)

        # Progress
        if fi % 1000 == 0:
            n_active = len(tracker.active_tracks)
            n_total = len(tracker.tracks)
            print(f"  Frame {fi}/{total_frames} ({fi/total_frames*100:.1f}%) — "
                  f"{n_active} active tracks, {n_total} total tracks")

    print(f"  Done. Total tracks: {len(tracker.tracks)}")
    print()

    # Step 1E: Extract bounce physics
    print("Step 4: Extracting bounce physics...")
    ball_physics = extract_bounce_physics(tracker.tracks, play_area, fps)
    for lvl, props in sorted(ball_physics.items()):
        print(f"  Level {lvl}: radius={props['radius_px']:.1f}px, "
              f"bounce_h={props['bounce_height_px']:.1f}px, "
              f"period={props['bounce_period_s']:.3f}s, "
              f"xspeed={props['xspeed_px_per_s']:.1f}px/s "
              f"({props['samples']} samples)")
    print()

    # Step 1E+: Gravity fitting
    print("Step 5: Fitting gravity per ball level...")
    gravity_data = fit_gravity_from_tracks(tracker.tracks, play_area, fps)
    for lvl, props in sorted(gravity_data.items()):
        print(f"  Level {lvl}: gravity={props['gravity_px_per_s2']:.1f} px/s² "
              f"({props['samples']} samples)")
    print()

    # Step 1F: Pop event detection
    print("Step 6: Detecting pop events...")
    pop_events, ceiling_pops = extract_pop_events(tracker.tracks, play_area, fps)
    print(f"  Found {len(pop_events)} split events, {len(ceiling_pops)} ceiling destructions")

    # Show a few examples
    for event in pop_events[:5]:
        print(f"    Frame {event['frame']}: L{event['parent_level']}→L{event['child_level']} "
              f"parent_vy={event['parent_yspeed']:.0f} child_vy={event['child_yspeed']:.0f}")
    print()

    # Step 1G: Fit pop physics model
    print("Step 7: Fitting pop physics model...")
    pop_model = fit_pop_physics_model(pop_events, play_area)
    print(f"  Inherit factor: {pop_model['inherit_factor']:.4f}")
    print(f"  Per-level impulse: {pop_model['per_level_impulse']}")
    print(f"  Parametric fit: base={pop_model['parametric_fit']['base']:.2f}, "
          f"exp={pop_model['parametric_fit']['exponent']:.4f}, "
          f"R²={pop_model['parametric_fit']['r_squared']:.4f}")
    print()

    # Step 1H: Laser speed measurement
    print("Step 8: Measuring laser speed...")
    laser_data = measure_laser_speed(cap, play_area, level_segments, fps)
    print(f"  Growth rate: {laser_data['growth_rate_px_per_s']} px/s "
          f"({laser_data['growth_rate_ratio_per_s']} × height/s)")
    print()

    # Step 1I: Level features
    print("Step 9: Detecting obstacles...")
    level_obstacles = detect_obstacles(cap, play_area, level_segments, fps)
    for lnum, obs in sorted(level_obstacles.items()):
        print(f"  Level {lnum}: {len(obs)} obstacles")
    print()

    # Detect initial balls per level
    print("Step 10: Detecting initial balls per level...")
    level_initial_balls = detect_level_initial_balls(cap, play_area, level_segments, fps)
    for lnum, balls in sorted(level_initial_balls.items()):
        ball_summary = ", ".join(f"L{b['level']}" for b in balls)
        print(f"  Level {lnum}: {len(balls)} balls ({ball_summary})")
    print()

    # Collect background colors
    level_backgrounds = {}
    for lnum, seg in level_segments.items():
        bg = sample_background_color(cap, play_area, seg["frame_start"] + int(fps * 3))
        level_backgrounds[lnum] = [int(bg[2]), int(bg[1]), int(bg[0])]  # BGR -> RGB

    # Build output JSON
    output = {
        "video_metadata": {
            "file": str(video_path),
            "width": width,
            "height": height,
            "fps": float(fps),
            "frames": total_frames,
        },
        "play_area": play_area,
        "ball_levels": ball_physics,
        "gravity": gravity_data,
        "pop_events": pop_events,
        "ceiling_pop_events": ceiling_pops,
        "pop_physics_model": pop_model,
        "laser": laser_data,
        "agent": measure_agent_speed(None, play_area, fps),
        "levels": {},
    }

    # Assemble per-level data
    for lnum in sorted(level_segments.keys()):
        seg = level_segments[lnum]
        level_data = {
            "frame_start": seg["frame_start"],
            "frame_end": seg["frame_end"],
            "background_rgb": level_backgrounds.get(lnum, [0, 0, 0]),
            "initial_balls": level_initial_balls.get(lnum, []),
            "obstacles": level_obstacles.get(lnum, []),
        }
        output["levels"][str(lnum)] = level_data

    # Write output
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written to {output_path}")
    print(f"Total tracks analyzed: {len(tracker.tracks)}")

    cap.release()


if __name__ == "__main__":
    main()
