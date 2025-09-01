# PVL_OpenPoseMatch - ComfyUI custom node
# Author: ChatGPT (for Pavel)
# License: MIT
#
# Features:
# - POSE_KEYPOINT in, POSE_KEYPOINT + IMAGE out
# - Angle-preserving NECK_OUTWARD retarget:
#     * Directions (angles) from ACTION pose
#     * Segment lengths from REST pose (optionally autoscaled)
# - Centering (BBOX) by default
# - Robust hand rendering + wrist anchoring
# - Missing-part interpolation (symmetry) for body + hands (with left/right flip for hands)
# - Optional shoulder perspective narrowing based on hip-width ratio
# - Final hard enforcement of ARM + LEG segment lengths
# - FACIAL RETARGETING FIX: Preserves facial proportions from rest pose
#
# Notes:
# - Coordinates are treated as NORMALIZED_0_1 internally for rendering and output.

import json, math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import cv2
import matplotlib

Point = Tuple[float, float, float]  # (x, y, confidence)


# ---------------- Parsing ----------------
def _flat_to_points(flat) -> List[Point]:
    pts: List[Point] = []
    for i in range(0, len(flat), 3):
        x = float(flat[i]); y = float(flat[i+1]); c = float(flat[i+2])
        # treat (0,0) or negative coords as invalid
        if x < 0 or y < 0 or (abs(x) < 1e-6 and abs(y) < 1e-6):
            c = 0.0
        pts.append((x, y, c))
    return pts


def _triplets_to_points(tris) -> List[Point]:
    pts: List[Point] = []
    for t in tris:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            x = float(t[0]); y = float(t[1]); c = float(t[2]) if len(t) >= 3 else 1.0
            if x < 0 or y < 0 or (abs(x) < 1e-6 and abs(y) < 1e-6):
                c = 0.0
            pts.append((x, y, c))
    return pts


def _points_to_flat(pts: List[Point]) -> List[float]:
    out: List[float] = []
    for x, y, c in pts:
        out.extend([float(x), float(y), float(c if c is not None else 1.0)])
    return out


def _parse_any_to_points(obj) -> Dict[str, List[Point]]:
    """
    Accepts:
      - POSE_KEYPOINT list of dicts with ["people"][0][...]
      - Direct body arrays (flat 3N or list-of-triplets)
    Returns dict with keys: 'body', 'face', 'hand_l', 'hand_r' (when found).
    """
    out: Dict[str, List[Point]] = {}
    if obj is None:
        return out

    data = obj
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'people' in data[0]:
        data = data[0]

    if isinstance(data, dict) and 'people' in data and isinstance(data['people'], list) and data['people']:
        person = data['people'][0]
        body = person.get('pose_keypoints_2d') or person.get('pose_keypoints')
        face = person.get('face_keypoints_2d')
        hl = person.get('hand_left_keypoints_2d')
        hr = person.get('hand_right_keypoints_2d')
        if isinstance(body, list):
            out['body'] = _flat_to_points(body) if (len(body) % 3 == 0 and (not body or isinstance(body[0], (int, float)))) else _triplets_to_points(body)
        if isinstance(face, list):
            out['face'] = _flat_to_points(face) if (len(face) % 3 == 0 and (not face or isinstance(face[0], (int, float)))) else _triplets_to_points(face)
        if isinstance(hl, list):
            out['hand_l'] = _flat_to_points(hl) if (len(hl) % 3 == 0 and (not hl or isinstance(hl[0], (int, float)))) else _triplets_to_points(hl)
        if isinstance(hr, list):
            out['hand_r'] = _flat_to_points(hr) if (len(hr) % 3 == 0 and (not hr or isinstance(hr[0], (int, float)))) else _triplets_to_points(hr)
        return out

    if isinstance(data, list):
        if data and isinstance(data[0], (int, float)):
            if len(data) % 3 == 0:
                out['body'] = _flat_to_points(data)
        elif data and isinstance(data[0], (list, tuple)):
            out['body'] = _triplets_to_points(data)

    return out



# ---------------- COCO -> BODY_25-like coercion ----------------
def _coerce_to_body25_like(pts: List[Point], skel: str) -> List[Point]:
    """Return points remapped so indices behave like BODY_25 for the subset we use.
    For COCO-17:
        0 nose -> 0
        neck (1) -> midpoint of shoulders (5,6)
        r_shoulder(2)-> COCO 6, r_elbow(3)->8, r_wrist(4)->10
        l_shoulder(5)-> COCO 5, l_elbow(6)->7, l_wrist(7)->9
        midhip(8) -> midpoint of hips (11,12)
        r_hip(9)->12, r_knee(10)->14, r_ankle(11)->16
        l_hip(12)->11, l_knee(13)->13, l_ankle(14)->15
    If BODY_25/BODY_135, return as-is.
    """
    if skel in ("BODY_25", "BODY_135"):
        return pts
    out_len = 15
    out = [(0.0, 0.0, 0.0) for _ in range(out_len)]
    def get(i):
        return pts[i] if i < len(pts) else (0.0, 0.0, 0.0)
    def mid(a, b):
        pa, pb = get(a), get(b)
        if pa[2] > 0 and pb[2] > 0:
            return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, min(pa[2], pb[2]))
        return (0.0, 0.0, 0.0)
    out[0] = get(0)
    out[1] = mid(5, 6)
    out[2] = get(6); out[3] = get(8); out[4] = get(10)
    out[5] = get(5); out[6] = get(7); out[7] = get(9)
    out[8] = mid(11, 12)
    out[9] = get(12); out[10] = get(14); out[11] = get(16)
    out[12] = get(11); out[13] = get(13); out[14] = get(15)
    return out

# ---------------- Skeleton & Geometry ----------------
def _skeleton_type(body_kpts_len: int, force: str) -> str:
    if force in ("BODY_25", "COCO", "BODY_135"):
        return force
    return "BODY_25" if body_kpts_len >= 25 else "COCO"


def _midhip(pts: List[Point], skel: str) -> Optional[Point]:
    if skel in ("BODY_25", "BODY_135") and len(pts) > 8 and pts[8][2] > 0:
        return pts[8]
    # Fallback average hips
    hips = []
    r_i = 9 if skel in ("BODY_25", "BODY_135") else 8
    l_i = 12 if skel in ("BODY_25", "BODY_135") else 11
    if len(pts) > r_i and pts[r_i][2] > 0:
        hips.append(pts[r_i])
    if len(pts) > l_i and pts[l_i][2] > 0:
        hips.append(pts[l_i])
    if hips:
        x = sum(p[0] for p in hips) / len(hips)
        y = sum(p[1] for p in hips) / len(hips)
        return (x, y, 1.0)
    return None


def _body_bones(skel: str):
    if skel in ("BODY_25", "BODY_135"):
        return [
            ("torso_midhip_neck", -1, 1),
            ("neck_head", 1, 0),
            ("r_clavicle", 1, 2), ("r_upper_arm", 2, 3), ("r_lower_arm", 3, 4),
            ("l_clavicle", 1, 5), ("l_upper_arm", 5, 6), ("l_lower_arm", 6, 7),
            ("r_upper_leg", 9, 10), ("r_lower_leg", 10, 11),
            ("l_upper_leg", 12, 13), ("l_lower_leg", 13, 14),
        ]
    else:
        return [
            ("torso_midhip_neck", -1, 1),
            ("neck_head", 1, 0),
            ("r_clavicle", 1, 2), ("r_upper_arm", 2, 3), ("r_lower_arm", 3, 4),
            ("l_clavicle", 1, 5), ("l_upper_arm", 5, 6), ("l_lower_arm", 6, 7),
            ("r_upper_leg", 8, 9), ("r_lower_leg", 9, 10),
            ("l_upper_leg", 11, 12), ("l_lower_leg", 12, 13),
        ]


def _len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], 1.0)


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], 1.0)


def _scale(v: Point, s: float) -> Point:
    return (v[0] * s, v[1] * s, 1.0)


def _unit(v: Point):
    mag = math.hypot(v[0], v[1])
    if mag == 0:
        return (0.0, 0.0, 1.0), 0.0
    return (v[0] / mag, v[1] / mag, 1.0), mag


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _copy_pts(pts: List[Point]) -> List[Point]:
    return [(p[0], p[1], p[2]) for p in pts]


def _leg_indices(skel: str, side: str):
    if skel in ("BODY_25", "BODY_135"):
        return (9, 10, 11) if side == "R" else (12, 13, 14)
    else:
        return (8, 9, 10) if side == "R" else (11, 12, 13)


def _shoulder_indices(skel: str):
    return (2, 5)


def _hip_indices(skel: str):
    return (9, 12) if skel in ("BODY_25", "BODY_135") else (8, 11)


def _wrist_indices(skel: str):
    # (right_wrist_idx, left_wrist_idx) in body array (COCO/BODY_25 compatible)
    return (4, 7)


def _align_hand_to_wrist(hand: List[Point], wrist: Point, conf_threshold: float) -> List[Point]:
    if not hand:
        return hand
    wx, wy, wc = wrist
    hx, hy, hc = hand[0]
    if wc <= conf_threshold:
        return hand
    dx, dy = wx - hx, wy - hy
    return [(x + dx, y + dy, c) for (x, y, c) in hand]


def _hand_missing(hand: Optional[List[Point]], conf_threshold: float) -> bool:
    if not hand or len(hand) < 21:
        return True
    good = sum(1 for (_, _, c) in hand if c > conf_threshold)
    # Require most joints to exist; partial hands degrade scaling accuracy
    return good < 20


def _mirror_hand_for_lengths(src: List[Point]) -> List[Point]:
    # Mirror around the palm root x (index 0). Only used to *borrow lengths*, exact coords don't matter.
    if not src:
        return []
    cx = src[0][0]
    return [(2.0 * cx - x, y, c) for (x, y, c) in src]


def _complete_hand_by_mirror(left_hand: List[Point], right_hand: List[Point], conf_threshold: float) -> tuple:
    """
    Returns possibly replaced (L, R) where a missing side is replaced by a mirrored counterpart entirely.
    """
    L_missing = _hand_missing(left_hand, conf_threshold)
    R_missing = _hand_missing(right_hand, conf_threshold)
    if L_missing and not R_missing:
        return (_mirror_hand_for_lengths(right_hand), right_hand)
    if R_missing and not L_missing:
        return (left_hand, _mirror_hand_for_lengths(left_hand))
    return (left_hand, right_hand)


# ---------------- Retarget ----------------
def _retarget_body(rest_body: List[Point], act_body: List[Point], conf_threshold: float, skel: str, auto_scale_lengths: bool) -> List[Point]:
    """
    Generic: rebuild each child from its parent, using ACTION direction and REST length.
    """
    if len(rest_body) != len(act_body):
        n = min(len(rest_body), len(act_body))
        rest_body, act_body = rest_body[:n], act_body[:n]

    bones = _body_bones(skel)
    rest_midhip = _midhip(rest_body, skel)
    act_midhip = _midhip(act_body, skel)
    ref_len: Dict[str, float] = {}
    act_len: Dict[str, float] = {}

    for name, a_idx, b_idx in bones:
        if name == "torso_midhip_neck":
            if rest_midhip is not None and len(rest_body) > 1 and rest_body[1][2] > conf_threshold:
                ref_len[name] = _len(rest_midhip, rest_body[1])
            if act_midhip is not None and len(act_body) > 1 and act_body[1][2] > conf_threshold:
                act_len[name] = _len(act_midhip, act_body[1])
        else:
            if min(a_idx, b_idx) >= 0 and a_idx < len(rest_body) and b_idx < len(rest_body):
                if rest_body[a_idx][2] > conf_threshold and rest_body[b_idx][2] > conf_threshold:
                    ref_len[name] = _len(rest_body[a_idx], rest_body[b_idx])
            if min(a_idx, b_idx) >= 0 and a_idx < len(act_body) and b_idx < len(act_body):
                if act_body[a_idx][2] > conf_threshold and act_body[b_idx][2] > conf_threshold:
                    act_len[name] = _len(act_body[a_idx], act_body[b_idx])

    scale_ratio = 1.0
    if auto_scale_lengths:
        common = [nm for nm in ref_len if nm in act_len]
        if common:
            med_rest = _median([ref_len[n] for n in common])
            med_act = _median([act_len[n] for n in common])
            if med_rest > 0 and med_act > 0:
                scale_ratio = med_act / med_rest

    new_pts = _copy_pts(act_body)

    def get_joint(pts: List[Point], idx: int) -> Optional[Point]:
        if idx == -1:
            return act_midhip
        if 0 <= idx < len(pts) and pts[idx][2] > conf_threshold:
            return pts[idx]
        return None

    # Neck from midhip
    if "torso_midhip_neck" in ref_len and act_midhip is not None and len(act_body) > 1 and act_body[1][2] > conf_threshold:
        parent = act_midhip
        child = act_body[1]
        dir_u, mag = _unit(_sub(child, parent))
        if mag > 0:
            length = ref_len["torso_midhip_neck"] * scale_ratio
            new_neck = _add(parent, _scale(dir_u, length))
            new_pts[1] = (new_neck[0], new_neck[1], act_body[1][2])

    # Head & clavicles
    for name, a_idx, b_idx in [("neck_head", 1, 0), ("r_clavicle", 1, 2), ("l_clavicle", 1, 5)]:
        if name not in ref_len:
            continue
        parent = get_joint(new_pts, a_idx)
        child = get_joint(new_pts, b_idx)
        if parent is None or child is None:
            continue
        dir_u, mag = _unit(_sub(child, parent))
        if mag == 0:
            continue
        new_child = _add(parent, _scale(dir_u, ref_len[name] * scale_ratio))
        new_pts[b_idx] = (new_child[0], new_child[1], child[2])

    # Remaining bones
    for name, a_idx, b_idx in bones:
        if name in ("torso_midhip_neck", "neck_head", "r_clavicle", "l_clavicle"):
            continue
        if name not in ref_len:
            continue
        parent = get_joint(new_pts, a_idx if a_idx != -1 else -1)
        child = get_joint(new_pts, b_idx)
        if parent is None or child is None:
            continue
        dir_u, mag = _unit(_sub(child, parent))
        if mag == 0:
            continue
        new_child = _add(parent, _scale(dir_u, ref_len[name] * scale_ratio))
        new_pts[b_idx] = (new_child[0], new_child[1], child[2])

    return new_pts


def _retarget_body_from_feet(rest_body: List[Point], act_body: List[Point], conf_threshold: float, skel: str, auto_scale_lengths: bool) -> List[Point]:
    """
    Optional: bottom-up build for legs first, then rest via _retarget_body.
    """
    new_pts = _copy_pts(act_body)
    hR, kR, aR = _leg_indices(skel, "R")
    hL, kL, aL = _leg_indices(skel, "L")

    def seg_len(body, a, b):
        if a < len(body) and b < len(body) and body[a][2] > conf_threshold and body[b][2] > conf_threshold:
            return _len(body[a], body[b])
        return None

    shinR = seg_len(rest_body, kR, aR); thighR = seg_len(rest_body, hR, kR)
    shinL = seg_len(rest_body, kL, aL); thighL = seg_len(rest_body, hL, kL)

    scale_ratio = 1.0
    if auto_scale_lengths:
        l_rest = [v for v in [shinR, thighR, shinL, thighL] if v is not None]
        l_act = []
        for (ka, aa, h, k) in [(kR, aR, hR, kR), (kL, aL, hL, kL)]:
            if ka < len(act_body) and aa < len(act_body) and act_body[ka][2] > conf_threshold and act_body[aa][2] > conf_threshold:
                l_act.append(_len(act_body[ka], act_body[aa]))
            if h < len(act_body) and k < len(act_body) and act_body[h][2] > conf_threshold and act_body[k][2] > conf_threshold:
                l_act.append(_len(act_body[h], act_body[k]))
        if l_rest and l_act:
            scale_ratio = _median(l_act) / _median(l_rest)

    def bottom_up_leg(h_idx, k_idx, a_idx, shin_len, thigh_len):
        nonlocal new_pts
        if shin_len is None or thigh_len is None:
            return
        if not (a_idx < len(act_body) and k_idx < len(act_body)):
            return
        ankle = act_body[a_idx]; knee = act_body[k_idx]
        if ankle[2] <= conf_threshold or knee[2] <= conf_threshold:
            return
        dir_k2a, mag = _unit(_sub(ankle, knee))
        if mag == 0:
            return
        new_knee = _add(ankle, _scale(dir_k2a, -shin_len * scale_ratio))
        new_pts[k_idx] = (new_knee[0], new_knee[1], knee[2])
        if h_idx < len(act_body):
            hip = act_body[h_idx]
            if hip[2] > conf_threshold:
                dir_k2h, mag2 = _unit(_sub(hip, knee))
            else:
                dir_k2h, mag2 = (0.0, -1.0, 1.0), 1.0
        else:
            dir_k2h, mag2 = (0.0, -1.0, 1.0), 1.0
        new_hip = _add(new_knee, _scale(dir_k2h, thigh_len * scale_ratio))
        if h_idx < len(new_pts):
            new_pts[h_idx] = (new_hip[0], new_hip[1], new_pts[h_idx][2] if h_idx < len(new_pts) else 1.0)

    if shinR is not None and thighR is not None:
        bottom_up_leg(hR, kR, aR, shinR, thighR)
    if shinL is not None and thighL is not None:
        bottom_up_leg(hL, kL, aL, shinL, thighL)

    return _retarget_body(rest_body, new_pts, conf_threshold, skel, auto_scale_lengths)


def _retarget_body_from_neck(rest_body: List[Point], act_body: List[Point], conf_threshold: float, skel: str,
                             auto_scale_lengths: bool, interpolate_missing: bool, shoulder_perspective: bool) -> List[Point]:
    """
    Neck-rooted angle-preserving retarget that uses rest lengths, with optional:
      - Symmetry interpolation for missing lengths
      - Shoulder-perspective narrowing (scale shoulders radially from neck by hip-width ratio)
    Final pass enforces ARM+LEG lengths to match rest (after any shoulder adjustments).
    """
    if len(rest_body) != len(act_body):
        n = min(len(rest_body), len(act_body))
        rest_body, act_body = rest_body[:n], act_body[:n]

    def length_if(a, b, pts):
        if a < len(pts) and b < len(pts) and pts[a][2] > conf_threshold and pts[b][2] > conf_threshold:
            return _len(pts[a], pts[b])
        return None

    NECK = 1
    if NECK >= len(act_body) or act_body[NECK][2] <= conf_threshold:
        return _copy_pts(act_body)

    # Collect reference lengths from REST
    ref: Dict[str, Optional[float]] = {}
    ref["neck_head"] = length_if(NECK, 0, rest_body)
    ref["neck_r_shoulder"] = length_if(NECK, 2, rest_body)
    ref["neck_l_shoulder"] = length_if(NECK, 5, rest_body)
    ref["r_upper_arm"] = length_if(2, 3, rest_body)
    ref["r_lower_arm"] = length_if(3, 4, rest_body)
    ref["l_upper_arm"] = length_if(5, 6, rest_body)
    ref["l_lower_arm"] = length_if(6, 7, rest_body)

    if skel in ("BODY_25", "BODY_135"):
        L_neck_midhip = length_if(NECK, 8, rest_body)
        if L_neck_midhip is None:
            rest_mid = _midhip(rest_body, skel)
            if rest_mid is not None and rest_body[NECK][2] > conf_threshold:
                L_neck_midhip = _len(rest_body[NECK], rest_mid)
        ref["neck_midhip"] = L_neck_midhip
        ref["midhip_rhip"] = length_if(8, 9, rest_body)
        ref["midhip_lhip"] = length_if(8, 12, rest_body)
        ref["r_thigh"] = length_if(9, 10, rest_body)
        ref["r_shin"] = length_if(10, 11, rest_body)
        ref["l_thigh"] = length_if(12, 13, rest_body)
        ref["l_shin"] = length_if(13, 14, rest_body)
    else:
        ref["neck_rhip"] = length_if(NECK, 8, rest_body)
        ref["neck_lhip"] = length_if(NECK, 11, rest_body)
        ref["r_thigh"] = length_if(8, 9, rest_body)
        ref["r_shin"] = length_if(9, 10, rest_body)
        ref["l_thigh"] = length_if(11, 12, rest_body)
        ref["l_shin"] = length_if(12, 13, rest_body)

    # Symmetry interpolation for missing segments
    if interpolate_missing:
        def copy_sym(a, b):
            if ref.get(a) is None and ref.get(b) is not None:
                ref[a] = ref[b]
            if ref.get(b) is None and ref.get(a) is not None:
                ref[b] = ref[a]
        copy_sym("neck_r_shoulder", "neck_l_shoulder")
        copy_sym("r_upper_arm", "l_upper_arm")
        copy_sym("r_lower_arm", "l_lower_arm")
        if skel in ("BODY_25", "BODY_135"):
            copy_sym("midhip_rhip", "midhip_lhip")
        else:
            copy_sym("neck_rhip", "neck_lhip")
        copy_sym("r_thigh", "l_thigh")
        copy_sym("r_shin", "l_shin")

    # Global scale (optional)
    ref_lengths = [v for v in ref.values() if v is not None]
    scale_ratio = 1.0
    if auto_scale_lengths and ref_lengths:
        act_lengths = []
        def add_act(a, b):
            if a < len(act_body) and b < len(act_body) and act_body[a][2] > conf_threshold and act_body[b][2] > conf_threshold:
                act_lengths.append(_len(act_body[a], act_body[b]))
        for (a, b) in [(NECK, 0), (NECK, 2), (2, 3), (3, 4), (NECK, 5), (5, 6), (6, 7)]:
            add_act(a, b)
        if skel in ("BODY_25", "BODY_135"):
            for (a, b) in [(8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]:
                add_act(a, b)
        else:
            for (a, b) in [(NECK, 8), (8, 9), (9, 10), (NECK, 11), (11, 12), (12, 13)]:
                add_act(a, b)
        if act_lengths:
            med_act = _median(act_lengths); med_ref = _median(ref_lengths)
            if med_act > 0 and med_ref > 0:
                scale_ratio = med_act / med_ref

    new = _copy_pts(act_body)

    def place_child(parent_idx, child_idx, ref_key):
        if parent_idx >= len(new) or child_idx >= len(new):
            return
        parent = new[parent_idx]
        if parent[2] <= conf_threshold:
            return
        if ref.get(ref_key) is None:
            return
        act_parent = act_body[parent_idx]
        act_child = act_body[child_idx] if child_idx < len(act_body) else None
        if act_child is None or act_child[2] <= conf_threshold:
            return
        dir_u, mag = _unit(_sub(act_child, act_parent))
        if mag == 0:
            return
        length = ref[ref_key] * scale_ratio
        cand = _add(parent, _scale(dir_u, length))
        new[child_idx] = (cand[0], cand[1], new[child_idx][2] if child_idx < len(new) else 1.0)

    # Head & shoulders
    place_child(NECK, 0, "neck_head")
    place_child(NECK, 2, "neck_r_shoulder")
    place_child(NECK, 5, "neck_l_shoulder")
    # Arms
    place_child(2, 3, "r_upper_arm")
    place_child(3, 4, "r_lower_arm")
    place_child(5, 6, "l_upper_arm")
    place_child(6, 7, "l_lower_arm")

    # ===== FACIAL RETARGETING FIX =====
    if skel in ("BODY_25", "BODY_135"):
        # Add facial bone constraints
        ref["nose_left_eye"] = length_if(0, 17, rest_body)
        ref["nose_right_eye"] = length_if(0, 18, rest_body)
        ref["nose_left_ear"] = length_if(0, 15, rest_body)
        ref["nose_right_ear"] = length_if(0, 16, rest_body)
        
        # Apply facial proportions from rest pose
        place_child(0, 17, "nose_left_eye")
        place_child(0, 18, "nose_right_eye")
        place_child(0, 15, "nose_left_ear")
        place_child(0, 16, "nose_right_ear")
    # =================================

    if skel in ("BODY_25", "BODY_135"):
        # Neck -> midhip
        if ref.get("neck_midhip") is not None and 8 < len(new):
            act_mid = _midhip(act_body, skel)
            if act_mid is not None:
                dir_u, mag = _unit(_sub(act_mid, act_body[NECK]))
                if mag > 0:
                    pos = _add(new[NECK], _scale(dir_u, ref["neck_midhip"] * scale_ratio))
                    new[8] = (pos[0], pos[1], new[8][2] if 8 < len(new) else 1.0)
        # midhip -> hips
        if 8 < len(new):
            if ref.get("midhip_rhip") is not None and 9 < len(new) and 9 < len(act_body):
                base = _midhip(act_body, skel) or act_body[8]
                dir_u, mag = _unit(_sub(act_body[9], base))
                if mag > 0:
                    pos = _add(new[8], _scale(dir_u, ref["midhip_rhip"] * scale_ratio))
                    new[9] = (pos[0], pos[1], new[9][2] if 9 < len(new) else 1.0)
            if ref.get("midhip_lhip") is not None and 12 < len(new) and 12 < len(act_body):
                base = _midhip(act_body, skel) or act_body[8]
                dir_u, mag = _unit(_sub(act_body[12], base))
                if mag > 0:
                    pos = _add(new[8], _scale(dir_u, ref["midhip_lhip"] * scale_ratio))
                    new[12] = (pos[0], pos[1], new[12][2] if 12 < len(new) else 1.0)
        # Legs
        for (h, k, a, key_thigh, key_shin) in [(9, 10, 11, "r_thigh", "r_shin"), (12, 13, 14, "l_thigh", "l_shin")]:
            place_child(h, k, key_thigh)
            place_child(k, a, key_shin)
    else:
        # COCO: Neck->hips directly
        place_child(1, 8, "neck_rhip")
        place_child(8, 9, "r_thigh")
        place_child(9, 10, "r_shin")
        place_child(1, 11, "neck_lhip")
        place_child(11, 12, "l_thigh")
        place_child(12, 13, "l_shin")

    # --- Optional: shoulder perspective narrowing based on hip width ratio ---
    if shoulder_perspective:
        Rhip, Lhip = _hip_indices(skel)
        Rsho, Lsho = _shoulder_indices(skel)
        def dist_if(a, b, pts):
            if a < len(pts) and b < len(pts) and pts[a][2] > conf_threshold and pts[b][2] > conf_threshold:
                return _len(pts[a], pts[b])
            return None
        hip_rest = dist_if(Rhip, Lhip, rest_body)
        hip_new = dist_if(Rhip, Lhip, new)
        if hip_rest is not None and hip_rest > 0 and hip_new is not None:
            ratio = hip_new / hip_rest
            # scale shoulders radially from neck by the same ratio
            def scale_from_neck(idx):
                nonlocal new
                if idx < len(new) and new[idx][2] > conf_threshold and NECK < len(new) and new[NECK][2] > conf_threshold:
                    vx, vy = new[idx][0] - new[NECK][0], new[idx][1] - new[NECK][1]
                    nx, ny = new[NECK][0] + vx * ratio, new[NECK][1] + vy * ratio
                    new[idx] = (nx, ny, new[idx][2])
            scale_from_neck(Rsho); scale_from_neck(Lsho)

            # Re-plant arms from the moved shoulders, preserving action angles & rest lengths
            def length_if2(a, b, pts):
                if a < len(pts) and b < len(pts) and pts[a][2] > conf_threshold and pts[b][2] > conf_threshold:
                    return _len(pts[a], pts[b])
                return None
            L_rua = length_if2(2, 3, rest_body); L_rla = length_if2(3, 4, rest_body)
            L_lua = length_if2(5, 6, rest_body); L_lla = length_if2(6, 7, rest_body)

            def place_from_action(parent_idx, child_idx, Lref):
                nonlocal new
                if Lref is None:
                    return
                if parent_idx >= len(new) or child_idx >= len(new):
                    return
                if parent_idx >= len(act_body) or child_idx >= len(act_body):
                    return
                par = new[parent_idx]; apar = act_body[parent_idx]; ach = act_body[child_idx]
                if par[2] <= conf_threshold or apar[2] <= conf_threshold or ach[2] <= conf_threshold:
                    return
                dir_u, mag = _unit(_sub(ach, apar))
                if mag == 0:
                    return
                target = _add(par, _scale(dir_u, Lref * scale_ratio))
                new[child_idx] = (target[0], target[1], new[child_idx][2] if child_idx < len(new) else 1.0)

            place_from_action(2, 3, L_rua); place_from_action(3, 4, L_rla)
            place_from_action(5, 6, L_lua); place_from_action(6, 7, L_lla)

    # Final enforcement: ensure arm segments match rest lengths (use global scale_ratio if enabled)
    def _len_if_arm(a, b, pts):
        if a < len(pts) and b < len(pts) and pts[a][2] > conf_threshold and pts[b][2] > conf_threshold:
            return _len(pts[a], pts[b])
        return None
    L_rua = _len_if_arm(2, 3, rest_body); L_rla = _len_if_arm(3, 4, rest_body)
    L_lua = _len_if_arm(5, 6, rest_body); L_lla = _len_if_arm(6, 7, rest_body)
    if interpolate_missing:
        if L_rua is None and L_lua is not None: L_rua = L_lua
        if L_lua is None and L_rua is not None: L_lua = L_rua
        if L_rla is None and L_lla is not None: L_rla = L_lla
        if L_lla is None and L_rla is not None: L_lla = L_rla
    def enforce_arm(parent_idx, child_idx, Lref):
        nonlocal new
        if Lref is None: return
        if parent_idx >= len(new) or child_idx >= len(new): return
        if parent_idx >= len(act_body) or child_idx >= len(act_body): return
        par = new[parent_idx]; apar = act_body[parent_idx]; ach = act_body[child_idx]
        if par[2] <= conf_threshold or apar[2] <= conf_threshold or ach[2] <= conf_threshold: return
        dir_u, mag = _unit(_sub(ach, apar))
        if mag == 0: return
        target = _add(par, _scale(dir_u, Lref * scale_ratio))
        new[child_idx] = (target[0], target[1], new[child_idx][2] if child_idx < len(new) else 1.0)
    enforce_arm(2, 3, L_rua); enforce_arm(3, 4, L_rla)
    enforce_arm(5, 6, L_lua); enforce_arm(6, 7, L_lla)

    # Final enforcement: ensure leg segments match rest lengths (with optional global scale_ratio)
    def _len_if(a, b, pts):
        if a < len(pts) and b < len(pts) and pts[a][2] > conf_threshold and pts[b][2] > conf_threshold:
            return _len(pts[a], pts[b])
        return None

    if skel in ("BODY_25", "BODY_135"):
        L_r_thigh = _len_if(9, 10, rest_body)
        L_r_shin = _len_if(10, 11, rest_body)
        L_l_thigh = _len_if(12, 13, rest_body)
        L_l_shin = _len_if(13, 14, rest_body)
        if interpolate_missing:
            if L_r_thigh is None and L_l_thigh is not None: L_r_thigh = L_l_thigh
            if L_l_thigh is None and L_r_thigh is not None: L_l_thigh = L_r_thigh
            if L_r_shin is None and L_l_shin is not None: L_r_shin = L_l_shin
            if L_l_shin is None and L_r_shin is not None: L_l_shin = L_r_shin
        def enforce(parent_idx, child_idx, Lref):
            nonlocal new
            if Lref is None: return
            if parent_idx >= len(new) or child_idx >= len(new): return
            if parent_idx >= len(act_body) or child_idx >= len(act_body): return
            par = new[parent_idx]; apar = act_body[parent_idx]; ach = act_body[child_idx]
            if par[2] <= conf_threshold or apar[2] <= conf_threshold or ach[2] <= conf_threshold: return
            dir_u, mag = _unit(_sub(ach, apar))
            if mag == 0: return
            target = _add(par, _scale(dir_u, Lref * scale_ratio))
            new[child_idx] = (target[0], target[1], new[child_idx][2] if child_idx < len(new) else 1.0)
        enforce(9, 10, L_r_thigh); enforce(10, 11, L_r_shin)
        enforce(12, 13, L_l_thigh); enforce(13, 14, L_l_shin)
    else:
        L_r_thigh = _len_if(8, 9, rest_body)
        L_r_shin = _len_if(9, 10, rest_body)
        L_l_thigh = _len_if(11, 12, rest_body)
        L_l_shin = _len_if(12, 13, rest_body)
        if interpolate_missing:
            if L_r_thigh is None and L_l_thigh is not None: L_r_thigh = L_l_thigh
            if L_l_thigh is None and L_r_thigh is not None: L_l_thigh = L_r_thigh
            if L_r_shin is None and L_l_shin is not None: L_r_shin = L_l_shin
            if L_l_shin is None and L_r_shin is not None: L_l_shin = L_r_shin
        def enforce(parent_idx, child_idx, Lref):
            nonlocal new
            if Lref is None: return
            if parent_idx >= len(new) or child_idx >= len(new): return
            if parent_idx >= len(act_body) or child_idx >= len(act_body): return
            par = new[parent_idx]; apar = act_body[parent_idx]; ach = act_body[child_idx]
            if par[2] <= conf_threshold or apar[2] <= conf_threshold or ach[2] <= conf_threshold: return
            dir_u, mag = _unit(_sub(ach, apar))
            if mag == 0: return
            target = _add(par, _scale(dir_u, Lref * scale_ratio))
            new[child_idx] = (target[0], target[1], new[child_idx][2] if child_idx < len(new) else 1.0)
        enforce(8, 9, L_r_thigh); enforce(9, 10, L_r_shin)
        enforce(11, 12, L_l_thigh); enforce(12, 13, L_l_shin)

    return new


# ---------------- Centering & scaling ----------------
def _bbox_center_valid(all_pts: list, conf_threshold: float, width: int, height: int) -> Optional[Tuple[float, float]]:
    xs = []; ys = []
    for (x, y, c) in all_pts:
        if c <= conf_threshold:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x < -0.01 * width or x > 1.01 * width or y < -0.01 * height or y > 1.01 * height:
            continue
        xs.append(x); ys.append(y)
    if not xs or not ys:
        return None
    return ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)


def _apply_offset(pts: Optional[List[Point]], dx: float, dy: float) -> Optional[List[Point]]:
    if not pts:
        return pts
    return [(p[0] + dx, p[1] + dy, p[2]) for p in pts]


def _sanitize_points_pixels(pts: List[Point], width: int, height: int) -> List[Point]:
    out = []
    eps = 1e-6
    for (x, y, c) in pts:
        if not (np.isfinite(x) and np.isfinite(y)) or (abs(x) < eps and abs(y) < eps):
            out.append((0.0, 0.0, 0.0)); continue
        if x < -0.01 * width or x > 1.01 * width or y < -0.01 * height or y > 1.01 * height:
            out.append((x, y, 0.0))
        else:
            out.append((x, y, c))
    return out


# ---------------- Rendering (OpenPose-like) ----------------
def _draw_bodypose(canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray, pose_marker_size: int) -> np.ndarray:
    H, W, _ = canvas.shape
    limbSeq = np.array([[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                        [1, 16], [16, 18], [3, 17], [6, 18]])
    colors = np.array([[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]], dtype=np.uint8)

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][limbSeq[i] - 1]
            if (index < 0).any():
                continue
            Y = candidate[index.astype(int), 0] * float(W)  # x
            X = candidate[index.astype(int), 1] * float(H)  # y
            mX = np.mean(X)
            mY = np.mean(Y)
            length = math.hypot(X[0] - X[1], Y[0] - Y[1])
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), pose_marker_size), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i].tolist())

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), pose_marker_size, colors[i % len(colors)].tolist(), thickness=-1)
    return canvas


def _draw_handpose(canvas: np.ndarray, all_hand_peaks, hand_marker_size: int, conf_threshold: float,
                   body_n: Optional[List[Point]], skel: str) -> np.ndarray:
    H, W, _ = canvas.shape
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                      [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]])
    rw_idx, lw_idx = _wrist_indices(skel)
    body_px = None
    if body_n:
        body_px = [(int(p[0] * W), int(p[1] * H), p[2]) for p in body_n]

    def _iter_hands(a):
        # Accept [('L', [[x,y,c],...]), ('R', ...)] or [[[x,y,c],...], ...]
        for item in a:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                yield item[0], item[1]
            else:
                yield '?', item

    for side, peaks in _iter_hands(all_hand_peaks):
        peaks_np = np.array(peaks, dtype=np.float32)
        if peaks_np.ndim != 2 or peaks_np.shape[0] < 1:
            continue
        if peaks_np.shape[1] == 2:
            c = np.ones((peaks_np.shape[0], 1), dtype=np.float32)
            peaks_np = np.concatenate([peaks_np, c], axis=1)
        elif peaks_np.shape[1] > 3:
            peaks_np = peaks_np[:, :3]

        # Edges
        for ie, e in enumerate(edges):
            x1, y1, c1 = peaks_np[e[0]]
            x2, y2, c2 = peaks_np[e[1]]
            if c1 <= conf_threshold or c2 <= conf_threshold:
                continue
            if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                continue
            x1p = int(x1 * W); y1p = int(y1 * H)
            x2p = int(x2 * W); y2p = int(y2 * H)
            col = (matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255).astype(np.uint8).tolist()
            cv2.line(canvas, (x1p, y1p), (x2p, y2p), col, thickness=1 if hand_marker_size == 0 else hand_marker_size)

        # Joints
        joint_size = (hand_marker_size + 1) if hand_marker_size < 2 else (hand_marker_size + 2)
        for i in range(peaks_np.shape[0]):
            x, y, c = peaks_np[i]
            if c <= conf_threshold:
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            cv2.circle(canvas, (int(x * W), int(y * H)), joint_size, (0, 0, 255), thickness=-1)

        # Bridge from body wrist -> hand[0]
        if body_px is not None and peaks_np.shape[0] > 0 and peaks_np[0, 2] > conf_threshold:
            wx = wy = None
            if side == 'R' and rw_idx < len(body_px) and body_px[rw_idx][2] > conf_threshold:
                wx, wy = body_px[rw_idx][0], body_px[rw_idx][1]
            if side == 'L' and lw_idx < len(body_px) and body_px[lw_idx][2] > conf_threshold:
                wx, wy = body_px[lw_idx][0], body_px[lw_idx][1]
            if wx is not None:
                x0 = int(peaks_np[0, 0] * W); y0 = int(peaks_np[0, 1] * H)
                cv2.line(canvas, (wx, wy), (x0, y0), (255, 255, 255), thickness=max(1, hand_marker_size))

    return canvas


def _render_pose_to_image_like_util(
    body: Optional[List[Point]], face: Optional[List[Point]], hand_l: Optional[List[Point]], hand_r: Optional[List[Point]],
    width: int, height: int, pose_marker_size: int, face_marker_size: int, hand_marker_size: int, conf_threshold: float, skel: str,
) -> torch.Tensor:
    # coords_mode is fixed to NORMALIZED_0_1 internally for renderer
    body_n = [(p[0] / width, p[1] / height, p[2]) for p in body] if body else None
    face_n = [(p[0] / width, p[1] / height, p[2]) for p in face] if face else None
    hand_l_n = [(p[0] / width, p[1] / height, p[2]) for p in hand_l] if hand_l else None
    hand_r_n = [(p[0] / width, p[1] / height, p[2]) for p in hand_r] if hand_r else None

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if body_n:
        candidate = [[x, y] for (x, y, c) in body_n]
        subset = [i if body_n[i][2] > conf_threshold else -1 for i in range(len(body_n))]
        candidate = np.array(candidate, dtype=np.float32)
        subset = np.array([subset], dtype=np.int32)
        canvas = _draw_bodypose(canvas, candidate, subset, pose_marker_size)

    hands = []
    if hand_l_n and len(hand_l_n) == 21:
        hands.append(('L', [[p[0], p[1], p[2]] for p in hand_l_n]))
    if hand_r_n and len(hand_r_n) == 21:
        hands.append(('R', [[p[0], p[1], p[2]] for p in hand_r_n]))
    if hands:
        canvas = _draw_handpose(canvas, hands, hand_marker_size, conf_threshold, body_n, skel)

    if face_n and len(face_n) >= 5:
        for (x, y, c) in face_n:
            if c <= conf_threshold:
                continue
            cv2.circle(canvas, (int(x * width), int(y * height)), face_marker_size, (255, 255, 255), thickness=-1)

    arr = (canvas.astype(np.float32) / 255.0)[None, ...]  # BHWC
    t = torch.from_numpy(arr).contiguous()
    t.clamp_(0.0, 1.0)
    return t


# ---------------- Node ----------------
class PVL_OpenPoseMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rest_pose_keypoints": ("POSE_KEYPOINT", {}),
                "action_pose_keypoints": ("POSE_KEYPOINT", {}),
                "conf_threshold": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "skeleton": (["AUTO", "BODY_25", "COCO", "BODY_135"], {"default": "AUTO"}),
                "render_width": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1}),
                "render_height": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1}),
                "pose_marker_size": ("INT", {"default": 4, "min": 0, "max": 100}),
                "face_marker_size": ("INT", {"default": 3, "min": 0, "max": 100}),
                "hand_marker_size": ("INT", {"default": 2, "min": 0, "max": 100}),
                "include_body": ("BOOLEAN", {"default": True}),
                "include_face": ("BOOLEAN", {"default": False}),
                "include_hands": ("BOOLEAN", {"default": True}),
                "debug_passthrough": ("BOOLEAN", {"default": False}),
                "auto_scale_lengths": ("BOOLEAN", {"default": False}),
                "center_output": ("BOOLEAN", {"default": True}),
                "interpolate_missing": ("BOOLEAN", {"default": True}),
                "shoulder_perspective": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE")
    RETURN_NAMES = ("retargeted_pose_keypoints", "pose_image")
    FUNCTION = "retarget"
    CATEGORY = "PVL/Pose"

    def retarget(
        self,
        rest_pose_keypoints,
        action_pose_keypoints,
        conf_threshold: float,
        skeleton: str,
        render_width: int,
        render_height: int,
        pose_marker_size: int,
        face_marker_size: int,
        hand_marker_size: int,
        debug_passthrough: bool,
        auto_scale_lengths: bool,
        center_output: bool,
        interpolate_missing: bool,
        shoulder_perspective: bool,
        include_body: bool,
        include_face: bool,
        include_hands: bool,
    ):
        # Hidden defaults requested:
        coords_mode = "PIXELS"  # patched: treat incoming keypoints as pixel coords
        center_mode = "BBOX"
        build_mode = "NECK_OUTWARD"
        root_mode = "FEET"  # internal only

        rest = _parse_any_to_points(rest_pose_keypoints)
        act = _parse_any_to_points(action_pose_keypoints)
        body_len = len(rest.get("body") or act.get("body") or [])
        skel = _skeleton_type(body_len, skeleton)

        body_rest = rest.get("body"); body_act = act.get("body")

        if debug_passthrough or not (body_rest and body_act):
            # Passthrough action if anything is missing
            new_body = body_act or []
            new_face = act.get("face", [])
            new_hand_l = act.get("hand_l", [])
            new_hand_r = act.get("hand_r", [])
        else:
            # Main body solve
            if build_mode == "NECK_OUTWARD":
                new_body = _retarget_body_from_neck(body_rest, body_act, conf_threshold, skel,
                                                    auto_scale_lengths, interpolate_missing, shoulder_perspective)
            elif build_mode == "FEET_UP":
                new_body = _retarget_body(body_rest, body_act, conf_threshold, skel, auto_scale_lengths)
            else:
                new_body = _retarget_body(body_rest, body_act, conf_threshold, skel, auto_scale_lengths)

            # Defer hand solve until body is FINAL (after centering)
            new_face = _scale_face_uniform(rest.get("face", []), act.get("face", []), conf_threshold)
            new_hand_l = None
            new_hand_r = None

        # Convert to pixel coords for centering
        def to_pixels(pts):
            if not pts:
                return pts
            if coords_mode == "NORMALIZED_0_1":
                return [(p[0] * render_width, p[1] * render_height, p[2]) for p in pts]
            else:
                return pts

        new_body = to_pixels(new_body or [])
        new_face = to_pixels(new_face or [])

        # Pre-sanitize for centering
        new_body = _sanitize_points_pixels(new_body or [], render_width, render_height)
        new_face = _sanitize_points_pixels(new_face or [], render_width, render_height)

        # Centering
        if center_output:
            all_pts = []
            if include_body and new_body: all_pts += new_body
            if include_face and new_face: all_pts += new_face
            target_cx, target_cy = render_width * 0.5, render_height * 0.5
            current_c = None
            if center_mode == "BBOX":
                current_c = _bbox_center_valid(all_pts, conf_threshold, render_width, render_height)
            if current_c is not None:
                dx = target_cx - current_c[0]; dy = target_cy - current_c[1]
                new_body = _apply_offset(new_body, dx, dy)
                new_face = _apply_offset(new_face, dx, dy)

        # Sanitize again
        new_body = _sanitize_points_pixels(new_body or [], render_width, render_height)
        new_face = _sanitize_points_pixels(new_face or [], render_width, render_height)

        # Now retarget hands after body is final, then anchor to final wrists
        rest_hand_l = rest.get("hand_l", [])
        rest_hand_r = rest.get("hand_r", [])
        act_hand_l = act.get("hand_l", [])
        act_hand_r = act.get("hand_r", [])
        if interpolate_missing:
            rest_hand_l, rest_hand_r = _complete_hand_by_mirror(rest_hand_l, rest_hand_r, conf_threshold)

        new_hand_l = _retarget_hand(rest_hand_l, act_hand_l, conf_threshold)
        new_hand_r = _retarget_hand(rest_hand_r, act_hand_r, conf_threshold)

        # Anchor palms to final wrists
        if new_body:
            rw, lw = _wrist_indices(skel)
            # Convert new_body back to normalized for anchor helpers, then back to pixels
            def to_norm(pts):
                if not pts:
                    return pts
                return [(p[0] / render_width, p[1] / render_height, p[2]) for p in pts]
            if include_hands and include_body:
                body_px = new_body  # already in pixels at this point
                if new_hand_r and rw < len(body_px):
                    new_hand_r = _align_hand_to_wrist(new_hand_r, body_px[rw], conf_threshold)
                if new_hand_l and lw < len(body_px):
                    new_hand_l = _align_hand_to_wrist(new_hand_l, body_px[lw], conf_threshold)

        # Anchor face to head (body index 0 ~ nose) so head stays attached
        if include_face and include_body and new_face and new_body and len(new_body) > 0:
            bx, by, bc = new_body[0] if len(new_body) > 0 else (0,0,0)  # nose
            if bc > conf_threshold:
                xs = [x for (x, y, c) in new_face if c > conf_threshold]
                ys = [y for (x, y, c) in new_face if c > conf_threshold]
                if xs and ys:
                    cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
                    dx, dy = bx - cx, by - cy
                    new_face = [(x + dx, y + dy, c) for (x, y, c) in new_face]

        # Convert hands to pixels & sanitize
        new_hand_l = to_pixels(new_hand_l or [])
        new_hand_r = to_pixels(new_hand_r or [])
        new_hand_l = _sanitize_points_pixels(new_hand_l or [], render_width, render_height)
        new_hand_r = _sanitize_points_pixels(new_hand_r or [], render_width, render_height)

        # Apply include toggles
        if not include_body:
            new_body = []
        if not include_face:
            new_face = []
        if not include_hands:
            new_hand_l = []
            new_hand_r = []

        # Build POSE_KEYPOINT structure in NORMALIZED_0_1
        def to_norm(pts):
            if not pts:
                return pts
            return [(p[0] / render_width, p[1] / render_height, p[2]) for p in pts]

        out_body = to_norm(new_body) if new_body else None
        out_l = to_norm(new_hand_l) if new_hand_l else None
        out_r = to_norm(new_hand_r) if new_hand_r else None
        out_face = to_norm(new_face) if new_face else None

        person: Dict[str, Any] = {}
        if out_body: person["pose_keypoints_2d"] = _points_to_flat(out_body)
        if out_face: person["face_keypoints_2d"] = _points_to_flat(out_face)
        if out_l: person["hand_left_keypoints_2d"] = _points_to_flat(out_l)
        if out_r: person["hand_right_keypoints_2d"] = _points_to_flat(out_r)

        # canvas metadata
        try:
            src_meta = action_pose_keypoints[0] if isinstance(action_pose_keypoints, list) else action_pose_keypoints
            canvas_w = int(src_meta.get("canvas_width", render_width)) if isinstance(src_meta, dict) else render_width
            canvas_h = int(src_meta.get("canvas_height", render_height)) if isinstance(src_meta, dict) else render_height
        except Exception:
            canvas_w, canvas_h = render_width, render_height

        out_pose = [{
            "people": [person] if person else [],
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
        }]

        img_tensor = _render_pose_to_image_like_util(
            new_body if new_body else None,
            new_face if new_face else None,
            new_hand_l if new_hand_l else None,
            new_hand_r if new_hand_r else None,
            render_width, render_height,
            pose_marker_size, face_marker_size, hand_marker_size, conf_threshold, skel
        )

        return (out_pose, img_tensor)


# Helper function for face scaling (placeholder implementation)
def _scale_face_uniform(rest_face, act_face, conf_threshold):
    if not rest_face or not act_face:
        return act_face or []
    # Simple uniform scaling based on bounding box
    def bbox(pts):
        xs = [p[0] for p in pts if p[2] > conf_threshold]
        ys = [p[1] for p in pts if p[2] > conf_threshold]
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)
    
    rest_bbox = bbox(rest_face)
    act_bbox = bbox(act_face)
    if not rest_bbox or not act_bbox:
        return act_face
    
    rest_w = rest_bbox[2] - rest_bbox[0]
    rest_h = rest_bbox[3] - rest_bbox[1]
    act_w = act_bbox[2] - act_bbox[0]
    act_h = act_bbox[3] - act_bbox[1]
    
    if rest_w == 0 or rest_h == 0 or act_w == 0 or act_h == 0:
        return act_face
    
    scale_x = rest_w / act_w
    scale_y = rest_h / act_h
    scale = (scale_x + scale_y) / 2
    
    # Find center of action face
    cx = sum(p[0] for p in act_face if p[2] > conf_threshold) / len([p for p in act_face if p[2] > conf_threshold])
    cy = sum(p[1] for p in act_face if p[2] > conf_threshold) / len([p for p in act_face if p[2] > conf_threshold])
    
    # Scale and translate
    scaled_face = []
    for x, y, c in act_face:
        nx = cx + (x - cx) * scale
        ny = cy + (y - cy) * scale
        scaled_face.append((nx, ny, c))
    
    return scaled_face


# Helper function for hand retargeting (placeholder implementation)
def _retarget_hand(rest_hand, act_hand, conf_threshold):
    if not rest_hand or not act_hand:
        return act_hand or []
    
    # Simple uniform scaling based on palm size
    def palm_size(pts):
        if len(pts) < 5:
            return None
        # Use distance between wrist (0) and middle finger base (9)
        if pts[0][2] > conf_threshold and pts[9][2] > conf_threshold:
            return _len(pts[0], pts[9])
        return None
    
    rest_palm = palm_size(rest_hand)
    act_palm = palm_size(act_hand)
    if not rest_palm or not act_palm:
        return act_hand
    
    scale = rest_palm / act_palm
    
    # Scale from wrist (0)
    wx, wy, wc = act_hand[0]
    scaled_hand = []
    for i, (x, y, c) in enumerate(act_hand):
        if i == 0:  # Keep wrist position
            scaled_hand.append((x, y, c))
        else:
            nx = wx + (x - wx) * scale
            ny = wy + (y - wy) * scale
            scaled_hand.append((nx, ny, c))
    
    return scaled_hand