# pvl_ImageStitch.py
# PVL – Image Stitch (ComfyUI custom node)
#
# Features:
# - 1 required image input + up to 7 optional image inputs (total 8).
# - If only one image provided: passthrough (drop alpha, keep size).
# - Linear arrangement: right / left / up / down.
# - Pack-to-square:
#     * pack_to_square=True ignores linear arrangement and performs a global pack.
#     * Two pack modes:
#         - "grid"       → tries ALL row/column partitions across ALL permutations (legacy).
#         - "guillotine" → MaxRects/guillotine-style free-rect packing (can fill L-shaped space).
#     * Optional post-compaction for guillotine: vertical & horizontal nudging with left/up slides.
# - Optional pre-normalization when packing:
#     * match_image_size=True (AR preserved), compute a single S and resize each image once:
#         - match_type="upscale"   → S = AVGmax = (max_w + max_h) / 2
#         - match_type="downscale" → S = AVGmin = (min_w + min_h) / 2
#       Then pack the already-normalized tiles; no additional per-row/col matching.
# - Resample: lanczos/bicubic/bilinear/area/nearest (Lanczos CPU-only).
# - Spacing (pixels) and spacing color "R,G,B" (0..255).
# - Centering: linear layouts center on the cross-axis; grid-pack centers rows/cols.
# - Square tolerance (%): allow extra area above absolute minimum to select a squarer layout,
#   applied to both grid and guillotine packers.
#
# Output: IMAGE [1,H,W,3] RGB float32 in [0..1]

from __future__ import annotations
import itertools
import math
import torch
from comfy.utils import common_upscale


# -----------------------------
# Utility helpers
# -----------------------------
def _render_linear_row(tiles: list[torch.Tensor], spacing: int, bg_rgb: torch.Tensor, direction: str) -> torch.Tensor:
    # tiles are [1,C,H,W]
    heights = [t.shape[2] for t in tiles]
    widths  = [t.shape[3] for t in tiles]
    H_out = max(heights)
    W_out = sum(widths) + spacing * max(0, len(tiles) - 1)

    device = tiles[0].device
    dtype  = tiles[0].dtype
    out = torch.ones((1, 3, H_out, W_out), device=device, dtype=dtype)
    out[:, 0].fill_(bg_rgb[0]); out[:, 1].fill_(bg_rgb[1]); out[:, 2].fill_(bg_rgb[2])

    if direction == "right":
        x = 0
        for tile in tiles:
            _, _, h, w = tile.shape
            y_off = max(0, (H_out - h) // 2)
            out[:, :, y_off:y_off + h, x:x + w] = tile
            x += w + spacing
    elif direction == "left":
        x = W_out
        for tile in tiles:
            _, _, h, w = tile.shape
            x -= w
            y_off = max(0, (H_out - h) // 2)
            out[:, :, y_off:y_off + h, x:x + w] = tile
            x -= spacing
    else:
        raise ValueError("_render_linear_row: direction must be 'right' or 'left'")
    return out


def _render_linear_col(tiles: list[torch.Tensor], spacing: int, bg_rgb: torch.Tensor, direction: str) -> torch.Tensor:
    # tiles are [1, C, H, W]
    heights = [t.shape[2] for t in tiles]
    widths  = [t.shape[3] for t in tiles]
    H_out = sum(heights) + spacing * max(0, len(tiles) - 1)
    W_out = max(widths)

    device = tiles[0].device
    dtype  = tiles[0].dtype
    out = torch.ones((1, 3, H_out, W_out), device=device, dtype=dtype)
    out[:, 0].fill_(bg_rgb[0]); out[:, 1].fill_(bg_rgb[1]); out[:, 2].fill_(bg_rgb[2])

    if direction == "down":
        y = 0
        for tile in tiles:
            _, _, h, w = tile.shape
            x_off = max(0, (W_out - w) // 2)
            out[:, :, y:y + h, x_off:x_off + w] = tile
            y += h + spacing
    elif direction == "up":
        y = H_out
        for tile in tiles:
            _, _, h, w = tile.shape
            y -= h
            x_off = max(0, (W_out - w) // 2)
            out[:, :, y:y + h, x_off:x_off + w] = tile
            y -= spacing
    else:
        raise ValueError("_render_linear_col: direction must be 'down' or 'up'")
    return out
    
def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]

def _drop_alpha(x_bhwc: torch.Tensor) -> torch.Tensor:
    return x_bhwc[..., :3] if x_bhwc.shape[-1] == 4 else x_bhwc

def _parse_rgb(s: str, device, dtype) -> torch.Tensor:
    if not isinstance(s, str):
        raise ValueError("pvl_ImageStitch: invalid spacing_color (not a string)")
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != 3:
        raise ValueError("pvl_ImageStitch: invalid spacing_color, expected 'R,G,B'")
    try:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        raise ValueError("pvl_ImageStitch: invalid spacing_color numbers")
    for v in (r, g, b):
        if not (0 <= v <= 255):
            raise ValueError("pvl_ImageStitch: spacing_color must be 0..255")
    return torch.tensor([r/255.0, g/255.0, b/255.0], device=device, dtype=dtype)

def _interp_mode(mode: str):
    m = (mode or "").lower().strip()
    if m in ("nearest", "nearest-exact"): return "nearest"
    if m in ("bilinear",):  return "bilinear"
    if m in ("bicubic",):   return "bicubic"
    if m in ("area",):      return "area"
    if m in ("lanczos","lancoz","lanczos3","lanczos_3"): return "lanczos"
    raise ValueError(f"Unsupported resample mode: {mode}")

def _resize_ar_nchw(img: torch.Tensor, target_h: int | None = None, target_w: int | None = None, resample: str = "bicubic") -> torch.Tensor:
    """AR-preserving resize to target_h or target_w (one must be provided). img: [1,C,H,W]."""
    assert img.ndim == 4 and img.shape[0] == 1, "_resize_ar_nchw expects [1,C,H,W]"
    _, _, h, w = img.shape
    if (target_h is None) == (target_w is None):
        raise ValueError("Provide exactly one of target_h or target_w")
    if target_h is not None:
        scale = target_h / float(h)
        new_h = int(round(target_h)); new_w = max(1, int(round(w * scale)))
    else:
        scale = target_w / float(w)
        new_w = int(round(target_w)); new_h = max(1, int(round(h * scale)))
    mode = _interp_mode(resample)
    if mode == "lanczos" and img.device.type == "cuda":
        raise Exception("Lanczos is not supported on the GPU")
    return common_upscale(img, new_w, new_h, "nearest" if mode=="nearest" else mode, crop="disabled")

def _resize_fit_square_nchw(img: torch.Tensor, S: int, mode: str, resample: str) -> torch.Tensor:
    """Fit [1,C,H,W] into S×S square. mode: 'downscale' (never up) or 'upscale' (never down)."""
    assert img.ndim == 4 and img.shape[0] == 1
    _, _, h, w = img.shape
    if S <= 0:
        return img
    scale = min(S/float(w), S/float(h))
    scale = min(scale, 1.0) if mode == "downscale" else max(scale, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if _interp_mode(resample) == "lanczos" and img.device.type == "cuda":
        raise Exception("Lanczos is not supported on the GPU")
    return common_upscale(img, new_w, new_h,
                          "nearest" if _interp_mode(resample)=="nearest" else _interp_mode(resample),
                          crop="disabled")


# -----------------------------
# GRID packing (legacy)
# -----------------------------

def _group_by_breaks(seq: list[int], mask: int) -> list[list[int]]:
    groups = []; cur = [seq[0]]
    for i in range(len(seq) - 1):
        if (mask >> i) & 1:
            groups.append(cur); cur = [seq[i+1]]
        else:
            cur.append(seq[i+1])
    groups.append(cur); return groups

def _simulate_rows(sizes: list[tuple[int,int]], groups: list[list[int]], spacing: int, match: bool, match_type: str) -> tuple[int,int,list[tuple[int,int]], list[list[tuple[int,int]]]]:
    row_dims: list[tuple[int,int]] = []; row_tiles_dims: list[list[tuple[int,int]]] = []
    for g in groups:
        tiles = [sizes[i] for i in g]
        if match and tiles:
            heights = [h for (_,h) in tiles]
            tgt_h = min(heights) if match_type == "downscale" else max(heights)
            tiles = [(w if h==tgt_h else max(1, int(round(w * (tgt_h/float(h))))), tgt_h) for (w,h) in tiles]
        row_w = sum(w for (w,_) in tiles) + spacing * max(0, len(tiles)-1)
        row_h = max(h for (_,h) in tiles)
        row_dims.append((row_w, row_h)); row_tiles_dims.append(tiles)
    canvas_w = max(w for (w,_) in row_dims)
    canvas_h = sum(h for (_,h) in row_dims) + spacing * max(0, len(row_dims)-1)
    return canvas_w, canvas_h, row_dims, row_tiles_dims

def _simulate_cols(sizes: list[tuple[int,int]], groups: list[list[int]], spacing: int, match: bool, match_type: str) -> tuple[int,int,list[tuple[int,int]], list[list[tuple[int,int]]]]:
    col_dims: list[tuple[int,int]] = []; col_tiles_dims: list[list[tuple[int,int]]] = []
    for g in groups:
        tiles = [sizes[i] for i in g]
        if match and tiles:
            widths = [w for (w,_) in tiles]
            tgt_w = min(widths) if match_type == "downscale" else max(widths)
            tiles = [(tgt_w, h if w==tgt_w else max(1, int(round(h * (tgt_w/float(w)))))) for (w,h) in tiles]
        col_w = max(w for (w,_) in tiles)
        col_h = sum(h for (_,h) in tiles) + spacing * max(0, len(tiles)-1)
        col_dims.append((col_w, col_h)); col_tiles_dims.append(tiles)
    canvas_w = sum(w for (w,_) in col_dims) + spacing * max(0, len(col_dims)-1)
    canvas_h = max(h for (_,h) in col_dims)
    return canvas_w, canvas_h, col_dims, col_tiles_dims

def _choose_best_candidate(
    cands: list[tuple[str, tuple[int,int], list[list[int]], list[list[tuple[int,int]]]]],
    area_tolerance: float = 0.15
) -> tuple[str, tuple[int,int], list[list[int]], list[list[tuple[int,int]]]]:
    # 1) min area
    areas = [W*H for _, (W,H), _, _ in cands]
    A_min = min(areas)
    thresh = int(math.ceil(A_min * (1.0 + max(0.0, area_tolerance))))
    # 2) near-minimal set
    kept = [c for c in cands if (c[1][0] * c[1][1]) <= thresh]
    # 3) prefer square, then smaller W/H, then fewer groups, rows first
    def keyfn(c):
        kind, (W,H), groups, _ = c
        return (abs(W-H), W, H, len(groups), 0 if kind=="rows" else 1)
    return min(kept, key=keyfn)

def _pack_grid(sizes: list[tuple[int,int]], spacing: int, allow_match: bool, match_type: str, area_tolerance: float = 0.15):
    N = len(sizes); idx = list(range(N)); cands = []
    for perm in itertools.permutations(idx, N):
        for mask in range(1 << (N-1)):
            groups = _group_by_breaks(list(perm), mask)
            W,H, _, row_tiles = _simulate_rows(sizes, groups, spacing, allow_match, match_type)
            cands.append(("rows", (W,H), groups, row_tiles))
            W,H, _, col_tiles = _simulate_cols(sizes, groups, spacing, allow_match, match_type)
            cands.append(("cols", (W,H), groups, col_tiles))
    return _choose_best_candidate(cands, area_tolerance)


# -----------------------------
# GUILLotine / MaxRects packing + compaction
# -----------------------------

class _Rect:
    __slots__ = ("x","y","w","h")
    def __init__(self, x:int, y:int, w:int, h:int): self.x=x; self.y=y; self.w=w; self.h=h
    def right(self):  return self.x + self.w
    def bottom(self): return self.y + self.h
    def area(self):   return self.w * self.h

def _rect_overlap(a:_Rect, b:_Rect) -> bool:
    return not (a.x >= b.x + b.w or a.x + a.w <= b.x or a.y >= b.y + b.h or a.y + a.h <= b.y)

def _split_free_rect(f:_Rect, used:_Rect) -> list[_Rect]:
    """Split free rect f by used rect; return list of non-overlapping remainders."""
    out = []
    # Above
    if used.y > f.y and used.y < f.y + f.h:
        out.append(_Rect(f.x, f.y, f.w, used.y - f.y))
    # Below
    if used.y + used.h < f.y + f.h:
        out.append(_Rect(f.x, used.y + used.h, f.w, (f.y + f.h) - (used.y + used.h)))
    # Left
    if used.x > f.x and used.x < f.x + f.w:
        out.append(_Rect(f.x, f.y, used.x - f.x, f.h))
    # Right
    if used.x + used.w < f.x + f.w:
        out.append(_Rect(used.x + used.w, f.y, (f.x + f.w) - (used.x + used.w), f.h))
    return [r for r in out if r.w > 0 and r.h > 0]

def _prune_free_list(free:list[_Rect]) -> list[_Rect]:
    """Remove contained rectangles."""
    pruned = []
    for i, r in enumerate(free):
        contained = False
        for j, s in enumerate(free):
            if i != j and r.x >= s.x and r.y >= s.y and r.right() <= s.right() and r.bottom() <= s.bottom():
                contained = True; break
        if not contained: pruned.append(r)
    return pruned

def _maxrects_pack_fixed_width(sizes:list[tuple[int,int]], gap:int, width:int):
    """
    MaxRects/BSSF packing into fixed width. Returns (ok, placements, W, H)
    placements = list[(idx, x, y, w_eff, h_eff)] where w_eff/h_eff include the gap on right/bottom.
    We expand each tile to (w+gap, h+gap), then subtract a final gap from W/H so there's no outer border gap.
    """
    eff = [(w + (gap if gap>0 else 0), h + (gap if gap>0 else 0)) for (w,h) in sizes]
    total_h_lim = sum(h for (_,h) in eff)  # generous height limit
    free = [_Rect(0,0,width,total_h_lim)]
    placements = []

    for idx, (w,h) in enumerate(eff):
        best_rect = None
        best_key  = None
        # Best Short Side Fit + tie on long side, then y, then x
        for fr in free:
            if w <= fr.w and h <= fr.h:
                ssf = min(fr.w - w, fr.h - h)
                lsf = max(fr.w - w, fr.h - h)
                key = (ssf, lsf, fr.y, fr.x)
                if best_key is None or key < best_key:
                    best_key = key; best_rect = fr
        if best_rect is None:
            return False, [], width, 0  # doesn't fit this width

        used = _Rect(best_rect.x, best_rect.y, w, h)
        # split all overlapping free rects
        new_free = []
        for fr in free:
            if _rect_overlap(fr, used):
                new_free.extend(_split_free_rect(fr, used))
            else:
                new_free.append(fr)
        free = _prune_free_list(new_free)
        placements.append((idx, used.x, used.y, w, h))

    # compute tight bbox; remove outer gap on right/bottom
    W = max((x + w for (_,x,_,w,_) in placements), default=0)
    H = max((y + h for (_,_,y,_,h) in placements), default=0)
    if gap > 0:
        W = max(0, W - gap)
        H = max(0, H - gap)
    # Return tight bbox (crop unused right/bottom space)
    return True, placements, W, H

# ---------- Post-packing compaction (bidirectional) ----------

def _tight_bbox_from_eff(placements):
    if not placements:
        return 0, 0
    W = max(x + w for (_, x, _, w, _) in placements)
    H = max(y + h for (_, _, y, _, h) in placements)
    return W, H

def _vertical_candidates(placements, i_idx, h_eff):
    ys = {0}
    for k, (_, _xk, yk, _wk, hk) in enumerate(placements):
        if k == i_idx:
            continue
        ys.add(yk + hk)
    return sorted(ys)

def _horizontal_candidates(placements, i_idx, w_eff):
    xs = {0}
    for k, (_idx, xk, _yk, wk, _hk) in enumerate(placements):
        if k == i_idx:
            continue
        xs.add(xk + wk)
    return sorted(xs)

def _min_left_x_at_y(placements, i_idx, y, w_eff, h_eff):
    left_bound = 0
    for k, (_idx, xk, yk, wk, hk) in enumerate(placements):
        if k == i_idx:
            continue
        if not (y + h_eff <= yk or y >= yk + hk):  # vertical overlap
            left_bound = max(left_bound, xk + wk)
    return left_bound

def _min_top_y_at_x(placements, i_idx, x, w_eff, h_eff):
    y_min = 0
    for k, (_idx, xk, yk, wk, hk) in enumerate(placements):
        if k == i_idx:
            continue
        # horizontal overlap?
        if not (x + w_eff <= xk or x >= xk + wk):
            y_min = max(y_min, yk + hk)  # must sit above this one
    return y_min

def _improve_by_nudging(placements, max_iters=10):
    """
    Bidirectional local search on *effective* rectangles (gap included):
    A) Vertical anchors -> slide left (reduce width).
    B) Horizontal anchors -> slide up (reduce height).
    Accept moves that improve (area, width, height). Repeat until stable or max_iters.
    """
    if not placements:
        return placements

    for _ in range(max_iters):
        improved = False
        W_cur, H_cur = _tight_bbox_from_eff(placements)
        A_cur = W_cur * H_cur

        # Pass A: down anchors then left slide
        for i in range(len(placements)):
            idx_i, xi, yi, wi, hi = placements[i]
            best = (A_cur, W_cur, H_cur, xi, yi)
            for y_cand in _vertical_candidates(placements, i, hi):
                x_cand = _min_left_x_at_y(placements, i, y_cand, wi, hi)
                old = placements[i]
                placements[i] = (idx_i, x_cand, y_cand, wi, hi)
                W_try, H_try = _tight_bbox_from_eff(placements)
                A_try = W_try * H_try
                if (A_try < best[0]) or (A_try == best[0] and (W_try < best[1] or (W_try == best[1] and H_try < best[2]))):
                    best = (A_try, W_try, H_try, x_cand, y_cand)
                placements[i] = old
            if (best[0], best[1], best[2]) < (A_cur, W_cur, H_cur):
                placements[i] = (idx_i, best[3], best[4], wi, hi)
                W_cur, H_cur = _tight_bbox_from_eff(placements)
                A_cur = W_cur * H_cur
                improved = True

        # Pass B: right anchors then up slide
        for i in range(len(placements)):
            idx_i, xi, yi, wi, hi = placements[i]
            best = (A_cur, W_cur, H_cur, xi, yi)
            for x_cand in _horizontal_candidates(placements, i, wi):
                y_cand = _min_top_y_at_x(placements, i, x_cand, wi, hi)
                old = placements[i]
                placements[i] = (idx_i, x_cand, y_cand, wi, hi)
                W_try, H_try = _tight_bbox_from_eff(placements)
                A_try = W_try * H_try
                if (A_try < best[0]) or (A_try == best[0] and (H_try < best[2] or (H_try == best[2] and W_try < best[1]))):
                    best = (A_try, W_try, H_try, x_cand, y_cand)
                placements[i] = old
            if (best[0], best[1], best[2]) < (A_cur, W_cur, H_cur):
                placements[i] = (idx_i, best[3], best[4], wi, hi)
                W_cur, H_cur = _tight_bbox_from_eff(placements)
                A_cur = W_cur * H_cur
                improved = True

        if not improved:
            break

    return placements

def _pack_guillotine(sizes:list[tuple[int,int]], gap:int, area_tolerance: float = 0.15):
    """
    Try candidate widths around sqrt(total_area) and heuristics.
    Explore all permutations; collect candidates, run local compaction, then choose among those
    within (1+area_tolerance)*min_area by squareness (then W, H, then area).
    Returns: ((W,H), placements) with placements [(orig_idx, x, y, w, h)] using *real* sizes.
    """
    A = sum(w*h for (w,h) in sizes)
    max_w = max(w for (w,_) in sizes)
    sum_w = sum(w for (w,_) in sizes) + gap * max(0, len(sizes)-1)
    S = int(math.ceil(math.sqrt(A)))
    cand_widths = sorted(set([max_w, S, S+1, S+2, max(max_w, S+3), sum_w]))

    candidates = []  # (W,H, mapped_placements)
    N = len(sizes)
    for perm in itertools.permutations(range(N), N):
        perm_sizes = [sizes[i] for i in perm]
        for W in cand_widths:
            ok, placements, _W_eff, _H_eff = _maxrects_pack_fixed_width(perm_sizes, gap, W)
            if not ok: continue

            # Post-pack compaction (works on effective placements)
            placements = _improve_by_nudging(placements)

            # Tight bbox after compaction
            W_eff, H_eff = _tight_bbox_from_eff(placements)

            # Map back to original indices, stripping gap from sizes (positions stay the same)
            mapped = []
            for (local_idx, x, y, w_eff, h_eff) in placements:
                orig_idx = perm[local_idx]
                w_real, h_real = sizes[orig_idx]
                mapped.append((orig_idx, x, y, w_real, h_real))
            candidates.append((W_eff, H_eff, mapped))

    if not candidates:
        # Fallback: vertical strip
        W = max(w for (w,_) in sizes)
        H = sum(h for (_,h) in sizes) + gap * (len(sizes)-1)
        y = 0
        mapped = []
        for i, (w, h) in enumerate(sizes):
            mapped.append((i, 0, y, w, h)); y += h + gap
        return (W, H), mapped

    # 1) minimal area
    areas = [W*H for (W,H,_) in candidates]
    A_min = min(areas)
    thresh = int(math.ceil(A_min * (1.0 + max(0.0, area_tolerance))))
    near = [(W,H,m) for (W,H,m) in candidates if W*H <= thresh]

    # 2) prefer squarer, then smaller W/H, then final tie by area
    near.sort(key=lambda t: (abs(t[0]-t[1]), t[0], t[1], t[0]*t[1]))
    W_best, H_best, placements = near[0]
    return (W_best, H_best), placements


# -----------------------------
# Rendering helpers
# -----------------------------

def _paste_rows(tiles: list[torch.Tensor], groups: list[list[int]], tiles_dims: list[list[tuple[int,int]]], canvas_wh: tuple[int,int], spacing: int, bg_rgb: torch.Tensor, resample: str) -> torch.Tensor:
    device = tiles[0].device; dtype = tiles[0].dtype
    W_out, H_out = canvas_wh
    out = torch.ones((1,3,H_out,W_out), device=device, dtype=dtype)
    out[:,0].fill_(bg_rgb[0]); out[:,1].fill_(bg_rgb[1]); out[:,2].fill_(bg_rgb[2])
    y = 0; t_index = 0
    for g_idx, g in enumerate(groups):
        row_tiles = tiles_dims[g_idx]
        row_h = max(h for (_,h) in row_tiles)
        row_w = sum(w for (w,_) in row_tiles) + spacing * max(0, len(row_tiles)-1)
        x = max(0, (W_out - row_w)//2)
        for i_in_row in range(len(g)):
            w_target, h_target = row_tiles[i_in_row]
            tile = tiles[t_index]
            _, _, h, w = tile.shape
            if (h != h_target) or (w != w_target):
                tile = _resize_ar_nchw(tile, target_h=h_target, resample=resample) if h != h_target else _resize_ar_nchw(tile, target_w=w_target, resample=resample)
                _, _, h, w = tile.shape
            y_off = y + max(0, (row_h - h)//2)
            out[:, :, y_off:y_off+h, x:x+w] = tile
            x += w + spacing; t_index += 1
        y += row_h + spacing
    return out

def _paste_cols(tiles: list[torch.Tensor], groups: list[list[int]], tiles_dims: list[list[tuple[int,int]]], canvas_wh: tuple[int,int], spacing: int, bg_rgb: torch.Tensor, resample: str) -> torch.Tensor:
    device = tiles[0].device; dtype = tiles[0].dtype
    W_out, H_out = canvas_wh
    out = torch.ones((1,3,H_out,W_out), device=device, dtype=dtype)
    out[:,0].fill_(bg_rgb[0]); out[:,1].fill_(bg_rgb[1]); out[:,2].fill_(bg_rgb[2])
    x = 0; t_index = 0
    for g_idx, g in enumerate(groups):
        col_tiles = tiles_dims[g_idx]
        col_w = max(w for (w,_) in col_tiles)
        col_h = sum(h for (_,h) in col_tiles) + spacing * max(0, len(col_tiles)-1)
        y = max(0, (H_out - col_h)//2)
        for i_in_col in range(len(g)):
            w_target, h_target = col_tiles[i_in_col]
            tile = tiles[t_index]
            _, _, h, w = tile.shape
            if (h != h_target) or (w != w_target):
                tile = _resize_ar_nchw(tile, target_w=w_target, resample=resample) if w != w_target else _resize_ar_nchw(tile, target_h=h_target, resample=resample)
                _, _, h, w = tile.shape
            x_off = x + max(0, (col_w - w)//2)
            out[:, :, y:y+h, x_off:x_off+w] = tile
            y += h + spacing; t_index += 1
        x += col_w + spacing
    return out

def _paste_absolute(tiles_by_index: dict[int, torch.Tensor], placements: list[tuple[int,int,int,int,int]], canvas_wh: tuple[int,int], bg_rgb: torch.Tensor) -> torch.Tensor:
    """placements: list of (orig_idx, x, y, w, h) with *real* sizes."""
    tile0 = next(iter(tiles_by_index.values()))
    device = tile0.device; dtype = tile0.dtype
    W_out, H_out = canvas_wh
    out = torch.ones((1,3,H_out,W_out), device=device, dtype=dtype)
    out[:,0].fill_(bg_rgb[0]); out[:,1].fill_(bg_rgb[1]); out[:,2].fill_(bg_rgb[2])

    for (orig_idx, x, y, w_t, h_t) in placements:
        tile = tiles_by_index[orig_idx]
        _, _, h, w = tile.shape
        if (h != h_t) or (w != w_t):
            if h != h_t:
                tile = _resize_ar_nchw(tile, target_h=h_t, resample="bicubic")
            else:
                tile = _resize_ar_nchw(tile, target_w=w_t, resample="bicubic")
            _, _, h, w = tile.shape
        out[:, :, y:y+h, x:x+w] = tile
    return out


# -----------------------------
# ComfyUI Node
# -----------------------------
class PVL_ImageStitch:
    """PVL – Image Stitch: stitch up to 8 images either linearly or via pack-to-square."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "arrangement": (["right","left","up","down"], {"default": "right"}),
                "pack_to_square": ("BOOLEAN", {"default": False}),
                "pack_mode": (["grid","guillotine"], {"default": "grid"}),
                "match_image_size": ("BOOLEAN", {"default": False}),
                "match_type": (["downscale","upscale"], {"default": "downscale"}),
                "resample": (["lanczos","bicubic","bilinear","area","nearest"], {"default": "bicubic"}),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "spacing_color": ("STRING", {"default": "0,0,0"}),
            },
            "optional": {
                "square_tolerance_pct": ("INT", {"default": 15, "min": 0, "max": 50,
                                                 "tooltip": "Allow up to this % extra area to pick a squarer layout."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "image/compose"

    def _validate_and_collect(self, imgs: list[torch.Tensor | None]) -> list[torch.Tensor]:
        out = []
        for t in imgs:
            if t is None: continue
            if not isinstance(t, torch.Tensor): raise TypeError("pvl_ImageStitch: expected IMAGE tensors")
            if t.ndim != 4: raise ValueError("pvl_ImageStitch: expected [B,H,W,C] tensors")
            b, _, _, c = t.shape
            if b != 1: raise ValueError(f"pvl_ImageStitch: batch > 1 is not supported (got B={b}).")
            if c not in (3,4): raise ValueError(f"pvl_ImageStitch: expected channels C=3 or 4, got C={c}.")
            out.append(_drop_alpha(t))
        return out

    def _maybe_match_linear(self, tiles_nchw: list[torch.Tensor], axis: str, match_type: str, resample: str) -> list[torch.Tensor]:
        if not tiles_nchw: return tiles_nchw
        if axis == "row":
            heights = [t.shape[2] for t in tiles_nchw]
            tgt_h = min(heights) if match_type=="downscale" else max(heights)
            return [t if t.shape[2]==tgt_h else _resize_ar_nchw(t, target_h=tgt_h, resample=resample) for t in tiles_nchw]
        elif axis == "col":
            widths = [t.shape[3] for t in tiles_nchw]
            tgt_w = min(widths) if match_type=="downscale" else max(widths)
            return [t if t.shape[3]==tgt_w else _resize_ar_nchw(t, target_w=tgt_w, resample=resample) for t in tiles_nchw]
        else:
            raise ValueError("axis must be 'row' or 'col'")

    def stitch(self,
               image_1: torch.Tensor,
               arrangement: str,
               pack_to_square: bool,
               pack_mode: str,
               match_image_size: bool,
               match_type: str,
               resample: str,
               spacing_width: int,
               spacing_color: str,
               square_tolerance_pct: int = 15,
               image_2: torch.Tensor | None = None,
               image_3: torch.Tensor | None = None,
               image_4: torch.Tensor | None = None,
               image_5: torch.Tensor | None = None,
               image_6: torch.Tensor | None = None,
               image_7: torch.Tensor | None = None,
               image_8: torch.Tensor | None = None):

        # Collect & validate
        img_list_bhwc = self._validate_and_collect([image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8])
        if len(img_list_bhwc) == 0:
            raise ValueError("pvl_ImageStitch: no valid images provided.")
        if len(img_list_bhwc) == 1:
            return (img_list_bhwc[0],)

        device = img_list_bhwc[0].device
        dtype  = img_list_bhwc[0].dtype
        bg_rgb = _parse_rgb(spacing_color, device=device, dtype=dtype)

        # NCHW
        tiles_nchw = [_to_nchw(t) for t in img_list_bhwc]

        # Linear path
        if not pack_to_square:
            if match_image_size:
                if arrangement in ("right","left"):
                    tiles_nchw = self._maybe_match_linear(tiles_nchw, "row", match_type, resample)
                elif arrangement in ("up","down"):
                    tiles_nchw = self._maybe_match_linear(tiles_nchw, "col", match_type, resample)
                else:
                    raise ValueError("Invalid arrangement")
            if arrangement in ("right","left"):
                out_nchw = _render_linear_row(tiles_nchw, spacing_width, bg_rgb, arrangement)
            else:
                out_nchw = _render_linear_col(tiles_nchw, spacing_width, bg_rgb, arrangement)
            return (_to_bhwc(out_nchw).contiguous(),)

        # Pack path (pre-normalize when requested)
        if match_image_size:
            widths  = [int(t.shape[3]) for t in tiles_nchw]
            heights = [int(t.shape[2]) for t in tiles_nchw]
            if match_type == "upscale":
                S = int(round((max(widths) + max(heights)) / 2))
            else:
                S = int(round((min(widths) + min(heights)) / 2))
            tiles_nchw = [_resize_fit_square_nchw(t, S, match_type, resample) for t in tiles_nchw]

        sizes = [(int(t.shape[3]), int(t.shape[2])) for t in tiles_nchw]  # (w,h)
        tol = max(0.0, float(square_tolerance_pct) / 100.0)

        if pack_mode == "grid":
            # After pre-normalization, do NOT perform per-row/column matching during grid pack
            kind, (W_out, H_out), groups, tiles_dims = _pack_grid(
                sizes, spacing_width, allow_match=False, match_type=match_type, area_tolerance=tol
            )
            perm_indices = [i for g in groups for i in g]
            tiles_perm = [tiles_nchw[i] for i in perm_indices]
            out_nchw = _paste_rows(tiles_perm, groups, tiles_dims, (W_out, H_out), spacing_width, bg_rgb, resample) if kind=="rows" \
                       else _paste_cols(tiles_perm, groups, tiles_dims, (W_out, H_out), spacing_width, bg_rgb, resample)
            return (_to_bhwc(out_nchw).contiguous(),)

        # pack_mode == "guillotine"
        (W_out, H_out), placements = _pack_guillotine(sizes, spacing_width, area_tolerance=tol)
        tiles_by_idx = {i: tiles_nchw[i] for i in range(len(tiles_nchw))}
        out_nchw = _paste_absolute(tiles_by_idx, placements, (W_out, H_out), bg_rgb)
        return (_to_bhwc(out_nchw).contiguous(),)