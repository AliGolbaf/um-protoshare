#!/usr/bin/env python3
"""
Pushing shared multi-scale prototypes (UM-ProtoShare)

Written by: Ali Golbaf (ali.golbaf71@gmail.com)
"""
################################################################################
################################################################################
""" Libraries """
import os
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import re
################################################################################
""" Functions """
# Note: Repository utilities (used for RF back-projection and saving)
from receptive_field import compute_rf_prototype
from utils import find_high_activation_crop, makedir

# Gpu or Cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################
################################################################################
""" Functions """

def pf_unpack_multi(out):
    """
    Accepts:
        out = ((PX3,PX2,PX1), (D3,D2,D1))                    # p_mode=4 (no p-maps)
           or ((PX3,PX2,PX1), (D3,D2,D1), (PM3,PM2,PM1))     # p_mode=5 (with p-maps)
    """
    if not (isinstance(out, (tuple, list)) and len(out) >= 2):
        raise ValueError("push_forward must return ((PX3,PX2,PX1),(D3,D2,D1),[(PM3,PM2,PM1)]) for p_mode>=4.")
    PXs = out[0]
    Ds  = out[1]
    PMs = out[2] if len(out) >= 3 else (None, None, None)

    if not (isinstance(PXs, (tuple, list)) and len(PXs) == 3 and
            isinstance(Ds,  (tuple, list)) and len(Ds)  == 3):
        raise ValueError("push_forward must provide 3 scales: (...3, ...2, ...1).")

    if PMs is not None and PMs != (None, None, None):
        if not (isinstance(PMs, (tuple, list)) and len(PMs) == 3):
            raise ValueError("p-maps must be a 3-tuple (PM3,PM2,PM1) if provided.")
    else:
        PMs = (None, None, None)

    return PXs, Ds, PMs


def is_px_per_proto(t: torch.Tensor) -> bool:
    """
    A: Pooled vectors per prototype
       Shape: [B, K, 128, 1, 1, 1]
    """
    return isinstance(t, torch.Tensor) and t.ndim == 6 and t.shape[2] == 128 and list(t.shape[3:]) == [1, 1, 1]


def is_embed_map(t: torch.Tensor) -> bool:
    """
    B: Per-location embedding map (128-d at each spatial position)
       Shape: [B, 128, H, W, D]  or [B, 128, H, W]
    """
    return isinstance(t, torch.Tensor) and t.ndim in (5, 4) and t.shape[1] == 128


def vec128_from_px_or_embed(px_or_e: torch.Tensor, b_best: int, k: int, h: int, w: int, d: int) -> np.ndarray:
    """
    Input:
        px_or_e : per-proto vectors [B,K,128,1,1,1] OR per-location embeddings [B,128,H,W,(D)]
        b_best  : best image in the batch
        k,h,w,d : best proto index and spatial location

    Output:
        numpy [128,1,1,1]
    """
    if is_px_per_proto(px_or_e):
        v = px_or_e[b_best, k, :, 0, 0, 0]
        v = v.detach().cpu().numpy()
        return v[:, None, None, None]

    if is_embed_map(px_or_e):
        if px_or_e.ndim == 5:
            v = px_or_e[b_best, :, h, w, d]
        else:
            v = px_or_e[b_best, :, h, w]
        v = v.detach().cpu().numpy()
        return v[:, None, None, None]

    raise ValueError(f"Unsupported PX/E tensor shape: {None if not isinstance(px_or_e, torch.Tensor) else tuple(px_or_e.shape)}")


def save_per_modality_overlays(out_dir: str,
                                prefix: str,
                                j_scale: int,
                                scale_idx: int,
                                orig_4c: np.ndarray,      # [4, H, W] slice (T1, T1c, T2, FLAIR)
                                act2d_full: np.ndarray,   # [H, W] activation upsampled to image plane
                                bounds_xyxy: Tuple[int, int, int, int],
                                modality_names: Tuple[str, str, str, str] = ("t1", "t1c", "t2", "t2flair"),
                                subject_id : str = "",
                                transpose_like_push: bool = True,
                                save_proto_grayscale: bool = False):
    """
    Save three PNGs per modality:
        1) overlay : heatmap over the modality slice
        2) act     : heatmap alone (grayscale)
        3) proto   : cropped prototype patch (from modality slice)
        4) img     : the modality slice itself (grayscale, full frame)
    Filenames:
        {prefix}-p{j:03d}-s{scale}-{mod}-[overlay|act|proto].png
    """
    y0, y1, x0, x1 = map(int, bounds_xyxy)

    hm = act2d_full - act2d_full.min()
    hm = hm / max(hm.max(), 1e-6)
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET) / 255.0
    heatmap_gray = (hm * 255.0).astype(np.uint8)

    for c, mname in enumerate(modality_names):
        plane = orig_4c[c]
        plane = plane - plane.min()
        plane = plane / max(plane.max(), 1e-6)
        
        # Build RGB base from grayscale plane, then overlay heatmap (like Push.py)
        base_rgb = np.repeat(plane[..., None], 3, axis=2)
        overlay = (0.5 * heatmap_rgb + 0.5 * base_rgb).clip(0, 1)
        
        # Crop (prototype patch) from the modality plane
        crop = plane[y0:y1, x0:x1]  # 2D

        # ---- Orientation to match Push.py (transpose = 90° rotation) ----
        if transpose_like_push:
            overlay_to_save = overlay.transpose(1, 0, 2)          # (W,H,3)
            act_to_save     = heatmap_gray.T                       # (W,H)
            img_to_save     = np.uint8(plane.T * 255.0)            # grayscale full slice
            if save_proto_grayscale:
                proto_to_save = np.uint8((crop.T * 255.0))         # grayscale crop
            else:
                proto_rgb = np.repeat(crop[..., None], 3, axis=2)
                proto_to_save = np.uint8(proto_rgb.transpose(1, 0, 2) * 255.0)
        else:
            overlay_to_save = overlay                              # (H,W,3)
            act_to_save     = heatmap_gray                         # (H,W)
            img_to_save     = np.uint8(plane * 255.0)              # grayscale full slice
            if save_proto_grayscale:
                proto_to_save = np.uint8(crop * 255.0)             # grayscale crop
            else:
                proto_rgb = np.repeat(crop[..., None], 3, axis=2)
                proto_to_save = np.uint8(proto_rgb * 255.0)

        # ---- Save (cv2 expects BGR for 3-channel) ------------------------
        out_overlay = os.path.join(out_dir, f"{prefix}-p{j_scale:03d}-s{scale_idx}-{subject_id[0]}-{mname}-overlay.png")
        cv2.imwrite(out_overlay, np.uint8(overlay_to_save[:, :, ::-1] * 255))

        # out_act = os.path.join(out_dir, f"{prefix}-p{j_scale:03d}-s{scale_idx}-{mname}-act.png")
        # cv2.imwrite(out_act, act_to_save)

        out_proto = os.path.join(out_dir, f"{prefix}-p{j_scale:03d}-s{scale_idx}-{subject_id[0]}-{mname}-proto.png")
        
        if proto_to_save.ndim == 2:
            cv2.imwrite(out_proto, proto_to_save)                  # grayscale
        else:
            cv2.imwrite(out_proto, proto_to_save[:, :, ::-1])     # BGR

        # NEW: save the modality image itself (grayscale)
        out_img = os.path.join(out_dir, f"{prefix}-p{j_scale:03d}-s{scale_idx}-{subject_id[0]}-{mname}-img.png")
        cv2.imwrite(out_img, img_to_save)

################################################################################
################################################################################
""" Main body: push_prototypes for UM-ProtoShare """

def push_prototypes_u_protoshare_ms(data_loader,
                                    model,
                                    root_dir_for_saving: Optional[str] = None,
                                    epoch_number: Optional[int] = None,
                                    prototype_img_filename_prefix: str = "uPSMS",
                                    log=print):
    """
    Multi-scale push for:
        p_mode = 4  : U-ProtoShare-MS (no p-maps)
        p_mode >= 5 : U-ProtoShare-MS (with p-maps if provided)

    Steps:
        1) For each prototype at each scale, find lowest-distance patch across dataset.
        2) Update prototype vectors with the winning local 128-d features.
        3) Save per-modality overlays/act/proto images for visual checks.

    Returns:
        proto_vecs: dict {3: [K3,128,1,1,1], 2: ..., 1: ...}
        rf_boxes  : dict {scale: [K_l, 5] -> (img_idx_global, y0,y1,x0,x1)}
        bounds    : dict {scale: [K_l, 5] high-activation crop bounds}
    """
    assert getattr(model, "p_mode", 0) >= 4, "This function requires p_mode >= 4."

    model.eval()
    dev = next(model.parameters()).device

    # Number of prototypes per scale
    K3 = int(getattr(model, "num_prototypes_03"))
    K2 = int(getattr(model, "num_prototypes_02"))
    K1 = int(getattr(model, "num_prototypes_01"))

    # Initialisation -----------------------------------------------------------
    best_dist = {3: np.full(K3, np.inf, dtype=np.float64),
                 2: np.full(K2, np.inf, dtype=np.float64),
                 1: np.full(K1, np.inf, dtype=np.float64)}

    proto_vecs = {3: np.zeros((K3, 128, 1, 1, 1), dtype=np.float32),
                  2: np.zeros((K2, 128, 1, 1, 1), dtype=np.float32),
                  1: np.zeros((K1, 128, 1, 1, 1), dtype=np.float32)}

    rf_boxes = {3: np.full((K3, 5), -1, dtype=np.int64),
                2: np.full((K2, 5), -1, dtype=np.int64),
                1: np.full((K1, 5), -1, dtype=np.int64)}

    bounds = {3: np.full((K3, 5), -1, dtype=np.int64),
              2: np.full((K2, 5), -1, dtype=np.int64),
              1: np.full((K1, 5), -1, dtype=np.int64)}

    # Output directory ---------------------------------------------------------
    if root_dir_for_saving is not None:
        out_dir = (root_dir_for_saving if epoch_number is None
                   else os.path.join(root_dir_for_saving, f"epoch-{epoch_number}"))
        makedir(out_dir)
    else:
        out_dir = None

    # Push over dataset --------------------------------------------------------
    batch_size = getattr(data_loader, "batch_size", 1)

    for batch_index, (subject_id, image, label, seg) in enumerate(tqdm(data_loader)):
        start_idx = batch_index * batch_size

        with torch.no_grad():
            image = image.to(dev)
            out = model.push_forward(image)

        # Unify p_mode 4/5 outputs
        PXs, Ds, PMs = pf_unpack_multi(out)

        # Convert to numpy where needed (distance/p-map); keep PXs as tensors (for vector extraction)
        D3 = None if Ds[0] is None else Ds[0].detach().cpu().numpy()
        D2 = None if Ds[1] is None else Ds[1].detach().cpu().numpy()
        D1 = None if Ds[2] is None else Ds[2].detach().cpu().numpy()
        PM3 = None if PMs[0] is None else PMs[0].detach().cpu().numpy()
        PM2 = None if PMs[1] is None else PMs[1].detach().cpu().numpy()
        PM1 = None if PMs[2] is None else PMs[2].detach().cpu().numpy()

        # Iterate scales: coarse -> fine
        for scale, K_l, D_l, PX_l, PM_l in ((3, K3, D3, PXs[0], PM3),
                                            (2, K2, D2, PXs[1], PM2),
                                            (1, K1, D1, PXs[2], PM1)):

            if D_l is None or PX_l is None or K_l == 0:
                continue

            # Fast argmin over B,H,W,(D)
            B, K, *spatial = D_l.shape
            D_flat = D_l.reshape(B, K, -1)                  # [B,K,H*W*D]
            flat_min = D_flat.min(axis=2)                   # [B,K]
            flat_idx = D_flat.argmin(axis=2)                # [B,K]

            # Image geometry
            H_img, W_img, D_img = int(image.shape[2]), int(image.shape[3]), int(image.shape[4])

            for k in range(K_l):
                b_best = int(flat_min[:, k].argmin())
                val = float(flat_min[b_best, k])

                # Only update if this is the best so far
                if val >= best_dist[scale][k]:
                    continue

                # Decode (h,w,d)
                idx = int(flat_idx[b_best, k])
                if len(spatial) == 3:
                    Hs, Ws, Ds_sp = spatial
                    h = idx // (Ws * Ds_sp)
                    w = (idx % (Ws * Ds_sp)) // Ds_sp
                    d = (idx % (Ws * Ds_sp)) % Ds_sp
                elif len(spatial) == 2:
                    Hs, Ws = spatial
                    h = idx // Ws
                    w = idx % Ws
                    d = 0
                else:
                    raise ValueError(f"Unexpected spatial dims in distance map: {spatial}")

                # 1) Best distance + vector
                best_dist[scale][k] = val
                vec128 = vec128_from_px_or_embed(PX_l, b_best=b_best, k=k, h=int(h), w=int(w), d=int(d))
                proto_vecs[scale][k] = vec128

                # 2) RF mapping to input coordinates (y0,y1,x0,x1)
                rf_info = getattr(model, f"proto_layer_rf_info_{scale}")
                rf_box = compute_rf_prototype(H_img, [b_best, int(h), int(w), int(d)], rf_info)

                rf_boxes[scale][k, 0] = rf_box[0] + start_idx
                rf_boxes[scale][k, 1:] = rf_box[1:5]

                # 3) Visualisations (optional)
                if out_dir is not None and prototype_img_filename_prefix is not None:
                    # Slice aligned with feature depth
                    if D_l.shape[-1] > 1:
                        depth_ratio = max(1, D_img // D_l.shape[-1])
                        axial_slice = int(d * depth_ratio + depth_ratio // 2)
                    else:
                        axial_slice = D_img // 2

                    # Base 4-channel plane: [4,H,W]
                    orig_4c = image[b_best, :, :, :, axial_slice].detach().cpu().numpy()

                    # Activation for viz:
                    #   p_mode>=5: use p-map if provided; else inverse distance
                    if PM_l is not None:
                        act3d = PM_l[b_best, k]     # [H_l,W_l,(D_l)]
                        if act3d.ndim == 3 and act3d.shape[-1] > 1:
                            act2d = act3d[:, :, d]
                        elif act3d.ndim == 3:
                            act2d = act3d[:, :, 0]
                        else:
                            act2d = act3d
                    else:
                        dmap = D_l[b_best, k]
                        if dmap.ndim == 3 and dmap.shape[-1] > 1:
                            act2d = dmap[:, :, d].max() - dmap[:, :, d]
                        elif dmap.ndim == 3:
                            act2d = dmap[:, :, 0].max() - dmap[:, :, 0]
                        else:
                            act2d = dmap.max() - dmap

                    # Upsample + crop box
                    up = cv2.resize(act2d, dsize=(W_img, H_img), interpolation=cv2.INTER_CUBIC)
                    y0, y1, x0, x1 = find_high_activation_crop(up)
                    bounds[scale][k] = np.array([rf_boxes[scale][k, 0], y0, y1, x0, x1], dtype=np.int64)

                    # Save overlays/act/proto for each modality
                    save_per_modality_overlays(out_dir,
                                                prototype_img_filename_prefix,
                                                j_scale=k,
                                                scale_idx=scale,
                                                orig_4c=orig_4c,
                                                act2d_full=up,
                                                bounds_xyxy=(y0, y1, x0, x1),
                                                modality_names=("t1", "t1c", "t2", "t2flair"),
                                                subject_id = subject_id)

    # Write-back into model ----------------------------------------------------
    with torch.no_grad():
        if K3 > 0:
            getattr(model, "prototype_vectors_3").data.copy_(torch.from_numpy(proto_vecs[3]).to(dev))
        if K2 > 0:
            getattr(model, "prototype_vectors_2").data.copy_(torch.from_numpy(proto_vecs[2]).to(dev))
        if K1 > 0:
            getattr(model, "prototype_vectors_1").data.copy_(torch.from_numpy(proto_vecs[1]).to(dev))

    print(f"Push (U-ProtoShare, p_mode={getattr(model,'p_mode')}) finished.")
    print(f"  K3/K2/K1 = {K3}/{K2}/{K1}")
    for s in (3, 2, 1):
        if best_dist[s].size > 0 and np.isfinite(best_dist[s]).any():
            print(f"  scale-{s}: mean(best_dist)={np.mean(best_dist[s][np.isfinite(best_dist[s])]):.4f}  "
                  f"min={np.min(best_dist[s]):.4f}")

    return proto_vecs, rf_boxes, bounds
