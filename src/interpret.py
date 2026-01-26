#!/usr/bin/env python3
"""
    Interpretation / Testing utilities

     Works with:
    - MProtoNet / ProtoPNet family (p_mode < 4)
    - UM-ProtoShare (p_mode = 4)
    - UM-ProtoShare-MS (p_mode ≥ 5)

    Written by: Ali Golbaf (ali.golbaf71@gmail.com)
"""
################################################################################
################################################################################
""" Libraries """
import warnings
from typing import Tuple, List, Optional, Union

import captum.attr as ctattr
import numpy as np
import torch
import torch.nn as nn

################################################################################
################################################################################
""" Attributes mapping """
# Keys used elsewhere to pick the method
attr_methods: dict = {'M': 'MProtoNet',              # Proto-like prototype attribution (unified here)
                      'D': 'Deconvolution',
                      'G': 'GradCAM',
                      'U': 'Guided GradCAM',
                      'O': 'Occlusion'}

################################################################################
""" Functions """

def pos_mask_from_classifier(net: nn.Module, target: torch.Tensor) -> torch.Tensor:
    """
    Get binary mask of 'positive' prototypes for each sample's class from last_layer weights.
    last_layer: [C, K_total], target: [B] -> mask: [B, K_total]
    """
    W = net.last_layer.weight                                   # [C, K]
    pos = (W > 0).float()                                       # [C, K]
    return pos[target]                                          # [B, K]

def combine_scales_to_image(act_maps: List[torch.Tensor],
                             data: torch.Tensor) -> torch.Tensor:
    """
    act_maps: list of [B,1,Hs,Ws,(Ds)] per-scale maps (already class-aggregated)
    data    : [B, C, H, W, D]
    Return  : [B, C, H, W, D] with same map copied to all channels
    """
    ups = []
    for m in act_maps:
        if m is None:
            continue
        if m.ndim == 5:
            u = ctattr.LayerAttribution.interpolate(m, data.shape[2:],
                                                    interpolate_mode='trilinear')
        else:
            u = ctattr.LayerAttribution.interpolate(m, data.shape[2:4],
                                                    interpolate_mode='bilinear')
            u = u.reshape((data.shape[0], 1, 1) + data.shape[2:4]).permute(0, 1, 3, 4, 2)
        ups.append(u)
    if not ups:
        # fallback: zeros
        return torch.zeros_like(data)
    # average across scales
    U = torch.stack([u for u in ups], dim=0).mean(0)    # [B,1,H,W,D]
    return U.expand_as(data).clone()

################################################################################
""" Upsample to input space (legacy helper) """
def upsample(attr, data):
    # Keep the original behaviour
    if attr.ndim == 5:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:],
                                                   interpolate_mode='trilinear')
    else:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:4],
                                                   interpolate_mode='bilinear')
        attr = attr.reshape((data.shape[0],
                             data.shape[4], 1) + data.shape[2:4]).permute(0, 2, 3, 4, 1)
    if attr.shape[0] // data.shape[0] == data.shape[1] // attr.shape[1]:
        return attr.reshape_as(data)
    else:
        return attr.expand_as(data).clone()

################################################################################
################################################################################
""" Main """
def attribute(net: nn.Module,
              data: torch.Tensor,
              target: torch.Tensor,
              device: Union[str, torch.device],
              method: str,
              show_progress: bool = False) -> torch.Tensor:
                  
    warnings.filterwarnings('ignore', message="Input Tensor \\d+ did not already require gradients,")
    warnings.filterwarnings('ignore', message="Setting backward hooks on ReLU activations.The hook")
    warnings.filterwarnings('ignore', message="Setting forward, backward hooks and attributes on n")
    # -------------------------------------------------------------------------
    # MProtoNet (unified for both families)
    if method == 'MProtoNet':
        if isinstance(net, nn.DataParallel):
            net = net.module

        p_mode = getattr(net, 'p_mode', 0)

        with torch.no_grad():
            # p_mode branches --------------------------------------------------
            if p_mode < 4:
                # Proto/MProto path
                out = net.push_forward(data)
                if p_mode >= 2:
                    # (p_x, distances, p_map)
                    # P_map as the spatial attribution; otherwise distance->similarity
                    p_x, distances, p_map = out
                    if p_map is not None:
                        # class mask using prototype_class_identity
                        proto_filter = net.prototype_class_identity[:, target].mT
                        # class-mean across positive prototypes
                        attr = (p_map * proto_filter[(...,) + (None,) * (p_map.ndim - 2)]).mean(1, keepdim=True)
                        return upsample(attr, data)
                    else:
                        attr = net.distance_to_similarity(distances)
                        proto_filter = net.prototype_class_identity[:, target].mT
                        attr = (attr * proto_filter[(...,) + (None,) * (attr.ndim - 2)]).mean(1, keepdim=True)
                        return upsample(attr, data)
                else:
                    # ProtoPNet
                    fmap, distances = out
                    sim = net.distance_to_similarity(distances)
                    proto_filter = net.prototype_class_identity[:, target].mT
                    attr = (sim * proto_filter[(...,) + (None,) * (sim.ndim - 2)]).mean(1, keepdim=True)
                    return upsample(attr, data)

            else:
                # UM-ProtoShare family (multi-scale)
                out = net.push_forward(data)
                # p_mode ≥ 5 may return 3-tuples (PXs, Ds, PMs), p_mode=4 returns (PXs, Ds)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    Ds = out[1]                                   # (D3, D2, D1)
                    PMs = out[2] if len(out) >= 3 else (None, None, None)
                else:
                    raise ValueError("Unexpected push_forward output for p_mode>=4.")

                # Convert to 'activation' maps per scale
                act_maps = []
                # Class mask from last layer weights (shared pool)
                # Note: last_layer.weight: [C, K_total], mask -> [B, K_total]
                class_mask = pos_mask_from_classifier(net, target)  # [B, K_total]

                # figure out K-slices per scale
                K1 = net.prototype_vectors_1.shape[0]
                K2 = net.prototype_vectors_2.shape[0]
                K3 = net.prototype_vectors_3.shape[0]
                off1 = 0
                off2 = K1
                off3 = K1 + K2

                for D_scale, P_scale, K_slice, off in [
                    (Ds[2], PMs[2] if isinstance(PMs, (tuple, list)) else None, K1, off1),  # scale-1 (fine)
                    (Ds[1], PMs[1] if isinstance(PMs, (tuple, list)) else None, K2, off2),  # scale-2
                    (Ds[0], PMs[0] if isinstance(PMs, (tuple, list)) else None, K3, off3),  # scale-3 (coarse)
                                                        ]:
                    if D_scale is None:
                        act_maps.append(None); continue
                    # Prefer provided p_map (p_mode≥5); else convert distance→similarity
                    if P_scale is not None:
                        # Weighted class mask for this slice
                        mask_slice = class_mask[:, off:off+K_slice]
                        # Aggregate across prototypes
                        m = (P_scale * mask_slice[:, :, None, None, None]).sum(1, keepdim=True)
                        act_maps.append(m)
                    else:
                        S = net.distance_to_similarity(D_scale)
                        mask_slice = class_mask[:, off:off+K_slice]
                        m = (S * mask_slice[:, :, None, None, None]).mean(1, keepdim=True)
                        act_maps.append(m)

                # Upsample + average across scales, then expand to 4 modalities
                return combine_scales_to_image(act_maps, data)

    # -------------------------------------------------------------------------
    # Deconvolution
    elif method == 'Deconvolution':
        deconv = ctattr.Deconvolution(net)
        return deconv.attribute(data, target=target)

    # -------------------------------------------------------------------------
    # GradCAM
    elif method == 'GradCAM':
        conv_name = [n for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))][-1]
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        return upsample(attr, data)

    # -------------------------------------------------------------------------
    # Guided GradCAM
    elif method == 'Guided GradCAM':
        conv_name = [n for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))][-1]
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        guided_bp = ctattr.GuidedBackprop(net)
        return guided_bp.attribute(data, target=target) * upsample(attr, data)

    # -------------------------------------------------------------------------
    # Occlusion
    elif method == 'Occlusion':
        occlusion = ctattr.Occlusion(net)
        sliding_window = (1,) + (11,) * len(data.shape[2:])
        strides = (1,) + (5,) * len(data.shape[2:])
        return occlusion.attribute(data, sliding_window, strides=strides, target=target,
                                   perturbations_per_eval=1, show_progress=show_progress)

    # -------------------------------------------------------------------------
    # Fallback
    else:
        raise ValueError(f"Unknown method: {method}. Options: {list(attr_methods.values())}")
