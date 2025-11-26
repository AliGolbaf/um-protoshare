#!/usr/bin/env python3
"""
The code for:
Prototype-based 3D CNNs (Backbones + UM_ProtoShare)

    Written by: Ali Golbaf (ali.golbaf71@gmail.com)
"""
################################################################################
################################################################################
""" Libraries """
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models

################################################################################
################################################################################
""" Functions """
# Receptive-field utilities used to map feature coordinates back to input space
from receptive_field import compute_proto_layer_rf_info_v2

################################################################################
""" A: Build ResNet models """
# TODO: Please see the torch website to underestant the reson behind this

# ResNet
# We use "Sequential OrderedDict" to build the model from the first layer to the end of layer 4 of resnet models.
# After layer 4 we have an average pooling and also a fc layer which we ignore.
def build_resnet_models(net):
    """
    Wrap a torchvision ResNet into a plain nn.Sequential up to layer4.
    Args:
        net: torch_models.resnetXX()
        Returns: nn.Sequential with conv1..layer4 (pool+fc removed).
    """
    return nn.Sequential(OrderedDict([('conv1', net.conv1),
                                      ('bn1', net.bn1),
                                      ('relu', net.relu),
                                      ('maxpool', net.maxpool),
                                      ('layer1', net.layer1),
                                      ('layer2', net.layer2),
                                      ('layer3', net.layer3),
                                      ('layer4', net.layer4)]))

################################################################################
""" B: Predefined ResNet models """
# Note: "_ri" variants start from random init (no ImageNet weights)
def build_predefined_resnet_models(features):

    if features == 'resnet18':
        return build_resnet_models(torch_models.resnet18(weights='IMAGENET1K_V1'))
    elif features == 'resnet18_ri':
        return build_resnet_models(torch_models.resnet18())
    elif features == 'resnet34':
        return build_resnet_models(torch_models.resnet34(weights='IMAGENET1K_V1'))
    elif features == 'resnet34_ri':
        return build_resnet_models(torch_models.resnet34())
    elif features == 'resnet50':
        return build_resnet_models(torch_models.resnet50(weights='IMAGENET1K_V2'))
    elif features == 'resnet50_ri':
        return build_resnet_models(torch_models.resnet50())
    elif features == 'resnet101':
        return build_resnet_models(torch_models.resnet101(weights='IMAGENET1K_V2'))
    elif features == 'resnet101_ri':
        return build_resnet_models(torch_models.resnet101())
    elif features == 'resnet152':
        return build_resnet_models(torch_models.resnet152(weights='IMAGENET1K_V2'))
    elif features == 'resnet152_ri':
        return build_resnet_models(torch_models.resnet152())

################################################################################
""" C: Replace modules in ResNet models"""
# TODO: Please check this links for more information 
# https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7
# https://stackoverflow.com/q/36901
# https://docs.python.org/3/tutorial/controlflow.html#more-on-defining-functions
""" For help:
    net = torch_models.resnet152()
    replace_module(net, [nn.Conv2d, {'in_channels': 3, 'out_channels': 64}], nn.Conv3d, 3, 64, 7, stride=2, padding=3, bias=False)
    1. net : net
    2. type_old : [nn.Conv2d, {'in_channels': 3, 'out_channels': 64}]
    3. type_new : nn.Conv3d
    4. *args : 3, 64, 7: which are used for type_new (conv3d: in_ch = 3, out_ch = 64, kernel_size = 7)
    5. **kwargs : stride=2, padding=3, bias=False: used for type_new
    >>>> type_new(*args, **kwargs)
"""

# ResNet
def replace_module(net, type_old, type_new, *args, **kwargs):
    
    # If type_old is a list
    if isinstance(type_old, list):
        
        # Check the size if 2
        assert len(type_old) == 2
        
        # Type and attribute
        type_old_, attrs = type_old
        
        # Check if the attribute is a dictionary
        assert isinstance(attrs, dict)
        # -------------------------------------------------------------------------------
        # Till the next sign we search for exact match of type_old to be replaced
        
        # Search for type_old
        for n, m in net.named_children():

            # If type_old matches within the network module
            if isinstance(m, type_old_):
                matched = True
                
                # Then search for attribute items to see if their values are not the same: to just replace the exact type_old
                for attr, value in attrs.items():

                    if getattr(m, attr) != value:
                        matched = False
                        break
       # ---------------------------------------------------------------------------------
                # If exact match: then replace           
                if matched:
                    
                    setattr(net, n, type_new(*args, **kwargs))
            
            # If type_old does not match within the network module
            else:
                replace_module(m, type_old, type_new, *args, **kwargs)
    
    # If type_old is not a list 
    else:
        
        for n, m in net.named_children():
            
            if isinstance(m, type_old):
                setattr(net, n, type_new(*args, **kwargs))
                
            else:
                replace_module(m, type_old, type_new, *args, **kwargs)

################################################################################
""" D: 2D to 3D of ResNet models """
# Note: Convert key 2D layers to 3D counterparts (first conv, other convs, BN, and maxpool)
def resnet2d_to_resnet3d(net, in_channels=3):
    
    # Replace the first conv2d
    replace_module(net, 
                   [nn.Conv2d,{'in_channels': 3, 'out_channels': 64}], 
                    nn.Conv3d, 
                    in_channels, 64, 7, 
                    stride=2, padding=3, bias=False)
    
    # Replace other conv2d modules
    for in_ch, out_c, ker_size, stride, padding in [(64, 64, 1, 1, 0), (64, 64, 3, 1, 1), (64, 128, 1, 2, 0),
                                                    (64, 128, 3, 2, 1), (64, 256, 1, 1, 0), (128, 128, 3, 1, 1),
                                                    (128, 128, 3, 2, 1), (128, 256, 1, 2, 0), (128, 256, 3, 2, 1),
                                                    (128, 512, 1, 1, 0), (256, 64, 1, 1, 0), (256, 128, 1, 1, 0),
                                                    (256, 256, 3, 1, 1), (256, 256, 3, 2, 1), (256, 512, 1, 2, 0),
                                                    (256, 512, 3, 2, 1), (256, 1024, 1, 1, 0), (512, 128, 1, 1, 0),
                                                    (512, 256, 1, 1, 0), (512, 512, 3, 1, 1), (512, 512, 3, 2, 1),
                                                    (512, 1024, 1, 2, 0), (512, 2048, 1, 1, 0), (1024, 256, 1, 1, 0),
                                                    (1024, 512, 1, 1, 0), (1024, 2048, 1, 2, 0), (2048, 512, 1, 1, 0)]:
        
        replace_module(net,
                       [nn.Conv2d, {'in_channels': in_ch, 'out_channels': out_c, 'kernel_size': (ker_size, ker_size),
                                    'stride': (stride, stride), 'padding': (padding, padding)}],
                       nn.Conv3d, 
                       in_ch, out_c, ker_size, 
                       stride=stride, padding=padding, bias=False)
        
        
    # Replace BatchNorm2d 
    for num_features in [64, 128, 256, 512, 1024, 2048]:
        
        replace_module(net, [nn.BatchNorm2d, {'num_features': num_features}], nn.BatchNorm3d, num_features)
        
    # Replace maxpool2d    
    replace_module(net, nn.MaxPool2d, nn.MaxPool3d, 3, stride=2, padding=1)

################################################################################
""" E: Initialisation of Conv weights """
# kaiming Initialisation: Adjust the initial weights of neural network layers to facilitate efficient training
# fan_out: preserve the magnitude of the variance of the weights in the forward pass.
# fan_in : preserve the magnitude of the variance of the weights in the bacward pass.
# Paper: Delving into rectifiers: Superpassing human-level on image-net classificaion.

def initialise_conv(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
################################################################################
""" F: Optional: Transfer learning """                
def load_resnet2d_into_3d(model3d: nn.Module,
                          back_bone_name,
                          include_prefixes=("conv1", "bn1", "layer1", "layer2"),  # extend to ("layer3","layer4") if you use them
                          scale_by_depth: bool = True,
                          freeze_bn: bool = False,
                          verbose: bool = True,
                          ):
    """
    Inflate & copy weights from a 2D torchvision ResNet into an already-constructed 3D ResNet (4-channel input).

    Works by transforming a pretrained 2D state_dict into a 3D-compatible one and loading it with strict=False.
    - Conv2d -> Conv3d: repeat along depth (kD) and optionally divide by kD.
    - First conv (3->4 in): the 4th channel is set to mean(R,G,B).
    - BN params & running stats are copied as-is.
    - Matches layers by *suffix* (so it works if your 3D model is nested, e.g., 'backbone.layer1.0.conv1.weight').

    Args:
        model3d: existing 3D ResNet model (already constructed).
        resnet2d: a pretrained torchvision ResNet (e.g., resnet152 with ImageNet weights).
        include_prefixes: tuple of top-level names to consider ('conv1','bn1','layer1','layer2', ...).
        scale_by_depth: divide repeated kernels by kD to preserve activation magnitude.
        freeze_bn: set BN affine params to requires_grad=False after loading.
        verbose: print a brief summary.

    Returns:
        report: dict with counts and a small list of skipped parameters.
    """
    if "resnet152" in back_bone_name:
        resnet2d = torch_models.resnet152(weights='IMAGENET1K_V2')
        
    else:
        assert "The model is not ResNet152"
        # TODO: develope it for other versions of ResNet
        
        
    tgt_sd = model3d.state_dict()
    src_sd = resnet2d.state_dict()

    def wants(key: str) -> bool:
        # Keep only the modules you're transferring (by prefix of the 2D key)
        return key.split(".", 1)[0] in include_prefixes

    # Build an index of 3D keys by suffix for easy matching
    tgt_keys = list(tgt_sd.keys())
    suffix_index = {}
    for k in tgt_keys:
        suffix_index.setdefault(k.split(".", 0)[-1], set()).add(k)  # trivial
        suffix_index.setdefault(k, set()).add(k)  # whole key
        # also index by full suffix after potential nesting:
        # e.g., 'backbone.layer1.0.conv1.weight' -> 'layer1.0.conv1.weight'
        parts = k.split(".")
        for i in range(len(parts)):
            suf = ".".join(parts[i:])
            suffix_index.setdefault(suf, set()).add(k)

    report = {
        "conv_copied": 0,
        "bn_copied": 0,
        "buffers_copied": 0,
        "skipped": [],
        "loaded_ok": False,
        }

    def find_target_key(src_key: str):
        # Prefer exact suffix match; if multiple, choose the shortest full name
        cands = list(suffix_index.get(src_key, []))
        if not cands:
            # try common case where keys already match by base (no nesting)
            cands = [k for k in tgt_keys if k.endswith(src_key)]
        if not cands:
            return None
        return min(cands, key=len)

    # Build a patch state_dict with transformed tensors ready for the 3D model
    patch = {}

    for k2d, w2d in src_sd.items():
        if not wants(k2d):
            continue

        k3d = find_target_key(k2d)
        if k3d is None:
            report["skipped"].append(f"no target for '{k2d}'")
            continue

        t3 = tgt_sd[k3d]
        # Same shape? direct copy.
        if w2d.shape == t3.shape:
            patch[k3d] = w2d.detach().to(dtype=t3.dtype, device=t3.device)
            if "running_" in k2d or k2d.endswith("num_batches_tracked"):
                report["buffers_copied"] += 1
            elif k2d.endswith(("weight", "bias")):
                if ".bn" in k2d or k2d.split(".")[-2].startswith("bn"):
                    report["bn_copied"] += 1
            continue

        # Try Conv2d -> Conv3d weight inflation
        if w2d.ndim == 4 and t3.ndim == 5 and k2d.endswith("weight"):
            # Inflate kernel along depth
            kd = t3.shape[2]
            w = w2d.detach().to(dtype=t3.dtype, device=t3.device)  # [out,in,kh,kw]
            w = w.unsqueeze(2).repeat(1, 1, kd, 1, 1)              # [out,in,kd,kh,kw]
            if scale_by_depth and kd > 1:
                w = w / float(kd)

            # Handle first conv (3->4 input channels)
            in2d = w.shape[1]
            in3d = t3.shape[1]
            if in2d == 3 and in3d == 4:
                out_c, _, kd_, kh, kw = w.shape
                w4 = torch.empty((out_c, 4, kd_, kh, kw), dtype=t3.dtype, device=t3.device)
                w4[:, :3] = w
                w4[:, 3] = w.mean(dim=1)  # 4th channel = mean of RGB
                w = w4

            # Final sanity: match target shape
            if w.shape != t3.shape:
                report["skipped"].append(f"shape mismatch after inflate: {k2d} -> {k3d} ({tuple(w.shape)} vs {tuple(t3.shape)})")
                continue

            patch[k3d] = w
            report["conv_copied"] += 1
            continue

        # BN tensors can differ only in dtype/device; shape must match to copy.
        if "running_" in k2d or k2d.endswith("num_batches_tracked"):
            if w2d.shape == t3.shape:
                patch[k3d] = w2d.detach().to(dtype=t3.dtype, device=t3.device)
                report["buffers_copied"] += 1
                continue

        # Otherwise skip
        report["skipped"].append(f"unsupported or mismatched: {k2d} ({tuple(w2d.shape)}) -> {k3d} ({tuple(t3.shape)})")

    # Load the patched weights into the 3D model
    missing, unexpected = model3d.load_state_dict(patch, strict=False)
    report["loaded_ok"] = (len(missing) == 0)

    # Optionally freeze BN affine params (common in transfer setups)
    if freeze_bn:
        for m in model3d.modules():
            if isinstance(m, nn.BatchNorm3d):
                for p in m.parameters():
                    p.requires_grad = False

    if verbose:
        print(f"[load_resnet2d_into_3d] conv:{report['conv_copied']}, bn:{report['bn_copied']}, "
              f"buffers:{report['buffers_copied']}, skipped:{len(report['skipped'])}")
        if report["skipped"]:
            for s in report["skipped"][:6]:
                print("  -", s)
        if missing:
            print("  missing (not loaded into model):", missing[:6])
        if unexpected:
            print("  unexpected (keys in patch not used):", unexpected[:6])

    return report                
               
################################################################################
""" G: Function to extract Conv information """
# Note: Used to compute receptive-field growth across layers

def conv_information(net):
    kernel_sizes, strides, paddings = [], [], []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if m.kernel_size[0] == 1 and m.stride[0] == 2:
                continue
            kernel_sizes += [m.kernel_size[0]]
            strides += [m.stride[0]]
            paddings += [m.padding[0]]
        elif isinstance(m, (nn.MaxPool2d, nn.MaxPool3d)):
            kernel_sizes += [m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]]
            strides += [m.stride if isinstance(m.stride, int) else m.stride[0]]
            paddings += [m.padding if isinstance(m.padding, int) else m.padding[0]]
    return kernel_sizes, strides, paddings


################################################################################
################################################################################
""" Classes """
################################################################################
""" A: A lightweight 3D U-Net decoder """

""" 3D-Block: Conv-Batchnormalise-Activaiton """
class Conv_Bn_Act(nn.Module):
    def __init__(self, in_ch, out_ch, kernel = 3, stride = 1, pad = 1, act = True):
        super(Conv_Bn_Act, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_ch, 
                              out_channels = out_ch, 
                              kernel_size = kernel, 
                              stride= stride, 
                              padding= pad,
                              bias= False)
        
        self.bn = nn.BatchNorm3d(num_features= out_ch)
        self.act = nn.ReLU() if act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
""" Up 3D-Block """
class Up_Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor= False):
        super(Up_Block3D, self).__init__()
        
        self.scale_factor = scale_factor
        
        if scale_factor:
            self.up = nn.Upsample(scale_factor = scale_factor, 
                                  mode= "trilinear", 
                                  align_corners= False)
        self.conv = Conv_Bn_Act(in_ch = in_ch,
                               out_ch = out_ch, 
                               kernel = 3, 
                               stride = 1, 
                               pad = 1)
        
    def forward(self, x):
        
        if self.scale_factor:
            return self.conv(self.up(x))
        else:
            return(self.conv(x))

""" U-Net decoder 3D """        
class UNet_Decoder3D(nn.Module):
    def __init__(self, ch1, ch2, ch3, base):
        super(UNet_Decoder3D, self).__init__()
        self.up_3 = Up_Block3D(in_ch = base, out_ch = ch1, scale_factor= False)
        self.up_2 = Up_Block3D(in_ch = ch1,  out_ch = ch2, scale_factor= 2)
        self.up_1 = Up_Block3D(in_ch = ch2, out_ch = ch3, scale_factor= 2)

    def forward(self, x):
        f_3_unet = self.up_3(x)
        f_2_unet = self.up_2(f_3_unet)
        f_1_unet = self.up_1(f_2_unet)
        
        return f_3_unet, f_2_unet, f_1_unet

################################################################################
"""
    B: Gated fusion
    Instead of plain concat, learn a tiny gate per-channel:
        alpha = sigmoid( MLP( GAP([B||U]) ) ), then F = alpha*B + (1-alpha)*U
        Note: keeps parameter count small; stabilizes using U-Net info.
"""
class Gated_Fuse3D_Channel(nn.Module):
    def __init__(self, channels, reduction=8, init_backbone_bias=0.5):
        super(Gated_Fuse3D_Channel, self).__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(nn.Conv3d(2*channels, mid, kernel_size=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv3d(mid, channels, kernel_size=1, bias=True))
        
        # init bias so sigmoid ≈ ~0.62 on backbone (tunable)
        nn.init.zeros_(self.mlp[0].weight); nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight); nn.init.constant_(self.mlp[2].bias, torch.logit(torch.tensor(init_backbone_bias)))

    def forward(self, f_backbone, f_unet):
        x = torch.cat([f_backbone, f_unet], dim=1)
        s = self.pool(x)  
        alpha = torch.sigmoid(self.mlp(s))
        return alpha * f_backbone + (1 - alpha) * f_unet, alpha

################################################################################
""" C: Add_On class """
class Add_On(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel, if_backbone = True):
        
        super(Add_On, self).__init__()
        self.if_backbone = if_backbone
        
        self.conv_1 = nn.Conv3d(in_channels= in_ch, 
                                out_channels= out_ch, 
                                kernel_size= 1, 
                                bias=False)
        self.bn_1 = nn.BatchNorm3d(num_features= out_ch)
        self.act_1 = nn.ReLU(inplace=True)
        
        self.conv_2 = nn.Conv3d(in_channels= out_ch, 
                                out_channels= out_ch, 
                                kernel_size= 1, 
                                bias=False)
        self.bn_2 = nn.BatchNorm3d(num_features= out_ch)
        
        if self.if_backbone:
            self.act_2 = nn.ReLU(inplace=True) 
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.flat = nn.Flatten(1)
        
        else:
            self.act_2 =  nn.Sigmoid()
            
    def forward(self, x):
        x = self.bn_2(self.conv_2(self.act_1(self.bn_1(self.conv_1(x)))))
        if self.if_backbone:
            x = self.flat(self.pool(self.act_2(x)))
        else:
            x = self.act_2(x)
        return x
        

################################################################################
""" C: P_Map class """

class P_Map(nn.Module):
    """
    Two-conv mapping head to produce per-prototype p-maps.
    """
    def __init__(self, in_ch, prototypes_ch, pototypes_num):
        super(P_Map,self).__init__()
        self.conv_1 = nn.Conv3d(in_channels = in_ch, 
                               out_channels = prototypes_ch, 
                               kernel_size = 1, 
                               bias=False)
        self.bn_1 = nn.BatchNorm3d(num_features= prototypes_ch)
        self.act_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv3d(in_channels= prototypes_ch, 
                                out_channels= pototypes_num, 
                                kernel_size= 1, 
                                bias=False)
        self.bn_2 = nn.BatchNorm3d(num_features= pototypes_num)
        self.act_2 = nn.Sigmoid()
        
    def forward(self, x):
        # Output: [B,K,H,W,D] in [0,1]
        return self.act_2(self.bn_2(self.conv_2(self.act_1(self.bn_1(self.conv_1(x))))))
        
################################################################################
################################################################################
"""
    ResNet_3D with predefined architectures 
    1. Note: here we train the model from the scratch or we use transfer learning
    2. Note: we added add-one to train it as well
    3. this model will be used for feature extraction for the UM-protoShare
    
"""

class ResNet_3D(nn.Module):
    
    def __init__(self, 
                 in_size=(4, 128, 128, 96), 
                 num_classes=2, 
                 backbone='resnet152_ri', 
                 n_layers=6,
                 p_mode = 2,
                 backbone_ch = [64,256,512],
                 use_unet = True,
                 freeze_unet = False,
                 fusion = "gated",
                 transfer_learning = True):
        
        super(ResNet_3D, self).__init__()
        
        self.in_size = in_size
        self.num_classes = num_classes
        self.p_mode = p_mode
        self.backbone_ch = backbone_ch
        self.use_unet = use_unet
        self.freeze_unet = freeze_unet
        self.transfer_learning = transfer_learning
        self.fusion  = fusion
        # ----------------------------------------------------------------------
        # B. Backbone ----------------------------------------------------------
        # Extract features 
        self.backbone = build_predefined_resnet_models(backbone)[:n_layers]
        self.backbone_name = backbone + f'[:{n_layers}]' if n_layers else backbone
                
        # Extarct the number of input chanenls ---------------------------------
        add_ons_channels = [m for m in self.backbone.modules()
                            if isinstance(m, nn.BatchNorm2d)][-1].num_features
        
        # Make a 3D model out of 2D model --------------------------------------
        resnet2d_to_resnet3d(self.backbone, in_channels=self.in_size[0])
        
        # Transfer Learning ----------------------------------------------------
        if self.transfer_learning:
            load_resnet2d_into_3d(self.backbone,
                                  backbone,
                                  include_prefixes=("conv1","bn1","layer1","layer2"),  # extend if needed
                                  scale_by_depth=True,
                                  freeze_bn=False,
                                  verbose=True,
                                  )
        
        # ----------------------------------------------------------------------
        # C. U-Net decoder -----------------------------------------------------
        if self.p_mode>=4:
            
            if self.use_unet: # UM-ProtoShare
                self.unet_decoder = UNet_Decoder3D(ch1 = self.backbone_ch[2], 
                                                   ch2 = self.backbone_ch[1], 
                                                   ch3 = self.backbone_ch[0], 
                                                   base = self.backbone_ch[2])
            # ------------------------------------------------------------------
            # D1. Gated-fusion per scale ---------------------------------------
                if self.fusion == "gated":
                    self.gated_fuse3d_channel_3 = Gated_Fuse3D_Channel(channels= self.backbone_ch[2])
                    self.gated_fuse3d_channel_2 = Gated_Fuse3D_Channel(channels= self.backbone_ch[1])
                    self.gated_fuse3d_channel_1 = Gated_Fuse3D_Channel(channels= self.backbone_ch[0])
                
                if self.fusion == "concat":
            # ------------------------------------------------------------------
            # D2. Concatenation per scale --------------------------------------
                    self.concat_proj_3 = nn.Conv3d(self.backbone_ch[2]*2, self.backbone_ch[2], kernel_size=1, bias=False)
                    self.concat_proj_2 = nn.Conv3d(self.backbone_ch[1]*2, self.backbone_ch[1], kernel_size=1, bias=False)
                    self.concat_proj_1 = nn.Conv3d(self.backbone_ch[0]*2, self.backbone_ch[0], kernel_size=1, bias=False)
                    
                else:
                    assert "the fusion mode should be either gated or concat"
                            
            # ------------------------------------------------------------------
            # Learned convex fusion across the three scales (keeps 128-dim)
            self.scale_logits = nn.Parameter(torch.zeros(3)) 
            
            # last layer -------------------------------------------------------
            self.fc = nn.Linear(128, self.num_classes, bias=False)
            # ------------------------------------------------------------------
            # E. Add-on layer --------------------------------------------------
        
            self.add_ons_3 = Add_On(in_ch= self.backbone_ch[2], 
                                out_ch = 128, 
                                kernel=1, 
                                if_backbone= True)
    
            self.add_ons_2 = Add_On(in_ch= self.backbone_ch[1], 
                                out_ch = 128, 
                                    kernel=1, 
                                    if_backbone= True)
    
            self.add_ons_1 = Add_On(in_ch= self.backbone_ch[0], 
                                out_ch = 128, 
                                kernel=1, 
                                if_backbone= True)
        
        # ---------------------------------------------------------------------
        if self.p_mode < 4:
            # ------------------------------------------------------------------
            # E. Add-on layer --------------------------------------------------
            self.add_ons = Add_On(in_ch= add_ons_channels, 
                                  out_ch = 128, 
                                  kernel=1, 
                                  if_backbone= True)
        
            # last layer -------------------------------------------------------
            self.fc = nn.Linear(128, self.num_classes, bias=False)
            
        self._initialize_weights()


    def get_multiscale_backbone(self, x):
        # Extract three levels from the backbone
        f_1_backbone = self.backbone[:3](x)
        f_2_backbone = self.backbone[3:5](f_1_backbone)
        f_3_backbone = self.backbone[5](f_2_backbone)
        return f_3_backbone, f_2_backbone, f_1_backbone
    
    def _initialize_weights(self):
        # Backbone
        if self.backbone_name.startswith('resnet'):
            if not self.transfer_learning:
                initialise_conv(self.backbone)
            
        if self.p_mode < 4:
            # add-ones
            initialise_conv(self.add_ons)
            
        if self.p_mode >=4:
            # add-ones
            initialise_conv(self.add_ons_3)
            initialise_conv(self.add_ons_2)
            initialise_conv(self.add_ons_1)
            
            if self.use_unet: # UM-ProtoShare
                # U-net encoder
                initialise_conv(self.unet_decoder)
                
                # Gated fuse 3D
                if self.fusion == "gated":
                    initialise_conv(self.gated_fuse3d_channel_3)
                    initialise_conv(self.gated_fuse3d_channel_2)
                    initialise_conv(self.gated_fuse3d_channel_1)
                
                if self.fusion == "concat":
                    initialise_conv(self.concat_proj_3)
                    initialise_conv(self.concat_proj_2)
                    initialise_conv(self.concat_proj_1)
                    
                if self.freeze_unet:
                    for p in self.unet_decoder.parameters():
                        p.requires_grad = False
                        
                    if self.fusion =="gate":
                        for p in self.gated_fuse3d_channel_3.parameters():
                            p.requires_grad = False
                        for p in self.gated_fuse3d_channel_2.parameters():
                            p.requires_grad = False
                        for p in self.gated_fuse3d_channel_1.parameters():
                            p.requires_grad = False
                    
                    if self.fusion =="concat":
                        for p in self.concat_proj_3.parameters():
                            p.requires_grad = False
                        for p in self.concat_proj_2.parameters():
                            p.requires_grad = False
                        for p in self.concat_proj_1.parameters():
                            p.requires_grad = False
            
            
            
    def forward(self, x, missing=None):
        
        if self.p_mode < 4:
            x = self.backbone(x)
            x = self.add_ons(x)
            x = self.fc(x)
            return x
        
        # p_mode >= 4: multi-scale path
        if self.p_mode >=4 :
            f_3_backbone, f_2_backbone, f_1_backbone = self.get_multiscale_backbone(x)
    
            if self.use_unet:
                # 2. U-Net (no internal skips) from low-res f_3_backbone
                f_3_unet, f_2_unet, f_1_unet = self.unet_decoder(f_3_backbone) 
                
                # 3. Gated fusion per scale
                if self.fusion == "gated":
                    f_3, _ = self.gated_fuse3d_channel_3(f_3_backbone, f_3_unet)    # [B, C3, ...]
                    f_2, _ = self.gated_fuse3d_channel_2(f_2_backbone, f_2_unet)    # [B, C2, ...]
                    f_1, _ = self.gated_fuse3d_channel_1(f_1_backbone, f_1_unet)    # [B, C1, ...]
                    
                if self.fusion =="concat":
                    f_3 = self.concat_proj_3(torch.cat([f_3_backbone, f_3_unet], dim=1))
                    f_2 = self.concat_proj_2(torch.cat([f_2_backbone, f_2_unet], dim=1))
                    f_1 = self.concat_proj_1(torch.cat([f_1_backbone, f_1_unet], dim=1))
                 
            else:
                # U-Net disabled: use backbone features ------------------------
                f_3, f_2, f_1 = f_3_backbone, f_2_backbone, f_1_backbone

            # Add ons per-scale to 128-d embeddings
            f_x_3 = self.add_ons_3(f_3)
            f_x_2 = self.add_ons_2(f_2)
            f_x_1 = self.add_ons_1(f_1)
            
            # Learned convex fusion across scales (keeps dim=128) --------------
            w = torch.softmax(self.scale_logits, dim=0)   # nn.Parameter([0,0,0]) defined in __init__
            v = w[0] * f_x_1 + w[1] * f_x_2 + w[2] * f_x_3 
            logits = self.fc(v)
            
            return logits

################################################################################
################################################################################
################################################################################
""" 
    UM_Proto_Share: shared multi-scale prototypes (p_mode >= 4) 
"""
class UM_Proto_Share(nn.Module):
    """
    Unified model for ProtoPNet-family with multi-scale shared prototypes.
    
    Modes:
        p_mode = 1 : ProtoPNet
        p_mode = 2 : XProtoNet
        p_mode = 3 : MProtoNet
        p_mode = 4 : UM-ProtoShare without Mapping
        p_mode >= 4: UM-ProtoShare with Mapping
    Note:
        use_unet changes the decoder/gated-fusion independently of p_mode.
        
    """

    def __init__(self, 
                 in_size=(4, 128, 128, 96), 
                 num_classes= 2, 
                 backbone = 'resnet152_ri', 
                 n_layers= 6,
                 backbone_ch = [64,256,512],
                 num_prototypes = 30, 
                 init_weights= True, 
                 f_dist= 'l2',
                 prototype_activation_function='log', 
                 p_mode= 0, # {1: ProtoPNet, 2: XProtoNet, 3:Mprotonet, 4: UM-Proto-Share} **
                 topk_p= 1,
                 use_unet = True,
                 freeze_unet = False,
                 fusion = "gated"):
        
        super(UM_Proto_Share, self).__init__()
        # ----------------------------------------------------------------------
        # Inputs ---------------------------------------------------------------
        self.in_size = in_size
        self.num_classes = num_classes
        self.backbone_ch = backbone_ch
        self.p_mode = p_mode
        self.num_prototypes = num_prototypes
        self.use_unet = use_unet
        self.freeze_unet = freeze_unet
        self.fusion = fusion

        if self.p_mode < 4:
            self.prototype_shape = (self.num_prototypes, 128, 1, 1, 1)
            
        elif self.p_mode >= 4: # UM-ProtoShare
            self.num_prototypes_03 = int(self.num_prototypes * 0.5)
            self.num_prototypes_02 = int(self.num_prototypes * 0.3)
            self.num_prototypes_01 = int(self.num_prototypes * 0.2)
            self.prototype_shape_3 = (self.num_prototypes_03, 128, 1, 1, 1)
            self.prototype_shape_2 = (self.num_prototypes_02, 128, 1, 1, 1)
            self.prototype_shape_1 = (self.num_prototypes_01, 128, 1, 1, 1)
            
        self.epsilon = 1e-4
        self.f_dist = f_dist        
        self.prototype_activation_function = prototype_activation_function
        
        # This is helpful for calculation of squared l2 distance
        if self.p_mode < 4:
            self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # ----------------------------------------------------------------------
        # Backbone -------------------------------------------------------------
        # Extract features 
        self.backbone = build_predefined_resnet_models(backbone)[:n_layers]
        self.backbone_name = backbone + f'[:{n_layers}]' if n_layers else backbone
        
        # Extarct the number of input chanenls ---------------------------------
        add_ons_channels = [m for m in self.backbone.modules()
                            if isinstance(m, nn.BatchNorm2d)][-1].num_features
        
        
        # Receptive-field info (for mapping prototype RF back to input) --------
        if self.p_mode < 4:
            layer_filter_sizes, layer_strides, layer_paddings = conv_information(self.backbone)
            # Receptive fields
            self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=self.in_size[1], 
                                                                        layer_filter_sizes=layer_filter_sizes,
                                                                        layer_strides=layer_strides, 
                                                                        layer_paddings=layer_paddings,
                                                                        prototype_kernel_size=self.prototype_shape[2])
            # Add-ons produce 128-D feature maps (no pooling here)
            self.add_ons = Add_On(in_ch= add_ons_channels, 
                                  out_ch = self.prototype_shape[1], 
                                  kernel=1, 
                                  if_backbone= False)
            
        # Multi-scale RF (backbone slices): [:6] ~ layer3, [:5] ~ layer2, [:3] ~ layer1
        if self.p_mode >= 4: # UM-ProtoShare
            layer_filter_sizes_3, layer_strides_3, layer_paddings_3 = conv_information(self.backbone[:6])
            layer_filter_sizes_2, layer_strides_2, layer_paddings_2 = conv_information(self.backbone[:5])
            layer_filter_sizes_1, layer_strides_1, layer_paddings_1 = conv_information(self.backbone[:3])
        
            # Receptive fields
            self.proto_layer_rf_info_3 = compute_proto_layer_rf_info_v2(img_size=self.in_size[1], 
                                                                        layer_filter_sizes=layer_filter_sizes_3,
                                                                        layer_strides=layer_strides_3, 
                                                                        layer_paddings=layer_paddings_3,
                                                                        prototype_kernel_size=self.prototype_shape_3[2])  
            self.proto_layer_rf_info_2 = compute_proto_layer_rf_info_v2(img_size=self.in_size[1], 
                                                                        layer_filter_sizes=layer_filter_sizes_2,
                                                                        layer_strides=layer_strides_2, 
                                                                        layer_paddings=layer_paddings_2,
                                                                        prototype_kernel_size=self.prototype_shape_2[2]) 
            self.proto_layer_rf_info_1 = compute_proto_layer_rf_info_v2(img_size=self.in_size[1], 
                                                                        layer_filter_sizes=layer_filter_sizes_1,
                                                                        layer_strides=layer_strides_1, 
                                                                        layer_paddings=layer_paddings_1,
                                                                        prototype_kernel_size=self.prototype_shape_1[2]) 
        # ----------------------------------------------------------------------
        # U-Net decoder + gated fusion ---------------------------------------
            if self.use_unet:
                self.unet_decoder = UNet_Decoder3D(ch1 = self.backbone_ch[2], 
                                                   ch2 = self.backbone_ch[1], 
                                                   ch3 = self.backbone_ch[0], 
                                                   base = self.backbone_ch[2])
            
                # ------------------------------------------------------------------
                # D1. Gated-fusion per scale ---------------------------------------
                if self.fusion == "gated":
                    self.gated_fuse3d_channel_3 = Gated_Fuse3D_Channel(channels= self.backbone_ch[2])
                    self.gated_fuse3d_channel_2 = Gated_Fuse3D_Channel(channels= self.backbone_ch[1])
                    self.gated_fuse3d_channel_1 = Gated_Fuse3D_Channel(channels= self.backbone_ch[0])
    
                if self.fusion == "concat":
                    # ------------------------------------------------------------------
                    # D2. Concatenation per scale --------------------------------------
                    self.concat_proj_3 = nn.Conv3d(self.backbone_ch[2]*2, self.backbone_ch[2], kernel_size=1, bias=False)
                    self.concat_proj_2 = nn.Conv3d(self.backbone_ch[1]*2, self.backbone_ch[1], kernel_size=1, bias=False)
                    self.concat_proj_1 = nn.Conv3d(self.backbone_ch[0]*2, self.backbone_ch[0], kernel_size=1, bias=False)  
                
                else:
                    assert "the fusion mode should be either gated or concat"
                
        # ----------------------------------------------------------------------
        # Add-ons per scale (keep 128-D channel count) -------------------------
        
            self.add_ons_3 = Add_On(in_ch= self.backbone_ch[2], 
                                    out_ch = self.prototype_shape_3[1], 
                                    kernel=1, 
                                    if_backbone= False)
            self.add_ons_2 = Add_On(in_ch= self.backbone_ch[1], 
                                    out_ch = self.prototype_shape_2[1], 
                                        kernel=1, 
                                        if_backbone= False)
            self.add_ons_1 = Add_On(in_ch= self.backbone_ch[0], 
                                    out_ch = self.prototype_shape_1[1], 
                                    kernel=1, 
                                    if_backbone= False)
        # ----------------------------------------------------------------------
        # Learnable prtotypes --------------------------------------------------
        if self.p_mode < 4:
            # Class-specific identity (legacy ProtoPNet-style)
            self.prototype_class_identity = nn.Parameter(torch.zeros(self.num_prototypes, self.num_classes), requires_grad=False)
            
            # Number of prototypes per class
            self.num_prototypes_per_class = self.num_prototypes // self.num_classes
            for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
            
            # Ensure 5D shape
            while len(self.prototype_shape) < 5:
                self.prototype_shape += (1,)
            
            # Prototype vectors 
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))
        
        if self.p_mode >= 4: # UM-ProtoShare
            self.prototype_vectors_3 = nn.Parameter(torch.rand(self.prototype_shape_3))
            self.prototype_vectors_2 = nn.Parameter(torch.rand(self.prototype_shape_2))
            self.prototype_vectors_1 = nn.Parameter(torch.rand(self.prototype_shape_1))
        
        # ----------------------------------------------------------------------
        # Mapping module -------------------------------------------------------
        if self.p_mode < 4:
            if self.p_mode >= 1:
                # Compute prototype patch sizes along W and D to set top-k
                p_size_w = compute_proto_layer_rf_info_v2(
                                                          img_size=self.in_size[2], 
                                                          layer_filter_sizes=layer_filter_sizes,
                                                          layer_strides=layer_strides, 
                                                          layer_paddings=layer_paddings,
                                                          prototype_kernel_size=self.prototype_shape[3]
                                                          )[0]
                
                p_size_a = compute_proto_layer_rf_info_v2(
                                                          img_size=self.in_size[3], 
                                                          layer_filter_sizes=layer_filter_sizes,
                                                          layer_strides=layer_strides, layer_paddings=layer_paddings,
                                                          prototype_kernel_size=self.prototype_shape[4]
                                                          )[0]
                
                self.p_size = (self.proto_layer_rf_info[0], p_size_w, p_size_a)
                self.topk_p = int(self.p_size[0] * self.p_size[1] * self.p_size[2] * topk_p / 100)
                            
                assert self.topk_p >= 1
            
            if self.p_mode >= 2:
                self.p_map = P_Map(in_ch = add_ons_channels, 
                                   prototypes_ch = self.prototype_shape[1],
                                   pototypes_num = self.prototype_shape[0])
        
        if self.p_mode >= 4: # UM-ProtoShare
            self.p_map_3 = P_Map(in_ch=self.backbone_ch[2],
                                 prototypes_ch=self.prototype_shape_3[1],
                                 pototypes_num=self.prototype_shape_3[0])
            
            self.p_map_2 = P_Map(in_ch=self.backbone_ch[1],
                                 prototypes_ch=self.prototype_shape_2[1],
                                 pototypes_num=self.prototype_shape_2[0])
            
            self.p_map_1 = P_Map(in_ch=self.backbone_ch[0],
                                 prototypes_ch=self.prototype_shape_1[1],
                                 pototypes_num=self.prototype_shape_1[0])

                
        # Last layer -----------------------------------------------------------
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        if init_weights:
                self._initialize_weights()

    
    """ General Functions """  
    def device(self):
        device, = set([p.device for p in self.parameters()] + [b.device for b in self.buffers()])
        return device 
    
    def scale(self, x, dim):
        # Min-max scale over given dims (avoid divide-by-zero)
        x = x - x.amin(dim, keepdim=True)
        return x / x.amax(dim, keepdim=True).clamp_min(self.epsilon)   

    def sigmoid(self, x, omega=10, sigma=0.5):
        # Sharpening nonlinearity used on p-maps
        return torch.sigmoid(omega * (x - sigma))
    
    """ UM_Proto_Share""" 
    def _scale(self, x, dim):
        """
        L1 normalise x over the given dims.
        """
        # x should already be non-negative because of ReLU in get_p_map_scale_sharp
        denom = x.sum(dim=dim, keepdim=True).clamp_min(self.epsilon)
        return x / denom

    def get_multiscale_backbone(self, x):
        f_1_backbone = self.backbone[:3](x)
        f_2_backbone = self.backbone[3:5](f_1_backbone)
        f_3_backbone = self.backbone[5](f_2_backbone)
        return f_3_backbone, f_2_backbone, f_1_backbone
    
    def masked_pool(self,f_x, p_m):
        # Einsum over spatial dims then average by spatial size (MProtoNet style)
        # f_x: [B,128,H,W,D], p_m: [B,K,H,W,D] -> [B,K,128,1,1,1]
        p_size = f_x.flatten(2).shape[2]
        return (torch.einsum('bphwd,bchwd->bpc', p_m, f_x) / p_size)[(...,) + (None,)*3]  # [B,K,128,1,1,1]
      
    def get_p_map_scale_sharp(self, x, p_map_module, sharpening = True):
        # Scale-specific p-map: BN/ReLU -> scale -> (optional) sharpen
        p_map = F.relu(p_map_module(x))
        p_map = self._scale(p_map, tuple(range(1, p_map.ndim)))  # dims 1..N-1
        if sharpening:
            return self.sigmoid(p_map)
        else: 
            return p_map
    
    def get_p_map(self, x, sharpening = True):
        # Get p_map then scale+sharpen
        if self.p_mode >=3:
            p_map = F.relu(self.p_map(x))
            p_map = self.scale(p_map, tuple(range(1, p_map.ndim)))
            return self.sigmoid(p_map)
        else:
            return self.p_map(x)
        
    def lse_pooling(self, x, r=10, dim=-1):
        # Log-sum-exp pooling (temperature r)
        return (torch.logsumexp(r * x, dim=dim) - torch.log(torch.tensor(x.shape[dim]))) / r
    
    def conv_features(self, x):
        # x: input -> backbone -> add_ons
        x = self.backbone(x)
        f_x = self.add_ons(x)

        if self.p_mode >= 2:
            p_map = self.get_p_map(x)
            return f_x, p_map, x
        else:
            return f_x
    
    def l2_convolution_3D(self, x):
        # L2 distance via conv3d (ProtoPNet-style)
        if x.shape[1:] == self.prototype_shape: # in somehow it means if p_mode>=2
            x_2 = (x ** 2).sum(2)
            xp = (x * self.prototype_vectors).sum(2)
        else:
            x_2 = F.conv3d(x ** 2, self.ones)
            xp = F.conv3d(x, self.prototype_vectors)
        p_2 = (self.prototype_vectors ** 2).sum((1, 2, 3, 4)).reshape(-1, 1, 1, 1)
        return F.relu(x_2 - 2 * xp + p_2)

    def cosine_convolution_3D(self, x):
        assert x.min() >= 0, f"{x.min():.16g} >= 0"
        prototype_vectors_unit = F.normalize(self.prototype_vectors, p=2, dim=1)
        if x.shape[1:] == self.prototype_shape: # in somehow it means if p_mode>=2
            x_unit = F.normalize(x, p=2, dim=2)
            return F.relu(1 - (x_unit * prototype_vectors_unit).sum(2))
        else:
            x_unit = F.normalize(x, p=2, dim=1)
            return F.relu(1 - F.conv3d(input=x_unit, weight=prototype_vectors_unit))
        
    def prototype_distances(self, x, p_map=None):
        if self.p_mode >= 2:
            p_size = x.flatten(2).shape[2]
            p_x = (torch.einsum('bphwa,bchwa->bpc', p_map, x) / p_size)[(...,) + (None,) * 3]
            
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(p_x), p_map, p_x
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(p_x), p_map, p_x
        else:
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(x)
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(x)
            
   # Scale-aware distances for UM_ProtoShare
    def _l2_conv_3D_scale(self, x, proto_vectors):
        if x.shape[2:] == proto_vectors.shape[2:]:
            x_2  = (x ** 2).sum(2)
            xp   = (x * proto_vectors).sum(2)
        else:
            ones = torch.ones_like(proto_vectors, device=x.device, dtype=x.dtype)
            x_2  = F.conv3d(x ** 2, ones)
            xp   = F.conv3d(x, proto_vectors)
        p_2 = (proto_vectors ** 2).sum((1, 2, 3, 4)).reshape(-1, 1, 1, 1)
        return F.relu(x_2 - 2 * xp + p_2)

    def _cosine_conv_3D_scale(self, x, proto_vectors):
        proto_unit = F.normalize(proto_vectors, p=2, dim=1)
        if x.shape[1:] == proto_vectors.shape:
            x_unit = F.normalize(x, p=2, dim=2)
            return F.relu(1 - (x_unit * proto_unit).sum(2))
        else:
            x_unit = F.normalize(x, p=2, dim=1)
            return F.relu(1 - F.conv3d(input= x_unit, weight= proto_unit))
        
    def _prototype_distances(self, x, proto_vectors, p_map=None):
        if self.p_mode >= 5: # For applying P_map
            p_size = x.flatten(2).shape[2]
            p_x = (torch.einsum('bphwa,bchwa->bpc', p_map, x) / p_size)[(...,) + (None,) * 3]
            
            if self.f_dist == 'l2':
                return self._l2_conv_3D_scale(p_x, proto_vectors), p_map, p_x
            elif self.f_dist == 'cos':
                return self._cosine_conv_3D_scale(p_x, proto_vectors), p_map, p_x
        else:
            if self.f_dist == 'l2':
                return self._l2_conv_3D_scale(x, proto_vectors)
            elif self.f_dist == 'cos':
                return self._cosine_conv_3D_scale(x, proto_vectors)
    
    # Convert distance to similarity 
    def distance_to_similarity(self, distances):
        if self.f_dist == 'cos':
            return F.relu(1 - distances)
        elif self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)            
            
    def forward(self, x, missing=None):
        
        # Single-scale branches ---------------------------------------
        if self.p_mode < 4:
            
            if self.p_mode >= 2:
                f_x, p_map, x = self.conv_features(x)
                distances, p_map, _ = self.prototype_distances(f_x, p_map)
            else:
                f_x = self.conv_features(x)
                distances = self.prototype_distances(f_x)
                        
            distances = distances.flatten(2)
            
            if self.p_mode == 1: # Xprotonet
                min_distances = distances.topk(self.topk_p, dim=2, largest=False)[0].mean(2)
            elif self.p_mode >= 2: # Mprotonet
                min_distances = distances.flatten(1) 
            else:                  # ProtoPNet
                min_distances = distances.amin(2)
            
            prototype_activations = self.distance_to_similarity(min_distances)
            logits = self.last_layer(prototype_activations)
            
            if self.p_mode >= 2 and self.training:
                return logits, min_distances, x, p_map
            elif self.training:
                return logits, min_distances
            else:
                return logits    
        # UM_ProtoShare ------------------------------------------------------   
        if self.p_mode >= 4: # UM-ProtoShare

            # Multi-scale backbone feature space
            f_3_backbone, f_2_backbone, f_1_backbone = self.get_multiscale_backbone(x)
            
            if self.use_unet:
                # UNet (no internal skips) from low-res f_3_backbone
                f_3_unet, f_2_unet, f_1_unet = self.unet_decoder(f_3_backbone) 
            
                # 3. Gated fusion per scale
                if self.fusion == "gated":
                    f_3, _ = self.gated_fuse3d_channel_3(f_3_backbone, f_3_unet)
                    f_2, _ = self.gated_fuse3d_channel_2(f_2_backbone, f_2_unet)
                    f_1, _ = self.gated_fuse3d_channel_1(f_1_backbone, f_1_unet)
    
                if self.fusion =="concat":
                    f_3 = self.concat_proj_3(torch.cat([f_3_backbone, f_3_unet], dim=1))
                    f_2 = self.concat_proj_2(torch.cat([f_2_backbone, f_2_unet], dim=1))
                    f_1 = self.concat_proj_1(torch.cat([f_1_backbone, f_1_unet], dim=1))
            else:
               f_3, f_2, f_1 = f_3_backbone, f_2_backbone, f_1_backbone
                
            # Add ons per-scale to 128-d embeddings
            f_x_3 = self.add_ons_3(f_3) 
            f_x_2 = self.add_ons_2(f_2)
            f_x_1 = self.add_ons_1(f_1)
            
            # P-maps per-scale (attention maps per prototype)
            if self.p_mode >= 5:
                p_map_3 = self.get_p_map_scale_sharp(f_3, self.p_map_3, sharpening = True)
                p_map_2 = self.get_p_map_scale_sharp(f_2, self.p_map_2, sharpening = True)
                p_map_1 = self.get_p_map_scale_sharp(f_1, self.p_map_1, sharpening = True)
            
            # Distances per scale to the corresponding prototype banks
            if self.p_mode >= 5:
                distances_3, p_map_3, _ = self._prototype_distances(f_x_3, 
                                                                    self.prototype_vectors_3, 
                                                                    p_map_3)
                distances_2, p_map_2, _ = self._prototype_distances(f_x_2, 
                                                                    self.prototype_vectors_2, 
                                                                    p_map_2)
                distances_1, p_map_1, _ = self._prototype_distances(f_x_1, 
                                                                    self.prototype_vectors_1, 
                                                                    p_map_1)
                
                # Concat distances across scales
                distances = torch.cat([distances_3, distances_2, distances_1], dim=1).flatten(1)
            
            else:
                distances_3 = self._prototype_distances(f_x_3, self.prototype_vectors_3)
                distances_2 = self._prototype_distances(f_x_2, self.prototype_vectors_2)
                distances_1 = self._prototype_distances(f_x_1, self.prototype_vectors_1)
                # Flatening
                distances_3 = distances_3.flatten(2).amin(2)
                distances_2 = distances_2.flatten(2).amin(2)
                distances_1 = distances_1.flatten(2).amin(2)
                # Concat distances across scales → [B,K_total]
                distances = torch.cat([distances_3,distances_2,distances_1], dim=1)
        
            # Similarity + logits
            prototype_activations = self.distance_to_similarity(distances)            
            logits = self.last_layer(prototype_activations)   
            
            if self.p_mode >= 5 and self.training:
                return logits, distances, (f_3, f_2, f_1), (p_map_3, p_map_2, p_map_1)
            
            elif self.training:
                return logits, distances
            else:
                return logits 
            
        
    def push_forward(self, x):
        # Return intermediate tensors needed by the push procedure
        if self.p_mode < 4:
            if self.p_mode >= 2:
                f_x, p_map, _ = self.conv_features(x)
                distances, p_map, p_x = self.prototype_distances(f_x, p_map)
                return p_x, distances, p_map
            else:
                f_x = self.conv_features(x)
                distances = self.prototype_distances(f_x)
                return f_x, distances
        
        if self.p_mode >= 4:
            # Multi-scale backbone feature space
            f_3_backbone, f_2_backbone, f_1_backbone = self.get_multiscale_backbone(x)
            
            if self.use_unet:
                # U-Net (no internal skips) from low-res f_3_backbone
                f_3_unet, f_2_unet, f_1_unet = self.unet_decoder(f_3_backbone) 
                            
                # 3. Gated fusion per scale
                if self.fusion == "gated":
                    f_3, _ = self.gated_fuse3d_channel_3(f_3_backbone, f_3_unet)    
                    f_2, _ = self.gated_fuse3d_channel_2(f_2_backbone, f_2_unet)   
                    f_1, _ = self.gated_fuse3d_channel_1(f_1_backbone, f_1_unet)  
                    
                if self.fusion =="concat":
                    f_3 = self.concat_proj_3(torch.cat([f_3_backbone, f_3_unet], dim=1))
                    f_2 = self.concat_proj_2(torch.cat([f_2_backbone, f_2_unet], dim=1))
                    f_1 = self.concat_proj_1(torch.cat([f_1_backbone, f_1_unet], dim=1))
            else: 
                f_3, f_2, f_1 = f_3_backbone, f_2_backbone, f_1_backbone
            
            # Add ons per-scale to 128-d embeddings
            f_x_3 = self.add_ons_3(f_3)   
            f_x_2 = self.add_ons_2(f_2)  
            f_x_1 = self.add_ons_1(f_1)
            
            # P-maps per-scale (attention maps per prototype)
            if self.p_mode >= 5:
                p_map_3 = self.get_p_map_scale_sharp(f_3, self.p_map_3, sharpening = True) 
                p_map_2 = self.get_p_map_scale_sharp(f_2, self.p_map_2, sharpening = True) 
                p_map_1 = self.get_p_map_scale_sharp(f_1, self.p_map_1, sharpening = True) 
            
            # Distances per scale to the corresponding prototype banks
            if self.p_mode >= 5:
                distances_3, p_map_3, p_x_3 = self._prototype_distances(f_x_3, 
                                                                    self.prototype_vectors_3, 
                                                                    p_map_3)
                distances_2, p_map_2, p_x_2 = self._prototype_distances(f_x_2, 
                                                                    self.prototype_vectors_2, 
                                                                    p_map_2)
                distances_1, p_map_1, p_x_1 = self._prototype_distances(f_x_1, 
                                                                    self.prototype_vectors_1, 
                                                                    p_map_1)
            else:
                distances_3 = self._prototype_distances(f_x_3, self.prototype_vectors_3)
                distances_2 = self._prototype_distances(f_x_2, self.prototype_vectors_2)
                distances_1 = self._prototype_distances(f_x_1, self.prototype_vectors_1)

            
            if self.p_mode >= 5:
                return (p_x_3, p_x_2, p_x_1), (distances_3, distances_2, distances_1), (p_map_3, p_map_2,p_map_1) 
            else:
                return (f_x_3, f_x_2, f_x_1),  (distances_3, distances_2, distances_1)
            
            

    def prune_prototypes(self, prototypes_to_prune):
        # TODO: Pruning
        pass
   
      
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = self.prototype_class_identity.mT
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
                                          correct_class_connection * positive_one_weights_locations
                                          + incorrect_class_connection * negative_one_weights_locations                                          )

    def _initialize_weights(self):
        
        # Backbone
        if self.backbone_name.startswith('resnet'):
            resnet2d_to_resnet3d(self.backbone, in_channels=self.in_size[0])
        
        if self.p_mode < 4:
            # add-ones
            initialise_conv(self.add_ons)
            # p-mode
            if self.p_mode >= 2:
                initialise_conv(self.p_map)
                
            self.set_last_layer_incorrect_connection(-0.5)
    
        if self.p_mode >= 4: # UM-ProtoShare
            # add-ones
            initialise_conv(self.add_ons_3)
            initialise_conv(self.add_ons_2)
            initialise_conv(self.add_ons_1)
            
            if self.p_mode>4:
                # p-map
                initialise_conv(self.p_map_3)
                initialise_conv(self.p_map_2)
                initialise_conv(self.p_map_1)
                
            if self.use_unet and self.freeze_unet:
                for p in self.unet_decoder.parameters():
                    p.requires_grad = False
                
                if self.fusion == "gated":
                    for p in self.gated_fuse3d_channel_3.parameters():
                        p.requires_grad = False
                    for p in self.gated_fuse3d_channel_2.parameters():
                        p.requires_grad = False
                    for p in self.gated_fuse3d_channel_1.parameters():
                        p.requires_grad = False
                        
                if self.fusion == "concat":
                    for p in self.concat_proj_3.parameters():
                        p.requires_grad = False
                    for p in self.concat_proj_2.parameters():
                        p.requires_grad = False
                    for p in self.concat_proj_1.parameters():
                        p.requires_grad = False
                        
# ###############################################################################
# ################################################################################
# from torchinfo import summary

# if __name__ =="__main__":
#     x = torch.rand(1, 4, 128,128,96)
#     # print(x.shape)
#     x = x.type(torch.FloatTensor).to("cuda")
#     model = UM_Proto_Share(in_size=(4, 128, 128, 96), 
#                           num_classes= 2, 
#                           backbone = 'resnet152_ri', 
#                           n_layers= 6,
#                           backbone_ch = [64,256,512],
#                           num_prototypes = 30, 
#                           init_weights= True, 
#                           f_dist= 'cos',
#                           prototype_activation_function='log', 
#                           p_mode= 4, # {1: ProtoPNet, 2: XProtoNet, 3:Mprotonet, 4: UM-Proto-Share, 5: UM_ProtoShare with Pmap} **
#                           topk_p= 1,
#                           use_unet = True,
#                           freeze_unet = True,
#                           fusion = "gated").to("cuda")
    
#     model_summary = summary(model = model, 
#                             input_size=(1, 4, 128, 128,96), 
#                             verbose = 1)
#     x = model(x)
#     del model, x
    # x,y,z = model.get_multiscale_backbone(x)




