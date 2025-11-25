This repository contains the official implementation of **UM-ProtoShare** from the paper “**UM-ProtoShare: UNet-Guided, Multi-Scale Shared Prototypes for Interpretable Brain Tumour Classification Using Multi-Sequence 3D MRI**” (under review at **MIDL 2026**) by **Ali Golbaf, Vivek Singh, Swen Gaudl, and Emmanuel Ifeachor**.
## Model Overview
### 🔄 UM-ProtoShare Workflow
![UM-ProtoShare Workflow](images/Workflow.png)
---
## Architecture
### 🧠 Backbone: 3D ResNet-152 + Lightweight UNet Decoder + Gated Fusions
![UM-ProtoShare Backbone](images/Backbone.png)
*Figure 2 – Backbone architecture. A 3D ResNet-152 encoder (truncated) produces three spatial scales of features.  
A lightweight 3D UNet-style decoder upsamples features via trilinear interpolation, followed by Conv–BatchNorm–ReLU blocks.  
Per-scale **gated fusion** modules combine encoder and decoder features to form spatially coherent feature maps used for prototypes and localisation.*
Key points:
- 3D ResNet-152 encoder (converted from 2D ImageNet weights)
- Truncated to preserve spatial resolution
- Lightweight decoder with trilinear upsampling (no heavy deconvs)
- **Gated encoder–decoder fusion** at each scale to balance semantics and localisation

---
### 📍 Localisation & Prototype Matching
![UM-ProtoShare Localisation](images/Localisation.png)
*Figure 3 – Localisation component. At each scale, fused feature maps are passed through:  
(1) an **add-on module** to get fixed-dimensional embeddings, and  
(2) a **mapping module** that predicts per-prototype attention maps.  
A soft-masked, normalised mapping produces prototype-specific descriptors that are compared (via cosine similarity) with shared prototypes. These similarities are then used for classification and for generating 3D activation maps.*
Core ideas:
- Shared **class-agnostic prototypes** at multiple scales
- Per-prototype attention maps with soft-masked normalisation
- Cosine similarity between prototype vectors and masked feature descriptors
- Online-CAM and diversity regularisation to sharpen and diversify prototype usage
