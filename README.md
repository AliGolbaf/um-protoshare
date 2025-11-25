This repository contains the official implementation of **UM-ProtoShare** from the paper “**UM-ProtoShare: UNet-Guided, Multi-Scale Shared Prototypes for Interpretable Brain Tumour Classification Using Multi-Sequence 3D MRI**” (under review at **MIDL 2026**) by **Ali Golbaf, Vivek Singh, Swen Gaudl, and Emmanuel Ifeachor**.
## Model Overview
### 🔄 UM-ProtoShare Workflow
![UM-ProtoShare Workflow](images/Workflow.png)
---
## Architecture
### 🧠 Backbone: 3D ResNet-152 + Lightweight UNet Decoder + Gated Fusions
![UM-ProtoShare Backbone](images/Backbone.png)
Key points:
- **3D ResNet-152** encoder (Truncated to preserve localisation).
- **Lightweight decoder** with trilinear upsampling.
- **Gated encoder–decoder fusion** at each scale to balance semantics and localisation.
---
### 📍 Localisation & Prototype Matching
![UM-ProtoShare Localisation](images/Localisation.png)

Key points:  
* **add-on module** to extract high level features and get fixed-dimensional embeddings.  
* **mapping module** that predicts per-prototype attention maps.  
---
**Core ideas:**
- **Shared, class-agnostic prototypes**  
  UM-ProtoShare learns a bank of shared, class-agnostic prototypes instead of class-specific ones. Each prototype can support multiple classes through soft class–prototype coefficients derived from Grad-CAM-style importance weights. This allows the model to efficiently reuse MRI features that genuinely occur across tumour grades (e.g. peritumoural oedema, necrotic cores, enhancing rims) and reduces redundancy in the prototype space.

- **Weakly supervised localisation with gated fusions**  
  The model improves localisation over prior case-based methods (MProtoNet, MAProtoNet) by adding a lightweight 3D UNet-style decoder with encoder–decoder **gated fusion** blocks. Trained only with image-level labels, this backbone produces spatially coherent feature maps so that prototype evidence aligns with tumour-related regions, while preserving strong classification performance.

- **Explicit multi-scale prototypes**  
  UM-ProtoShare learns separate prototype sets at multiple spatial scales, capturing tumour appearance from fine to coarse resolutions. By varying how many prototypes are allocated to each scale, we can explicitly study how emphasising different spatial scales trades off between classification accuracy and interpretability.

---
