# UM-ProtoShare
This repository contains the official implementation of **UM-ProtoShare** from the paper “**UM-ProtoShare: UNet-Guided, Multi-Scale Shared Prototypes for Interpretable Brain Tumour Classification Using Multi-Sequence 3D MRI**” (under review at **MIDL 2026**) by **Ali Golbaf, Vivek Singh, Swen Gaudl, and Emmanuel Ifeachor**.
### 🔄 UM-ProtoShare Workflow
![UM-ProtoShare Workflow](images/Workflow.png)
---
### 🧠 Core ideas
* **Shared, class-agnostic prototypes**  
  UM-ProtoShare learns a bank of shared, class-agnostic prototypes instead of class-specific ones. Each prototype can support multiple classes through soft class–prototype coefficients derived from Grad-CAM-style importance weights. This allows the model to efficiently reuse MRI features that genuinely occur across tumour grades (e.g. peritumoural oedema, necrotic cores, enhancing rims) and reduces redundancy in the prototype space.
* **Weakly supervised localisation with gated fusions**  
  The model improves localisation over prior case-based methods (MProtoNet, MAProtoNet) by adding a lightweight 3D UNet-style decoder with encoder–decoder gated fusion blocks. Trained only with image-level labels, this backbone produces spatially coherent feature maps so that prototype evidence aligns with tumour-related regions, while preserving strong classification performance.
* **Explicit multi-scale prototypes**  
  UM-ProtoShare learns separate prototype sets at multiple spatial scales, capturing tumour appearance from fine to coarse resolutions. By varying how many prototypes are allocated to each scale, we can explicitly study how emphasising different spatial scales trades off between classification accuracy and interpretability.

### Experiment
### Configure data paths
Before running the code, edit `main_Code.py` and set:
```python
path_to_data_images   = "/path/to/MICCAI_BraTS2020_TrainingData"
path_to_data_clinical = "/path/to/BraTS2020_clinical_csv"
### Acknowledgment
This repository contains modified source code from [MProtoNet](https://github.com/aywi/mprotonet) by Yuanyuan Wei, Roger Tam, and Xiaoying Tang.

### Citation
```bibtex
@inproceedings{Golbaf2026UMProtoShare,
  title     = {UM-ProtoShare: UNet-Guided, Multi-Scale Shared Prototypes for Interpretable Brain Tumour Classification Using Multi-Sequence 3D MRI},
  author    = {Golbaf, Ali and Singh, Vivek and Gaudl, Swen and Ifeachor, Emmanuel},
  booktitle = {Proceedings of the International Conference on Medical Imaging with Deep Learning (MIDL)},
  year      = {2026},
  note      = {Full paper, under review}
}


