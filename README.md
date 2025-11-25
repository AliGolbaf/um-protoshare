This repository contains the official implementation of **UM-ProtoShare** from the paper  
“**UM-ProtoShare: UNet-Guided, Multi-Scale Shared Prototypes for Interpretable Brain Tumour Classification Using Multi-Sequence 3D MRI**” (under review at **MIDL 2026**) by **Ali Golbaf, Vivek Singh, Swen Gaudl, and Emmanuel Ifeachor**.

## Model Overview

### 🔄 UM-ProtoShare Workflow
![UM-ProtoShare Workflow](images/Workflow.png)
*Figure 1 – High-level workflow of UM-ProtoShare. The model takes multi-sequence 3D MRI as input, extracts multi-scale features with a 3D ResNet-152 backbone plus UNet-style decoder with gated fusions, matches them against a bank of shared prototypes, and aggregates prototype similarities into tumour grade predictions with prototype-based visual explanations.*
