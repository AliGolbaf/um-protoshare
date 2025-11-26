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
#### Training the Backbone
```python
python main_Code.py \
  -di /path/to/MICCAI_BraTS2020_TrainingData \
  -dc /path/to/clinical_dir \
  --clinical-csv name_mapping.csv \
  --seed 0 \
  --cv-folds 5 \
  --transfer-learning 1 \
  --cv-repeats 1 \
  --backbone resnet152_ri \
  --n-layers 6 \
  --train-backbone True \
  --epochs-bb 50 \
  --batch-size-bb 1 \
  --num-workers-bb 1 \
  --lr-bb 1e-3 \
  --wd-bb 1e-2 \
  --class-loss-bb focal \
  --optim-bb Adam \
  --save-model 1
```
#### Key configuration options
- **Enable / disable UNet decoder (`--use-unet`)**
  - `--use-unet 1`: enables the lightweight 3D UNet-style decoder (full UM-ProtoShare).
  - `--use-unet 0`: disables the UNet decoder and uses only the backbone features.
  - Example:
    ```bash
    --use-unet 1    # with UNet decoder
    --use-unet 0   # backbone-only
    
- **Fusion type (`--fusion`)**
  - `--fusion gated`: uses **gated encoder–decoder fusions** (UM-ProtoShare default).
  - `--fusion concat`: uses **simple concatenation** of encoder and decoder features.
  - Example:
    ```bash
    --fusion gated     # gated fusion
    --fusion concat    # simple concatenation
    ```  
#### Training UM-ProtoShare
```python
python main_Code.py \
  -di /path/to/MICCAI_BraTS2020_TrainingData \
  -dc /path/to/clinical_dir \
  --clinical-csv name_mapping.csv \
  --seed 0 \
  --cv-folds 5 \
  --cv-repeats 1 \
  --p-mode 5 \
  --backbone resnet152_ri \
  --n-layers 6 \
  --num-prototypes 30 \
  --use-unet True \
  --freeze-unet 1 \
  --fusion gated \
  --train-um True \
  --epochs-um 100 \
  --batch-size-um 1 \
  --num-workers-um 1 \
  --lr-um 1e-3 \
  --wd-um 1e-2 \
  --class-loss-um focal \
  --optim-um AdamW \
  --warmup 1 \
  --warmup-ratio 0.2 \
  --class-assignments soft\
  --use-augmentation 1 \
  --save-model True \
  --coefs "{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05, 'div': 0.01}"
```
#### Key configuration options
- **Freeze UNet decoder (`--freeze-unet`)**
  - `--freeze-unet 1`: keep UNet decoder weights fixed.
  - `--freeze-unet 0`: train the UNet decoder jointly with the rest of the model.
  - Example:
    ```bash
    --freeze-unet 0
    ```
- **Pretrained backbone checkpoint**
  - To **add a pretrained backbone** into UM-ProtoShare, use:
    ```bash
    --add-pretrained-backbone 1 \
    --pretrained-backbone-path Model_Checkpoints/backbone_best.pt
    ```
  - This loads `Model_Checkpoints/backbone_best.pt` before training prototypes.
- **Hard vs soft class assignment (`--class-assignments`)**

  UM-ProtoShare uses **shared, class-agnostic prototypes**. Each prototype can support multiple classes through class–prototype coefficients. You can choose how these coefficients are used:
  - `--class-assignments soft` (recommended)  
    - Each prototype has a **soft weight** for every class (e.g. `[0.2, 0.8]`).
    - These weights are derived from Grad-CAM–style importance scores and normalised.
    - During prediction, a prototype’s similarity contributes **fractionally** to multiple classes.
    - Intuitively: a prototype can be “mostly HGG” but still partially support LGG.
  - `--class-assignments hard`  
    - Each prototype is assigned to the **single class with the highest weight** (argmax).
    - Its similarity then contributes **only to that class**, like classic ProtoPNet/MProtoNet.
  - Example:
    ```bash
    --class-assignments soft   # shared prototypes (UM-ProtoShare default)
    --class-assignments hard   # class-specific prototypes, ProtoPNet-like
    ```
- **Loss functions and coefficients**
  - **Classification loss** (`--class-loss-um`)
    - `focal` (default): focal loss with class weights.
    - `cross_ent`: standard cross-entropy.
    - Example:
      ```bash
      --class-loss-um focal
      --class-loss-um cross_ent
      ```
  - **Prototype-related losses** (cluster, separation, mapping, Online-CAM, diversity, L1)  

    These are controlled via the `--coefs` argument, using a dictionary:
    - `cls`  → classification loss coefficient  
    - `clst` → cluster loss coefficient  
    - `sep`  → separation loss coefficient  
    - `L1`   → L1 regularisation of the last layer  
    - `map`  → mapping loss coefficient  
    - `OC`   → Online-CAM loss coefficient  
    - `div`  → prototype diversity loss coefficient  
    Default (used in the paper):
    ```bash
    --coefs "{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05, 'div': 0.01}"
    ```
- **Prototype mode (`--p-mode`)**

  The `--p-mode` flag switches between different prototype model variants:
  - `--p-mode 1` – **ProtoPNet-style**
    - Class-specific prototypes.
    - No mapping loss (`map ≈ 0`), no Online-CAM (`OC ≈ 0`).
    - Use this to mimic classic ProtoPNet in 3D.
  - `--p-mode 2` – **XProtoNet-style**
    - Adds a **mapping module** that predicts prototype attention maps.
    - Mapping loss on (`map > 0`), Online-CAM off (`OC ≈ 0`).
  - `--p-mode 3` – **MProtoNet**
    - Mapping module + mapping loss (`map > 0`).
    - Online-CAM loss enabled (`OC > 0`), but **class-specific** prototypes.
    - Use this to reproduce the original MProtoNet-style behaviour.
  - `--p-mode 4` – **UM-ProtoShare without Online-CAM**
    - Multi-scale **shared** prototypes.
    - UNet-guided features with encoder–decoder fusion.
    - Mapping loss on (`map > 0`), **Online-CAM loss off** (`OC ≈ 0`).
    - Use this as an ablation of UM-ProtoShare **without** Online-CAM regularisation.
  - `--p-mode 5` – **UM-ProtoShare with Online-CAM (full model)**
    - Multi-scale **shared** prototypes.
    - UNet-guided features with encoder–decoder fusion.
    - Mapping loss + **Online-CAM loss** + diversity loss.
    - This is the **main configuration** used in the UM-ProtoShare paper.

- **Prototype scale allocation (multi-scale prototypes)**

  For UM-ProtoShare (`--p-mode 4` or `--p-mode 5`), the total number of prototypes  
  `--num-prototypes K` is automatically split across **three spatial scales**:
  - scale 3 (coarsest feature map) → 50% of K  
  - scale 2 (mid-level feature map) → 30% of K  
  - scale 1 (finest feature map) → 20% of K
  
  Changing `--num-prototypes` on the command line changes the total number of prototypes K, and the model will automatically recompute the per-scale counts using the 50/30/20 split.
  If you want to use a different per-scale allocation (e.g. 40/40/20 or 33/33/34), please edit the multipliers in `models.py`.


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


