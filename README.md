UM-ProtoShare/
│
├─ models/
│   ├─ um_protoshare.py        # UM-ProtoShare model (encoder, decoder, gated fusions, prototypes)
│   └─ resnet3d.py             # 3D ResNet-152 backbone
│
├─ scripts/
│   ├─ train_um_protoshare.py  # main training + cross-validation script
│   ├─ eval_um_protoshare.py   # evaluation / metrics on held-out data
│   └─ visualize_prototypes.py # prototype and activation-map visualisation
│
├─ utils/
│   ├─ data.py                 # load_data, dataset construction, transforms
│   ├─ losses.py               # Focal loss, clustering/separation, mapping, Online-CAM, diversity
│   ├─ metrics.py              # BAC, AP, IDS, confusion matrix helpers
│   ├─ cv.py                   # RepeatedStratifiedKFold helpers
│   └─ viz.py                  # plotting utilities
│
├─ configs/
│   └─ um_protoshare_brats2020.yaml  # hyper-parameters and paths
│
├─ data/
│   └─ name_mapping.csv        # subject IDs and grades (not tracked; user provided)
│
├─ Model_Checkpoints/          # saved models / best checkpoints (gitignored)
│
├─ Prototype_Images/           # prototype patches and activation maps (gitignored)
│
├─ README.md                   # this file
└─ LICENSE
