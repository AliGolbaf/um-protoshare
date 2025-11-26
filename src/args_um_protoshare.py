#!/usr/bin/env python3
"""
The code for:
    args
Author: Ali Golbaf
"""
################################################################################
################################################################################
""" Libraries """
import argparse

def parse_arguments():
    """
    Parse command-line arguments for UM-ProtoShare main runner.
    The defaults mirror the current values in main_Code.py.
    """
    parser = argparse.ArgumentParser(
        description="UM-ProtoShare: UNet-guided, multi-scale shared prototypes for 3D MRI glioma grading.")

    # -------------------------------------------------------------------------
    # General / paths
    # -------------------------------------------------------------------------
    parser.add_argument("-m", "--model-name",
                        type=str,
                        choices={"um_protoshare", "backbone", "both"},
                        default="um_protoshare",
                        help="Which part to train: 'um_protoshare', 'backbone', or 'both'.")

    # Path to images
    parser.add_argument("-di", "--data-images",
                        type=str,
                        default='/home/agolbaf/Desktop/Code/Prototype_ACE_TCAV/Datasets/Glioma/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
                        help="Path to the BraTS2020 image data (directory with BraTS20_Training_xxx folders)."
                        )

    # Path to data clinical
    parser.add_argument("-dc", "--data-clinical",
                        type=str,
                        default='/home/agolbaf/Desktop/Code/Prototype_ACE_TCAV/Datasets/Glioma/BraTS2020_TrainingData',
                        help="Path to the directory containing the clinical CSV file."
                        )

    # clinical csv name
    parser.add_argument("--clinical-csv",
                        type=str,
                        default="name_mapping.csv",
                        help="Filename of the clinical CSV (e.g. name_mapping.csv).")
    # -------------------------------------------------------------------------
    # Preprocessing & augmentation
    # -------------------------------------------------------------------------
    parser.add_argument("--use-augmentation",
                        type=int,
                        choices={0, 1},
                        default=1,
                        help="Whether to apply data augmentation during training (1) or not (0).")

    parser.add_argument("--target-label-aug",
                        type=int,
                        default=0,
                        help="Target label index for augmentation, if used (default: 0).")
    # -------------------------------------------------------------------------
    # P_mode
    # -------------------------------------------------------------------------
    parser.add_argument("--p-mode",
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=5,
                        help="Prototype mode: 1=ProtoPNet, 2=ProtoPNet+top-k, 3=MProtoNet+p_map, 4/5=UM-ProtoShare variants.")
    
    # -------------------------------------------------------------------------
    # Unet/ Gated fusions
    # -------------------------------------------------------------------------
    parser.add_argument("--use-unet",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Whether to use the UNet-style decoder (1) or not (0).")

    parser.add_argument("--freeze-unet",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Whether to freeze the UNet decoder weights (1) or train them (0).")

    parser.add_argument("--fusion",
                        type=str,
                        choices={"gated", "concat"},
                        default="gated",
                        help='Encoder–decoder fusion mode: "gated" (default) or "concat".')
    
    # -------------------------------------------------------------------------
    # Architecture & Backbone
    # -------------------------------------------------------------------------
    parser.add_argument("--backbone",
                        type=str,
                        default="resnet152_ri",
                        help="3D backbone name (default: resnet152_ri).")

    parser.add_argument("--n-layers",
                        type=int,
                        default=6,
                        help="Number of backbone layers/stages to use (default: 6).")

    # -------------------------------------------------------------------------
    # Transfer learning
    # -------------------------------------------------------------------------
    parser.add_argument("--transfer-learning",
                        type=int,
                        choices={1, 0},
                        default=1,
                        help="Whether to use transfer learning / pretrained backbone initialisation (1) or not (0).")

    # -------------------------------------------------------------------------
    # Backbone training hyperparameters (ResNet_3D)
    # -------------------------------------------------------------------------
    parser.add_argument("--epochs-bb",
                        type=int,
                        default=50,
                        help="Number of backbone training epochs (default: 50).")

    parser.add_argument("--batch-size-bb",
                        type=int,
                        default=1,
                        help="Backbone batch size (default: 1).")

    parser.add_argument("--num-workers-bb",
                        type=int,
                        default=1,
                        help="Number of DataLoader workers for backbone training (default: 1).")

    parser.add_argument("--lr-bb",
                        type=float,
                        default=1e-3,
                        help="Learning rate for backbone (default: 1e-3).")

    parser.add_argument("--wd-bb",
                        type=float,
                        default=1e-2,
                        help="Weight decay for backbone (default: 1e-2).")

    parser.add_argument("--class-loss-bb",
                        type=str,
                        choices={"focal", "cross_ent"},
                        default="focal",
                        help='Backbone classification loss: "focal" (default) or "cross_ent".')

    parser.add_argument("--optim-bb",
                        type=str,
                        choices={"Adam", "AdamW", "SGD"},
                        default="Adam",
                        help='Backbone optimizer: "Adam" (default), "AdamW", or "SGD".')
        
    # -------------------------------------------------------------------------
    # Cross validation
    # -------------------------------------------------------------------------
    parser.add_argument("-s", "--seed",
                        type=int,
                        default=0,
                        help="Random seed for cross-validation and training.")

    parser.add_argument("--cv-folds",
                        type=int,
                        default=5,
                        help="Number of cross-validation folds (default: 5).")

    parser.add_argument("--cv-repeats",
                        type=int,
                        default=1,
                        help="Number of CV repetitions (default: 1).")
    
    # -------------------------------------------------------------------------
    # UM-ProtoShare 
    # -------------------------------------------------------------------------
    parser.add_argument("--num-prototypes",
                        type=int,
                        default=30,
                        help="Total number of prototypes K (default: 30).")

    parser.add_argument("--f-dist",
                        type=str,
                        choices={"cos", "l2"},
                        default="cos",
                        help="Distance function for prototypes (default: cosine).")

    parser.add_argument("--topk-p",
                        type=int,
                        default=1,
                        help="Top-k parameter for ProtoPNet-style pooling (default: 1).")

    parser.add_argument("--prototype-act",
                        type=str,
                        choices={"log", "linear"},
                        default="log",
                        help="Prototype activation function (default: log).")
    # -------------------------------------------------------------------------
    # If pretrained backbone
    # -------------------------------------------------------------------------  
    parser.add_argument("--train-backbone",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Whether to train the ResNet-3D backbone (1) or skip it (0).")

    parser.add_argument("--add-pretrained-backbone",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Whether to load a pretrained backbone checkpoint into UM-ProtoShare (1) or not (0).")

    parser.add_argument("--pretrained-backbone-path",
                        type=str,
                        default=None,
                        help="Path to a pretrained backbone checkpoint to load (used if --add-pretrained-backbone=1).")

    # -------------------------------------------------------------------------
    # UM-ProtoShare training hyperparameters
    # -------------------------------------------------------------------------
    parser.add_argument("--train-um",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Whether to train UM-ProtoShare (1) or skip it (0).")

    parser.add_argument("-n", "--epochs-um",
                        type=int,
                        default=100,
                        help="Number of UM-ProtoShare training epochs (default: 100).")

    parser.add_argument("--batch-size-um",
                        type=int,
                        default=1,
                        help="UM-ProtoShare batch size (default: 1).")

    parser.add_argument("--num-workers-um",
                        type=int,
                        default=1,
                        help="Number of DataLoader workers for UM-ProtoShare (default: 1).")

    parser.add_argument("--lr-um",
                        type=float,
                        default=1e-3,
                        help="Learning rate for UM-ProtoShare (default: 1e-3).")

    parser.add_argument("--wd-um",
                        type=float,
                        default=1e-2,
                        help="Weight decay for UM-ProtoShare (default: 1e-2).")

    parser.add_argument("--class-loss-um",
                        type=str,
                        choices={"focal", "cross_ent"},
                        default="focal",
                        help='UM-ProtoShare classification loss: "focal" (default) or "cross_ent".')

    parser.add_argument("--optim-um",
                        type=str,
                        choices={"Adam", "AdamW", "SGD"},
                        default="AdamW",
                        help='UM-ProtoShare optimizer: "AdamW" (default), "Adam", or "SGD".')

    parser.add_argument("--warmup",
                        type=int,
                        choices={1, 0},
                        default=1,
                        help="Whether to use warm-up before cosine annealing LR (1) or not (0).")

    parser.add_argument("--warmup-ratio",
                        type=float,
                        default=0.2,
                        help="Fraction of epochs used for warm-up in the LR scheduler (default: 0.2).")

    parser.add_argument("--class-assignments",
                        type=str,
                        choices={"soft", "hard"},
                        default="soft",
                        help="soft and hard class assignment")
    
    parser.add_argument("--save-model",
                        type=int,
                        choices={0, 1},
                        default=1,
                        help="Whether to save model checkpoints and prototype images (1) or not (0).")
    
    
    parser.add_argument("--coefs",
                        type=str,
                        default="{'cls': 1.0, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05, 'div': 0.01}",
                        help=(
                            "Loss coefficients in MProtoNet style dict, e.g. "
                        "\"{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05, 'div': 0.01}\""),)
    
    
    # -------------------------------------------------------------------------
    # Testing
    # -------------------------------------------------------------------------
    parser.add_argument("--test-model",
                        type=int,
                        choices={1, 0},
                        default=0,
                        help="Run in testing/evaluation mode only (no training).")
    
    parser.add_argument("--attr-methods",
                        type=str,
                        default="MGU",
                        help=("Attribution methods to use, as a string of characters from {M,D,G,U,O}. "
                              "M=MProtoNet, D=Deconvolution, G=GradCAM, U=Guided GradCAM, O=Occlusion."),)
    
    
    return parser.parse_args()