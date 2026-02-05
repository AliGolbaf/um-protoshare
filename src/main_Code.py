#!/usr/bin/env python3

"""
    UM-ProtoShare
     Written by: Ali Golbaf (ali.golbaf71@gmail.com)
"""
################################################################################
################################################################################
""" Libraries """
import os
import time
from functools import partial
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from tqdm import tqdm
from torchmetrics.classification import (MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, 
                                         MulticlassRecall, MulticlassConfusionMatrix )
################################################################################
################################################################################
""" Classes """
# Models / Train / Push / Interpret / Data / Utils
from args_um_protoshare import parse_arguments
import models
from models import ResNet_3D, UM_Proto_Share
from train import train_CNN_3D, train_one_epoch_UM_ProtoShare
import push as Push_MProto
import push_um_protoshare as Push_UM_ProtoShare
from interpret import attr_methods, attribute
from data_loader import SubjectsDataset
from utils import makedir, read_csv_exl, missing_modalities, load_data, preprocessing, get_loss_coefs, loc_coherence, inc_add_del
from utils import FocalLoss, warmup_cosine_scheduler
from utils import process_iad, print_results 

if __name__ == '__main__':
 ###############################################################################
    """ Args """
    args = parse_arguments()
    ################################################################################
    ################################################################################
    """ Device """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ################################################################################
    """ Paths """
    path_to_data_images = args.data_images
    files = os.listdir(path_to_data_images)
    print(path_to_data_images)
    path_to_data_clinical = args.data_clinical

    ################################################################################
    """ Clinical Data """
    data_clinical_name = args.clinical_csv
    data_clinical = read_csv_exl(path_to_data_clinical, data_clinical_name)

    subjects = data_clinical["BraTS_2020_subject_ID"].values

    glioma_grades = data_clinical["Grade"]
    # Replace ["LGG", "HGG"] with [0,1]
    glioma_grades = np.astype(glioma_grades.replace(["LGG", "HGG"], [0,1]).values, "float32")

    # MRI modalities
    modalities = ['t1', 't1ce', 't2', 'flair']

    ################################################################################
    """ Data Structuring """

    # Check for missing modalities
    missing_modalities = missing_modalities(path_to_data_images , subjects)
    assert missing_modalities != 0  

    # Data Loading on tio
    data = load_data(path_to_data_images, subjects, glioma_grades)

    # Preprocessing
    preprocessing_args = ["crop", 
                          "resample", 
                          "z_normalise"]

    # Augmentation
    target_label = args.target_label_aug
    augmentation_args = ["random_affine_0", 
                         "random_noise", 
                         "random_blur", 
                         "augment_brightness", 
                         "augment_contrast", 
                         "random_anisotropy",
                         "augment_gamma",
                         "random_flip_0"]

    ################################################################################
    ################################################################################
    """ Main Body """
    # Shared variables -------------------------------------------------------------
    # Input details ----------------------------------------------------------------
    input_size = (4, 128, 128, 96)
    # Output details ---------------------------------------------------------------
    output_size = num_classes = 2
    # P_mode -----------------------------------------------------------------------
    # Explanations:
    # p_mode = 1   : ProtoPnet
    # p_mode = 2   : XProtoNet
    # p_mode = 3   : MProtoNet
    # p_mode = 4   :  UM-ProtoShare without Online-CAM
    # p_mode = 5   :  UM-ProtoShare with Online-CAM (full model)
    p_mode = args.p_mode

    # U-NET ------------------------------------------------------------------------
    use_unet = args.use_unet
    freeze_unet = args.freeze_unet

    # Fusion mode ------------------------------------------------------------------
    fusion = args.fusion

    ################################################################################
    """ Backbone """
    # Backbone details -------------------------------------------------------------
    backbone   = args.backbone
    number_of_layers = args.n_layers

    # Transfer_learning ------------------------------------------------------------
    transfer_learning = args.transfer_learning

    # Model ------------------------------------------------------------------------
    model_bb = ResNet_3D(in_size = input_size, 
                               num_classes = output_size, 
                               backbone = backbone, 
                               n_layers = number_of_layers,
                               p_mode = p_mode,
                               backbone_ch = [64,256,512], 
                               use_unet= use_unet,
                               freeze_unet= freeze_unet,
                               fusion= fusion,
                               transfer_learning= transfer_learning).to(device)
       
    # Model summary
    model_summary = summary(model = model_bb, 
                               input_size=(1, 4, 128, 128,96), 
                               verbose = 1)
    # Training variables -----------------------------------------------------------
    train_backbone = args.train_backbone
    num_workers_bb = args.num_workers_bb
    batch_size_bb =  args.batch_size_bb # (num_workers_bb + 1) // 2
    number_of_epochs_bb = args.epochs_bb

    class_weights_bb = torch.tensor([(293)/(293+76), (76)/(293+76)]).to(device)
    learning_rate_bb = args.lr_bb
    weight_decay_bb = args.wd_bb  

    class_loss_bb = args.class_loss_bb

    if class_loss_bb == "focal":
        if class_weights_bb is not None:
            loss_class_bb = FocalLoss(weight=class_weights_bb).to(device)    
    if class_loss_bb == "cross_ent":  
        if class_weights_bb is not None:
            loss_class_bb = torch.nn.CrossEntropyLoss(weight= class_weights_bb)
    else:
        loss_class_bb = torch.nn.CrossEntropyLoss()

    optimiser_bb = args.optim_bb
    if optimiser_bb == "AdamW" and weight_decay_bb:
        optimiser_bb = torch.optim.AdamW(model_bb.parameters(), lr= learning_rate_bb, weight_decay= weight_decay_bb)

    if optimiser_bb == "AdamW" and weight_decay_bb is None:
        optimiser_bb = torch.optim.AdamW(model_bb.parameters(), lr= learning_rate_bb)

    if optimiser_bb == "Adam" and weight_decay_bb:
        optimiser_bb = torch.optim.Adam(model_bb.parameters(), lr= learning_rate_bb, weight_decay= weight_decay_bb)

    if optimiser_bb == "Adam" and weight_decay_bb is None:
        optimiser_bb = torch.optim.Adam(model_bb.parameters(), lr= learning_rate_bb)
        
    if optimiser_bb == "SGD" and weight_decay_bb:
        optimiser_bb = torch.optim.SGD(model_bb.parameters(), lr= learning_rate_bb, weight_decay= weight_decay_bb, momentum = 0.9)
        
    elif optimiser_bb == "SGD" and weight_decay_bb is None:
        optimiser_bb = torch.optim.SGD(model_bb.parameters(), lr= learning_rate_bb, momentum = 0.9)

    else:
        assert "optimiser is not defined"

    metric_acc = MulticlassAccuracy(num_classes=num_classes)
    metric_acc.to(device)
    metric_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
    metric_f1.to(device)
    metric_presicion = MulticlassPrecision(num_classes = num_classes, average=None)
    metric_presicion.to(device)
    metric_recall = MulticlassRecall(num_classes = num_classes, average=None)
    metric_recall.to(device)
    confusion_mat = MulticlassConfusionMatrix(num_classes = num_classes)
    confusion_mat.to(device)

    # Training of Backbone ---------------------------------------------------------
    # Train/ test split
    # 5-fold CV
    random_seed = args.seed
    n_splits = args.cv_folds
    n_repeats = args.cv_repeats
    # 5-fold CV, repeated once (set n_repeats>1 if you want multiple runs)
    cv = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=random_seed)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(subjects, glioma_grades)):

        train_idx = np.array(train_idx)
        val_idx   = np.array(val_idx)
        
        # ---- Data per fold ----
        data_train = [data[i] for i in train_idx]
        data_val   = [data[i] for i in val_idx]

        # Custom Dataset -----------------------------------------------------------
        data_train = preprocessing(data_train, preprocessing_args)
        data_val   = preprocessing(data_val, preprocessing_args)

        # Number of classes: Used for one-hot    
        dataset_train = SubjectsDataset(data_train, 
                                        num_classes = num_classes,
                                        augment = augmentation_args,
                                        target_label_aug = target_label)
        
        dataset_val = SubjectsDataset(data_val,
                                      num_classes = num_classes,
                                      augment = None,
                                      target_label_aug = target_label)
        
        # Delete data_train and data_val
        del data_train, data_val

        # Data loaders
        data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                        batch_size =  batch_size_bb, 
                                                        num_workers = num_workers_bb)
        
        data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                      batch_size =  batch_size_bb, 
                                                      num_workers = num_workers_bb)
        
        # Delete data_train and data_val
        del dataset_train, dataset_val

        if train_backbone:
            train_CNN_3D(model = model_bb,
                         data_loader_train = data_loader_train,
                         data_loader_val = data_loader_val,
                         number_of_epochs = number_of_epochs_bb,
                         num_classes = output_size,
                         class_weights = class_weights_bb,
                         loss = loss_class_bb,
                         learning_rate = learning_rate_bb,
                         weight_decay = weight_decay_bb,
                         optimiser = optimiser_bb,
                         metric_acc = metric_acc,
                         metric_f1 = metric_f1,
                         metric_presicion = metric_presicion,
                         metric_recall = metric_recall,
                         confusion_mat = confusion_mat,
                         lr_scheduler = True)
        
        del data_loader_train, data_loader_val

    del model_bb
    
    ################################################################################
    """ UM-ProtoShare """
    # Details ----------------------------------------------------------------------
    backbone = args.backbone
    number_of_layers = args.n_layers
    # Prototypes shape -------------------------------------------------------------
    prototype_shape = (30, 128, 1, 1, 1)
    num_prototypes = prototype_shape[0]
    # Distance functions -----------------------------------------------------------
    f_dist = args.f_dist
    # Top_k: ProtoPnet -------------------------------------------------------------
    topk_p = args.topk_p
    # Prototype activation function: for converting distance to similarity----------
    prototype_activation_function = args.prototype_act

    # Model UM-ProtoShare-----------------------------------------------------------
    model_um = UM_Proto_Share(in_size = input_size, 
                              num_classes  = num_classes, 
                              backbone  = backbone, 
                              n_layers  = number_of_layers,
                              backbone_ch = [64,256,512],
                              num_prototypes = num_prototypes, 
                              init_weights= True, 
                              f_dist = f_dist,
                              prototype_activation_function= prototype_activation_function, 
                              p_mode = p_mode, 
                              topk_p = topk_p, 
                              use_unet= use_unet,
                              freeze_unet= True,
                              fusion = fusion)    
        
    # Model Summary
    model_summary = summary(model = model_um, 
                            input_size=(1, 4, 128, 128,96), 
                            verbose = 1)
    # Add pre-trained model -------------------------------------------------------
    add_pretrained_backbone = args.add_pretrained_backbone
    if add_pretrained_backbone:
        path_to_backbone = args.pretrained_backbone_path
        model_um.load_state_dict(torch.load(path_to_backbone, map_location=device), strict= False) 

    # Training variables -----------------------------------------------------------
    train_um_protoshare = args.train_um
    num_workers_um = args.num_workers_um
    batch_size_um =  args.batch_size_um
    number_of_epochs_um = args.epochs_um

    class_weights_um = torch.tensor([(293)/(293+76), (76)/(293+76)]).to(device)
    learning_rate_um = args.lr_um
    weight_decay_um = args.wd_um

    class_loss_um = args.class_loss_um
    if class_loss_um == "focal":
        if class_weights_um is not None:
            loss_class_um = FocalLoss(weight=class_weights_um).to(device)    
    if class_loss_um == "cross_ent":  
        if class_weights_bb is not None:
            loss_class_um = torch.nn.CrossEntropyLoss(weight= class_weights_um)
    else:
        loss_class_um = torch.nn.CrossEntropyLoss()

    optimiser_um = args.optim_um
    if optimiser_um == "AdamW" and weight_decay_um:
        optimiser_um = torch.optim.AdamW(model_um.parameters(), lr= learning_rate_um, weight_decay= weight_decay_um)
        optimiser_last_layer = torch.optim.Adam(model_um.last_layer.parameters(), lr= learning_rate_um)

    if optimiser_um == "AdamW" and weight_decay_um is None:
        optimiser_um = torch.optim.AdamW(model_um.parameters(), lr= learning_rate_um)
        optimiser_last_layer = torch.optim.Adam(model_um.last_layer.parameters(), lr= learning_rate_um)

    if optimiser_um == "Adam" and weight_decay_um:
        optimiser_um = torch.optim.Adam(model_um.parameters(), lr= learning_rate_um, weight_decay= weight_decay_um)
        optimiser_last_layer = torch.optim.Adam(model_um.last_layer.parameters(), lr= learning_rate_um)
    
    if optimiser_um == "Adam" and weight_decay_um is None:
        optimiser_um = torch.optim.Adam(model_um.parameters(), lr= learning_rate_um)
        optimiser_last_layer = torch.optim.Adam(model_um.last_layer.parameters(), lr= learning_rate_um)

    if optimiser_um == "SGD" and weight_decay_um:
        optimiser_um = torch.optim.SGD(model_um.parameters(), lr= learning_rate_um, weight_decay= weight_decay_um, momentum = 0.9)
        optimiser_last_layer = torch.optim.SGD(model_um.last_layer.parameters(), lr= learning_rate_um, weight_decay= weight_decay_um, momentum = 0.9)

    elif optimiser_um == "SGD" and weight_decay_um is None:
        optimiser_um = torch.optim.SGD(model_um.parameters(), lr= learning_rate_um, momentum = 0.9)
        optimiser_last_layer = torch.optim.SGD(model_um.last_layer.parameters(), lr= learning_rate_um, momentum = 0.9)

    else:
        assert "optimiser is not defined"

    warm_up = args.warmup
    warmup_ratio = args.warmup_ratio
    scheduler = warmup_cosine_scheduler(optimiser= optimiser_um,
                                        num_epochs=number_of_epochs_um,
                                        use_warmup = warm_up,
                                        warmup_ratio = warmup_ratio)
        
    scaler = torch.cuda.amp.GradScaler(enabled=False) 
    
    # Loss coefs ---------------------------------------------------------------
    loss_coefs = get_loss_coefs(args)
        
    ## TODO:  Whether to use deterministic algorithms
    determin_algth = True # To be implemented

    save_model = args.save_model

    # For saving prototypes
    path_to_prototype_iamges = os.path.join(os.getcwd(), "prototype_images")
    makedir(path_to_prototype_iamges)
    prototype_img_filename_prefix = 'prototype-img'        
    prototype_self_act_filename_prefix = 'prototype-self-act'       
    proto_bound_boxes_filename_prefix = 'bb'

    # Training of UM-ProtoShare -----------------------------------------------------
    # Train/ test split
    # 5-fold CV
    random_seed = args.seed
    n_splits = args.cv_folds
    n_repeats = args.cv_repeats
    # 5-fold CV, repeated once (set n_repeats>1 if you want multiple runs)
    cv = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=random_seed)
    
    class_assignments = args.class_assignments
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(subjects, glioma_grades)):

        train_idx = np.array(train_idx)
        val_idx   = np.array(val_idx)
        
        # ---- Data per fold ----
        data_train = [data[i] for i in train_idx]
        data_val   = [data[i] for i in val_idx]

        # Custom Dataset -----------------------------------------------------------
        data_train = preprocessing(data_train, preprocessing_args)
        data_val   = preprocessing(data_val, preprocessing_args)

        # Number of classes: Used for one-hot    
        dataset_train = SubjectsDataset(data_train, 
                                        num_classes = num_classes,
                                        augment = augmentation_args,
                                        target_label_aug = target_label)
        
        dataset_push = SubjectsDataset(data_train, 
                                        num_classes = num_classes,
                                        augment = None, #augmentation_args,
                                        target_label_aug = target_label)
        
        dataset_val = SubjectsDataset(data_val,
                                      num_classes = num_classes,
                                      augment = None,
                                      target_label_aug = target_label)
        
        # Delete data_train and data_val
        del data_train, data_val

        # Data loaders
        data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                        batch_size =  batch_size_um, 
                                                        num_workers = num_workers_um)
        
        data_loader_push = torch.utils.data.DataLoader(dataset_push, 
                                                       batch_size =  batch_size_um, 
                                                       num_workers = num_workers_um)
        
        data_loader_val = torch.utils.data.DataLoader(dataset_val, 
                                                      batch_size =  batch_size_um, 
                                                      num_workers = num_workers_um)
        
        # Delete data_train and data_val
        del dataset_train, dataset_val
        
        if train_um_protoshare:
            for epoch_number in range(number_of_epochs_um):
                print('EPOCH {}:'.format(epoch_number + 1))
                stage = "joint"
                
                    
                train_one_epoch_UM_ProtoShare(stage,
                                              model_um, 
                                              data_loader_train,
                                              optimiser_um,
                                              class_weights_um,
                                              loss_class_um,
                                              loss_coefs,
                                              use_l1_mask = True,
                                              scaler = scaler,
                                              lr_scheduler = scheduler,
                                              class_assignments= class_assignments)
                
                if (epoch_number + 1) >= 10 and (epoch_number + 1) in [i for i in range(number_of_epochs_um + 1) if i % 10 == 0]:
                    stage = ""
                    if model_um.p_mode>=4:
                        Push_UM_ProtoShare.push_prototypes_u_protoshare_ms(data_loader_push,
                                                              model_um,
                                                              root_dir_for_saving= path_to_prototype_iamges,
                                                              epoch_number = epoch_number,
                                                              prototype_img_filename_prefix = "UM_ProtoShare_")
                        
                    elif model_um.p_mode <4:    
                        Push_MProto.push_prototypes(data_loader_push,
                                            model_um,
                                            root_dir_for_saving_prototypes= path_to_prototype_iamges,
                                            prototype_img_filename_prefix=prototype_img_filename_prefix,
                                            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix)
                    
                    for j in range(10):
                        stage = ""
                        train_one_epoch_UM_ProtoShare(stage,
                                                      model_um, 
                                                      data_loader_train,
                                                      optimiser_last_layer,
                                                      class_weights_um,
                                                      loss_class_um,
                                                      loss_coefs,
                                                      use_l1_mask = True,
                                                      scaler = scaler,
                                                      lr_scheduler = scheduler,
                                                      class_assignments= class_assignments)
                if save_model:
                    model_path = 'model_checkpoints/UM_ProtoShare_epoch_{}'.format(epoch_number)
                    torch.save(model_um.state_dict(), model_path)
                    
        del data_loader_train, data_loader_push

    # #############################################################################
    # # Phase 3:  Testing ----------------------------------------------------------
    # test_model = args.test_model # 'U' -> Guided Grad-CAM in attr_methods
    # methods = args.attr_methods
    
    # model_um.eval()
    
    # f_x, y_true = [], []     # predictions and one-hot labels
    # m_f_xes, lcs, iads = {}, {}, {}

    # # Disable gradient computation and reduce memory consumption.
    # with torch.no_grad():
    #     for batch_index, (subject_id, image, label, seg) in enumerate(tqdm(data_loader_val)):
    #         # Data
    #         image = image.to(device)
    #         label = label.to(device)          # one-hot [B, num_classes]
    #         label_target = label.argmax(dim=1)
    #         seg = seg.to(device)

    #         # Store classification probabilities and labels
    #         logits = model_um(image)
    #         f_x.append(F.softmax(logits, dim=1).cpu().numpy())
    #         y_true.append(label.cpu().numpy())

    #         # Attribution-based metrics
    #         for method_i in methods:
    #             method = attr_methods[method_i]

    #             # Initialise dicts once per method
    #             if method not in lcs:
    #                 lcs[method] = {
    #                     f'(WT, Th=0.5) {m}': [] for m in ['AP', 'DSC']
    #                 }
    #             if method not in iads:
    #                 iads[method] = {m: [] for m in ['IA', 'ID', 'IAD']}

    #             # Attribution map
    #             attr = attribute(model_um, image, label_target, device, method)

    #             # Localisation coherence (whole tumour, annos=[1,2,4])
    #             lcs[method]['(WT, Th=0.5) AP'].append(
    #                 loc_coherence(attr, seg, annos=[1, 2, 4],
    #                               threshold=0.5, metric='AP'))
    #             lcs[method]['(WT, Th=0.5) DSC'].append(
    #                 loc_coherence(attr, seg, annos=[1, 2, 4],
    #                               threshold=0.5, metric='DSC'))

    #             # Incremental addition / deletion curves
    #             iads[method]['IA'].append(
    #                 inc_add_del(model_um, image, attr,
    #                             n_intervals=50, quantile=True, addition=True))
    #             iads[method]['ID'].append(
    #                 inc_add_del(model_um, image, attr,
    #                             n_intervals=50, quantile=True, addition=False))

    # # Convert lists to numpy arrays --------------------------------------------
    # f_x = np.vstack(f_x)
    # y_true = np.vstack(y_true)

    # for method, lcs_ in lcs.items():
    #     for metric, lcs_list in lcs_.items():
    #         lcs[method][metric] = np.vstack(lcs_list)   # shape [N, 4] as in MProtoNet code

    # for method, iads_ in iads.items():
    #     for metric, iads_list in iads_.items():
    #         if metric == 'IAD':
    #             continue
    #         # inc_add_del returns [N, n_intervals] per batch; concat on axis=1
    #         iads[method][metric] = np.concatenate(iads_list, axis=1)

    # # Process IAD curves and print summary results -----------------------------

    # # Compute IAD summary scores + save IA/ID curves
    # process_iad(iads, y_true, save_plot=True, model_name="UM_ProtoShare")

    # # Nicely formatted classification + localisation + IAD metrics
    # print_results(
    #     dataset="Validation",
    #     f_x=f_x,
    #     y=y_true,
    #     m_f_xes=None,        # no missing-modality experiments here
    #     lcs=lcs,
    #     n_prototypes=None,   # can be filled if you want (e.g. 30)
    #     iads=iads,
    #     splits=None          # no CV splits here 
    #     )    
    

        
