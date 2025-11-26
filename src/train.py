#!/usr/bin/env python3
"""
    Train models
    Written by: Ali Golbaf (ali.golbaf71@gmail.com)
"""
################################################################################
################################################################################
""" Libraries """
import numpy as np
import math
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torchio as tio
import torch.nn.functional as F
from torchmetrics.classification import (MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, 
                                         MulticlassRecall, MulticlassConfusionMatrix )

from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             log_loss, recall_score, roc_auc_score)


from typing import Dict, List, Optional, Tuple

# Gpu or Cpu
device = "cuda" if torch.cuda.is_available() else "cpu"


################################################################################
         
# The Training Loop
def train_one_epoch(model, 
                    num_classes,
                    train_loader,
                    optimiser,
                    metric_acc,
                    metric_f1,
                    metric_presicion,
                    metric_recall,
                    confusion_mat,
                    loss_,
                    lr_scheduler,
                    epoch_index, 
                    tb_writer):
    
    running_loss = 0
    last_loss = 0
    runnung_accuracy = 0
    runnung_f1_score = torch.zeros(num_classes).to(device)
    runnung_precision = torch.zeros(num_classes).to(device)
    runnung_recall = torch.zeros(num_classes).to(device)
    
    # For confusion matrix
    label_target_confusion = []
    label_predicted_confusion = []
    
    for batch_index, (subject_id, image, label, seg) in enumerate(tqdm(train_loader)):
                
        # Import image and label
        image = image.to(device)
        label = label.to(device)
        label_target = label.argmax(dim = 1)        
        # Zero your gradients for every batch!
        optimiser.zero_grad()
        
        # Make predictions for this batch
        label_predicted_logit = model(image)
        label_predicted_proba = torch.softmax(label_predicted_logit, dim = 1)  
        label_predicted       = label_predicted_proba.argmax(dim = 1)
        
        # Compute accuracy
        accuracy = metric_acc(preds = label_predicted, target = label_target)
        
        # Compute f1_score
        f1_score = metric_f1(preds = label_predicted, target = label_target)
    
        # Compute presicion
        precision = metric_presicion(preds = label_predicted, target = label_target)
        
        # Compute recall
        recall = metric_recall(preds = label_predicted, target = label_target)
        
        # Confusion matrix
        label_target_confusion.extend(label_target.data)
        label_predicted_confusion.extend(label_predicted.data)
        
        
        # Compute the loss and its gradients
        loss = loss_(label_predicted_logit, label_target)

        loss.backward()
                
        # Adjust learning weights
        optimiser.step()
        
        # Gather data and report
        running_loss += loss.item()
        runnung_accuracy += accuracy.item()
        runnung_f1_score += f1_score
        runnung_precision += precision
        runnung_recall += recall
        
        #
        divide  = len(train_loader)//3
        if batch_index % divide == (divide - 1):
            
            last_loss = running_loss / divide # loss per batch
            last_accuracy = runnung_accuracy /divide  # accuracy per batch
            last_f1_score  = runnung_f1_score/ divide
            last_presicion = runnung_precision/ divide
            last_recall = runnung_recall / divide
            
            print("\n")
            print('  batch {}'.format(batch_index + 1))
            print('  loss: {}'.format(last_loss))
            print('  accuracy: {}'.format(last_accuracy))
            print('  f1_score: {}'.format(last_f1_score.data))
            print('  presicion: {}'.format(last_presicion.data))
            print('  recall: {}'.format(last_recall.data))
            print("\n")
            
            tb_x = epoch_index * batch_index + 1
            
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0
            runnung_accuracy = 0
            runnung_f1_score = torch.zeros(num_classes).to(device)
            runnung_precision = torch.zeros(num_classes).to(device)
            runnung_recall = torch.zeros(num_classes).to(device)
            
    # Optimise lr after each epoch        
    lr_scheduler.step()    
    
    # For confusion matrix
    target_confusion = torch.tensor(label_target_confusion).to(device)
    predicted_confusion = torch.tensor(label_predicted_confusion).to(device)
    conf_mat = confusion_mat(preds = predicted_confusion, target = target_confusion)
        
    return last_loss, last_accuracy, last_f1_score, last_presicion, last_recall, conf_mat

################################################################################
################################################################################
""" Main body: train the backbone CNN """

def train_CNN_3D(
                 model = None,
                 data_loader_train = None,
                 data_loader_val = None,
                 number_of_epochs = None,
                 num_classes = None,
                 class_weights = None,
                 loss = None,
                 learning_rate = None,
                 weight_decay = None,
                 optimiser = None,
                 metric_acc = None,
                 metric_f1 = None,
                 metric_presicion = None,
                 metric_recall = None,
                 confusion_mat = None,
                 lr_scheduler = None,
                 ):
    # Loss ---------------------------------------------------------------------
    if loss is not None:
        loss_ = loss
    else:
        loss_ = torch.nn.CrossEntropyLoss()
        
    # Optimiser ----------------------------------------------------------------
    if optimiser is not None:
        optimiser = optimiser

    else:
        assert "optimiser is not defined"
        
    # Metrices -----------------------------------------------------------------
    # Accuracy
    metric_acc = metric_acc
    metric_acc.to(device)
    
    # F1 score
    metric_f1 = metric_f1
    metric_f1.to(device)
    
    # Precision
    metric_presicion = metric_presicion
    metric_presicion.to(device)
    
    # MultilabelRecall
    metric_recall = metric_recall
    metric_recall.to(device)

    # MultilabelConfusionMatrix 
    confusion_mat = confusion_mat
    confusion_mat.to(device)

    if lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, 
                                            start_factor=0.1, 
                                            end_factor=1.0, 
                                            total_iters=30)
    else:
        assert "Please define lr scheduler"
        # TODO: Define other approches
        
    # Model fit ----------------------------------------------------------------
    # Initializing in a separate cell so we can easily add more epochs to the same run
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    writer = SummaryWriter('runs/Trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000
    #
    # for plot the loss
    whole_loss_train = []
    whole_accuracy_trian = []

    whole_loss_valid = []
    whole_accuracy_valid = []
    
    for epoch in range(number_of_epochs):
        
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        
        avg_loss, avg_accuracy, avg_f1_score, avg_precision, avg_recall, conf_mat = train_one_epoch(model, 
                                                                                                    num_classes,
                                                                                                    data_loader_train,
                                                                                                    optimiser,
                                                                                                    metric_acc,
                                                                                                    metric_f1,
                                                                                                    metric_presicion,
                                                                                                    metric_recall,
                                                                                                    confusion_mat,
                                                                                                    loss_,
                                                                                                    lr_scheduler,
                                                                                                    epoch_number, 
                                                                                                    writer)
            
        running_vloss = 0.0
        runnung_vaccuracy = 0.0
        runnung_vf1_score = torch.zeros(num_classes).to(device)
        runnung_vprecision = torch.zeros(num_classes).to(device)
        runnung_vrecall = torch.zeros(num_classes).to(device)
        
        # For confusion matrix
        label_target_confusion = []
        label_predicted_confusion = []
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            
            for batch_index, (subject_id, image, label, seg) in enumerate(tqdm(data_loader_val)):
                            
                # Read image and label
                image = image.to(device)
                label = label.to(device)
                label_target = label.argmax(dim = 1)  
                
                # Make predictions for this batch
                # Make predictions for this batch
                label_predicted_logit = model(image)
                label_predicted_proba = torch.softmax(label_predicted_logit, dim = 1)  
                label_predicted       = label_predicted_proba.argmax(dim = 1)
                
                # compute accuracy
                vaccuracy = metric_acc(preds = label_predicted, target = label_target)
                runnung_vaccuracy += vaccuracy
                
                # Compute f1_score
                vf1_score = metric_f1(preds = label_predicted, target = label_target)
                runnung_vf1_score += vf1_score
                
                # Compute presicion
                vprecision = metric_presicion(preds = label_predicted, target = label_target)
                runnung_vprecision += vprecision
                
                # Compute recall
                vrecall = metric_recall(preds = label_predicted, target = label_target)
                runnung_vrecall += vrecall
                
                # Confusion matrix
                label_target_confusion.extend(label_target.data)
                label_predicted_confusion.extend(label_predicted.data)
                
                # Compute loss
                vloss =  loss_(label_predicted_logit, label_target)
                running_vloss += vloss
         
                       
        # For confusion matrix
        target_confusion = torch.tensor(label_target_confusion).to(device)
        predicted_confusion = torch.tensor(label_predicted_confusion).to(device)
        vconf_mat = confusion_mat(preds = predicted_confusion, target = target_confusion)
            
        avg_vloss = running_vloss / (batch_index + 1)
        avg_vaccuracy = runnung_vaccuracy / (batch_index + 1)
        avg_vf1_score = runnung_vf1_score / (batch_index + 1)
        avg_vprecision = runnung_vprecision / (batch_index + 1)
        avg_vrecall = runnung_vrecall / (batch_index + 1)
        
        print("\n")
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Accuracy train {} valid {}'.format(avg_accuracy, avg_vaccuracy))
        print('F1-score train {} valid {}'.format(avg_f1_score, avg_vf1_score))
        print('Precision train {} valid {}'.format(avg_precision, avg_vprecision))
        print('Recall train {} valid {}'.format(avg_recall, avg_vrecall))
        print('Confusion train {} valid {}'.format(conf_mat, vconf_mat))
        print("\n")
        
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_checkpoints/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
        
        # For plot the results
        whole_loss_train.append(avg_loss)
        whole_accuracy_trian.append(avg_accuracy)
        
        # Validation
        whole_loss_valid.append(avg_vloss.to("cpu"))
        whole_accuracy_valid.append(avg_vaccuracy.to("cpu"))
        
        # Loss
        plt.plot(np.array(whole_loss_train), label='loss_train')
        plt.plot(np.array(whole_loss_valid),label='loss_valid')
        plt.legend()
        plt.show()
        
        # Accuracy
        plt.plot(np.array(whole_accuracy_trian), label='accuracy_train')
        plt.plot(np.array(whole_accuracy_valid),label='accuracy_valid')
        plt.legend()
        plt.show()



################################################################################
################################################################################
""" Main body: train_one_epoch_UM-ProtoShare """

# Slice classifier columns for OC per-scale (K1,K2,K3)
def scale_slice(model, scale: int):
    K1 = model.prototype_vectors_1.shape[0]
    K2 = model.prototype_vectors_2.shape[0]
    K3 = model.prototype_vectors_3.shape[0]
    if scale == 1:
        return 0, K1
    if scale == 2:
        return K1, K2
    if scale == 3:
        return K1 + K2, K3
    raise ValueError("scale must be 1,2, or 3")
    

def lse_pool_prelast_conv(F_l, pmap_module, model):
    # features BEFORE last conv of the mapping head (bn1->relu output)
    Z = pmap_module.act_1(pmap_module.bn_1(pmap_module.conv_1(F_l)))  # [B,128,D,H,W]
    return model.lse_pooling(Z.flatten(2))                             # [B,128]

def online_cam_logits_single(x_feat, p_map_module, last_layer, model):
    # Single-scale OC (p_mode in {2,3}), no slicing needed
    u = lse_pool_prelast_conv(x_feat, p_map_module, model)            # [B,128]
    W_conv = p_map_module.conv_2.weight.flatten(1).t()                 # [128,K]
    k_scores = u @ W_conv                                              # [B,K]
    return k_scores @ last_layer.weight.t()                            # [B,C]

def online_cam_logits_multi(F_l, pmap_module, last_layer, model, scale_idx):
    # Multi-scale OC: slice classifier to this scale’s K
    u = lse_pool_prelast_conv(F_l, pmap_module, model)                # [B,128]
    W_conv = pmap_module.conv_2.weight.flatten(1).t()                  # [128,Kℓ]
    k_scores = u @ W_conv                                              # [B,Kℓ]
    off, K = scale_slice(model, scale_idx)
    W_slice = last_layer.weight[:, off:off+K]                          # [C,Kℓ]
    return k_scores @ W_slice.t()                                      # [B,C]

def map_loss_equiv(F_l, P_l, pmap_module, model, scale_factor: float):
    # Compare map(Aug(F)) vs Aug(map(F)) at the same spatial size
    if (F_l is None) or (P_l is None):
        return 0.0
    size = tuple(int(s * scale_factor) for s in F_l.shape[-3:])
    F_aug  = F.interpolate(F_l, size=size, mode='trilinear', align_corners=False)
    P_augA = model.get_p_map_scale_sharp(F_aug, pmap_module, sharpening=True)  # [B,K,D',H',W']
    P_augB = F.interpolate(P_l, size=size, mode='trilinear', align_corners=False)
    return (P_augA - P_augB).abs().mean()

def diversity_cos(P_bank):
    """
    Diversity penalty for a prototype bank: average off-diagonal cosine similarity.
    P_bank: [K,128,1,1,1] or [K,128]
    """
    if P_bank.dim() == 5:
        P = P_bank[:, :, 0, 0, 0]
    else:
        P = P_bank
    Pn = F.normalize(P, p=2, dim=1)               # [K,128]
    S = Pn @ Pn.t()                               # [K,K]
    return S.triu(1).abs().mean()                 # scalar

def pdist(P): 
    return F.pdist(P[:, :, 0, 0, 0]).mean()


# --- Soft-assign α from classifier weight -------------------------------------
def alpha_from_classifier(last_layer, eps: float = 1e-6):
    # last_layer.weight: [C, K_total]
    W = last_layer.weight.detach()          # stop-grad
    A = F.relu(W)                           # non-negative supports
    A = A / (A.sum(dim=0, keepdim=True) + eps)   # column-normalise across classes
    return A                                # [C, K_total]
# -----------------------------------------------------------------------------


def train_one_epoch_UM_ProtoShare(stage,
                                  model,
                                  data_loader,
                                  optimiser,
                                  class_weights,
                                  loss_class,
                                  loss_coefs,
                                  use_l1_mask,
                                  scaler,
                                  lr_scheduler,
                                  class_assignments):
    
    model.train()
    # running stats
    n_image = n_correct = n_batches = 0
    tot, tot_cls = 0.0, 0.0
    tot_clst = tot_sep = tot_avg_sep = 0.0
    tot_l1 = 0.0
    tot_map = tot_oc = 0.0
    
    for batch_index, (subject_id, image, label, seg) in enumerate(tqdm(data_loader)):
        # Data
        image = image.to(device)
        label = label.to(device)
        label_target = label.argmax(dim = 1)  
        # p-mode
        p_mode = getattr(model, 'p_mode')
        
        # ---- forward ---------------------------------------------------------
        if p_mode >= 5 and model.training:
            # U-ProtoShare-MS (multi-scale + p_maps)
            logits, distances, feats, pmaps = model(image)   # feats=(F3,F2,F1); pmaps=(P3,P2,P1)
            F3, F2, F1 = feats
            P3, P2, P1 = pmaps
            
        elif p_mode >= 4 and model.training:
            # U-ProtoShare (multi-scale, no p_maps path)
            logits, distances = model(image)
            F3 = F2 = F1 = None
            P3 = P2 = P1 = None
            
        elif p_mode >= 2 and model.training:
            # UM_ProtoShare style
            logits, distances, x_feat, p_map = model(image)
            F3 = F2 = F1 = None
            P3 = P2 = P1 = None
            
        else:
            # ProtoPNet / eval
            out = model(image)
            if isinstance(out, tuple):
                logits, distances = out[0], out[1]
            else:
                logits, distances = out, None
                
        # Classification loss --------------------------------------------------
        L_cls = loss_class(logits, label_target) * loss_coefs['loss_class_coe']
        tot_cls += L_cls.item()
        
        if stage == "joint":
            # Prototype regularizers (shared & multi-scale) --------------------
            # Convert distances -> activations (scale-invariant) ---------------
            # distances is [B, K_total] 
            proto_act = model.distance_to_similarity(distances)
            
            # Class re-weight (unchanged)
            cw = class_weights[label_target].to(device)
            cw = cw / cw.sum().clamp_min(1e-6)

            margin = 0.0  # keep your hinge-style margin

            if p_mode < 4: 
                
                # Hard class assigmnet for former versions of ProtoPnet
                if class_assignments == "hard":
                    # Calculate cluster loss
                    max_dist = torch.prod(torch.tensor(model.prototype_shape[1:])).to(device)
                    target_weight = class_weights.to(device)[label_target]
                    target_weight = target_weight / target_weight.sum()
                    prototypes_correct = model.prototype_class_identity[:, label_target].mT
                    inv_distances_correct = ((max_dist - distances) * prototypes_correct).amax(1)
                    clst_term = ((max_dist - inv_distances_correct) * target_weight).sum()
                    
                    # Calculate coefsseparation loss
                    prototypes_wrong = 1 - prototypes_correct
                    inv_distances_wrong = ((max_dist - distances) * prototypes_wrong).amax(1)
                    sep_term = ((max_dist - inv_distances_wrong) * target_weight).sum()
                    # Calculate average separation
                    avg_sep = (distances * prototypes_wrong).sum(1) / prototypes_wrong.sum(1)
                    avg_sep = (avg_sep * target_weight).sum()
                 
                # SOFT (Refer to Paper): α from non-negative classifier weights, per-proto normalised
                if class_assignments == "soft":
                    
                    alpha = alpha_from_classifier(model.last_layer)   # [C,K]
                    alpha_y   = alpha[label_target]                   # [B,K]
                    alpha_not = 1.0 - alpha_y                         # [B,K]

                    # Weighted SUM over prototypes (soft pooling)
                    act_correct = (alpha_y   * proto_act).sum(dim=1)  # [B]
                    act_wrong   = (alpha_not * proto_act).sum(dim=1)  # [B]

                    clst_term = (margin - act_correct).clamp_min(0.0) * cw
                    sep_term  = (act_wrong  - margin).clamp_min(0.0) * cw

                    # logging (α-weighted)
                    avg_sep = (alpha_not * proto_act).sum(1) / alpha_not.sum(1).clamp_min(1e-6) 
                    
            
            if p_mode>=4: # Soft class assigment for UM-Protoshare
            
                # SOFT (paper Eq.14): α from non-negative classifier weights, per-proto normalised
                alpha = alpha_from_classifier(model.last_layer)   # [C,K]
                alpha_y   = alpha[label_target]                   # [B,K]
                alpha_not = 1.0 - alpha_y                         # [B,K]

                # Weighted SUM over prototypes (soft pooling)
                act_correct = (alpha_y   * proto_act).sum(dim=1)  # [B]
                act_wrong   = (alpha_not * proto_act).sum(dim=1)  # [B]

                clst_term = (margin - act_correct).clamp_min(0.0) * cw
                sep_term  = (act_wrong  - margin).clamp_min(0.0) * cw

                # logging (α-weighted)
                avg_sep = (alpha_not * proto_act).sum(1) / alpha_not.sum(1).clamp_min(1e-6) 
                
            L_clst = loss_coefs['loss_clust_coe'] * clst_term.sum()
            L_sep  = loss_coefs['loss_sepra_coe'] * sep_term.sum()
            tot_clst += L_clst.item()
            tot_sep += L_sep.item()
            tot_avg_sep += (avg_sep * cw).sum().item()
            # ---------------------------------------------------------------------------
            
            # Mapping equivariance + Online-CAM --------------------------------
            L_map = 0.0
            L_oc  = 0.0
            
            if p_mode >= 4: # UM_ProtoShare
                if p_mode >= 5: # UM_ProtoShare with Online-cam and P_Map
                    # multi-scale maps available
                    ri = torch.randint(2, (1,), device=device).item()
                    scale = (0.75, 0.875)[ri]
    
                    L_map_3 = map_loss_equiv(F3, P3, model.p_map_3, model, scale)
                    L_map_2 = map_loss_equiv(F2, P2, model.p_map_2, model, scale)
                    L_map_1 = map_loss_equiv(F1, P1, model.p_map_1, model, scale)
    
                    β3, β2, β1 = 0.3, 0.4, 0.3
                    L_map = loss_coefs['loss_map_coe'] * (β3*L_map_3 + β2*L_map_2 + β1*L_map_1)
                    tot_map += float(L_map)
    
                    # OC per scale
                    oc_logits_3 = online_cam_logits_multi(F3, model.p_map_3, model.last_layer, model, scale_idx=3) if F3 is not None else None
                    oc_logits_2 = online_cam_logits_multi(F2, model.p_map_2, model.last_layer, model, scale_idx=2) if F2 is not None else None
                    oc_logits_1 = online_cam_logits_multi(F1, model.p_map_1, model.last_layer, model, scale_idx=1) if F1 is not None else None
    
                    for oc_logits in (oc_logits_1, oc_logits_2, oc_logits_3):
                        if isinstance(oc_logits, torch.Tensor):
                            L_oc += loss_class(oc_logits, label_target)
                    L_oc = loss_coefs['loss_online_cam_coe'] * L_oc
                    tot_oc += float(L_oc)
                    
                    # total (joint) --------------------------------------------
                    loss = L_cls + L_clst + L_sep + L_map + L_oc 
                
                elif p_mode < 5: # UM_ProtoShare without Online-cam and P_Map
                    # total (joint) --------------------------------------------
                    loss = L_cls + L_clst + L_sep 
                    
                # Diversity between prototypes ---------------------------------
                L_div = 0.0
                λ_div = loss_coefs.get('loss_div_coe', 0.0)
                if λ_div > 0.0:
                    if hasattr(model, 'prototype_vectors_03'):
                        L_div = λ_div * (diversity_cos(model.prototype_vectors_03)
                                         + diversity_cos(model.prototype_vectors_02)
                                         + diversity_cos(model.prototype_vectors_01)) / 3.0
                    elif hasattr(model, 'prototype_vectors'):
                        L_div = λ_div * diversity_cos(model.prototype_vectors)
                        
                loss += L_div
                
            elif p_mode >= 2:
                if p_mode >= 3: # UM_ProtoShare with Online-cam and P_Map
                    # single-scale (x_feat, p_map)
                    ri = torch.randint(2, (1,), device=device).item()
                    scale = (0.75, 0.875)[ri]
                    size = tuple(max(1, int(s * scale)) for s in x_feat.shape[-3:])
                    x_aug  = F.interpolate(x_feat, size=size, mode='trilinear', align_corners=True)
                    p_augA = model.get_p_map(x_aug)  # your sharpening path for single-scale
                    p_augB = F.interpolate(p_map, size=size, mode='trilinear', align_corners=True)
                    L_map = loss_coefs['loss_map_coe'] * (p_augA - p_augB).abs().mean()
                    tot_map += float(L_map)
    
                    oc_logits = online_cam_logits_single(x_feat, model.p_map, model.last_layer, model)
                    L_oc = loss_coefs['loss_online_cam_coe'] * loss_class(oc_logits, label_target)
                    tot_oc += float(L_oc)
                
                    # total (joint) ------------------------------------------------
                    loss = L_cls + L_clst + L_sep + L_map + L_oc
            
                else: # UM_ProtoShare without Online-cam and P_Map
                    # total (joint) ------------------------------------------------
                    loss = L_cls + L_clst + L_sep
            else: 
                # total (joint) --------------------------------------------
                loss = L_cls + L_clst + L_sep 

        # If not joint
        else:
            # L1 sparsity on classifier (last stage) ---------------------------
            if use_l1_mask and hasattr(model, 'prototype_class_identity'):
                # legacy class-assigned prototypes
                l1_mask = 1 - model.prototype_class_identity.mT
                L1 = torch.linalg.vector_norm(model.last_layer.weight * l1_mask, ord=1)
            else:
                L1 = torch.linalg.vector_norm(model.last_layer.weight, ord=1)
    
            loss = L_cls + loss_coefs['loss_L1_reg_coe'] * L1
            tot_l1 += (loss_coefs['loss_L1_reg_coe'] * L1).item()
       
        # Optimise -------------------------------------------------------------
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        lr_scheduler.step()
        
        # Metrics --------------------------------------------------------------
        n_image += label_target.shape[0]
        n_correct  += (logits.argmax(1) == label_target).sum().item()
        n_batches += 1
        tot += loss.item()
        
    # End-of-epoch -------------------------------------------------------------
    with torch.no_grad():
        if hasattr(model, 'prototype_vectors'):
            p_avg_pdist = F.pdist(model.prototype_vectors.flatten(1)).mean().item()
        elif all(hasattr(model, n) for n in ['prototype_vectors_01','prototype_vectors_02','prototype_vectors_03']):
            p_avg_pdist = float(torch.stack([pdist(model.prototype_vectors_1),
                                             pdist(model.prototype_vectors_2),
                                             pdist(model.prototype_vectors_3)]).mean())
        else:
            p_avg_pdist = float('nan')
            
    print(f" acc: {n_correct / n_image:.4f},"
          f" loss: {tot / n_batches:.4f},"
          f" cls: {tot_cls / n_batches:.4f},"
          f" clst: {tot_clst / n_batches:.4f},"
          f" sep: {tot_sep / n_batches:.4f},"
          f" avg_sep: {tot_avg_sep / n_batches:.4f},"
          f" L1: {tot_l1 / n_batches:.4f},"
          f" map: {tot_map / n_batches:.4f},"
          f" OC: {tot_oc / n_batches:.4f},"
          f" p_avg_pdist: {p_avg_pdist:.4f}")
            
        
            
            
            
            
            
            