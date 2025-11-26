#!/usr/bin/env python3
"""
The code for:
    utils
Author: Ali Golbaf
"""
################################################################################
################################################################################
""" Libraries """
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import copy
import itertools
import os
import socket
import time
import zlib
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ast 

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             log_loss, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

################################################################################
################################################################################
""" Make Directory """

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

################################################################################
################################################################################
""" Data """

# Read csv / excal file
def read_csv_exl(path, name):
    root, ext = os.path.splitext(name)
    if ext == ".csv":
        file = pd.read_csv(os.path.join(path, name))
    if ext == ".xlsx":
        file = pd.read_excel(os.path.join(path, name))
    return file  

# Check for missing modalities
def missing_modalities(path_to_data_images, subjects):
    
    missing_modalities = []

    for i in subjects:
        t1 = os.path.join(path_to_data_images, i)
        image_names = os.listdir(t1)
        image_names_ = [i + "_seg.nii",
                        i + "_t1.nii",
                        i + "_t1ce.nii",
                        i + "_t2.nii",
                        i + "_flair.nii"]
        
        t2 = list(set(image_names_) - set(image_names))

        if t2:
            missing_modalities.append(t2)

    return missing_modalities

# Load data using torchio
def load_data(path_to_data_images, subjects, glioma_grades):
    data = []
    for i in range(len(subjects)):
        subject = subjects[i]
        label = glioma_grades[i]
        t1 = os.path.join(path_to_data_images, subject)
        
        image_t1  = os.path.join(t1, subject + "_t1.nii")
        image_t1ce = os.path.join(t1, subject + "_t1ce.nii")
        image_t2  = os.path.join(t1, subject + "_t2.nii")
        image_flair  = os.path.join(t1, subject + "_flair.nii")
        image_seg = os.path.join(t1, subject + "_seg.nii")

        subj = tio.Subject(t1=tio.ScalarImage(image_t1),
                           t1ce=tio.ScalarImage(image_t1ce),
                           t2=tio.ScalarImage(image_t2),
                           flair=tio.ScalarImage(image_flair),
                           seg=tio.LabelMap(image_seg),
                           subject_id=subject, 
                           label=label)
        data.append(subj)
    return data

# Preprocessing
def preprocessing(data, preprocessing_args):
    
    transformation = [tio.ToCanonical(),]
    for i in preprocessing_args:
        if i == "crop":
            transformation  += [tio.CropOrPad(target_shape=(192,192, 144))]
        if i == "resample":
            transformation  += [tio.Resample(target=(1.5, 1.5, 1.5))]
        if i == "z_normalise":
            transformation  += [tio.ZNormalization()]
            
    transformation = tio.Compose(transformation) 
    return tio.SubjectsDataset(data, transform = transformation)

################################################################################
################################################################################
""" Augmentation """

def augment_brightness(t, multiplier_range=(0.5, 2)):
    multiplier = torch.empty(1).uniform_(multiplier_range[0], multiplier_range[1])
    return t * multiplier


def augment_contrast(t, contrast_range=(0.75, 1.25), preserve_range=True):
    if torch.rand(1) < 0.5 and contrast_range[0] < 1:
        factor = torch.empty(1).uniform_(contrast_range[0], 1)
    else:
        factor = torch.empty(1).uniform_(max(contrast_range[0], 1), contrast_range[1])
    t_mean = t.mean()
    if preserve_range:
        return ((t - t_mean) * factor + t_mean).clamp(t.min(), t.max())
    else:
        return (t - t_mean) * factor + t_mean

def augment_gamma(t, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-12, retain_stats=True):
    if invert_image:
        t = -t
    if retain_stats:
        t_mean, t_std = t.mean(), t.std()
    if torch.rand(1) < 0.5 and gamma_range[0] < 1:
        gamma = torch.empty(1).uniform_(gamma_range[0], 1)
    else:
        gamma = torch.empty(1).uniform_(max(gamma_range[0], 1), gamma_range[1])
    t_min = t.min()
    t_range = (t.max() - t_min).clamp_min(epsilon)
    t = ((t - t_min) / t_range) ** gamma * t_range + t_min
    if retain_stats:
        t = (t - t.mean()) / t.std().clamp_min(epsilon) * t_std + t_mean
    if invert_image:
        t = -t
    return t

transform_augs: dict = {
    'random_affine_0': tio.OneOf(
        {
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(0, 0), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.16,
            tio.RandomAffine(scales=(1, 1), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.16,
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.04,
        },
        p=0.36,
    ),
    'random_affine_1': tio.OneOf(
        {
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(0, 0), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
            tio.RandomAffine(scales=(1, 1), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
            tio.RandomAffine(scales=(0.7, 1.4), degrees=(-30, 30), isotropic=True,
                             default_pad_value='otsu', exclude=['seg']): 0.25,
        },
        p=0.75,
    ),
    'mo1': tio.RandomMotion(p=0.1),
    'bi1': tio.RandomBiasField(coefficients=(0, 0.1), p=0.1),
    'random_noise': tio.RandomNoise(std=(0, 0.1), p=0.15),
    'random_blur': tio.Compose(
        [
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t1']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t1ce']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['t2']),
            tio.RandomBlur(std=(0.5, 1.5), p=0.5, include=['flair']),
        ],
        p=0.2,
    ),
    'augment_brightness': tio.Lambda(partial(augment_brightness, multiplier_range=(0.7, 1.3)),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'augment_contrast': tio.Lambda(partial(augment_contrast, contrast_range=(0.65, 1.5)),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'random_anisotropy': tio.Compose(
        [
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t1']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t1ce']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['t2']),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.5, include=['flair']),
        ],
        p=0.25,
    ),
    'gi0': tio.Lambda(partial(augment_gamma, gamma_range=(0.7, 1.5), invert_image=True),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'augment_gamma': tio.Lambda(partial(augment_gamma, gamma_range=(0.7, 1.5), invert_image=False),
                      types_to_apply=[tio.INTENSITY], p=0.15),
    'random_flip_0': tio.RandomFlip(axes=(0, 1, 2)),
    'random_flip_1': tio.RandomFlip(),
}

def get_transform_augment(augmentation_sequence='af0-no0-bl0-br0-co0-an0-gi0-ga0-fl0'):
    
    transformation = [transform_augs[aug] for aug in augmentation_sequence]
    return tio.Compose(transformation)



################################################################################
################################################################################
""" Push Attributes """  

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1
         
################################################################################
################################################################################
""" Training Attributes """

def get_loss_coefs(args=None):
    """
    Return loss coefficients.
    """
    # default values (your original hard-coded ones)
    default = {
        'loss_class_coe': 1.0,
        'loss_clust_coe': 0.8,
        'loss_sepra_coe': -0.08,
        'loss_L1_reg_coe': 0.01,
        'loss_map_coe': 0.5,
        'loss_online_cam_coe': 0.05,
        'loss_div_coe': 0.01,
    }

    if args is None or getattr(args, "coefs", None) is None:
        return default

    # Parse string from CLI
    coefs_str = args.coefs
    try:
        parsed = ast.literal_eval(coefs_str)
    except Exception:
        # fall back to defaults if parsing fails
        return default

    # Handle either a dict or a list like [{'cls': ...}]
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        c = parsed[0]
    elif isinstance(parsed, dict):
        # if it's wrapped as {'coefs': [{...}]}, unwrap
        if 'coefs' in parsed and isinstance(parsed['coefs'], list) and len(parsed['coefs']) > 0:
            c = parsed['coefs'][0]
        else:
            c = parsed
    else:
        # weird format -> use defaults
        return default

    return {'loss_class_coe':      c.get('cls',  default['loss_class_coe']),
            'loss_clust_coe':      c.get('clst', default['loss_clust_coe']),
            'loss_sepra_coe':      c.get('sep',  default['loss_sepra_coe']),
            'loss_L1_reg_coe':     c.get('L1',   default['loss_L1_reg_coe']),
            'loss_map_coe':        c.get('map',  default['loss_map_coe']),
            'loss_online_cam_coe': c.get('OC',   default['loss_online_cam_coe']),
            'loss_div_coe':        c.get('div',  default['loss_div_coe']),}

class FocalLoss(nn.modules.loss._WeightedLoss):
    __constants__ = ['gamma', 'ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        log_prob = F.log_softmax(input, dim=0 if input.dim() == 1 else 1)
        return F.nll_loss((1 - log_prob.exp()) ** self.gamma * log_prob, target, weight=self.weight,
                          ignore_index=self.ignore_index, reduction=self.reduction)
    
def cross_entropy(f_x, y):
    return log_loss(y, f_x)    
    
    
def accuracy(f_x, y):
    return accuracy_score(y.argmax(1), f_x.argmax(1))


def balanced_accuracy(f_x, y):
    return balanced_accuracy_score(y.argmax(1), f_x.argmax(1))


def sensitivity(f_x, y):
    return recall_score(y.argmax(1), f_x.argmax(1), pos_label=1)


def specificity(f_x, y):
    return recall_score(y.argmax(1), f_x.argmax(1), pos_label=0)


def auroc(f_x, y):
    return roc_auc_score(y, f_x)


def auprc(f_x, y):
    return average_precision_score(y, f_x)   

def f_splits(splits, f_metric, f_x, y):
    indexes = np.unique(splits)
    metrics = np.zeros(indexes.shape)
    for i, index in enumerate(indexes):
        metrics[i] = f_metric(f_x[splits == index], y[splits == index])
    return metrics


def warmup_cosine_scheduler(optimiser,
                            num_epochs: int,
                            use_warmup: bool = True,
                            warmup_ratio: float = 0.2,):
    """
    Args:
        optimizer: torch.optim.Optimizer
            The optimizer whose LR will be scheduled.
        num_epochs: int
            Total number of training epochs.
        use_warmup: bool
            If True, use linear warm-up for the first `warmup_ratio * num_epochs` epochs.
        warmup_ratio: float
            Fraction of total epochs used for warm-up (e.g., 0.2 -> 20%).

    Returns:
        scheduler: torch.optim.lr_scheduler._LRScheduler
            A scheduler to call .step() on after each epoch.
    """
    if not use_warmup:
        # Pure cosine for the whole training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=num_epochs,
        )
        return scheduler

    # Number of warm-up epochs (e.g., 100 * 0.2 = 20)
    warmup_epochs = max(1, int(num_epochs * warmup_ratio))
    # Remaining epochs for cosine annealing
    cosine_epochs = num_epochs - warmup_epochs
    if cosine_epochs <= 0:
        raise ValueError("warmup_ratio is too large: no epochs left for cosine annealing.")

    # 1) Linear warm-up: LR goes from (1 / warmup_epochs) * base_lr → base_lr
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=1.0 / warmup_epochs,  # e.g. if warmup_epochs=20, start at lr/20
        end_factor=1.0,
        total_iters=warmup_epochs,)

    # 2) Cosine annealing for the remaining epochs
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=cosine_epochs,)

    # 3) Combine them in sequence
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],)  # switch after warmup_epochs

    return scheduler

################################################################################
################################################################################
""" Location coherence """
def loc_coherence(prob, seg, annos=None, threshold=None, topk_v=None, metric='AP'):
    lc_is = []
    for prob_i, seg_i in zip(prob, seg):
        lc_ijs = []
        anno = torch.zeros_like(seg_i[0])
        if isinstance(annos, list):
            for a in annos:
                anno[seg_i[0] == a] = 1
        else:
            anno[seg_i[0] > 0] = 1
        if threshold is None:
            if topk_v:
                quantile = 1 - topk_v / 100
            else:
                quantile = 1 - anno.sum() / anno.numel()
            threshold = torch.quantile(prob_i, quantile)
        else:
            prob_i = prob_i - prob_i.min()
            prob_i = prob_i / prob_i.max().clamp_min(1e-12)
        for prob_ij in prob_i:
            attr = torch.ones_like(prob_ij)
            attr[prob_ij < threshold] = 0
            if metric == 'AP':
                lc_ijs.append((anno * attr).sum() / attr.sum().clamp_min(1))
            elif metric == 'DSC':
                lc_ijs.append(2 * (anno * attr).sum() / (anno.sum() + attr.sum()).clamp_min(1))
            elif metric == 'IoU':
                lc_ijs.append((anno * attr).sum() / torch.logical_or(anno, attr).sum().clamp_min(1))
        lc_is.append(torch.hstack(lc_ijs))
    return torch.vstack(lc_is).cpu().numpy()


# Incremental addition/deletion
def inc_add_del(net, data, attr, n_intervals=100, uniform=True, quantile=True, addition=True):
    iads = []
    quantiles = torch.linspace(1, 0, n_intervals + 1).to(attr.device)
    if uniform:
        attr = attr.mean(1, keepdim=True)
    if quantile:
        thresholds = torch.quantile(attr.flatten(1), quantiles, dim=1)
    else:
        attr = attr - attr.amin(tuple(range(1, attr.ndim)), keepdim=True)
        attr = attr / attr.amax(tuple(range(1, attr.ndim)), keepdim=True).clamp_min(1e-12)
        thresholds = quantiles[:, None].tile(1, attr.shape[0])
    for i, threshold in enumerate(thresholds):
        if addition:
            mask = torch.zeros_like(attr)
            if i > 0:
                mask[attr >= threshold[(...,) + (None,) * (attr.ndim - 1)]] = 1
        else:
            mask = torch.ones_like(attr)
            if i > 0:
                mask[attr >= threshold[(...,) + (None,) * (attr.ndim - 1)]] = 0
        with torch.no_grad():
            iads.append(F.softmax(net(data * mask), dim=1))
    return torch.stack(iads).cpu().numpy()


def plot_iad(curve, ratio, method, metric, x_label, y_label, show=True, save=False, path=None):
    metric = {'IA': "Incremental Addition", 'ID': "Incremental Deletion"}[metric]
    x0 = np.linspace(0, 100, curve.shape[0])
    x = np.linspace(0, 100, (curve.shape[0] - 1) * 10 + 1)
    curve = np.interp(x, x0, curve)
    bounds = sorted([(curve[0], 'Start'), (curve[-1], 'End')])
    if show:
        # https://stackoverflow.com/a/56320309
        matplotlib.use('qt5agg')
    else:
        matplotlib.use('agg')
    plt.figure(figsize=(6, 4))
    h_c, = plt.plot(x, curve, label=metric)
    h_r = plt.fill_between(x, curve.clip(bounds[0][0], bounds[1][0]), bounds[0][0],
                           where=curve > bounds[0][0], color='dodgerblue', alpha=0.3,
                           label=f"Score: {ratio:.3f}")
    h_l = plt.axhline(bounds[0][0], color='r', alpha=0.7, linestyle=':',
                      label=f"Lower Bound ({bounds[0][1]})")
    h_u = plt.axhline(bounds[1][0], color='r', alpha=0.7, linestyle=(0, (2, 3)),
                      label=f"Upper Bound ({bounds[1][1]})")
    plt.legend(handles=[h_c, h_u, h_r, h_l])
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if path[-3:] != 'pdf':
        plt.title(f"{method}: {metric} ({y_label})")
    if save and path is not None:
        plt.savefig(path, dpi=300, pad_inches=0)
        # print(f"Output {path}")
    if show:
        plt.show()
    else:
        plt.close()


def process_iad(iads, y, save_plot=True, model_name=None):
    for method, iads_ in iads.items():
        for metric, iads__ in iads_.items():
            if metric == 'IAD':
                continue
            curves = np.zeros((iads__.shape[0], 2))
            for i in range(iads__.shape[0]):
                curves[i, 0] = accuracy(iads__[i], y)
                curves[i, 1] = balanced_accuracy(iads__[i], y)
            ratios = curves.clip(curves[[0, -1], :].min(0), curves[[0, -1], :].max(0))
            ratios = ratios - ratios.min(0)
            ratios = ratios / ratios.max(0).clip(1e-12)
            iads[method][metric] = ((ratios[:-1] + ratios[1:]) / 2).mean(0, keepdims=True)
            if save_plot and model_name is not None:
                iads_dir = f'../results/iads/{model_name}/'
                makedir(iads_dir)
                plot_iad(curves[:, 0], iads[method][metric][0, 0], method, metric, "Top Voxels (%)",
                         "Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_ACC.pdf')
                plot_iad(curves[:, 0], iads[method][metric][0, 0], method, metric, "Top Voxels (%)",
                         "Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_ACC.png')
                plot_iad(curves[:, 1], iads[method][metric][0, 1], method, metric, "Top Voxels (%)",
                         "Balanced Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_BAC.pdf')
                plot_iad(curves[:, 1], iads[method][metric][0, 1], method, metric, "Top Voxels (%)",
                         "Balanced Accuracy", show=False, save=True,
                         path=f'{iads_dir}{method}_{metric}_BAC.png')
        if 'IAD' in iads_:
            iads[method]['IAD'] = (iads[method]['IA'] + (1 - iads[method]['ID'])) / 2


def print_results(dataset, f_x, y, m_f_xes=None, lcs=None, n_prototypes=None, iads=None,
                  splits=None):
    print(f"{dataset}", end='', flush=True)
    if splits is not None:
        accs = f_splits(splits, accuracy, f_x, y)
        bacs = f_splits(splits, balanced_accuracy, f_x, y)
        sens = f_splits(splits, sensitivity, f_x, y)
        spes = f_splits(splits, specificity, f_x, y)
        aucs = f_splits(splits, auroc, f_x, y)
        print(f" ACC: {accs.mean():.3f}±{accs.std():.3f},"
              f" BAC: {bacs.mean():.3f}±{bacs.std():.3f},"
              f" SEN: {sens.mean():.3f}±{sens.std():.3f},"
              f" SPE: {spes.mean():.3f}±{spes.std():.3f},"
              f" AUC: {aucs.mean():.3f}±{aucs.std():.3f}")
    else:
        print(f" ACC: {accuracy(f_x, y):.3f},"
              f" BAC: {balanced_accuracy(f_x, y):.3f},"
              f" SEN: {sensitivity(f_x, y):.3f},"
              f" SPE: {specificity(f_x, y):.3f},"
              f" AUC: {auroc(f_x, y):.3f}")

        maxlen_method, maxlen_metric = 0, 0
        for method, lcs_ in lcs.items():
            for metric, lcs__ in lcs_.items():
                maxlen_method = max(len(method), maxlen_method)
                maxlen_metric = max(len(metric), maxlen_metric)
        for method, lcs_ in lcs.items():
            for metric, lcs__ in lcs_.items():
                print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                      f" {lcs__.mean(1).mean():.3f}±{lcs__.mean(1).std():.3f}"
                      f" (T1: {lcs__[:, 0].mean():.3f}±{lcs__[:, 0].std():.3f},"
                      f" T1CE: {lcs__[:, 1].mean():.3f}±{lcs__[:, 1].std():.3f},"
                      f" T2: {lcs__[:, 2].mean():.3f}±{lcs__[:, 2].std():.3f},"
                      f" FLAIR: {lcs__[:, 3].mean():.3f}±{lcs__[:, 3].std():.3f})")
    if n_prototypes is not None:
        for n_prototype in n_prototypes:
            print(f"        No. of Prototypes\t"
                  f" All: {n_prototype.sum():.0f},"
                  f" HGG: {n_prototype[1]:.0f},"
                  f" LGG: {n_prototype[0]:.0f}")
        if len(n_prototypes) > 1:
            print(f"Average No. of Prototypes\t"
                  f" All: {n_prototypes.sum(1).mean():.1f}±{n_prototypes.sum(1).std():.1f},"
                  f" HGG: {n_prototypes[:, 1].mean():.1f}±{n_prototypes[:, 1].std():.1f},"
                  f" LGG: {n_prototypes[:, 0].mean():.1f}±{n_prototypes[:, 0].std():.1f}")
    if iads:
        maxlen_method, maxlen_metric = 0, 0
        for method, iads_ in iads.items():
            for metric, iads__ in iads_.items():
                maxlen_method = max(len(method), maxlen_method)
                maxlen_metric = max(len(metric), maxlen_metric)
        for method, iads_ in iads.items():
            for metric, iads__ in iads_.items():
                print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                      f" {iads__[:, 0].mean():.3f}±{iads__[:, 0].std():.3f} (ACC),"
                      f" {iads__[:, 1].mean():.3f}±{iads__[:, 1].std():.3f} (BAC)")


