"""
* Data loader

* Written by:
    Ali Golbaf (ali.golbaf@plymouth.ac.uk)
"""
################################################################################
################################################################################
""" Libraries """
import torch
import torchio as tio
################################################################################
################################################################################
""" Classes """
from torch.utils.data import Dataset
from utils import get_transform_augment

################################################################################
################################################################################
# Main Body

class SubjectsDataset(Dataset):
    
    def __init__(self,
                 data,
                 num_classes = None,
                 augment = None,
                 target_label_aug = None
                 ):
        
        self.data = data
        self.num_classes = num_classes
        self.augment = augment
        self.target_label = target_label_aug
                
        # Check for augmentation
        if self.augment is not None:
            self.transform_augment = get_transform_augment(augmentation_sequence = self.augment)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        subject = self.data[index]
        subject.load()
        
        # Subject ID
        subject_id  = subject.subject_id
        
        # Extract label
        label = subject.label
        
        # Check for augmentation
        if self.augment is not None:
            
            if label == self.target_label:
                subject = self.transform_augment(subject)
        
        # Image 4 channels
        image = torch.cat((subject.t1.data, subject.t1ce.data, subject.t2.data, subject.flair.data), dim=0)
        
        # # Onehot
        label = torch.tensor(label).long()
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)

        # Segmentation
        seg = subject.seg.data

        del subject
        
        return subject_id, image, label, seg

    
    
    
    
    