import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
from datasets.preprocess_data import *
from datasets.dataset import *
from datasets.dataset_tsn import TSNDataSet
import copy


def get_train_val_test_dataloader(opt, model):
    print("###################### preprocessing train+val and test start #########################")
    print("Preprocessing train + val data ...")
    train_data = globals()['{}_test'.format(opt.datasets.name)](split = opt.datasets.val_split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))
    val_data = copy.deepcopy(train_data)
    val_data.data = val_data.val_data
    val_data.train_val_test = 2
    print("Length of validation data = ", len(val_data))

    print("Preprocessing test data ...")
    test_data   = globals()['{}_test'.format(opt.datasets.name)](split = opt.datasets.val_split, train = 2, opt = opt)
    print("Length of test data = ", len(test_data))

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.training.batch_size, shuffle=True, num_workers = opt.datasets.num_workers, pin_memory = True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size = opt.training.batch_size, shuffle=True, num_workers = opt.datasets.num_workers, pin_memory = True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size = opt.training.batch_size, shuffle=True, num_workers = opt.datasets.num_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of val datatloader = ",len(val_dataloader))
    print("Length of test datatloader = ",len(test_dataloader))
    print("###################### preprocessing train+val and test loader done #########################")
    return train_dataloader, val_dataloader, test_dataloader

def get_train_val_dataloader(opt, model):
    print("###################### preprocessing train and val start #########################")
    print("Preprocessing train data ...")
    train_data = globals()['{}_search'.format(opt.datasets.name)](split = opt.datasets.val_split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing val data ...")
    val_data = copy.deepcopy(train_data)
    val_data.data = val_data.val_data
    print("Length of val data = ", len(val_data))

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.training.batch_size, shuffle=True, num_workers = opt.datasets.num_workers, pin_memory = True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size = opt.training.batch_size, shuffle=True, num_workers = opt.datasets.num_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))
    print("###################### preprocessing train and val loader done #########################")
    return train_dataloader, val_dataloader


def get_train_val_dataloader_tsn(opt,model):
    
    train_dataloader = DataLoader(
        TSNDataSet(opt.datasets.rgb_flow_path,opt.datasets.pose_path, opt.datasets.data_annot_path, num_segments=opt.training.num_segments,
                   new_length=1, train = 1, opt = opt),
        batch_size=opt.training.batch_size, shuffle=True,
        num_workers=opt.datasets.num_workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU
                 
    val_dataloader = DataLoader(
        TSNDataSet(opt.datasets.rgb_flow_path,opt.datasets.pose_path, opt.datasets.data_annot_path, num_segments=opt.training.num_segments,
                   new_length=1, train = 2, opt = opt,
                   random_shift=False,),
        batch_size=opt.training.batch_size, shuffle=False,
        num_workers=opt.datasets.num_workers, pin_memory=True)
                   
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))  
    
    return train_dataloader, val_dataloader
