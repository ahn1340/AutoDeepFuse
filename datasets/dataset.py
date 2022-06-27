from __future__ import division
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from .preprocess_data import *
from PIL import Image, ImageFilter
import pickle
import glob
#import dircache
import pdb
import numpy as np
import math
import random


def get_test_video(opt, rgb_flow_path,pose_path, Total_frames):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : list of all video frames
        """
    clip_rgb = []
    clip_flow = []
    clip_pose = []
    
    # Case 1: total num_frame is larger than segments
    if Total_frames >= opt.training.num_segments:
        # First divide video into n segments of equal frames
        len_segment = math.floor(Total_frames / opt.training.num_segments)

        # Compute start_frame such that the sampled frames are at the center of the video
        rest = Total_frames % opt.training.num_segments
        start_frame = math.floor(rest / 2)

        # current segment in which sample a frame
        current_segment = 0
        rand = np.random.randint(1, len_segment + 1)

        # Then, from each segments sample a frame such that all sampled frames are equidistant
        for i in range(opt.training.num_segments):
            frame_to_sample = start_frame + current_segment * len_segment + rand
            im = Image.open(os.path.join(rgb_flow_path, 'frames/img_%05d.jpg'%(frame_to_sample)))
            im_x = Image.open(os.path.join(rgb_flow_path, 'flow_x/x_%05d.jpg'%(frame_to_sample)))
            im_y = Image.open(os.path.join(rgb_flow_path, 'flow_y/y_%05d.jpg'%(frame_to_sample)))
            im_pose = Image.open(os.path.join(pose_path, 'frames/img_%05d.jpg'%(frame_to_sample)))

            clip_rgb.append(im.copy())
            clip_flow.append(im_x.copy())
            clip_flow.append(im_y.copy())
            clip_pose.append(im_pose.copy())
            #im.close()
            #im_x.close()
            #im_y.close()
            #im_pose.close()

            current_segment += 1

    # Case 2: video is shorter than num_segments
    else:
        frame_to_sample = 1

        for i in range(opt.training.num_segments):
            im = Image.open(os.path.join(rgb_flow_path, 'frames/img_%05d.jpg'%(frame_to_sample)))
            im_x = Image.open(os.path.join(rgb_flow_path, 'flow_x/x_%05d.jpg'%(frame_to_sample)))
            im_y = Image.open(os.path.join(rgb_flow_path, 'flow_y/y_%05d.jpg'%(frame_to_sample)))
            im_pose = Image.open(os.path.join(pose_path, 'frames/img_%05d.jpg'%(frame_to_sample)))

            clip_rgb.append(im.copy())
            clip_flow.append(im_x.copy())
            clip_flow.append(im_y.copy())
            clip_pose.append(im_pose.copy())
            #im.close()
            #im_x.close()
            #im_y.close()
            #im_pose.close()

            frame_to_sample += 1

            if frame_to_sample > Total_frames:
                frame_to_sample = 1


    return clip_rgb,clip_flow,clip_pose

def get_train_video(opt, rgb_flow_path, pose_path, Total_frames):
    """
        Chooses a random clip from a video for training/ validation
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    clip_rgb = []
    clip_flow = []
    clip_pose = []
    
    # Case 1: total num_frame is larger than segments
    if Total_frames >= opt.training.num_segments:
        # First divide video into n segments of equal frames
        len_segment = math.floor(Total_frames / opt.training.num_segments)

        # Compute start_frame such that the sampled frames are at the center of the video
        rest = Total_frames % opt.training.num_segments
        start_frame = math.floor(rest / 2)

        # Segment in which a frame is sampled
        current_segment = 0

        # Then, from each segments sample a random frame and append
        for i in range(opt.training.num_segments):
            frame_to_sample = start_frame + current_segment * len_segment + np.random.randint(1, len_segment + 1)
            im = Image.open(os.path.join(rgb_flow_path, 'frames/img_%05d.jpg'%(frame_to_sample)))
            im_x = Image.open(os.path.join(rgb_flow_path, 'flow_x/x_%05d.jpg'%(frame_to_sample)))
            im_y = Image.open(os.path.join(rgb_flow_path, 'flow_y/y_%05d.jpg'%(frame_to_sample)))
            im_pose = Image.open(os.path.join(pose_path, 'frames/img_%05d.jpg'%(frame_to_sample)))

            clip_rgb.append(im.copy())
            clip_flow.append(im_x.copy())
            clip_flow.append(im_y.copy())
            clip_pose.append(im_pose.copy())
            #im.close()
            #im_x.close()
            #im_y.close()
            #im_pose.close()

            current_segment += 1
        
        #print("len_segment: {}, total_frames: {}".format(len_segment, Total_frames))

    # Case 2: video is shorter than num_segments
    else:
        frame_to_sample = 1

        for i in range(opt.training.num_segments):
            im = Image.open(os.path.join(rgb_flow_path, 'frames/img_%05d.jpg'%(frame_to_sample)))
            im_x = Image.open(os.path.join(rgb_flow_path, 'flow_x/x_%05d.jpg'%(frame_to_sample)))
            im_y = Image.open(os.path.join(rgb_flow_path, 'flow_y/y_%05d.jpg'%(frame_to_sample)))
            im_pose = Image.open(os.path.join(pose_path, 'frames/img_%05d.jpg'%(frame_to_sample)))

            clip_rgb.append(im.copy())
            clip_flow.append(im_x.copy())
            clip_flow.append(im_y.copy())
            clip_pose.append(im_pose.copy())
            #im.close()
            #im_x.close()
            #im_y.close()
            #im_pose.close()

            frame_to_sample += 1

            if frame_to_sample > Total_frames:
                frame_to_sample = 1

        #print("video shorter than len_segment. Total_frames: {}".format(Total_frames))


    return clip_rgb,clip_flow,clip_pose


class JHMDB51_test(Dataset):
    """HMDB51 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.datasets.data_annot_path)]))

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 21

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.datasets.data_annot_path) if file.strip('.txt')[-1] ==str(split)])
       
        self.data = []                                     # (filename , lab_id)
        
        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.datasets.data_annot_path, file), 'r')
            for line in f: 
                # If training data
                if train==1 and line.split(' ')[1] == '1':
                    frame_path = os.path.join(opt.datasets.rgb_flow_path, class_id, line.split(' ')[0][:-4])
                    if os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))


                # Elif validation/test data        
                elif train!=1 and line.split(' ')[1] == '2':
                    frame_path = os.path.join(opt.datasets.rgb_flow_path, class_id, line.split(' ')[0][:-4])
                    if os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))


                
            f.close()

    def __len__(self):
        '''
        returns number of test/train set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.datasets.rgb_flow_path, video[1], video[0])
        pose_path = os.path.join(self.opt.datasets.pose_path, video[1], video[0])
        #print(frame_path)
        #print(glob.escape(frame_path) +  'frames/*.jpg',"=="*30)
        #if self.opt.only_RGB:
        #    Total_frames = len(glob.glob(glob.escape(frame_path) +  '/*.jpg'))  
        #else:
        #    Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/x_*.jpg'))
        #print("Total frames: ", Total_frames)
        Total_frames = len(glob.glob(glob.escape(frame_path) +  '/frames/*.jpg')) 
        if self.train_val_test == 0: 
            clip_rgb,clip_flow,clip_pose = get_test_video(self.opt, frame_path, pose_path, Total_frames)
        else:
            clip_rgb,clip_flow,clip_pose = get_train_video(self.opt, frame_path, pose_path, Total_frames)
        self.opt.modality = 'RGB'
        clip_rgb_t = scale_crop(clip_rgb, self.train_val_test, self.opt)
        
        self.opt.modality = 'Flow'
        clip_flow_t = scale_crop(clip_flow, self.train_val_test, self.opt)
        
        self.opt.modality = 'RGB'
        clip_pose_t = scale_crop(clip_pose, self.train_val_test, self.opt)
        return((clip_rgb_t,clip_flow_t,clip_pose_t, label_id))
    

class HMDB51_test(Dataset):
    """HMDB51 Dataset for training the found architecture"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 1 for training, 2 for test
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.datasets.data_annot_path)]))

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 51

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.datasets.data_annot_path) if file.strip('.txt')[-1] ==str(split)])
       
        self.train_val_data = []  # train+val data (filename , lab_id)
        self.data = []  # test data

        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.datasets.data_annot_path, file), 'r')
            for line in f:
                # If training data
                if train==1 and (line.split(' ')[1] == '1' or line.split(' ')[1] == '2'):
                    frame_path = os.path.join(opt.datasets.rgb_flow_path, class_id, line.split(' ')[0][:-4])
                    if os.path.exists(frame_path):
                        self.train_val_data.append((line.split(' ')[0][:-4], class_id))

                # Elif test data
                elif train==2 and line.split(' ')[1] == '0':
                    frame_path = os.path.join(opt.datasets.rgb_flow_path, class_id, line.split(' ')[0][:-4])
                    if os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))

            f.close()

        if train == 1:  # if train+val
            # Count total number of train+val data
            num_samples = len(self.train_val_data)
            # Split point (90/10)
            split_point = num_samples // 10 * 9
            # shuffle and split train and val data into equal sizes
            random.shuffle(self.train_val_data)
            self.data = self.train_val_data[:split_point]
            self.val_data = self.train_val_data[split_point:]

    def __len__(self):
        '''
        returns number of test/train set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.datasets.rgb_flow_path, video[1], video[0])
        pose_path = os.path.join(self.opt.datasets.pose_path, video[1], video[0])
        #print(frame_path)
        #print(glob.escape(frame_path) +  'frames/*.jpg',"=="*30)
        #if self.opt.only_RGB:
        #    Total_frames = len(glob.glob(glob.escape(frame_path) +  '/*.jpg'))  
        #else:
        #    Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/x_*.jpg'))
        #print("Total frames: ", Total_frames)

        # Sometimes the total number of frames for each modality is not the same.
        # In that case, set the total number of frames as the smallest of the modalities.
        rgb_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/frames/*.jpg'))
        flow_x_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/*.jpg'))
        flow_y_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_y/*.jpg'))
        pose_Total_frames = len(glob.glob(glob.escape(pose_path) +  '/frames/*.jpg'))
        total_frames_list  = [rgb_Total_frames, flow_x_Total_frames, flow_y_Total_frames, pose_Total_frames]
        
        # Check that all modalities have same number of frames
        if not all(val == rgb_Total_frames for val in total_frames_list):
            Total_frames = min(total_frames_list)
            #print("The modalities of sample {} have different number of total number of frames. Setting its total_frames to {}".format(video[0], Total_frames))

        else:
            Total_frames = rgb_Total_frames

        if self.train_val_test == 2:
            clip_rgb,clip_flow,clip_pose = get_test_video(self.opt, frame_path, pose_path, Total_frames)
        else:
            clip_rgb,clip_flow,clip_pose = get_train_video(self.opt, frame_path, pose_path, Total_frames)
            
        #print("here1"*20)
        self.opt.modality = 'RGB'
        clip_rgb_t = scale_crop(clip_rgb, self.train_val_test, self.opt)

        #print("here2"*20)
        self.opt.modality = 'Flow'
        clip_flow_t = scale_crop(clip_flow, self.train_val_test, self.opt)

        #print("here3"*20)
        self.opt.modality = 'RGB'
        clip_pose_t = scale_crop(clip_pose, self.train_val_test, self.opt)
        return((clip_rgb_t,clip_flow_t,clip_pose_t, label_id))

class HMDB51_search(Dataset):
    """HMDB51 Dataset for searching the architecture"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 1 for training, 2 for search
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.datasets.data_annot_path)]))

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 51

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.datasets.data_annot_path) if file.strip('.txt')[-1] ==str(split)])

        self.train_val_data = []  # (filename, lab_id) train+val_set

        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.datasets.data_annot_path, file), 'r')
            for line in f:
                # If training data
                if train == 1 and line.split(' ')[1] == '1' or line.split(' ')[1] == '2':
                    frame_path = os.path.join(opt.datasets.rgb_flow_path, class_id, line.split(' ')[0][:-4])
                    if os.path.exists(frame_path):
                        self.train_val_data.append((line.split(' ')[0][:-4], class_id))

            f.close()

        # Count total number of train+val data
        num_samples = len(self.train_val_data)
        # Split point (50/50)
        split_point = num_samples // 2
        # shuffle and split train and val data into equal sizes
        random.shuffle(self.train_val_data)
        self.data = self.train_val_data[:split_point]
        self.val_data = self.train_val_data[split_point:]

    def __len__(self):
        '''
        returns number of test/train set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.datasets.rgb_flow_path, video[1], video[0])
        pose_path = os.path.join(self.opt.datasets.pose_path, video[1], video[0])
        #print(frame_path)

        # Sometimes the total number of frames for each modality is not the same.
        # In that case, set the total number of frames as the smallest of the modalities.
        rgb_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/frames/*.jpg'))
        flow_x_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/*.jpg'))
        flow_y_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_y/*.jpg'))
        pose_Total_frames = len(glob.glob(glob.escape(pose_path) +  '/frames/*.jpg'))
        total_frames_list  = [rgb_Total_frames, flow_x_Total_frames, flow_y_Total_frames, pose_Total_frames]

        # Check that all modalities have same number of frames
        if not all(val == rgb_Total_frames for val in total_frames_list):
            Total_frames = min(total_frames_list)
            #print("The modalities of sample {} have different number of total number of frames. Setting its total_frames to {}".format(video, Total_frames))

        else:
            Total_frames = rgb_Total_frames

        clip_rgb,clip_flow,clip_pose = get_train_video(self.opt, frame_path, pose_path, Total_frames)
            
        #print("here1"*20)
        self.opt.modality = 'RGB'
        clip_rgb_t = scale_crop(clip_rgb, self.train_val_test, self.opt)
        
        #print("here2"*20)
        self.opt.modality = 'Flow'
        clip_flow_t = scale_crop(clip_flow, self.train_val_test, self.opt)
        
        #print("here3"*20)
        self.opt.modality = 'RGB'
        clip_pose_t = scale_crop(clip_pose, self.train_val_test, self.opt)
        return((clip_rgb_t,clip_flow_t,clip_pose_t, label_id))


class UCF101_search(Dataset):
    """UCF101 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt

        with open(os.path.join(self.opt.datasets.data_annot_path, "classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        with open(os.path.join(self.opt.datasets.data_annot_path, "classInd.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        self.class_idx = dict(zip(self.lab_names, index))   # Each label is mappped to a number
        self.idx_class = dict(zip(index, self.lab_names))   # Each number is mappped to a label

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.datasets.data_annot_path) if file.strip('.txt')[-1] ==str(split)])

        # 1 = train/train+val
        # 2 = val/test

        self.train_val_data = []                                     # (filename , lab_id)

        # train + val data. Needs to be dvided into 50/50 train/val split
        split_lab_filenames = [f for f in split_lab_filenames if 'train' in f]

        f = open(os.path.join(self.opt.datasets.data_annot_path, split_lab_filenames[0]), 'r')
        for line in f:
            class_id = self.class_idx.get(line.split('/')[0]) - 1
            sample_folder = line.strip('\n').split('/')[1].split(' ')[0][:-4]
            if os.path.exists(os.path.join(self.opt.datasets.rgb_flow_path, line.split('/')[0], sample_folder)) == True:
                self.train_val_data.append((sample_folder, class_id))

        f.close()

        # Count total number of train+val data
        num_samples = len(self.train_val_data)
        # Split point (50/50)
        split_point = num_samples // 2
        # shuffle and split train and val data into equal sizes
        random.shuffle(self.train_val_data)
        self.data = self.train_val_data[:split_point]
        self.val_data = self.train_val_data[split_point:]

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = os.path.join(self.opt.datasets.rgb_flow_path, self.idx_class.get(label_id + 1), video[0])
        pose_path = os.path.join(self.opt.datasets.pose_path, self.idx_class.get(label_id + 1), video[0])

        # Sometimes the total number of frames for each modality is not the same.
        # In that case, set the total number of frames as the smallest of the modalities.
        rgb_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/frames/*.jpg'))
        flow_x_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/*.jpg'))
        flow_y_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_y/*.jpg'))
        pose_Total_frames = len(glob.glob(glob.escape(pose_path) +  '/frames/*.jpg'))
        total_frames_list  = [rgb_Total_frames, flow_x_Total_frames, flow_y_Total_frames, pose_Total_frames]

        # Check that all modalities have same number of frames
        if not all(val == rgb_Total_frames for val in total_frames_list):
            Total_frames = min(total_frames_list)
            #print("The modalities of sample {} have different number of total number of frames. Setting its total_frames to {}".format(video, Total_frames))

        else:
            Total_frames = rgb_Total_frames

        # Sample training/validation data
        clip_rgb, clip_flow, clip_pose = get_train_video(self.opt, frame_path, pose_path, Total_frames)

        #print("here1"*20)
        self.opt.modality = 'RGB'
        clip_rgb_t = scale_crop(clip_rgb, self.train_val_test, self.opt)

        #print("here2"*20)
        self.opt.modality = 'Flow'
        clip_flow_t = scale_crop(clip_flow, self.train_val_test, self.opt)

        #print("here3"*20)
        self.opt.modality = 'RGB'
        clip_pose_t = scale_crop(clip_pose, self.train_val_test, self.opt)
        return((clip_rgb_t,clip_flow_t,clip_pose_t, label_id))


class UCF101_test(Dataset):
    """UCF101 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation
            split : 1,2,3
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt

        with open(os.path.join(self.opt.datasets.data_annot_path, "classInd.txt")) as lab_file:
            self.lab_names = [line.strip('\n').split(' ')[1] for line in lab_file]

        with open(os.path.join(self.opt.datasets.data_annot_path, "classInd.txt")) as lab_file:
            index = [int(line.strip('\n').split(' ')[0]) for line in lab_file]

        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 101

        self.class_idx = dict(zip(self.lab_names, index))   # Each label is mappped to a number
        self.idx_class = dict(zip(index, self.lab_names))   # Each number is mappped to a label

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.datasets.data_annot_path) if file.strip('.txt')[-1] ==str(split)])

        self.train_val_data = []                                     # (filename , lab_id)
        self.data = []  # test data

        # train + val data. Needs to be dvided into 50/50 train/val split
        if train == 1:
            split_lab_filenames = [f for f in split_lab_filenames if 'train' in f]
        elif train == 2:
            split_lab_filenames = [f for f in split_lab_filenames if 'test' in f]

        f = open(os.path.join(self.opt.datasets.data_annot_path, split_lab_filenames[0]), 'r')
        for line in f:
            class_id = self.class_idx.get(line.split('/')[0]) - 1
            sample_folder = line.strip('\n').split('/')[1].split(' ')[0][:-4]
            if os.path.exists(os.path.join(self.opt.datasets.rgb_flow_path, line.split('/')[0], sample_folder)) == True:
                if train == 1:
                    self.train_val_data.append((sample_folder, class_id))
                elif train == 2:
                    self.data.append((sample_folder, class_id))

        f.close()

        if train == 1:  # if train+val
            # Count total number of train+val data
            num_samples = len(self.train_val_data)
            # Split point (90/10)
            split_point = num_samples // 10 * 9
            # shuffle and split train and val data into equal sizes
            random.shuffle(self.train_val_data)
            self.data = self.train_val_data[:split_point]
            self.val_data = self.train_val_data[split_point:]

    def __len__(self):
        '''
        returns number of test set
        '''
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = os.path.join(self.opt.datasets.rgb_flow_path, self.idx_class.get(label_id + 1), video[0])
        pose_path = os.path.join(self.opt.datasets.pose_path, self.idx_class.get(label_id + 1), video[0])
        #if self.opt.only_RGB:   # We do not use this
        #Total_frames = len(glob.glob(glob.escape(frame_path) +  '/*.jpg'))
        #else:
        #    Total_frames = len(glob.glob(glob.escape(frame_path) +  'flow_x/x_*.jpg'))

        # Sometimes the total number of frames for each modality is not the same.
        # In that case, set the total number of frames as the smallest of the modalities.
        rgb_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/frames/*.jpg'))
        flow_x_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_x/*.jpg'))
        flow_y_Total_frames = len(glob.glob(glob.escape(frame_path) +  '/flow_y/*.jpg'))
        pose_Total_frames = len(glob.glob(glob.escape(pose_path) +  '/frames/*.jpg'))
        total_frames_list  = [rgb_Total_frames, flow_x_Total_frames, flow_y_Total_frames, pose_Total_frames]

        # Check that all modalities have same number of frames
        if not all(val == rgb_Total_frames for val in total_frames_list):
            Total_frames = min(total_frames_list)
            #print("The modalities of sample {} have different number of total number of frames. Setting its total_frames to {}".format(video, Total_frames))

        else:
            Total_frames = rgb_Total_frames

        # Sample training/validation data
        if self.train_val_test == 1:
            clip_rgb, clip_flow, clip_pose = get_train_video(self.opt, frame_path, pose_path, Total_frames)
        elif self.train_val_test == 2:
            clip_rgb, clip_flow, clip_pose = get_test_video(self.opt, frame_path, pose_path, Total_frames)

        #print("here1"*20)
        self.opt.modality = 'RGB'
        clip_rgb_t = scale_crop(clip_rgb, self.train_val_test, self.opt)

        #print("here2"*20)
        self.opt.modality = 'Flow'
        clip_flow_t = scale_crop(clip_flow, self.train_val_test, self.opt)

        #print("here3"*20)
        self.opt.modality = 'RGB'
        clip_pose_t = scale_crop(clip_pose, self.train_val_test, self.opt)
        return(clip_rgb_t, clip_flow_t, clip_pose_t, label_id)


class Kinetics_test(Dataset):
    def __init__(self, split, train, opt):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation 
            split : 'val' or 'train'
        Returns:
            (tensor(frames), class_id ) : Shape of tensor C x T x H x W
        """
        self.split = split
        self.opt = opt
        self.train_val_test = train
              
        # joing labnames with underscores
        self.lab_names = sorted([f for f in os.listdir(os.path.join(self.opt.datasets.rgb_flow_path, "train"))])        
       
        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == 400
        
        # indexes for validation set
        if train==1:
            label_file = os.path.join(self.opt.datasets.data_annot_path, 'Kinetics_train_labels.txt')
        else:
            label_file = os.path.join(self.opt.datasets.data_annot_path, 'Kinetics_val_labels.txt')

        self.data = []                                     # (filename , lab_id)
    
        f = open(label_file, 'r')
        for line in f:
            class_id = int(line.strip('\n').split(' ')[-2])
            nb_frames = int(line.strip('\n').split(' ')[-1])
            self.data.append((os.path.join(self.opt.datasets.rgb_flow_path,' '.join(line.strip('\n').split(' ')[:-2])), class_id, nb_frames))
        f.close()
            
    def __len__(self):
        '''
        returns number of test set
        '''          
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = video[1]
        frame_path = video[0]
        Total_frames = video[2]

        if self.opt.only_RGB:
            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))  
        else:
            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))

        if self.train_val_test == 0: 
            clip = get_test_video(self.opt, frame_path, Total_frames)
        else:
            clip = get_train_video(self.opt, frame_path, Total_frames)


        return((scale_crop(clip, self.train_val_test, self.opt), label_id))

    
