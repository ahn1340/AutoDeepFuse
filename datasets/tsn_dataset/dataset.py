# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
from pathlib import Path
import json
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import glob
from ..preprocess_data import *


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(os.path.join(root_path, label, key))

    return video_ids, video_paths, annotations



class TSNDataSet(data.Dataset):
    def __init__(self, root_path, annotation_path,opt,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='image_%05d.jpg', train = 1,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.train_val_test = train
        self.opt = opt
        #self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        #self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff
        if train == 1:
            subset = 'training' #
        else:
            subset = 'validation'
            
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset)

        
    def __make_dataset(self, root_path, annotation_path, subset):
        #print(annotation_path,"==="*30)
        annotation_path = Path(annotation_path)
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        self.n_videos = len(video_ids)
        dataset = []
        for i in range(self.n_videos):
            if i % (self.n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = Path(video_paths[i])
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue
            
            Total_frames = len(glob.glob(glob.escape(video_path) +  '/*.jpg'))  
            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id,
                'len': Total_frames
            }
            dataset.append(sample)

        return dataset, idx_to_class
    
    def _load_image(self, directory, idx):
        #print(directory,idx,self.image_tmpl.format(idx),idx,"=="*20)
        #print(self.image_tmpl%1)
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(directory, self.image_tmpl%idx)).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(directory, self.image_tmpl%idx))
                return [Image.open(os.path.join(directory, self.image_tmpl%1)).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(directory, self.image_tmpl%('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(directory, self.image_tmpl%('y', idx))).convert(
                    'L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(directory, self.image_tmpl%idx)).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(directory, self.image_tmpl%idx))
                    flow = Image.open(os.path.join(directory, self.image_tmpl%1)).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]



    def _sample_indices(self, num_frames):
        """

        :param num_frames: number of frames in video
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif num_frames > self.num_segments:
                offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, num_frames):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if num_frames > self.num_segments + self.new_length - 1:
                tick = (num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, num_frames):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        #record = self.video_list[index]
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        num_frames = self.data[index]['len']
        #print(path,frame_indices,num_frames,"=="*20)
        # check this is a legit video folder

        #clip = self.__loading(path, frame_indices)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        #return clip, target
    
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl%('x', 1)
            full_path = os.path.join(path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl%('x', 1)
            full_path = os.path.join(path, file_name)
        else:
            file_name = self.image_tmpl%1
            full_path = os.path.join(path, file_name)

        while not os.path.exists(path):
            print('################## Not Found:', path)
            index = np.random.randint(self.n_videos)
            path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            num_frames = self.data[index]['len']
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl%('x', 1)
                full_path = os.path.join(path, file_name)
            elif self.image_tmpl == 'image_{:05d}.jpg':
                file_name = self.image_tmpl%('x', 1)
                full_path = os.path.join(path, file_name)
            else:
                file_name = self.image_tmpl%1
                full_path = os.path.join(path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(num_frames)
        else:
            segment_indices = self._get_test_indices(num_frames)
        return self.get(self.data[index], segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record['video'], p)
                images.extend(seg_imgs)
                if p < record['len']:
                    p += 1
        #print(len(images),"=="*30)
        #process_data = self.transform(images)
        process_data = scale_crop(images, self.train_val_test, self.opt)
        return process_data, record['label']

    def __len__(self):
        return self.n_videos
