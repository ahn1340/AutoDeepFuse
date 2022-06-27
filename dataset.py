"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, _, label = line.strip().split()
                #print(video)
                #video, label = line.strip().split()

                #print(video)
                #print(self._data_root,video[:-4])
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                vid_len = get_num_frames(video_path)
                #print(vid_len)
                if vid_len > 16:
                    self._video_list.append((
                        video_path,
                        int(label),vid_len
                        ))

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                                 representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def show_sequence(self, sequence, label):
        from matplotlib import pyplot as plt
        import matplotlib.gridspec as gridspec
        columns = 4
        print(sequence[0].shape)
        COUNT = 4
        rows = (COUNT + 1) // (columns)
        fig = plt.figure(figsize=(32, (16 // columns) * rows))
        gs = gridspec.GridSpec(rows, columns)
        j=0
        imgN = np.zeros((sequence[0].shape[0], sequence[0].shape[1], 3))
        for img in sequence:
            if sequence[0].shape[2] == 2:
                imgN[:,:,0] = img[:,:,0]
                imgN[:,:,1] =  img[:,:,1]
                imgN[:,:,2] = (img[:,:,0]^2 + img[:,:,1]^2)/2
                img = imgN
            plt.subplot(gs[j])
            plt.axis("off")
            # print(label)
            plt.suptitle("label " + str(label), fontsize=30)
            plt.imshow(img.astype('uint8'))
            plt.savefig('images/img_%04d.png'%j)
            #plt.close(fig)
            j +=1

        plt.show()

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0


        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames_rgb = []
        frames_mv = []
        frames_residual = []

        #print(video_path,'==' ,num_frames)
        for seg in range(self._num_segments):

            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)
            #print(num_frames, gop_index, gop_pos)

            img_rgb = load(video_path, gop_index, gop_pos,
                       0, False)

            img_mv = load(video_path, gop_index, gop_pos,
                       1, True)

            img_residual = load(video_path, gop_index, gop_pos,
                       2, True)

            if img_rgb is None:
                print('Error: loading video %s failed.' % video_path)
                img_mv = np.zeros((256, 256, 2))
                img_rgb = np.zeros((256, 256, 3))
                img_residual = np.zeros((256, 256, 3))

            else:
                img_mv = clip_and_scale(img_mv, 20)
                img_mv += 128
                img_mv = (np.minimum(np.maximum(img_mv, 0), 255)).astype(np.uint8)

                img_residual += 128
                img_residual = (np.minimum(np.maximum(img_residual, 0), 255)).astype(np.uint8)

                img_rgb = color_aug(img_rgb)
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img_rgb = img_rgb[..., ::-1]

            frames_rgb.append(img_rgb)
            frames_mv.append(img_mv)
            frames_residual.append(img_residual)

        #self.show_sequence(frames_rgb, label)
        #self.show_sequence(frames_residual, label)
        #self.show_sequence(frames_mv, label)

        #print(len(frames_mv))
        #print(frames_mv.shape)
        frames_rgb = self._transform(frames_rgb)
        frames_mv = self._transform(frames_mv)
        frames_residual = self._transform(frames_residual)

        frames_rgb = np.array(frames_rgb)
        frames_rgb = np.transpose(frames_rgb, (0, 3, 1, 2))
        input_rgb = torch.from_numpy(frames_rgb).float() / 255.0

        frames_mv = np.array(frames_mv)
        frames_mv = np.transpose(frames_mv, (0, 3, 1, 2))
        input_mv = torch.from_numpy(frames_mv).float() / 255.0

        frames_residual = np.array(frames_residual)
        frames_residual = np.transpose(frames_residual, (0, 3, 1, 2))
        input_residual = torch.from_numpy(frames_residual).float() / 255.0

        input_rgb = (input_rgb - self._input_mean) / self._input_std
        input_residual = (input_residual - 0.5) / self._input_std
        input_mv = (input_mv - 0.5)

        return input_rgb, input_mv, input_residual, label

    def __len__(self):
        return len(self._video_list)
