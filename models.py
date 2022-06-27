import sys
import copy
import torch
import torchvision
from torch import nn
from transforms import *

import torch.nn.functional as F
from arch_search.operations import OPS, ReLUConvBN
from generate_model import generate_model
from torch.autograd import Variable
from arch_search.operations import Identity


########################## Cell implementation ######################################
class Cell(nn.Module):
    def __init__(self, genotype, C1, C2, C3, C):
        """
        Basic cell class used during the architecture search.
        Params:
            :genotype: architecture encoding
            :C1: number of input channels coming from the first modality
            :C2: number of input channels coming from the second modality
            :C3: number of input channels coming from the third modality
            :C: number of output channels for this cell
        """

        super(Cell, self).__init__()

        op_names, indices = zip(*genotype.first)
        concat = genotype.first_concat

        self.preprocess1 = ReLUConvBN(C1, C, 1, 1, 0, 1, affine=False)
        self.preprocess2 = ReLUConvBN(C2, C, 1, 1, 0, 1, affine=False)
        self.preprocess3 = ReLUConvBN(C3, C, 1, 1, 0, 1, affine=False)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        #self._nodes = len(op_names) // 3
        self._nodes = 4
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name in op_names:
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s1, s2, s3, drop_prob=0):
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        s3 = self.preprocess3(s3)

        states = [s1, s2, s3]
        offset = 0
        for i in range(self._nodes):
            if self.training:
                s = sum(
                    drop_path(
                        self._ops[offset+j](states[self._indices[offset+j]]),
                        drop_prob,
                        self._ops[offset+j],
                    )
                    for j in range(len(states))
                )
            else:
                s = sum(
                    self._ops[offset+j](states[self._indices[offset+j]])
                    for j in range(len(states))
                )
            offset += len(states)
            states.append(s)

        x = torch.cat([states[i] for i in self._concat], dim=1)
        return x


########################## Cell_topk implementation ######################################
class Cell_topk(nn.Module):
    def __init__(self, genotype, C1, C2, C3, C, topk):
        """
        Basic cell class used during the architecture search.
        Params:
            :genotype: architecture encoding
            :C1: number of input channels coming from the first modality
            :C2: number of input channels coming from the second modality
            :C3: number of input channels coming from the third modality
            :C: number of output channels for this cell
        """

        super(Cell_topk, self).__init__()

        self.topk = topk
        op_names, indices = zip(*genotype.first)
        concat = genotype.first_concat

        self.preprocess1 = ReLUConvBN(C1, C, 1, 1, 0, 1, affine=False)
        self.preprocess2 = ReLUConvBN(C2, C, 1, 1, 0, 1, affine=False)
        self.preprocess3 = ReLUConvBN(C3, C, 1, 1, 0, 1, affine=False)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        #self._nodes = len(op_names) // 3
        self._nodes = 4
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name in op_names:
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s1, s2, s3, drop_prob=0):
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        s3 = self.preprocess3(s3)

        states = [s1, s2, s3]
        for i in range(self._nodes):
            if self.training:
                s = sum(
                    drop_path(
                        self._ops[self.topk*i+j](states[self._indices[self.topk*i+j]]),
                        drop_prob,
                        self._ops[self.topk*i+j],
                    )
                    for j in range(self.topk)
                )
            else:
                s = sum(
                    self._ops[self.topk*i+j](states[self._indices[self.topk*i+j]])
                    for j in range(self.topk)
                )
            states.append(s)

        x = torch.cat([states[i] for i in self._concat], dim=1)
        return x

# For one input
#class Cell(nn.Module):
#    def __init__(self, genotype, C_in, C_out):
#        """
#        Basic cell class used during the architecture search.
#        Params:
#            :genotype: architecture encoding
#            :C1: number of input channels coming from the first modality
#            :C2: number of input channels coming from the second modality
#            :C3: number of input channels coming from the third modality
#            :C: number of output channels for this cell
#        """
#
#        super(Cell, self).__init__()
#
#        op_names, indices = zip(*genotype.first)
#        concat = genotype.first_concat
#
#        self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine=False)
#
#        self._compile(C_out, op_names, indices, concat)
#
#    def _compile(self, C, op_names, indices, concat):
#        assert len(op_names) == len(indices)
#        #self._nodes = len(op_names) // 3
#        self._nodes = 4
#        self._concat = concat
#        self._multiplier = len(concat)
#
#        self._ops = nn.ModuleList()
#        for name in op_names:
#            stride = 1
#            op = OPS[name](C, stride, True)
#            self._ops.append(op)
#        self._indices = indices
#
#    def forward(self, s, drop_prob=0):
#        s = self.preprocess(s)
#
#        states = [s]
#        offset = 0
#        for i in range(self._nodes):
#            s = sum(
#                self._ops[offset+j](h)
#                for j, h in enumerate(states)
#            )
#            offset += len(states)
#            states.append(s)
#
#        x = torch.cat([states[i] for i in self._concat], dim=1)
#        return x
########################## Cell implementation ######################################


########################### Input Cell ######################################
class InputCell(nn.Module):
    def __init__(self, index, C):
        """
        Basic cell class used during the architecture search.
        Params:
            :genotype: index of the layer from which we take the input.
            :C: number of output channels for this cell
        """

        super(InputCell, self).__init__()
        self.channels = [64, 128, 256, 512]
        self.stride_sizes = [(1, 8, 8),
                             (1, 4, 4),
                             (1, 2, 2),
                             (1, 1, 1)]  #resolutions are [56, 28, 14, 7]
        self.kernel_sizes = [(1, 9, 9),
                             (1, 5, 5),
                             (1, 3, 3),
                             (1, 3, 3)]  #resolutions are [56, 28, 14, 7]
        self.padding_size = (0, 1, 1)
        self.n_output_channel = self.channels[-1]

        self._ops = nn.ModuleList()

        for c, k, s in zip(self.channels, self.kernel_sizes, self.stride_sizes):
            op = ReLUConvBN(C_in=c,
                            C_out=self.n_output_channel,
                            kernel_size=k,
                            stride=s,
                            padding=self.padding_size,
                            dilation=1,
                            )
            self._ops.append(op)

        self.index = index

    def forward(self, inputs):
        processed_inputs = []
        # extract the temporal features before starting the architecture search
        for i in inputs:
            processed = i.view(-1, self.num_segments, *i.shape[1:]).permute(0, 2, 1, 3, 4)
            if processed.shape[2] == 3:
                processed = F.pad(processed, (0, 0, 0, 0, 0, 1))
            processed_inputs.append(processed)

        return self._ops[self.index](processed_inputs[self.index])
########################## Input Cell ######################################


########################## Network ######################################
class select_model(nn.Module):
    def __init__(
        self,
        config,
        num_classes,
        num_segments,
        arch_config,
        dropout=0.1,
        genotype=None,
        input_genotype=None,
        truncated_pretrained=True,
        discretization='all',
        topk=None,
    ):
        super(select_model, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.genotype = genotype
        #self.input_genotype = input_genotype
        self._layers = arch_config.arch_search.cells
        self._nodes = arch_config.arch_search.nodes
        self._multiplier = arch_config.arch_search.multiplier
        self._C = arch_config.arch_search.channels
        if not hasattr(nn.Module, 'num_segments'):
            nn.Module.num_segments = num_segments

        self.drop_path_prob = 0  # Drop_path probability. Changes over epochs in cov_model.py
        self.truncated = truncated_pretrained  # Determines whether we take the final layer of the pretrained model or intermediate layer

        self._input_size = config.datasets.input_size
        print("Input size: ", self._input_size)

        if discretization not in ['all', 'topk']:
            raise ValueError("discretization should be 'all or 'topk', but got {}".format(discretization))
        else:
            self.discretization = discretization # 'all' or 'topk'
            self.topk = topk

        ###############################################
        self.prepare_network_arch(config, arch_config)
        ###############################################

        try:
            self.feature_dim_rgb = self.base_rgb.num_channels
            print("rgb num_channels: {}".format(self.feature_dim_rgb))
        except:
            print("num_channels for RGB does not exist. feature_dim set to 512")
            self.feature_dim_rgb = 512

        try:
            self.feature_dim_flow = self.base_flow.num_channels
            print("flow num_channels: {}".format(self.feature_dim_flow))
        except:
            print("num_channels for flow does not exist. feature_dim set to 512")
            self.feature_dim_flow = 512

        try:
            self.feature_dim_pose = self.base_pose.num_channels
            print("pose num_channels: {}".format(self.feature_dim_pose))
        except:
            print("num_channels for pose does not exist. feature_dim set to 512")
            self.feature_dim_pose = 512

        C_rgb = self.feature_dim_rgb
        C_flow = self.feature_dim_flow
        C_pose = self.feature_dim_pose
        C_curr = self._C

        if genotype is not None:
            # Stem cell
            self.stem = nn.Sequential(
                nn.Conv3d(C_rgb + C_flow + C_pose, C_curr, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(C_curr)
            )

            C_prev_prev_prev, C_prev_prev, C_prev = C_curr, C_curr, C_curr
            self.cells = nn.ModuleList()

            # Normal cells
            for i in range(self._layers):
                if self.discretization == 'all':
                    cell = Cell(self.genotype[i], C_prev_prev_prev, C_prev_prev, C_prev, C_curr)
                elif self.discretization == 'topk':
                    cell = Cell_topk(self.genotype[i], C_prev_prev_prev, C_prev_prev, C_prev, C_curr, self.topk)
                self.cells += [cell]
                C_prev_prev_prev, C_prev_prev, C_prev = C_prev_prev, C_prev, C_curr*self._multiplier

            self.fc = nn.Linear(
                self.cells[-1]._multiplier*arch_config.arch_search.channels,
                num_classes
            )
        else:
            if self.truncated:
                # Baseline with truncated pretrained models (intermediate layer)
                self.fc = nn.Linear(
                    self.feature_dim_rgb + self.feature_dim_flow + self.feature_dim_pose,
                    num_classes
                )
            else:
                pass

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # preserve time resoultion, just squeeze spatial resolution

        #self.fc_rgb = nn.Linear(self.feature_dim_rgb, num_classes)  # kernels * segments = 40*16 = 640
        #self.fc_flow = nn.Linear(self.feature_dim_flow, num_classes)  # kernels * segments = 40*16 = 640
        #self.fc_pose = nn.Linear(self.feature_dim_pose, num_classes)  # kernels * segments = 40*16 = 640

    def prepare_network_arch(self, config, network_config):
        # Load model for each modulity stream
        config.input_channels = 3
        self.base_rgb = copy.deepcopy(generate_model(config, network_config.arch_rgb, config.finetune.f_rgb_nclass,config.finetune.f_rgb_path))

        config.input_channels = 2
        self.base_flow = copy.deepcopy(generate_model(config, network_config.arch_flow, config.finetune.f_flow_nclass,config.finetune.f_flow_path))

        config.input_channels = 3
        self.base_pose = copy.deepcopy(generate_model(config, network_config.arch_pose, config.finetune.f_pose_nclass,config.finetune.f_pose_path))

    def forward(self, input_rgb, input_flow, input_pose):
        x_b_rgb = self.base_rgb(input_rgb)
        x_b_flow = self.base_flow(input_flow)
        x_b_pose = self.base_pose(input_pose)

        if self.genotype is None:  # baseline
            if self.truncated:
                x = torch.cat([x_b_rgb, x_b_flow, x_b_pose], dim=1)  # [N, C, 8, 7, 7]
                x = self.avgpool(x) #[N, C, 8, 1, 1]
                x = torch.mean(x, dim=2)  #[N, C, 1, 1, 1]
                x = x.view(x.size(0), -1) #[N, C]

            else:
                pass  # not implemented
        else:  # architecture search
            input = torch.cat([x_b_rgb, x_b_flow, x_b_pose], dim=1)

            # For three inputs
            s1 = s2 = s3 = self.stem(input)
            for i, cell in enumerate(self.cells):
                s1, s2, s3 = s2, s3, cell(s1, s2, s3, self.drop_path_prob)

            x = self.avgpool(s3)
            x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 112

    @property
    def input_size(self):
        return self._input_size


def drop_path(x, drop_prob, op):
    if drop_prob > 0. and not isinstance(op, Identity):
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


