import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

from arch_search.operations import OPS, ReLUConvBN
from models import select_model as Streams

PRIMITIVES = [
    'none',
    #'max_pool_3x3',
    'skip_connect',
    '3d_conv_1x1',
    '2_1d_conv_3x3',
    '3d_conv_3x3',
    'dil_conv_3x3',
    'sep_conv_3x3',
]

Genotype = namedtuple(
    'Genotype',
    'first first_concat'
)


################################ Input Cell #######################################3
class InputCell(nn.Module):
    def __init__(self, channels):
        """
        Cell class used to determine at which layer of the feature extractor to get the input.
        It receives inputs of different resolutions, convert them such that they have same resolution.
        and outputs the sum of them. The output resolution is the same as that of the last input feature.

        For now, this class is made for resnet18 architecture. TODO: Adopt this to the state-of-the-art backbone that we use
        :param channels: list of ints, number of channels for each input
        """
        super(InputCell, self).__init__()
        self.channels = channels  #[64, 128, 256, 512]
        self.stride_sizes = [(1, 8, 8),
                             (1, 4, 4),
                             (1, 2, 2),
                             (1, 1, 1)]  #resolutions are [56, 28, 14, 7]
        self.kernel_sizes = [(1, 9, 9),
                             (1, 5, 5),
                             (1, 3, 3),
                             (1, 3, 3)]  #resolutions are [56, 28, 14, 7]
        self.padding_size = (0, 1, 1)
        self.n_output_channel = channels[-1]

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

    def forward(self, inputs, input_alphas):
        """
        :param inputs: list of features
        :param alphas: shape: [4]
        :return:
        """
        processed_inputs = []
        # extract the temporal features before starting the architecture search
        # shape of s1, s2, s3: [batch * frame?, 512, 1, 1]
        # num_segment = 4
        for i in inputs:
            processed = i.view(-1, self.num_segments, *i.shape[1:]).permute(0, 2, 1, 3, 4)
            if processed.shape[2] == 3:
                processed = F.pad(processed, (0, 0, 0, 0, 0, 1))
            processed_inputs.append(processed)

        s = sum(self._ops[k](input) * input_alphas[k] for k, input in enumerate(processed_inputs))
        return s


################################ DARTS #######################################3
class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm3d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, nodes, multiplier, C1, C2, C3, C):
        """
        Basic cell class used during the architecture search.
        Params:
            :nodes: number of intermediate nodes in the cell
            :multiplier: number of intermediate nodes concatenated to the
            output node
            :C1: number of input channels coming from the first modality
            :C2: number of input channels coming from the second modality
            :C3: number of input channels coming from the third modality
            :C: number of output channels for this cell
            :is_last: whether the cell is the last cell before FC layer. If True, do avgpooling in the end
        """

        super(Cell, self).__init__()
        self._nodes = nodes
        self._multiplier = multiplier

        self.preprocess1 = ReLUConvBN(C1, C, 1, 1, 0, 1, affine=False)
        self.preprocess2 = ReLUConvBN(C2, C, 1, 1, 0, 1, affine=False)
        self.preprocess3 = ReLUConvBN(C3, C, 1, 1, 0, 1, affine=False)

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._nodes):
            for j in range(3+i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s1, s2, s3, alphas):
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        s3 = self.preprocess3(s3)

        states = [s1, s2, s3]
        offset = 0
        for i in range(self._nodes):
            s = sum(
                self._ops[offset+j](h, alphas[offset+j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        x = torch.cat(states[-self._multiplier:], dim=1)
        return x


################################ GDAS #######################################3
class MixedOp_GDAS(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp_GDAS, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm3d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, index):
        return self._ops[index](x) * weights[index]


class Cell_GDAS(nn.Module):
    def __init__(self, nodes, multiplier, C1, C2, C3, C):
        """
        Basic cell class used during the architecture search.
        Params:
            :nodes: number of intermediate nodes in the cell
            :multiplier: number of intermediate nodes concatenated to the
            output node
            :C1: number of input channels coming from the first modality
            :C2: number of input channels coming from the second modality
            :C3: number of input channels coming from the third modality
            :C: number of output channels for this cell
        """

        super(Cell_GDAS, self).__init__()
        self._nodes = nodes
        self._multiplier = multiplier

        self.preprocess1 = ReLUConvBN(C1, C, 1, 1, 0, 1, affine=False)
        self.preprocess2 = ReLUConvBN(C2, C, 1, 1, 0, 1, affine=False)
        self.preprocess3 = ReLUConvBN(C3, C, 1, 1, 0, 1, affine=False)

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._nodes):
            for j in range(3+i):
                stride = 1
                op = MixedOp_GDAS(C, stride)
                self._ops.append(op)


    def forward(self, s1, s2, s3, alphas, indexs):
        # shape of s1, s2, s3: [batch 512, frame, 1, 1]
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        s3 = self.preprocess3(s3)

        states = [s1, s2, s3]
        offset = 0
        for i in range(self._nodes):
            clist = []
            for j, h in enumerate(states):
                weight_i_j = alphas[offset+j]
                index_i_j = indexs[offset+j].item()
                clist.append(self._ops[offset+j](h, weight_i_j, index_i_j))
            offset += len(states)
            states.append(sum(clist))

        x = torch.cat(states[-self._multiplier:], dim=1)
        return x


################################ PC-DARTS #######################################3
class MixedOp_PCDarts(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp_PCDarts, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C//4, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm3d(C//4, affine=False))
            self._ops.append(op)
            #self._ops.append(nn.Dropout(0.2))  # This is wrong

    def approx_channel_shuffle(self, x, groups):
        batchsize, num_channels, num_segments, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, num_segments, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, num_segments, height, width)

        return x

    def forward(self, x, weights):
        #channel proportion k=4
        #x = channel_shuffle(x)
        x = self.approx_channel_shuffle(x, 4)
        dim_2 = x.shape[1]
        xtemp = x[:, :dim_2//4, :, :, :]
        xtemp2 = x[:, dim_2//4:, :, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans


class Cell_PCDarts(nn.Module):

    def __init__(self, nodes, multiplier, C1, C2, C3, C):
        super(Cell_PCDarts, self).__init__()
        self._nodes = nodes
        self._multiplier = multiplier

        self.preprocess1 = ReLUConvBN(C1, C, 1, 1, 0, 1, affine=False)
        self.preprocess2 = ReLUConvBN(C2, C, 1, 1, 0, 1, affine=False)
        self.preprocess3 = ReLUConvBN(C3, C, 1, 1, 0, 1, affine=False)

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._nodes):
            for j in range(3+i):
                stride = 1
                op = MixedOp_PCDarts(C, stride)
                self._ops.append(op)

    def forward(self, s1, s2, s3, alphas, betas):
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)
        s3 = self.preprocess3(s3)

        states = [s1, s2, s3]
        offset = 0
        for i in range(self._nodes):
            s = sum(betas[offset+j] * self._ops[offset+j](h, alphas[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        x = torch.cat(states[-self._multiplier:], dim=1)
        return x

class Cell_PCDarts_one_input(nn.Module):

    def __init__(self, nodes, multiplier, C_in, C_out):
        super(Cell_PCDarts_one_input, self).__init__()
        self._nodes = nodes
        self._multiplier = multiplier

        self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine=False)

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._nodes):
            for j in range(3+i):
                stride = 1
                op = MixedOp_PCDarts(C_out, stride)
                self._ops.append(op)

    def forward(self, s, alphas, betas):
        s = self.preprocess(s)

        states = [s]
        offset = 0
        for i in range(self._nodes):
            s = sum(betas[offset+j] * self._ops[offset+j](h, alphas[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        x = torch.cat(states[-self._multiplier:], dim=1)
        return x

####################### Network ###################################
class Network(Streams):
    def __init__(self, config, arch_config, num_classes, num_segments,
                 criterion, dropout=0.1, **kwargs):
        super(Network, self).__init__(config, num_classes, num_segments,
                                      arch_config, dropout,
                                      )

        self._C = arch_config.arch_search.channels
        self._layers = arch_config.arch_search.cells
        self._criterion = criterion
        self._nodes = arch_config.arch_search.nodes
        self._multiplier = arch_config.arch_search.multiplier
        self._arch_config = arch_config
        self._num_modalities = 3  # Used for stem cells
        if not hasattr(nn.Module, 'num_segments'):
            nn.Module.num_segments = num_segments

        # GDAS
        self.tau = 10

        C_rgb = self.feature_dim_rgb   # 1024
        C_flow = self.feature_dim_flow  #1024
        C_pose = self.feature_dim_pose  # 256
        C_curr = self._C  # 256

        # Stem cell
        self.stem = nn.Sequential(
            nn.Conv3d(C_rgb + C_flow + C_pose, C_curr, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(C_curr)
        )

        C_prev_prev_prev, C_prev_prev, C_prev = C_curr, C_curr, C_curr
        self.cells = nn.ModuleList()

        # Normal cells
        for i in range(self._layers):
            cell = Cell_PCDarts(self._nodes, self._multiplier, C_prev_prev_prev, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev_prev, C_prev_prev, C_prev = C_prev_prev, C_prev, C_curr*self._multiplier

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self._multiplier*C_curr, num_classes)

        self._initialize_alphas()

    def forward(self, input_rgb, input_flow, input_pose):
        # Modality extraction
        x_b_rgb = self.base_rgb(input_rgb)
        x_b_flow = self.base_flow(input_flow)
        x_b_pose = self.base_pose(input_pose)

        ###### PC-DARTS implementation #####
        # Softmax all architecture parameters (alphas and betas)
        sm_alphas_list = []
        sm_betas_list = []

        for i in range(self._layers):
            sm_alphas_list.append(F.softmax(self.alphas_list[i], dim=-1))
            sm_betas = F.softmax(self.betas_list[i][0:3], dim=-1)
            start = 3
            n = 4
            for j in range(self._nodes-1):
                end = start + n
                temp_betas = F.softmax(self.betas_list[i][start:end], dim=-1)
                start = end
                n += 1
                sm_betas = torch.cat([sm_betas, temp_betas], dim=-1)
            sm_betas_list.append(sm_betas)

        # Stem cell
        input = torch.cat([x_b_rgb, x_b_flow, x_b_pose], dim=1)
        s1 = s2 = s3 = self.stem(input)

        # normal cells
        for i, cell in enumerate(self.cells):
            s1, s2, s3 = s2, s3, cell(s1, s2, s3, sm_alphas_list[i], sm_betas_list[i])
        ###### PC-DARTS implementation #####

        # get rid of spatial resolution
        x = self.avgpool(s3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def get_stream_out_shape(self):
        x = torch.ones((1, self.num_segments, self.channel, self._input_size,
                        self._input_size), requires_grad=False)
        x, _ = self.base_rgb(x)

        return x.shape[-1]

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def new(self):
        model_new = Network(self._C, self._layers, self.num_classes,
                            self.num_segments, self._representation,
                            self._arch_config, self._criterion)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._nodes) for n in range(3+i))
        # For one input cell
        #k2 = sum(1 for i in range(self._nodes) for n in range(1+i))
        num_ops = len(PRIMITIVES)

        # Input cell alphas
        self.rgb_input_alphas = Variable(5e-3*torch.randn(4).cuda(),
                                     requires_grad=True)
        self.flow_input_alphas = Variable(5e-3*torch.randn(4).cuda(),
                                     requires_grad=True)
        self.pose_input_alphas = Variable(5e-3*torch.randn(4).cuda(),
                                     requires_grad=True)

        self.alphas_list = []
        self.betas_list = []

        for i in range(self._layers):
            ## Normal cell alphas
            #self.alphas_list.append(Variable(5e-3*torch.randn(k, num_ops).cuda(), requires_grad=True))
            ## PC-DART betas
            #self.betas_list.append(Variable(5e-3*torch.randn(k).cuda(), requires_grad=True))
            # For using DataParallel all paramters should be nn.Parameter() and not Variable()
            self.alphas_list.append(nn.Parameter(5e-3*torch.randn(k, num_ops), requires_grad=True))
            self.betas_list.append(nn.Parameter(5e-3*torch.randn(k), requires_grad=True))

        self._arch_parameters = [
            *self.alphas_list,
            *self.betas_list,
        ]

    # GDAS
    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    # For all connections
    def get_genotype(self):
        def _parse(weights):
            gene = []
            n = 3
            start = 0
            for i in range(self._nodes):
                end = start + n
                W = weights[start: end].copy()
                edges = sorted(range(i+3), key=lambda x: -max(W[x][k] for k in
                                                              range(len(W[x]))
                                                              if k !=
                                                              PRIMITIVES.index('none')))
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        genotypes = []

        for alphas in self.alphas_list:
            gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy())
            concat = range(3+self._nodes-self._multiplier, self._nodes+3)
            genotype = Genotype(
                first=gene, first_concat=concat
            )
            genotypes.append(genotype)

        return genotypes

    # For top-k (PC-DARTS)
    def get_genotype_topk(self, topk):
        def _parse(weights, weights2, topk):
            gene = []
            n = topk
            start = 0
            for i in range(self._nodes):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + topk),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:topk]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        genotypes = []

        for alphas, betas in zip(self.alphas_list, self.betas_list):
            # softmax betas
            sm_betas = F.softmax(betas[0:topk], dim=-1)
            start = topk
            n = topk+1
            for i in range(self._nodes-1):
                end = start + n
                temp_betas = F.softmax(betas[start:end], dim=-1)
                start = end
                n += 1
                sm_betas = torch.cat([sm_betas, temp_betas], dim=-1)

            # parse genes
            gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy(), sm_betas.data.cpu().numpy(), topk=topk)
            concat = range(topk + self._nodes-self._multiplier, self._nodes + topk)
            genotype = Genotype(
                first=gene, first_concat=concat,
            )

            genotypes.append(genotype)

        return genotypes

    # For oneinput
    #def get_genotype(self):
    #    def _parse(weights):
    #        gene = []
    #        n = 1
    #        start = 0
    #        for i in range(self._nodes):
    #            end = start + n
    #            W = weights[start: end].copy()
    #            edges = sorted(range(i+1), key=lambda x: -max(W[x][k] for k in
    #                                                          range(len(W[x]))
    #                                                          if k !=
    #                                                          PRIMITIVES.index('none')))
    #            for j in edges:
    #                k_best = None
    #                for k in range(len(W[j])):
    #                    if k_best is None or W[j][k] > W[j][k_best]:
    #                        k_best = k
    #                gene.append((PRIMITIVES[k_best], j))
    #            start = end
    #            n += 1
    #        return gene

    #    genotypes = []

    #    for alphas in self.alphas_list:
    #        gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy())
    #        concat = range(1+self._nodes-self._multiplier, self._nodes+1)
    #        genotype = Genotype(
    #            first=gene, first_concat=concat
    #        )
    #        genotypes.append(genotype)

    #    return genotypes


    def get_input_genotype(self):
        """
        Returns [index of maximum alpha value for rgb, for mv, for pose]
        :return:
        """
        input_genotypes = [self.rgb_input_alphas,
                           self.flow_input_alphas,
                           self.pose_input_alphas,
                           ]
        input_genotypes = [int(torch.argmax(alphas)) for alphas in input_genotypes]

        return input_genotypes
