import numpy as np
from collections import namedtuple
import torch.nn.functional as F
import torch
import copy
import os

"""
Create random cell architecture. Script gets number of cells as input.
"""

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

# Creates list of genotype
def get_genotype(alphas_list):
    nodes = 4
    multiplier = 4
    def _parse(weights):
        gene = []
        n = 3
        start = 0
        for i in range(nodes):
            end = start + n
            W = weights[start: end].copy()
            edges = sorted(range(i+3), key=lambda x: -max(W[x][k] for k in
                                                          range(len(W[x]))))
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

    for alphas in alphas_list:
        gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy())
        concat = range(3+nodes-multiplier, nodes+3)
        genotype = Genotype(
            first=gene, first_concat=concat
        )
        genotypes.append(genotype)

    return genotypes


def get_genotype_topk(alphas_list, topk):
    nodes = 4
    multiplier = 4
    def _parse(weights, topk):
        gene = []
        n = topk
        start = 0
        for i in range(nodes):
            end = start + n
            print(start)
            print(end)
            W = weights[start:end].copy()
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

    for alphas in alphas_list:
        gene = _parse(F.softmax(alphas, dim=-1).data.cpu().numpy(), topk)
        concat = range(topk+nodes-multiplier, nodes+topk)
        genotype = Genotype(
            first=gene, first_concat=concat
        )
        genotypes.append(genotype)

    return genotypes


def create_random_architecture(num_cells, topk=None):
    """
    Assumes three input nodes and four hidden nodes
    :param num_cells: int
    :return: list of genotypes
    """
    alphas_list = []
    for i in range(num_cells):
        rand_mat = torch.Tensor(np.random.rand(18, 7))
        alphas_list.append(rand_mat)

    if topk is None:
        genotypes = get_genotype(alphas_list)
    elif type(topk) is int:
        genotypes = get_genotype_topk(alphas_list, topk)


    return genotypes

