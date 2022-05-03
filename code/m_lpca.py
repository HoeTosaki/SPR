import scipy as sp
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import m_encoder
import os

def gen_sp_mat(datasets,data_names):
    for dataset,data_name in zip(datasets,data_names):
        g, _ = dgl.load_graphs(dataset)
        g = g[0]
        g = dgl.to_networkx(g)
        g = nx.to_scipy_sparse_matrix(g)
        io.savemat('../datasets/other/adj-{}.mat'.format(data_name),{'network':g})
        print('transformed data mat for {}'.format(data_name))

class LPCAEncoder(m_encoder.Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False):
        super(LPCAEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)

    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return

        raise NotImplementedError

    def save(self):
        pass

if __name__ == '__main__':
    print('hello lpca')
    # gen_sp_mat(datasets=['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc'],data_names = ['cr','fb','gq'])
    # gen_sp_mat(datasets=['../datasets/dst/DBLP'],data_names=['db'])
    gen_sp_mat(datasets=['../cls/data'])