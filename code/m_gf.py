import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
from subprocess import call
from time import time

import dgl
import torch as tc
from m_encoder import *

class GraphFactorization:
    """`Graph Factorization`_.
    Graph Factorization factorizes the adjacency matrix with regularization.
    Ref:
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf
    """
    def __init__(self, **kwargs):
        ''' Initialize the GraphFactorization class
        Args:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        '''
        hyper_params = {
            'print_step': 10000,
            'method_name': 'graph_factor_sgd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def _get_f_value(self, graph):
        f1 = 0
        # edges_cnt = len(graph.edges(data='weight', default=1))
        # cnt = 0
        for i, j, w in graph.edges(data='weight', default=1):
            f1 += (w - np.dot(self._X[i, :], self._X[j, :])) ** 2
            # cnt += 1
            # if cnt % 100 == 0:
            #     print('get f value {}/{}'.format(cnt,edges_cnt))
        f2 = self._regu * (np.linalg.norm(self._X) ** 2)
        return [f1, f2, f1 + f2]

    def learn_embedding(self, graph):
        self._node_num = len(graph.nodes())
        self._X = 0.01 * np.random.randn(self._node_num, self._d)
        for iter_id in range(self._max_iter):
            if not iter_id % self._print_step:
                [f1, f2, f] = self._get_f_value(graph)
                print('\t\tIter id: %d / %d, Objective: %g, f1: %g, f2: %g' % (
                    iter_id,
                    self._max_iter,
                    f,
                    f1,
                    f2
                ))
            # edges_cnt = len(graph.edges(data='weight', default=1))
            # cnt = 0
            for i, j, w in graph.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[i, :], self._X[j, :])) * self._X[j, :]
                term2 = self._regu * self._X[i, :]
                delPhi = term1 + term2
                self._X[i, :] -= self._eta * delPhi
                # cnt += 1
                # if cnt % 10000 == 0:
                #     print('get f value {}/{}'.format(cnt, edges_cnt // 2))
        return self._X

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r

class GFEncoder(Encoder):
    def __init__(self, g, emb_sz=128, workers=1, out_dir='../outputs', out_file='encoder', force=False,iter=1e4,r=1.0,lr=1e-4,print_step=5):
        super(GFEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.iter = iter
        self.r = r
        self.lr = lr
        self.print_step = print_step
    def train(self):
        self.gf = GraphFactorization(d=self.emb_sz, max_iter=int(self.iter), regu=self.r, eta=self.lr, print_step=self.print_step)

        if not self.force and self.check_file():
            print('encoder cache file checked')
            self.load()
            return
        G = dgl.to_networkx(self.g)
        self.gf.learn_embedding(graph=G)
        with open(os.path.join(self.out_dir,self.out_file),'w') as f:
            lins = []
            for row in self.gf.get_embedding():
                assert len(row) >= 2
                line = str(row[0])
                for idx,ele in enumerate(row):
                    if idx != 0:
                        line += ' ' + str(ele)
                lins.append(line+'\n')
            f.writelines(lins)
    def load(self):
        with open(os.path.join(self.out_dir,self.out_file),'r') as f:
            feats = []
            lins = f.readlines()
            for line in lins:
                assert len(line.strip())
                lst = line.strip().split()
                assert len(lst) == self.emb_sz
                feats.append([float(ele) for ele in lst])
            self.g.ndata['emb'] = tc.FloatTensor(feats)
        return self.g

if __name__ == '__main__':
    g, _ = dgl.load_graphs('../datasets/dst/facebook')
    g = g[0]
    encoder = GFEncoder(g=g,emb_sz=4,workers=8,out_dir='../tmp',out_file='gf-encoder',force=False,iter=100,r=1.0,lr=1e-3,print_step=5)
    st_time = time()
    encoder.train()
    print('encoder consume {:.2f}'.format(time()-st_time))
    out_g = encoder.load()
    print('emb output:',out_g.ndata['emb'][:3,:])

