import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
import time
from m_encoder import *
import dgl
import networkx as nx
import torch as tc

class LocallyLinearEmbedding:
    def __init__(self, **kwargs):
        hyper_params = {}
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def learn_embedding(self, graph):
        graph = graph.to_undirected()
        A = nx.to_scipy_sparse_matrix(graph)
        B = normalize(A, norm='l1', axis=1, copy=True)
        I_n = sp.eye(len(graph.nodes()))
        I_min_B = I_n - B
        # print('L', I_min_B)
        u, s, vt = lg.svds(I_min_B, k=self._d + 1, which='SM')

        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X.real

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )

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

class LLEEncoder(Encoder):
    def __init__(self, g, emb_sz=128, workers=1, out_dir='../outputs', out_file='encoder', force=False):
        super(LLEEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)

    def train(self):
        self.lle = LocallyLinearEmbedding(d=self.emb_sz)

        if not self.force and self.check_file():
            print('encoder cache file checked')
            self.load()
            return
        G = dgl.to_networkx(self.g)
        self.lle.learn_embedding(graph=G)
        with open(os.path.join(self.out_dir,self.out_file),'w') as f:
            lins = []
            for row in self.lle.get_embedding():
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
    encoder = LLEEncoder(g=g,emb_sz=4,workers=8,out_dir='../tmp',out_file='lle-encoder',force=False)
    st_time = time.time()
    encoder.train()
    print('encoder consume {:.2f}'.format(time.time()-st_time))
    out_g = encoder.load()
    print('emb output:',out_g.ndata['emb'][:3,:])
