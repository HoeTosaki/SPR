import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time
import dgl
import torch as tc
from m_encoder import *

class LaplacianEigenmaps:
    def __init__(self, **kwargs):
        ''' Initialize the LaplacianEigenmaps class
        Args:
            d: dimension of the embedding
        '''
        hyper_params = {
            'method_name': 'lap_eigmap_svd'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])

    def learn_embedding(self, graph):
        graph = graph.to_undirected()

        L_sym = nx.normalized_laplacian_matrix(graph)

        w, v = lg.eigs(L_sym, k=self._d + 1, which='SM')
        idx = np.argsort(w) # sort eigenvalues
        w = w[idx]
        v = v[:, idx]
        self._X = v[:, 1:]

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - L_sym)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X.real

    def get_embedding(self):
        # return self._X
        return self._X.real
        # 这里使用Laplacian分解，理论上不会产生虚分量，可以只使用实值作为emb的表达而不丢失数据特征

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

class LEEncoder(Encoder):
    def __init__(self, g, emb_sz=128, workers=1, out_dir='../outputs', out_file='encoder', force=False):
        super(LEEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)

    def train(self):
        self.le = LaplacianEigenmaps(d=self.emb_sz)

        if not self.force and self.check_file():
            print('encoder cache file checked')
            self.load()
            return
        G = dgl.to_networkx(self.g)
        self.le.learn_embedding(graph=G)
        with open(os.path.join(self.out_dir,self.out_file),'w') as f:
            lins = []
            for row in self.le.get_embedding():
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
    encoder = LEEncoder(g=g,emb_sz=4,workers=8,out_dir='../tmp',out_file='le-encoder',force=True)
    st_time = time()
    encoder.train()
    print('encoder consume {:.2f}'.format(time()-st_time))
    out_g = encoder.load()
    print('emb output:',out_g.ndata['emb'][:3,:])
