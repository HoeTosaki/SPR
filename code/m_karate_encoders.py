from embedding_proc import Embedding
import m_deepwalk
import os
import dgl
import m_node2vec
import time
import m_word2vec
import random
import torch as tc
import networkx as nx
import numpy as np
import karateclub
from sklearn.decomposition import PCA
from m_encoder import Encoder

class NodeSketchEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,seed=42):
        super(NodeSketchEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.seed = seed
    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        self.model = karateclub.NodeSketch(dimensions=self.emb_sz,iterations=2,decay=0.01,seed=self.seed)
        G = dgl.to_networkx(self.g)
        G = nx.to_undirected(G)
        self.model.fit(G)
        self.out_emb = self.model.get_embedding()

        with open(os.path.join(self.out_dir, self.out_file), 'w') as f:
            f.write('{} {}\n'.format(int(self.out_emb.shape[0]), int(self.out_emb.shape[1])))
            for idx,line in enumerate(self.out_emb):
                f.write(str(idx) + ' ' + ' '.join([str(ele) for ele in line]) + '\n')
        return

    def save(self):
        pass


class BoostNEEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,seed=42):
        super(BoostNEEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.seed = seed
    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        self.model = karateclub.BoostNE(dimensions=self.emb_sz,iterations=16,order=2,alpha=0.01)
        G = dgl.to_networkx(self.g)
        G = nx.to_undirected(G)
        self.model.fit(G)
        self.out_emb = self.model.get_embedding()
        pca = PCA(n_components=self.emb_sz)
        self.out_emb = pca.fit_transform(self.out_emb)
        assert self.out_emb.shape[1] == self.emb_sz
        with open(os.path.join(self.out_dir, self.out_file), 'w') as f:
            f.write('{} {}\n'.format(int(self.out_emb.shape[0]), int(self.out_emb.shape[1])))
            for idx,line in enumerate(self.out_emb):
                f.write(str(idx) + ' ' + ' '.join([str(ele) for ele in line]) + '\n')
        return

    def save(self):
        pass


class GLEEEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,seed=42):
        super(GLEEEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.seed = seed
    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        self.model = karateclub.GLEE(dimensions=self.emb_sz-1,seed=self.seed)
        G = dgl.to_networkx(self.g)
        G = nx.to_undirected(G)
        self.model.fit(G)
        self.out_emb = self.model.get_embedding()
        with open(os.path.join(self.out_dir, self.out_file), 'w') as f:
            f.write('{} {}\n'.format(int(self.out_emb.shape[0]), int(self.out_emb.shape[1])))
            for idx,line in enumerate(self.out_emb):
                f.write(str(idx) + ' ' + ' '.join([str(ele) for ele in line]) + '\n')
        return

    def save(self):
        pass