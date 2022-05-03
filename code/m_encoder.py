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
import gensim as gs
import karateclub

from sklearn.decomposition import PCA
class Encoder:
    def __init__(self,g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False):
        self.g = g
        self.emb_sz=emb_sz
        self.workers = workers
        self.out_dir = out_dir
        self.out_file = out_file
        self.force=force
    def train(self):
        '''
        定义训练逻辑，根据创建时策略，可能使用历史的缓存加速这一过程
        :return:
        '''
        pass
    def load(self):
        '''
        从磁盘缓存加载数据
        update self.state
        可覆盖
        '''
        emb = Embedding(os.path.join(self.out_dir,self.out_file))
        if not emb.check_with_graph(self.g):
            print('warning: cur emb not match graph')
            emb.debug_with_graph(self.g)
        else:
            emb.add_to_graph(self.g)
        return self.g

    def check_file(self):
        '''
        检查是否有可利用的缓存文件
        在force=False条件下使用，节约时间
        可覆盖
        :return: BOOL
        '''
        return os.path.exists(os.path.join(self.out_dir,self.out_file))

class RandomEncoder(Encoder):
    def __init__(self,g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,scale=1.0,sample_method='uniform',neg_permit=True):
        super(RandomEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.scale = scale
        self.sample_method = sample_method
        self.neg_permit = neg_permit

    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return

        self._X = None
        n = self.g.num_nodes()
        if self.sample_method == 'uniform':
            # [-1,1]
            self._X = (tc.rand(n,self.emb_sz) - 0.5) * 2 * self.scale
        elif self.sample_method == 'gaussian':
            self._X = tc.randn(n,self.emb_sz) * self.scale
        assert self._X.shape[0] == n
        if not self.neg_permit:
            self._X = tc.abs(self._X)
        self.save()

    def save(self):
        with open(os.path.join(self.out_dir,self.out_file),'w') as f:
            lins = []
            _X = self._X.tolist()
            for row in _X:
                assert len(row) >= 2
                line = str(row[0])
                for idx, ele in enumerate(row):
                    if idx != 0:
                        line += ' ' + str(ele)
                lins.append(line + '\n')
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


class DeepWalkEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False, num_walks=40, walk_lens=30,window_sz=20, max_mem=0, seed=0, is_dense_degree=False):
        super(DeepWalkEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.num_walks = num_walks
        self.walk_lens = walk_lens
        self.window_sz = window_sz
        self.max_mem = max_mem
        self.seed = seed
        self.is_dense_degree = is_dense_degree
        # graph adaptor
        # G = load_matfile('/Users/why/Documents/DiskCache/doc/研究资料/代码资源/code-ref/deepwalk/example_graphs/blogcatalog.mat', undirected=True)

    def train(self):
        G = m_deepwalk.Graph()
        for node in self.g.nodes().tolist():
            G[node] = self.g.successors(node).tolist()
        G.make_consistent()

        # %% in case point to G been destroyed.
        self.G = G
        dgl.save_graphs('../tmp/current_graph.tmp',[self.g])
        # %%

        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        print("Number of nodes: {}".format(len(self.G.nodes())))

        num_walks = len(self.G.nodes()) * self.num_walks

        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * self.walk_lens

        print("Data size (walks*length): {}".format(data_size))

        if data_size < self.max_mem:
            print("Walking...")
            walks = m_deepwalk.build_deepwalk_corpus(self.G, num_paths=self.num_walks, path_length=self.walk_lens, alpha=0,
                                          rand=random.Random(self.seed))
            print("Training...")
            model = m_deepwalk.Word2Vec(walks, size=self.emb_sz, window=self.window_sz, min_count=0, sg=1, hs=1,
                             workers=self.workers)
        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
                data_size, self.max_mem))
            print("Walking...")
            walks_filebase = os.path.join(self.out_dir, self.out_file) + ".walks"
            walk_files = m_deepwalk.write_walks_to_disk(self.G, walks_filebase, num_paths=self.num_walks
                                             , path_length=self.walk_lens, alpha=0, rand=random.Random(self.seed),
                                             num_workers=self.workers)

            print("Counting vertex frequency...")
            vertex_counts = 0
            if not self.is_dense_degree:
                vertex_counts = m_deepwalk.count_textfiles(walk_files, self.workers)
            else:
                # use degree distribution for frequency in tree
                vertex_counts = self.G.degree(nodes=self.g.iterkeys())

            print("Training...")
            # print('vertex_counts = ',vertex_counts)
            walks_corpus = m_deepwalk.WalksCorpus(walk_files)
            model = m_deepwalk.Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=self.emb_sz,
                             window=self.window_sz, min_count=0, trim_rule=None, workers=self.workers)
        model.wv.save_word2vec_format(os.path.join(self.out_dir, self.out_file))
    def save(self):
        pass


class Node2VecEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False, num_walks=40, walk_lens=30,window_sz=20, p=1, q=1, iter=1, is_directed=False, is_weighted=False, weight_arr=None):
        super(Node2VecEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.num_walks = num_walks
        self.walk_lens = walk_lens
        self.window_sz = window_sz
        self.p = p
        self.q = q
        self.iter = iter
        self.is_directed = is_directed
        self.is_weighted = is_weighted
        self.weight_arr = weight_arr


    def train(self):
        st_time = time.time()
        print('handle with node2vec graph...')
        nx_G = dgl.to_networkx(self.g)
        new_nx_G = nx.DiGraph()
        org_node,org_edge = nx_G.number_of_nodes(),nx_G.number_of_edges()
        print('\torg node',org_node,'edge',org_edge)
        if not self.is_weighted:
            for edge in nx_G.edges():
                # print((edge[0],edge[1],1))
                new_nx_G.add_weighted_edges_from(ebunch_to_add = [(edge[0],edge[1],1)],weight='weight')
                # nx_G.edges[edge[0]][edge[1]]['weight'] = 1
            nx_G = new_nx_G
        else:
            print('current Node2Vec Model NOT support weighted graph.')
            raise NotImplementedError
        new_node,new_edge = nx_G.number_of_nodes(),nx_G.number_of_edges()
        print('\tnew node', new_node, 'edge', new_edge)
        assert new_node == org_node and new_edge == org_edge

        self.G = m_node2vec.Graph(nx_G=nx_G, is_directed=self.is_directed,p=self.p,q=self.q,out_file=os.path.join(self.out_dir,self.out_file)+'.prob')

        # print('handle with walk graph...')
        # self.G_walk = m_deepwalk.Graph()
        # for node in self.g.nodes().tolist():
        #     self.G_walk[node] = self.g.successors(node).tolist()
        # self.G_walk.make_consistent()

        print('all input graph checked, time consumed:{:.2f}'.format(time.time()-st_time))

        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        self.G.preprocess_transition_probs()

        print("Number of nodes: {}".format(self.g.num_nodes()))

        num_walks = self.g.num_nodes() * self.num_walks

        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * self.walk_lens

        print("Data size (walks*length): {}".format(data_size))

        print("Walking...")
        # self.walks = m_deepwalk.build_deepwalk_corpus(self.G_walk, num_paths=self.num_walks, path_length=self.walk_lens, alpha=0,
        #                               rand=random.Random(self.seed))
        walks = self.G.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_lens)

        print("Training...")
        # model = m_word2vec.Word2Vec(walks, size=self.emb_sz, window=self.window_sz, min_count=0, sg=1, workers=self.workers,
        #                  iter=self.iter)
        model = gs.models.Word2Vec(walks, vector_size=self.emb_sz, window=self.window_sz, min_count=0, sg=1,workers=self.workers,epochs=self.iter)
        model.wv.save_word2vec_format(os.path.join(self.out_dir, self.out_file))
    def save(self):
        pass

class OrionEncoder(RandomEncoder):
    def __init__(self, g, emb_sz=128, workers=1, out_dir='../outputs', out_file='encoder', force=False, scale=1.0,
                 sample_method='uniform', neg_permit=True):
        super(OrionEncoder, self).__init__(g=g, emb_sz=emb_sz, workers=workers, out_dir=out_dir, out_file=out_file,
                                            force=force,scale=scale,sample_method=sample_method,neg_permit=neg_permit)

    def train(self):
        super(OrionEncoder, self).train()
        # if not self.force and self.check_file():
        #     print('encoder cache file checked')
        #     return

        # self.save()


class NetMFEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,order=2,iteration=10,neg_sz=1,seed=42):
        super(NetMFEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.order = order
        self.iteration = iteration
        self.neg_sz = neg_sz
        self.seed = seed
    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return

        # self.netmf = karateclub.NetMF(dimensions=self.emb_sz, iteration=self.iteration, order=self.order,
        #                               negative_samples=self.neg_sz, seed=self.seed)
        self.netmf = None
        assert False
        G = dgl.to_networkx(self.g)
        G = nx.to_undirected(G)
        self.netmf.fit(G)
        self.out_emb = self.netmf.get_embedding()

        with open(os.path.join(self.out_dir, self.out_file), 'w') as f:
            f.write('{} {}\n'.format(int(self.out_emb.shape[0]), int(self.out_emb.shape[1])))
            for idx,line in enumerate(self.out_emb):
                f.write(str(idx) + ' ' + ' '.join([str(ele) for ele in line]) + '\n')
        return

    def save(self):
        pass

class VerseEncoder(Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,force_fit_emb_sz=False):
        super(VerseEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.force_fit_emb_sz = force_fit_emb_sz
    def train(self):
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        print('check file:{}'.format(os.path.join(self.out_dir,'dense-' + self.out_file)))
        if os.path.exists(os.path.join(self.out_dir,'dense-' + self.out_file)):
            embeddings = np.fromfile(os.path.join(self.out_dir,'dense-' + self.out_file), dtype=np.float32).reshape(self.g.num_nodes(),-1)
            if embeddings.shape[1] != self.emb_sz:
                print('warn:unmatched emb sz (file){} != (expect){}'.format(embeddings.shape[1],self.emb_sz))
                if self.force_fit_emb_sz:
                    print('force emb sz by PCA ...')
                    pca = PCA(n_components=self.emb_sz)
                    embeddings = pca.fit_transform(embeddings)
                    assert embeddings.shape[1] == self.emb_sz
                else:
                    assert False
            with open(os.path.join(self.out_dir,self.out_file),'w') as f:
                f.write(str(int(embeddings.shape[0]))+' '+str(int(embeddings.shape[1])) + '\n')
                for idx,emb in enumerate(embeddings):
                    f.write(str(idx) + ' ' + ' '.join([str(ele) for ele in emb]) + '\n')
                f.flush()
            return
        raise NotImplementedError

    def save(self):
        pass


if __name__ == '__main__':
    print('hello encoder.')
    # g, _ = dgl.load_graphs('../datasets/dst/cora')
    # g = g[0]
    # netmf = karateclub.NetMF(dimensions=16, iteration=10, order=2,
    #                          negative_samples=1, seed=42)
    # netmf.fit(dgl.to_networkx(g))
    # out_emb = netmf.get_embedding()
    # print(out_emb[5, :])
    # print('emb shape:', out_emb.shape)
    embeddings = np.fromfile('../tmp/dense-cr-vs-emb=16', dtype=np.float32).reshape(-1,128)
    print(embeddings.shape)
