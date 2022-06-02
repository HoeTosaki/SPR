import dgl
import numpy as np
import networkx as nx
import os
import random
import json
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import copy

import m_node2vec
import m_word2vec
import karateclub as kc
import gensim as gs
import time
import pickle as pk
import tqdm
from keras.preprocessing.sequence import skipgrams
from numpy import float32 as real
from karateclub.utils.walker import BiasedRandomWalker

# device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')
device = 'cpu'

'''
Global.
'''
# path2vec.
neighbors_dict = dict()
current_pos_samples = [[], []]

'''
Model.
'''
class DistanceQueryModel:
    def __init__(self,nx_g=None,model_name='dqm',tmp_dir='../tmp',force=False,**kwargs):
        self.nx_g = nx_g
        self.model_name=model_name
        self.tmp_dir = tmp_dir
        self.force = force
        self.storage_lst = []
        self.train_var_lst = []
        self.query_var_lst = []
        self.train_var_id_set = set()
        self.query_var_id_set = set()
        self.add_train_mem = 0.
        self.add_query_mem = 0.
    def generate(self):
        if not self.force and self.load():
            print('--{}-- tmp file checked&loaded.'.format(self.model_name))
            return
        # force or fail loading.
        ret = self._generate() # if any
        print('--{}-- tmp file checked&loaded.'.format(self.model_name))
        return ret
    def load(self):
        print('--{}-- load tmp file...'.format(self.model_name))
        succ = self._load()
        if succ:
            print('--{}-- tmp file loaded'.format(self.model_name))
        else:
            print('--{}-- tmp file not exist'.format(self.model_name))

    def query(self,srcs,dsts):
        return self._query(srcs,dsts)

    # @Optional
    def get_disk_usage(self):
        return sum([os.stat(file).st_size / 1024/1024 for file in self.storage_lst])

    # @Optional
    def get_mem_usage(self,is_train=True):
        if is_train:
            return (sum([self._ex_getsizeof(ele) for ele in self.train_var_lst]) + self.add_train_mem) / 1024 / 1024
        else:
            return (sum([self._ex_getsizeof(ele) for ele in self.query_var_lst]) + self.add_query_mem) / 1024 / 1024

    def _ex_getsizeof(self,ele):
        if type(ele) is tc.Tensor:
            return sys.getsizeof(np.array(ele))
        return  sys.getsizeof(ele)

    # @Optional
    def query_path(self,srcs,dsts):
        raise NotImplementedError

    # @Optional
    def add_var_lst(self,var_id,var,is_train=True):
        if is_train:
            if var_id not in self.train_var_id_set:
                self.train_var_id_set.add(var_id)
                self.train_var_lst.append(var)
        else:
            if var_id not in self.query_var_id_set:
                self.query_var_id_set.add(var_id)
                self.query_var_lst.append(var)

    def _generate(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def _query(self,srcs,dsts):
        raise NotImplementedError

    def pwd(self):
        return os.path.join(self.tmp_dir,self.model_name)

    def __str__(self):
        return 'DQM:' + self.model_name

    def _save_param_pickle(self):
        raise NotImplementedError

    def _load_param_pickle(self):
        raise NotImplementedError

class ADO(DistanceQueryModel):
    def __init__(self,k=2,**kwargs):
        super(ADO, self).__init__(**kwargs)
        self.k = k
        self.As = []

        # self.storage_lst = [self.pwd() + '.json', self.pwd() + '.ps.npy', self.pwd() + '.deltas.npy']
        self.storage_lst = [self.pwd()+'.pkl']

    def __str__(self):
        return 'ADO:' + self.model_name

    def _generate(self):
        # self.nx_g = nx.Graph()
        lst_nodes = list(self.nx_g.nodes())
        num_nodes = len(lst_nodes)
        for i in range(self.k):
            if i == 0:
                self.As.append(lst_nodes)
                continue
            self.As.append([])
            for nid in self.As[-2]:
                if random.random() < num_nodes ** (-1/self.k):
                    self.As[-1].append(nid)
            if len(self.As[-1]) == 0:
                print('Warning: A_{} randomized with zero shot, whole seq is {}'.format(i,'-'.format([str(len(ele)) for ele in self.As])))
                del self.As[-1]
                break
        self.real_k = len(self.As)
        self.ps = [{} for _ in range(self.real_k)]
        self.deltas = [{} for _ in range(self.real_k)]
        self.c = {}
        self.w2v2d = {} # belong to 2-level hash table of B(v).

        for i in reversed((range(self.real_k))):
            # construct p(v).
            print(f'{self} construct p(v)')
            if i != 0:
                trace_map = -np.ones(shape=(num_nodes,)).astype(dtype=np.int32)
                dist_map = -np.ones(shape=(num_nodes,)).astype(dtype=np.int8)
                search_lst = []
                for nid in tqdm.tqdm(self.As[i]):
                    trace_map[nid] = nid # store from which of w does nid step from.
                    dist_map[nid] = 0 # store distance to s (any w).
                    search_lst.append(nid)
                while len(search_lst) != 0:
                    cur_nid = search_lst.pop(0)
                    for nnid in self.nx_g.neighbors(cur_nid):
                        if dist_map[nnid] < 0:
                            search_lst.append(nnid)
                            dist_map[nnid] = dist_map[nid] + 1
                            trace_map[nnid] = nid
                self.ps[i] = trace_map
                self.deltas[i] = dist_map
                if i != self.real_k - 1:
                    for v in self.ps[i]:
                        if self.deltas[i][v] == self.deltas[i+1][v]:
                            self.ps[i][v] = self.ps[i+1][v] # accelerate by skip some layers.
            else:
                self.ps[i] = np.array(lst_nodes).astype(dtype=np.int32) # self loop at 0 index.
                self.deltas[i] = np.zeros(shape=(num_nodes,)).astype(dtype=np.int8)
            # construct c(w).
            print(f'{self} construct c(w)')
            for w in tqdm.tqdm(self.As[i]):
                w = int(w)
                if w in self.c:
                    continue # has been traversed in A_{i+1}.
                # belong to A_{i-1} - A_{i}.
                # self.w2A[w] = self.w2A.get(w, i) # meets in A_i at first from higher.
                # if i == self.real_k - 1:
                #     self.c[w] = lst_nodes # since delta(A_k, .) = inf
                #     continue
                dist_map2 = {w:0}
                relax_from_nodes = [w]
                while len(relax_from_nodes) != 0:
                    cur_nid = int(relax_from_nodes.pop(0))
                    for nnid in self.nx_g.neighbors(cur_nid):
                        nnid = int(nnid)
                        if nnid not in dist_map2:
                            if i == self.real_k - 1 or dist_map2[cur_nid] + 1 < self.deltas[i+1][nnid]:
                                dist_map2[nnid] = dist_map2[cur_nid] + 1
                                relax_from_nodes.append(nnid)
                self.c[w] = list(dist_map2.keys())
                self.w2v2d[w] = dist_map2

        # construct b(v).
        print(f'{self} construct b(v)')
        self.b = {}
        for w in tqdm.tqdm(self.c):
            w = int(w)
            for nid in self.c[w]:
                nid = int(nid)
                if nid not in self.b:
                    self.b[nid] = set()
                if w not in self.b[nid]:
                    self.b[nid].add(w)
        # transform b to list for saving json.
        for v in self.b:
            self.b[v] = list(self.b[v])
        # del self.c
        # del self.As

        self._save_param_pickle()

        ed_time = time.time()

        # for anal train mem.
        self.train_var_lst.extend(self.As)
        self.train_var_lst.append(self.k)
        self.train_var_lst.append(self.real_k)
        self.train_var_lst.append(self.deltas)
        for delta in self.deltas:
            self.train_var_lst.append(delta) # inner numpy array.
        self.train_var_lst.append(self.ps)
        for p in self.ps:
            self.train_var_lst.append(p)  # inner numpy array.
        self.train_var_lst.append(self.w2v2d)
        for w in self.w2v2d:
            self.train_var_lst.append(self.w2v2d[w]) # inner dict.
        self.train_var_lst.append(self.b)
        for v in self.b:
            self.train_var_lst.append(self.b[v])  # inner set.
        self.train_var_lst.append(self.c)
        for w in self.c:
            self.train_var_lst.append(self.c[w])  # inner list.

        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        dists = []
        for src,dst in zip(srcs,dsts):
            src,dst = int(src),int(dst)
            w = src
            i = 0
            while w not in self.b[dst]:
                i += 1
                src,dst = dst,src
                w = self.ps[i][src]
            dists.append(self.deltas[i][src]+self.w2v2d[w][dst])

        ed_time = time.time()

        # anal query mem.
        self.query_var_lst = []
        self.query_var_lst.append(dists)
        self.query_var_lst.append(self.ps)
        for p in self.ps:
            self.query_var_lst.append(p)  # inner numpy array.
        self.query_var_lst.append(self.deltas)
        for delta in self.deltas:
            self.query_var_lst.append(delta)  # inner numpy array.
        self.query_var_lst.append(self.w2v2d)
        for w in self.w2v2d:
            self.query_var_lst.append(self.w2v2d[w])  # inner dict.

        return dists,ed_time
    def _save_param(self):
        save_dict = {'real_k':self.real_k,
                     'ps_path':self.pwd()+'.ps.npy',
                     'deltas_path':self.pwd()+'.deltas.npy',
                     'b':self.b,
                     'w2v2d':self.w2v2d
                     }
        with open(self.pwd()+'.json','w') as f:
            json.dump(save_dict,f)
        np.save(self.pwd()+'.ps.npy',self.ps)
        np.save(self.pwd() + '.deltas.npy', self.deltas)

    def _save_param_pickle(self):
        with open(self.pwd()+'.pkl','wb') as f:
            pk.dump(self.real_k,f)
            pk.dump(self.ps, f)
            pk.dump(self.deltas, f)
            pk.dump(self.b, f)
            pk.dump(self.w2v2d, f)

    def _load_param_pickle(self):
        with open(self.pwd()+'.pkl','rb') as f:
            self.real_k = pk.load(f)
            self.ps = pk.load(f)
            self.deltas = pk.load(f)
            self.b = pk.load(f)
            self.w2v2d = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.real_k = load_dict['real_k']
        self.ps = np.load(load_dict['ps_path'])
        self.deltas = np.load(load_dict['deltas_path'])
        self.b = load_dict['b']
        self.w2v2d = load_dict['w2v2d']

class LandmarkSelection(DistanceQueryModel):
    def __init__(self,landmark_sz=16,use_sel='degree',margin=-1,use_partition=None,use_inner_sampling=200,**kwargs):
        '''
        :param use_sel: ['degree','cc','random']
        :param use_partition: None
        :param kwargs: ...
        '''
        super(LandmarkSelection, self).__init__(**kwargs)
        self.landmark_sz = landmark_sz
        self.use_sel = use_sel
        self.margin = 0 if margin < 0 else margin
        self.use_partition = use_partition
        self.v2emb = {}
        self.storage_lst = [self.pwd() + '.pkl']
        self.use_inner_sampling = use_inner_sampling

    def __str__(self):
        return 'LandmarkSelection:' + self.model_name

    def _generate(self):
        # self.nx_g = nx.Graph()

        self.train_var_lst = []
        landmarks = []
        print(f'{self} select landmarks')
        if self.use_partition is None:
            num_nodes = self.nx_g.number_of_nodes()
            deg_nodes = self._sel_nodes()
            if self.use_sel != 'random':
                idx_deg = np.argsort(deg_nodes)[::-1]
            else:
                idx_deg = list(range(num_nodes))
                random.shuffle(idx_deg)
                idx_deg = np.array(idx_deg)
            for i in tqdm.tqdm(range(num_nodes)):
                if i == self.landmark_sz:
                    break
                assert deg_nodes[idx_deg[i]] != 0 # assert connected graph.
                if deg_nodes[idx_deg[i]] > 0:
                    landmarks.append(idx_deg[i]) # add current landmark node.

                    # preclude in-margin-proximity neighbors.
                    deg_nodes[idx_deg[i]] = - deg_nodes[idx_deg[i]]
                    dist_map = {idx_deg[i]:0}
                    search_lst = [idx_deg[i]]
                    while len(search_lst) > 0:
                        cur_nid = search_lst.pop(0)
                        if dist_map[cur_nid] <= self.margin - 1:
                            # may have  at least once another transition chance.
                            for nnid in self.nx_g.neighbors(cur_nid):
                                nnid = int(nnid)
                                if nnid not in dist_map:
                                    # not be visited by this turn.
                                    dist_map[nnid] = dist_map[cur_nid] + 1
                                    search_lst.append(nnid)
                    for nid in dist_map:
                        if deg_nodes[nid] > 0:
                            deg_nodes[nid] = -deg_nodes[nid]
            if len(landmarks) < self.landmark_sz:
                # may NOT be executed at most cases.
                print('insufficient sampling to match the max landmark size from {} to {},and take uniform sampling'.format(len(landmarks),self.landmark_sz))
                id_nodes = np.array(list(range(num_nodes)))
                deg_nodes = self._sel_nodes()
                for lid in landmarks:
                    deg_nodes[lid] = -1
                unsel_nids = id_nodes[deg_nodes > 0]
                random.shuffle(unsel_nids)
                landmarks.extend(unsel_nids[:self.landmark_sz - len(landmarks)])
        else:
            assert NotImplementedError

        # calculate dist from landmarks to any nodes.
        self.v2emb = {}
        print(f'{self} construct landmark coordinate')
        for lid in tqdm.tqdm(landmarks):
            dist_map = {lid:0}
            search_lst = [lid]
            while len(search_lst) > 0:
                cur_nid = search_lst.pop(0)
                for nnid in self.nx_g.neighbors(cur_nid):
                    nnid = int(nnid)
                    if nnid not in dist_map:
                        search_lst.append(nnid)
                        dist_map[nnid] = dist_map[cur_nid] + 1
            for k_nid in dist_map:
                k_nid = int(k_nid)
                if k_nid not in self.v2emb:
                    self.v2emb[k_nid] = []
                self.v2emb[k_nid].append(dist_map[k_nid])

        self._save_param_pickle()

        ed_time = time.time()

        # for anal train mem.
        self.train_var_lst.append(landmarks)
        self.train_var_lst.append(deg_nodes)
        self.train_var_lst.append(dist_map)
        self.train_var_lst.append(self.v2emb)
        for nid in self.v2emb:
            self.train_var_lst.append(self.v2emb[nid]) # inner list

        return ed_time
    def _sel_nodes(self):
        if self.use_sel == 'degree':
            num_nodes = self.nx_g.number_of_nodes()
            deg_dict = dict(self.nx_g.degree)
            deg_nodes = np.array([deg_dict[nid] for nid in range(num_nodes)])
            return deg_nodes
        elif self.use_sel == 'random':
            num_nodes = self.nx_g.number_of_nodes()
            return np.ones(shape=(num_nodes,))
        elif self.use_sel == 'cc':
            num_nodes = self.nx_g.number_of_nodes()
            id_nodes = list(range(num_nodes))
            random.shuffle(id_nodes)
            dists_nodes = np.ones(shape=(num_nodes,))
            for nid in id_nodes[:self.use_inner_sampling]:
                dist_map = {nid:0}
                search_lst = [nid]
                while len(search_lst) > 0:
                    cur_nid = search_lst.pop(0)
                    for nnid in self.nx_g.neighbors(cur_nid):
                        nnid = int(nnid)
                        if nnid not in dist_map:
                            search_lst.append(nnid)
                            dist_map[nnid] = dist_map[nid] + 1
                for k_nid in dist_map:
                    dists_nodes[k_nid] += dist_map[k_nid]
            dists_nodes /= self.use_inner_sampling
            self.train_var_lst.append(dist_map)
            self.train_var_lst.append(id_nodes)
            self.train_var_lst.append(dists_nodes)
            return dists_nodes

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        dists = []
        for src,dst in zip(srcs,dsts):
            src,dst = int(src),int(dst)
            dist = min([d1+d2 for d1,d2 in zip(self.v2emb[src],self.v2emb[dst])])
            dists.append(dist)

        ed_time = time.time()

        # anal query mem.
        self.query_var_lst = []
        self.query_var_lst.append(dists)
        self.query_var_lst.append(self.v2emb)
        for nid in self.v2emb:
            self.query_var_lst.append(self.v2emb[nid]) # inner list

        return dists,ed_time

    def _save_param(self):
        save_dict = {'v2emb':self.v2emb,}
        with open(self.pwd()+'.json','w') as f:
            json.dump(save_dict,f)

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.v2emb, f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.v2emb = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.v2emb = load_dict['v2emb']

class Orion(DistanceQueryModel):
    def __init__(self,use_sel='random',emb_sz=16,init_sz=16,landmark_sz=100,batch_node_sz=1,max_iter=[5000,1000,100],step_len=5,tol=1e-5,**kwargs):
        '''
        :param use_sel: in ['random','degree']
        '''
        super(Orion, self).__init__(**kwargs)
        self.use_sel = use_sel
        self.emb_sz = emb_sz
        self.init_sz = init_sz
        self.landmark_sz = landmark_sz
        self.batch_node_sz = batch_node_sz
        self.max_iter = max_iter
        self.step_len = step_len
        self.tol = tol

        # self.storage_lst = [self.pwd() + '.json', self.pwd() + '.embs.npy']
        self.storage_lst = [self.pwd() + '.pkl']
    def __str__(self):
        return 'Orion:' + self.model_name

    def _generate(self):
        # self.nx_g = nx.DiGraph()
        self.train_var_lst = []
        self.train_var_id_set = set()

        deg_nodes,idx_deg = self._sel_nodes()
        landmarks_init = idx_deg[:self.init_sz]
        landmarks_sec = idx_deg[self.init_sz:self.landmark_sz]
        all_nodes = idx_deg[self.landmark_sz:]
        # generate dist map.
        print('{} gen dist map..'.format(self))
        self.dist_maps = self._gen_dist_map(np.concatenate([landmarks_init,landmarks_sec],axis=0))

        # optimize inner initial nodes.
        print('{} optimizes initial nodes emb..'.format(self))
        init_node_verts = np.random.randn(self.init_sz*self.emb_sz)
        _init_node_verts = self._vertice_init(init_node_verts, self.step_len)
        static_bias_init = self._func_init(landmarks_init)
        vertice_max_list, vertice_min_list = self._optim_dsim(static_bias=static_bias_init,f=self._func_obj, vertice=_init_node_verts, maxit=self.max_iter[0], step_length=self.step_len, tol=self.tol,optim_type='init')
        print('{} initial nodes emb optimized with loss:{:4f}'.format(self,self._func_obj(optim_vertice=vertice_min_list[-1],static_bias=static_bias_init,optim_type='init')))
        init_node_verts = vertice_min_list[-1].reshape(self.init_sz,self.emb_sz)

        # optimize sec nodes.
        print('{} optimizes sec nodes emb..'.format(self))
        sec_node_verts = np.random.randn(len(landmarks_sec) * self.emb_sz)
        _sec_node_verts = self._vertice_init(sec_node_verts, self.step_len)
        static_bias_sec = self._func_other(fixed_node_vertice=init_node_verts,fixed_nodes=landmarks_init,other_nodes=landmarks_sec)
        vertice_max_list, vertice_min_list = self._optim_dsim(static_bias=static_bias_sec, f=self._func_obj,vertice=_sec_node_verts,maxit=self.max_iter[1],step_length=self.step_len, tol=self.tol,optim_type='sec')
        print('{} sec nodes emb optimized with loss:{:4f}'.format(self,self._func_obj(optim_vertice=vertice_min_list[-1], static_bias=static_bias_sec,optim_type='sec')))
        sec_node_verts = vertice_min_list[-1].reshape(len(landmarks_sec),self.emb_sz)

        # optimize all nodes.
        print('{} optimizes all nodes emb..'.format(self))
        st_idx = 0
        batch_loss = 0.
        all_node_verts = []
        while st_idx < len(all_nodes):
            cur_nodes = all_nodes[st_idx:min(len(all_nodes),st_idx + self.batch_node_sz)]
            cur_len = st_idx + len(cur_nodes)
            cur_node_verts = np.random.randn(len(cur_nodes) * self.emb_sz)
            _cur_node_verts = self._vertice_init(cur_node_verts, self.step_len)
            static_bias_cur = self._func_other(fixed_node_vertice=np.concatenate([init_node_verts,sec_node_verts],axis=0), fixed_nodes=np.concatenate([landmarks_init,landmarks_sec],axis=0),other_nodes=cur_nodes)
            vertice_max_list, vertice_min_list = self._optim_dsim(static_bias=static_bias_cur, f=self._func_obj,vertice=_cur_node_verts, maxit=self.max_iter[2],step_length=self.step_len, tol=self.tol,optim_type='all')
            batch_loss += self._func_obj(optim_vertice=vertice_min_list[-1],static_bias=static_bias_cur,optim_type='all')
            all_node_verts.append(vertice_min_list[-1].reshape(len(cur_nodes),self.emb_sz))
            print('{} - {}/{} traversed nodes emb optimized with avg loss:{:4f}'.format(self,cur_len,len(all_nodes),batch_loss /cur_len /self.batch_node_sz/self.batch_node_sz))
            self.add_var_lst(var_id='iter-cur-all-static',var=static_bias_cur,is_train=True)
            self.add_var_lst(var_id='iter-cur-all-verts(eye)', var=_cur_node_verts, is_train=True)
            st_idx += self.batch_node_sz

        all_node_verts = np.concatenate(all_node_verts,axis=0)
        print('{} all nodes emb optimized with final avg loss:{:4f}'.format(self,batch_loss / len(all_nodes) / self.batch_node_sz / self.batch_node_sz))

        embs = np.concatenate([init_node_verts,sec_node_verts,all_node_verts],axis=0)
        self.embs = np.zeros(shape=(self.nx_g.number_of_nodes(),self.emb_sz))
        for i in range(embs.shape[0]):
            self.embs[idx_deg[i]] = embs[i]
        self.embs = tc.from_numpy(self.embs)

        self._save_param_pickle()

        ed_time = time.time()
        # anal mem.
        self.train_var_lst.append(idx_deg)
        self.train_var_lst.append(deg_nodes)
        self.train_var_lst.append(landmarks_init)
        self.train_var_lst.append(landmarks_sec)
        self.train_var_lst.append(all_nodes)
        self.train_var_lst.append(self.dist_maps)
        for nid in self.dist_maps:
            self.train_var_lst.append(self.dist_maps[nid]) # inner dict

        self.train_var_lst.append(init_node_verts)
        self.train_var_lst.append(_init_node_verts)
        for ele in _init_node_verts:
            self.train_var_lst.append(ele) # inner matrix.
        self.train_var_lst.append(static_bias_init)

        self.train_var_lst.append(sec_node_verts)
        self.train_var_lst.append(_sec_node_verts)
        for ele in _sec_node_verts:
            self.train_var_lst.append(ele)  # inner matrix.
        self.train_var_lst.append(static_bias_sec)

        self.train_var_lst.append(all_node_verts)

        self.train_var_lst.append(self.embs)

        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        dists = tc.sum(self.embs[srcs]*self.embs[dsts],dim=1)

        ed_time = time.time()
        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        return dists,ed_time

    def _save_param(self):
        save_dict = {'embs_path': self.pwd() + '.embs.npy'}
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(save_dict, f)
        np.save(self.pwd() + '.embs.npy', self.embs.numpy())

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.embs = tc.from_numpy(np.load(load_dict['embs_path']))

    def _sel_nodes(self):
        if self.use_sel == 'degree':
            num_nodes = self.nx_g.number_of_nodes()
            deg_dict = dict(self.nx_g.degree)
            deg_nodes = np.array([deg_dict[nid] for nid in range(num_nodes)])
            idx_deg = np.argsort(deg_nodes)[::-1]
            return deg_nodes,idx_deg
        elif self.use_sel == 'random':
            num_nodes = self.nx_g.number_of_nodes()
            idx_deg = list(range(num_nodes))
            random.shuffle(idx_deg)
            idx_deg = np.array(idx_deg)
            return np.ones(shape=(num_nodes,)),idx_deg

    def _gen_dist_map(self,landmarks):
        dist_maps = {}
        for nid in tqdm.tqdm(landmarks):
            dist_map = {nid:0}
            search_lst = [nid]
            while len(search_lst) != 0:
                cur_nid = search_lst.pop(0)
                for nnid in self.nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        search_lst.append(nnid)
            dist_maps[nid] = dist_map
        return dist_maps

    def _func_init(self,init_nodes):
        static_bias = np.zeros(shape=(self.init_sz,self.init_sz))
        for i in range(self.init_sz):
            for j in range(self.init_sz):
                static_bias[i,j] = self.dist_maps[init_nodes[i]][init_nodes[j]]
        return static_bias

    def _func_other(self,fixed_node_vertice,fixed_nodes,other_nodes):
        fixed_nodes_sz = len(fixed_nodes)
        other_nodes_sz = len(other_nodes)
        static_bias = {}
        static_bias['dist'] = np.zeros(shape=(other_nodes_sz,fixed_nodes_sz))
        static_bias['fixed_node_vertice'] = fixed_node_vertice
        for i in range(other_nodes_sz):
            for j in range(fixed_nodes_sz):
                static_bias['dist'][i,j] = self.dist_maps[fixed_nodes[j]][other_nodes[i]]
        return static_bias

    def _func_obj(self,optim_vertice,static_bias,optim_type='init'):
        if optim_type == 'init':
            optim_vertice = optim_vertice.reshape(self.init_sz,self.emb_sz)
            return 0.5 * np.sum((((np.matmul(optim_vertice,optim_vertice.T) - static_bias).reshape(-1)) ** 2))
        elif optim_type in ['sec','all']:
            fixed_node_vertice = static_bias['fixed_node_vertice']
            fixed_node_vertice.reshape(-1,self.emb_sz) # unk&flexible for fixed num.
            optim_vertice = optim_vertice.reshape(-1,self.emb_sz)
            return 0.5 * np.sum(((np.matmul(optim_vertice,fixed_node_vertice.T) - static_bias['dist']).reshape(-1)) ** 2)
        assert False,print('unk optim type:{}'.format(optim_type))

    def _optim_dsim(self,static_bias=None,f=None, vertice=None, maxit=1000, step_length=100, tol=1e-3,optim_type='init'):
        '''
            downhill-simplex optimization.
        '''

        vertice_max_list = []  # store the max vertex during each iteration
        vertice_min_list = []  # store the min vertex during each iteration
        for jj in range(maxit):
            if optim_type in ['init','sec'] and jj % int((maxit * 0.1)) == 0 and len(vertice_min_list) > 0:
                print('{} - {} iter {}/{} optim loss:{:.4f}'.format(self, optim_type, jj, maxit,
                                                                    f(optim_vertice=vertice_min_list[-1],
                                                                      static_bias=static_bias, optim_type=optim_type)))
            y = []
            for ii in vertice:
                y.append(f(optim_vertice=ii,static_bias=static_bias,optim_type=optim_type))
            y = np.array(y)
            #  only the highest (worst), next-highest, and lowest (best) vertice are needed
            idx = np.argsort(y)
            vertice_max_list.append(vertice[idx[-1]])
            vertice_min_list.append(vertice[idx[0]])

            # centroid of the best n vertice
            # NOTE: the worst vertex should be excluded, but for simplicity we don't do this
            v_mean = np.mean(vertice)

            # compute the candidate vertex and corresponding function value
            v_ref = self._line(-1, v_mean, vertice[idx[-1]])
            y_ref = f(optim_vertice=v_ref,static_bias=static_bias,optim_type=optim_type)
            if y_ref >= y[idx[0]] and y_ref < y[idx[-2]]:
                # y_0<=y_ref<y_n, reflection (replace v_n+1 with v_ref)
                vertice[idx[-1]] = v_ref
                # print('reflection1')
            elif y_ref < y[idx[0]]:
                # y_ref<y_0, expand
                v_ref_e = self._line(-2, v_mean, vertice[idx[-1]])
                y_ref_e = f(optim_vertice=v_ref_e,static_bias=static_bias,optim_type=optim_type)
                if y_ref_e < y_ref:
                    vertice[idx[-1]] = v_ref_e
                    # print('expand')
                else:
                    vertice[idx[-1]] = v_ref
                    # print('reflection2')
            elif y_ref >= y[idx[-2]]:
                if y_ref < y[idx[-1]]:
                    # y_ref<y_{n+1}, outside contraction
                    v_ref_c = self._line(-0.5, v_mean, vertice[idx[-1]])
                    y_ref_c = f(optim_vertice=v_ref_c,static_bias=static_bias,optim_type=optim_type)
                    if y_ref_c < y_ref:
                        vertice[idx[-1]] = v_ref_c
                    # print('outside contraction')
                else:
                    # y_ref>=y_{n+1} inside contraction
                    v_ref_c = self._line(0.5, v_mean, vertice[idx[-1]])
                    y_ref_c = f(optim_vertice=v_ref_c,static_bias=static_bias,optim_type=optim_type)
                    if y_ref_c < y_ref:
                        vertice[idx[-1]] = v_ref_c
                        # print('inside contraction')
                        continue
                    # shrinkage
                    for ii in range(1, len(vertice)):
                        vertice[ii] = 0.5 * (vertice[0] + vertice[ii])
                        # print('shrinkage')
                    vertice = self._vertice_init(vertice[idx[0]], step_length)
            # restart
            # restarting is very important during iteration, for the simpex
            # can easily stucked into a nonoptimal point
            rtol = 2.0 * abs(y[idx[0]] - y[idx[-1]]) / (
                    abs(y[idx[0]]) + abs(y[idx[-1]]) + 1e-9)
            if rtol <= tol:
                vertice = self._vertice_init(vertice[idx[0]], step_length)

        # anal mem, log each optim routine for once at first.
        self.add_var_lst(optim_type+'-optim.min_list',vertice_min_list,is_train=True)
        self.add_var_lst(optim_type+'-optim.max_list',vertice_max_list,is_train=True)

        return vertice_max_list, vertice_min_list

    def _line(self,t, v1, v2):
        return (1 - t) * v1 + t * v2

    def _vertice_init(self,vertex_0, step_length):
        '''
            initialize vertice of the simplex, using: $xi=x0+step_length*ei$
        '''
        emat = np.eye(vertex_0.size) * step_length
        vertice = [vertex_0]
        for ii in range(vertex_0.size):
            vertice.append(vertex_0 + emat[:, ii])
        return vertice

class Rigel(Orion):
    def __init__(self,curvature=-1,**kwargs):
        super(Rigel, self).__init__(**kwargs)
        self.curvature = curvature

    def __str__(self):
        return 'Rigel:' + self.model_name

    def _func_obj(self,optim_vertice,static_bias,optim_type='init'):
        if optim_type == 'init':
            optim_vertice = optim_vertice.reshape(self.init_sz,self.emb_sz)
            return 0.5 * np.sum((((np.matmul(optim_vertice,optim_vertice.T) - static_bias).reshape(-1)) ** 2))
        elif optim_type in ['sec','all']:
            fixed_node_vertice = static_bias['fixed_node_vertice']
            fixed_node_vertice.reshape(-1,self.emb_sz) # unk&flexible for fixed num.
            optim_vertice = optim_vertice.reshape(-1,self.emb_sz)
            return 0.5 * np.sum(((np.matmul(optim_vertice,fixed_node_vertice.T) - static_bias['dist']).reshape(-1)) ** 2)
        assert False,print('unk optim type:{}'.format(optim_type))

    def _hyper_dist(self,x,y):
        dist_emb = np.zeros(x.shape[0],y.shape[0])
        for j in y.shape[0]:
            dist_emb[:,j] = np.arccosh( ( np.sqrt((1 + np.sum(x*x,axis=1))(1+np.sum(y[j]*y[j]))) - np.sum(x*y,axis=1) )*abs(self.curvature) )
        return dist_emb

    def _query(self,srcs,dsts):
        dists = tc.sum(self.embs[srcs]*self.embs[dsts],dim=1)
        for idx,(src,dst) in enumerate(zip(srcs,dsts)):
            src,dst = int(src),int(dst)
            if src == dst:
                dists[idx] = 0
            elif self.nx_g.has_edge(src,dst):
                dists[idx] = 1
        ed_time = time.time()
        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        # self.query_var_lst.append(self.nx_g)
        self.query_var_lst.append(list(self.nx_g.edges()))

        return dists,ed_time

'''
Pruned Landmark Labeling.
'''
class PLL(DistanceQueryModel):
    def __init__(self,use_order='degree',use_inner_sampling=200,**kwargs):
        '''
        :param use_sel: ['degree','cc','random']
        :param kwargs: ...
        '''
        super(PLL, self).__init__(**kwargs)
        self.use_order = use_order
        self.use_inner_sampling = use_inner_sampling

        self.v2lb = {}
        self.storage_lst = [self.pwd() + '.pkl']

    def __str__(self):
        return 'PLL:' + self.model_name

    def _generate(self):
        # self.nx_g = nx.Graph()
        self.train_var_lst = []
        num_nodes = self.nx_g.number_of_nodes()
        idx_deg,deg_nodes = self._sel_nodes() # node order.

        # gen labels for every node.
        self.v2lb = {v:[] for v in range(num_nodes)}
        for idx,nid in enumerate(idx_deg):
            if idx == 1 or idx % int(num_nodes * 0.01):
                print('{} - generate landmarks for {}/{} completed.'.format(self,idx,num_nodes))
            dist_map = {nid:0}
            search_lst = [nid]
            self.ins_landmark(int(nid),(int(nid),0))
            while len(search_lst) > 0:
                cur_nid = search_lst.pop(0)
                for nnid in self.nx_g.neighbors(cur_nid):
                    nnid = int(nnid)
                    if nnid not in dist_map:
                        if self._meta_query(nid,nnid,cur_nid=nid) > dist_map[cur_nid] + 1:
                            search_lst.append(nnid)
                            dist_map[nnid] = dist_map[cur_nid] + 1
                            self.ins_landmark(nnid,(int(nid),int(dist_map[nnid])))

        self._save_param_pickle()
        print('{} - generate landmarks {} completed.'.format(self,num_nodes))

        ed_time = time.time()

        # for anal train mem.
        self.train_var_lst.append(idx_deg)
        self.train_var_lst.append(deg_nodes)
        self.train_var_lst.append(dist_map)
        self.train_var_lst.append(self.v2lb)
        for v in self.v2lb:
            self.train_var_lst.append(self.v2lb[v]) # inner list
        return ed_time

    def _meta_query(self,src,dst,cur_nid = -1):
        if src not in self.v2lb or dst not in self.v2lb:
            return 9999
        pnt1,pnt2 = 0,0
        dist = 9999
        # mergesort.
        while pnt1 < len(self.v2lb[src]) and pnt2 < len(self.v2lb[dst]):
            if self.v2lb[src][pnt1][0] > self.v2lb[dst][pnt2][0]:
                pnt2 += 1
            elif self.v2lb[src][pnt1][0] < self.v2lb[dst][pnt2][0]:
                pnt1 += 1
            else:
                if self.v2lb[src][pnt1][0] != cur_nid:
                    dist = min(dist,self.v2lb[src][pnt1][1] + self.v2lb[dst][pnt2][1])
                pnt1 += 1
                pnt2 += 1
        if dist == 9999:
            print('error for:{}-{}'.format(src,dst))
        return dist

    def ins_landmark(self,nid,lm):
        length = len(self.v2lb[nid])
        if length == 0:
            self.v2lb[nid].append(lm)
        else:
            pnt = length - 1
            lst = self.v2lb[nid]
            lst.append(None)
            while pnt >= 0 and lst[pnt][0] > lm[0]:
                lst[pnt + 1] = lst[pnt]
                pnt -= 1
            lst[pnt + 1] = lm

    def _sel_nodes(self):
        if self.use_order == 'degree':
            num_nodes = self.nx_g.number_of_nodes()
            deg_dict = dict(self.nx_g.degree)
            deg_nodes = np.array([deg_dict[nid] for nid in range(num_nodes)])
            idx_deg = np.argsort(deg_nodes)[::-1]
            return idx_deg,deg_nodes
        elif self.use_order == 'random':
            num_nodes = self.nx_g.number_of_nodes()
            deg_nodes = np.ones(shape=(num_nodes,))
            idx_deg = np.array(list(range(num_nodes)))
            np.random.shuffle(idx_deg)
            return idx_deg,deg_nodes
        elif self.use_order == 'cc':
            num_nodes = self.nx_g.number_of_nodes()
            id_nodes = list(range(num_nodes))
            random.shuffle(id_nodes)
            dists_nodes = np.ones(shape=(num_nodes,))
            for nid in id_nodes[:self.use_inner_sampling]:
                dist_map = {nid:0}
                search_lst = [nid]
                while len(search_lst) > 0:
                    cur_nid = search_lst.pop(0)
                    for nnid in self.nx_g.neighbors(cur_nid):
                        nnid = int(nnid)
                        if nnid not in dist_map:
                            search_lst.append(nnid)
                            dist_map[nnid] = dist_map[nid] + 1
                for k_nid in dist_map:
                    dists_nodes[k_nid] += dist_map[k_nid]
            dists_nodes /= self.use_inner_sampling
            idx_deg = np.argsort(dists_nodes)
            return idx_deg,dists_nodes

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        dists = []
        for src,dst in zip(srcs,dsts):
            src,dst = int(src),int(dst)
            dists.append(self._meta_query(src,dst,cur_nid=-1))
        ed_time = time.time()
        # anal query mem.
        self.query_var_lst = []
        self.query_var_lst.append(self.v2lb)
        for v in self.v2lb:
            self.query_var_lst.append(self.v2lb[v]) # inner list

        return np.array(dists),ed_time

    def _save_param(self):
        save_dict = {'v2lb':self.v2lb,}
        with open(self.pwd()+'.json','w') as f:
            json.dump(save_dict,f)

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.v2lb, f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.v2lb = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        v2lb = load_dict['v2lb']

        # re-param for unexpected format error.
        self.v2lb = {}
        for k in v2lb:
            self.v2lb[int(k)] = v2lb[k]

'''
Sampling-based Path Greedy
'''
class SamPG(PLL):
    def __init__(self,boost_sel=1,**kwargs):
        '''
        :param kwargs: ...
        '''
        super(SamPG, self).__init__(**kwargs)
        self.boost_sel = boost_sel
        self.storage_lst = [self.pwd() + '.pkl']
        self.hubs = []

    def __str__(self):
        return 'SamPG:' + self.model_name

    def _generate(self):
        num_nodes = self.nx_g.number_of_nodes()
        self.v2lb = {v: [] for v in range(num_nodes)}
        self.tsam = TreeSampling(nx_g=self.nx_g,num_nodes=num_nodes,boost_sel=self.boost_sel)

        node_lst = list(self.nx_g.nodes())
        is_hub = [False]*num_nodes
        is_sel = [True] * num_nodes
        random.shuffle(node_lst)

        const_log_n = 4
        log_n = math.ceil(math.log2(num_nodes))
        desired_samples = log_n*const_log_n*num_nodes
        desired_pairs = log_n*const_log_n*self.boost_sel

        idx = 0
        while idx < num_nodes:
            if self.tsam.size_deleted() > max(self.tsam.size_all(),desired_samples):
                print('clear trees {} with respect to max(size_all:{},desired_samples:{}).'.format(self.tsam.size_deleted(),self.tsam.size_all(),desired_samples))
                self.tsam.clear()
                idx = 0
            while (self.tsam.size_all() < desired_samples or self.tsam.size_pairs() < desired_pairs) and idx < num_nodes:
                while idx < num_nodes and is_hub[node_lst[idx]]:
                    idx += 1
                if idx < num_nodes:
                    node_visited,parent_visited = self._meta_generate(node_lst[idx],is_add_hub=False)
                    for v_nid in node_visited:
                        if v_nid == node_lst[idx]:
                            continue
                        assert is_hub[v_nid] is False,print('a hub {} visited when traverse from {}'.format(v_nid,node_lst[idx]))
                    idx += 1
                    self.tsam.add_tree(node_visited,parent_visited,is_sel)


            if self.tsam.queue_empty():
                if idx < num_nodes:
                    print('no sel pair covered')
                    self.tsam.clear()
                    idx = 0
                continue

            # found the best cover hub node.
            cover_nid,_ = self.tsam.pop_best_cover()
            assert is_hub[cover_nid] is False
            self._meta_generate(cover_nid,is_add_hub=True)

            self.tsam.remove_subtrees(cover_nid)

            idx += 1


    def _meta_generate(self,nid,is_add_hub=False):
        dist_map = {nid: 0}
        search_lst = [nid]
        node_visited = []
        parent_visited = []
        if is_add_hub:
            self.ins_landmark(int(nid), (int(nid), 0))
            self.hubs.append(nid)
        node_visited.append(nid)
        parent_visited.append(nid)
        while len(search_lst) > 0:
            cur_nid = search_lst.pop(0)
            for nnid in self.nx_g.neighbors(cur_nid):
                nnid = int(nnid)
                if nnid not in dist_map:
                    if self._meta_query(nid, nnid, cur_nid=nid) > dist_map[cur_nid] + 1:
                        search_lst.append(nnid)
                        node_visited.append(nnid)
                        parent_visited.append(cur_nid)
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        if is_add_hub:
                            self.ins_landmark(nnid, (int(nid), int(dist_map[nnid])))
        return node_visited,parent_visited

    def _save_param(self):
        save_dict = {'v2lb':self.v2lb,}
        with open(self.pwd()+'.json','w') as f:
            json.dump(save_dict,f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        v2lb = load_dict['v2lb']

        # re-param for unexpected format error.
        self.v2lb = {}
        for k in v2lb:
            self.v2lb[int(k)] = v2lb[k]

class TreeSampling:
    def __init__(self,nx_g,num_nodes,boost_sel=1):
        self.nx_g = nx_g
        self.num_nodes = num_nodes
        self.boost_sel = boost_sel

        self.trees = []
        self.trees_vtx = [] # vertex in trees
        self.trees_sel = [] # sel in trees
        self.in_trees = [None]*num_nodes # index in trees
        self.n_subtree = [0]*num_nodes
        self.n_labs = [0]*num_nodes # num of labels
        self.n_pairs = [0]*num_nodes # num of pairs

        self.n_all_trees = 0
        self.n_deleted = 0
        self.n_all_pairs = 0

        self.queue = utils.MinMaxHeap()


    def queue_empty(self):
        return self.queue.size == 0

    def size_deleted(self):
        return self.n_deleted

    def size_all(self):
        return self.n_all_trees

    def size_pairs(self):
        return self.n_all_pairs

    def add_tree(self,node_visited,parent_visited,is_sel):
        # self.nx_g = nx.Graph()
        t = len(self.trees)
        tn = len(node_visited)
        vtx = copy.deepcopy(node_visited)
        pvtx = copy.deepcopy(parent_visited)
        for i in range(tn):
            self.n_subtree[vtx[i]] = 0
        self.trees.append(self.nx_g.subgraph(node_visited))
        self.trees_vtx.append(vtx)
        assert len(self.trees) == len(self.trees_vtx)
        self.trees_sel.append(is_sel)
        assert len(self.trees) == len(self.trees_sel)
        self.n_all_trees += tn
        for i in range(tn-1,-1,-1):
            u = vtx[i]
            if self.in_trees[u] is None:
                self.in_trees[u] = []
            self.in_trees[u].append((t,i))
            if is_sel[u]:
                self.n_subtree[u] += self.boost_sel
            else:
                self.n_subtree[u] += 1
            if pvtx[i] != u:
                self.n_subtree[pvtx[i]] += self.n_subtree[u]
            if self.n_subtree[u] > 0:
                self.n_labs[u] += 1
                self.n_pairs[u] += self.n_subtree[u]
                self.n_all_pairs += self.n_subtree[u]
                self.queue.insert((u,None))

    def dfs_del(self,tree,vtx,sel,i):
        assert tree.number_of_nodes() == len(vtx),print('{}-{}'.format(tree.number_of_nodes(),len(vtx)))
        count = 0
        if vtx[i] > -1:
            v = vtx[i]
            vtx[i] = -1
            if sel[v]:
                count = self.boost_sel
            else:
                count = 1
            for j in tree.neighbors(v):
                if j in vtx:
                    j = vtx.index(int(j))
                    if j > i:
                        count += self.dfs_del(tree,vtx,sel,j)
            self.n_labs[v] -= 1
            self.n_pairs[v] -= count
            self.n_all_pairs -= count
            assert self.n_labs[v] >= 0 and self.n_pairs[v] >= 0
            if self.n_labs[v] == 0 or self.n_pairs[v] == 0:
                idx_del = -1
                for idx,ele in enumerate(self.queue.a):
                    if ele[0] == v:
                        idx_del=idx
                assert idx_del != -1
                self.queue.remove(idx_del)
            else:
                self.queue.insert((v,None))
        return count

    def dfs_count(self,tree,vtx,i):
        count = 0
        if vtx[i] > -1:
            count = 1
            for j in tree.neighbors(vtx[i]):
                j = vtx.index(int(j))
                if j > i:
                    count += self.dfs_count(tree,vtx,j)
        return count

    def clear(self):
        self.trees = []
        self.trees_vtx = []
        self.trees_sel = []
        for i in range(len(self.in_trees)):
            self.in_trees[i] = None
            self.n_labs[i] = 0
            self.n_pairs[i] = 0
        self.n_all_trees = 0
        self.n_deleted = 0
        self.n_all_pairs = 0
        self.queue = utils.MinMaxHeap()


    def pop_best_cover(self):
        # TODO: pop what?
        return self.queue.popmin()

    def remove_subtrees(self,nid):
        nd = 0
        for pair in self.in_trees[nid]:
            if pair is None:
                continue
            x, ix = pair
            tre = self.trees[x]
            vtx = self.trees_vtx[x]
            sel = self.trees_sel[x]
            if vtx[ix] == -1:
                continue # already removed from tre.
            nd+=1
            x_subtree_size = self.dfs_count(tre,vtx,ix)
            x_subtree_score = self.dfs_del(tre,vtx,sel,ix)
            self.n_all_trees -= x_subtree_size
            self.n_deleted += x_subtree_size
            ip = ix
            while True:
                for j in tre.neighbors(ix):
                    if j in vtx:
                        j = vtx.index(int(j))
                        if j < ix:
                            ip = j
                            break
                if ip >= ix:
                    break
                v = vtx[ip]
                assert v != -1
                self.n_pairs[v] -= x_subtree_score
                self.n_all_pairs -= x_subtree_score
                assert self.n_labs[v] > 0 and self.n_pairs[v] >= 0
                if self.n_labs[v] == 0 or self.n_pairs[v] == 0:
                    idx_del = -1
                    for idx, ele in enumerate(self.queue.a):
                        if ele[0] == v:
                            idx_del = idx
                    assert idx_del != -1
                    self.queue.remove(idx_del)
                else:
                    self.queue.insert((v,None))
                ix = ip
        assert self.n_labs[x] == 0 and self.n_pairs[x] == 0
        self.in_trees[x] = None

class DistDecoder_DADL(nn.Module):
    def __init__(self, emb_sz=16):
        super(DistDecoder_DADL, self).__init__()
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
        self.lin2 = nn.Linear(emb_sz, 1)

    def forward(self, src, dst):
        out = self.lin1(tc.cat((src, dst), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class DADL(DistanceQueryModel):
    def __init__(self,emb_sz=16,landmark_sz=100,lr=0.01,iters=15,p=1,q=1,l=80,k=5,num_walks=10,num_workers=8,batch_landmark_sz=5,**kwargs):
        '''
        :param emb_sz: model embedding size
        :param landmark_sz: landmark size for training node pair
        :param lr: learning rate in NN.
        :param iters: total iterations in NN.
        :param p: control parameter in node2vec random walk
        :param q: control parameter in node2vec random walk
        :param l: length of random walk
        :param k: negative sampling size in skip-gram.
        :param num_walks: number of walks in skip-gram.
        :param kwargs:
        '''
        super(DADL, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.landmark_sz = landmark_sz
        self.lr = lr
        self.iters = iters
        self.p = p
        self.q = q
        self.l = l
        self.k = k
        self.num_walks = num_walks
        self.num_workers=num_workers
        self.batch_landmark_sz = batch_landmark_sz
        # self.storage_lst = [self.pwd() + '.json', self.pwd() + '.embs.npy',self.pwd()+'.model']
        self.storage_lst = [self.pwd() + '.pkl']

    def __str__(self):
        return 'DADL:' + self.model_name

    @staticmethod
    def _s_gen_train_pair(landmarks,nx_g,**kwargs):
        # nx_g = nx.Graph()
        pid = kwargs['__PID__']
        train_pairs = []
        for landmark in landmarks:
            search_lst = [landmark]
            dist_map = {landmark:0}
            while len(search_lst) != 0:
                cur_nid = search_lst.pop(0)
                for nnid in nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        search_lst.append(nnid)
            for nid in dist_map:
                if dist_map[nid] >= 1:
                    train_pairs.append([landmark,nid,dist_map[nid]])
        return train_pairs

    def _generate(self):
        # self.nx_g = nx.DiGraph()
        # gen train pairs.
        mpm = utils.MPManager(batch_sz=self.batch_landmark_sz,num_workers=self.num_workers,use_shuffle=False)
        nodes = list(self.nx_g.nodes())
        random.shuffle(nodes)
        landmarks = nodes[:self.landmark_sz]
        train_pairs = mpm.multi_proc(DADL._s_gen_train_pair,[landmarks],nx_g=self.nx_g,auto_concat=True)
        train_pairs = tc.FloatTensor(train_pairs)
        # train embs with Node2Vec.

        print('{} start to train embeddings with Node2Vec...'.format(self))
        st_time = time.time()
        # walker = BiasedRandomWalker(self.l, self.num_walks, self.p, self.q)
        # walker.do_walks(self.nx_g)
        # model = Word2Vec(
        #     walker.walks,
        #     hs=1,
        #     alpha=self.learning_rate,
        #     epochs=self.epochs,
        #     vector_size=self.dimensions,
        #     window=self.window_size,
        #     min_count=self.min_count,
        #     workers=self.workers,
        #     seed=self.seed,
        # )
        # n2v = kc.Node2Vec(walk_number=self.num_walks,walk_length=self.l,p=self.p,q=self.q,dimensions=self.emb_sz,workers=self.num_workers,min_count=self.k)
        # n2v.fit(self.nx_g)
        # self.embs = n2v.get_embedding()
        # self.embs = tc.FloatTensor(self.embs)
        # model = gs.models.Word2Vec(walker.walks, vector_size=self.emb_sz, window=5, min_count=0, sg=1,workers=self.num_workers)
        # model = gs.models.Word2Vec(walks, vector_size=self.emb_sz, window=walks.shape[0], min_count=0, sg=1, hs=0, negative=5,workers=self.num_workers)
        # self.embs = tc.FloatTensor([model.wv[str(ele)] for ele in range(len(nodes))])
        encoder = m_node2vec.Node2VecEncoder(g=dgl.from_networkx(self.nx_g), emb_sz=self.emb_sz, workers=self.num_workers, out_dir='../tmp', out_file='node2vec-encoder', force=True,
                                  num_walks=self.num_walks, walk_lens=self.l, window_sz=self.l, p=1, q=1, iter=1, is_directed=False,
                                  is_weighted=False, weight_arr=None)
        st_time = time.time()
        mem_usage = encoder.train()
        out_g = encoder.load()
        self.embs = out_g.ndata['emb']
        print('encoder consume {:.2f}'.format(time.time() - st_time))

        print('{} train embeddings finished with time {:.4f}s'.format(self,time.time()-st_time))

        # train NN.
        print('{} start to train NN Distance Decoder...'.format(self))
        st_time = time.time()
        self.model = DistDecoder_DADL(emb_sz=self.emb_sz)
        loss = nn.MSELoss(reduction='sum')
        optim = tc.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(device)
        pair_idx = list(range(len(train_pairs)))
        self.model.train()
        for e_iter in range(self.iters):
            random.shuffle(pair_idx)
            train_pairs = train_pairs[pair_idx]
            train_batch_sz = 128
            pnt = 0
            train_loss = 0.
            cur_len = 0
            while pnt < len(train_pairs):
                optim.zero_grad()
                batch_idx = pnt,min(pnt + train_batch_sz,len(train_pairs))
                pnt += batch_idx[1] - batch_idx[0]
                batch_in = train_pairs[batch_idx[0]:batch_idx[1]]
                srcs = batch_in[:,0].long()
                dsts = batch_in[:,1].long()
                dists = batch_in[:,2].float()
                emb_srcs = self.embs[srcs]
                emb_dsts = self.embs[dsts]
                dists = dists.to(device)
                emb_srcs = emb_srcs.to(device)
                emb_dsts = emb_dsts.to(device)
                pred_dists = self.model(emb_srcs,emb_dsts)
                batch_loss = loss(pred_dists,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                print('\titer {} | batch {}/{} | loss:{:.5f}'.format(e_iter,pnt,len(train_pairs),train_loss / pnt))
            print('iter {} finished | loss:{:.5f}'.format(e_iter,train_loss / pnt))
        print('{} train NN Distance Decoder finished with time {:.4f}s'.format(self,time.time() - st_time))
        self._save_param_pickle()

        ed_time = time.time()

        # anal mem.
        self.train_var_lst.append(self.embs)
        self.train_var_lst.append(list(self.nx_g.edges()))
        self.train_var_lst.append(train_pairs)
        self.train_var_lst.append(landmarks)
        self.add_train_mem = 0.
        self.add_train_mem = mem_usage
        for p in self.model.parameters():
            self.train_var_lst.append(p) # inner torch parameters.

        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        emb_srcs = self.embs[srcs].to(device)
        emb_dsts = self.embs[dsts].to(device)
        dists = self.model(emb_srcs, emb_dsts)
        for idx in range(srcs.shape[0]):
            if int(srcs[idx]) == int(dsts[idx]):
                dists[idx] = 0.
            elif int(dsts[idx]) in set(self.nx_g.neighbors(int(srcs[idx]))):
                dists[idx] = 1.
        ed_time = time.time()

        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        for p in self.model.parameters():
            self.query_var_lst.append(p) # inner torch parameters.
        self.query_var_lst.append(list(self.nx_g.edges()))

        return dists.view(-1),ed_time

    def _save_param(self):
        save_dict = {'embs_path': self.pwd() + '.embs.npy','model_path':self.pwd()+'.model'}
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(save_dict, f)
        np.save(self.pwd() + '.embs.npy', self.embs.numpy())
        tc.save(self.model,self.pwd()+'.model')

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)
            pk.dump(self.model, f)
            pk.dump(self.nx_g,f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)
            self.model = pk.load(f)
            self.nx_g = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.embs = tc.from_numpy(np.load(load_dict['embs_path']))
        self.model = tc.load(load_dict['model_path'],map_location=device)

class DistDecoder_HALK(nn.Module):
    def __init__(self, emb_sz=16):
        super(DistDecoder_HALK, self).__init__()
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
        self.lin2 = nn.Linear(emb_sz, 1)

    def forward(self, src, dst):
        out = self.lin1(tc.cat((src, dst), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class HALK(DistanceQueryModel):
    def __init__(self,emb_sz=16,landmark_sz=100,lr=0.01,iters=15,p=1,q=1,l=80,k=5,num_walks=10,num_workers=8,batch_landmark_sz=5,batch_node_sz=100,init_fraction=10,batch_walk_sz=100,**kwargs):
        '''
        :param emb_sz: model embedding size
        :param landmark_sz: landmark size for training node pair
        :param lr: learning rate in NN.
        :param iters: total iterations in NN.
        :param p: control parameter in node2vec random walk
        :param q: control parameter in node2vec random walk
        :param l: length of random walk
        :param k: negative sampling size in skip-gram.
        :param num_walks: number of walks in skip-gram.
        :param kwargs:
        '''
        super(HALK, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.landmark_sz = landmark_sz
        self.lr = lr
        self.iters = iters
        self.p = p
        self.q = q
        self.l = l
        self.k = k
        self.num_walks = num_walks
        self.num_workers=num_workers
        self.batch_landmark_sz = batch_landmark_sz
        self.batch_node_sz = batch_node_sz
        self.init_fraction = init_fraction
        self.batch_walk_sz = batch_walk_sz

        self.storage_lst = [self.pwd() + '.pkl']

    def __str__(self):
        return 'HALK:' + self.model_name

    @staticmethod
    def _s_gen_train_pair(landmarks,nx_g,**kwargs):
        # nx_g = nx.Graph()
        pid = kwargs['__PID__']
        train_pairs = []
        for landmark in landmarks:
            search_lst = [landmark]
            dist_map = {landmark:0}
            while len(search_lst) != 0:
                cur_nid = search_lst.pop(0)
                for nnid in nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        search_lst.append(nnid)
            for nid in dist_map:
                if dist_map[nid] >= 1:
                    train_pairs.append([landmark,nid,dist_map[nid]])
        return train_pairs

    @staticmethod
    def _s_gen_walks(roots,nx_g,l,num_walks,**kwargs):
        walks = []
        for root in roots:
            for _ in range(num_walks):
                cur_l = 0
                cur_walk = [str(root)]
                search_map = {root}
                cur_nid = root
                while len(cur_walk) < l:
                    is_added = False
                    for nid in nx_g.neighbors(cur_nid):
                        if nid not in search_map:
                            search_map.add(nid)
                            cur_walk.append(str(nid))
                            is_added = True
                            break
                    if not is_added:
                        break
                walks.append(cur_walk)
        return walks

    def _generate(self):
        # self.nx_g = nx.DiGraph()
        # gen train pairs.
        mpm = utils.MPManager(batch_sz=self.batch_landmark_sz, num_workers=self.num_workers, use_shuffle=False)
        nodes = list(self.nx_g.nodes())
        random.shuffle(nodes)
        landmarks = nodes[:self.landmark_sz]
        train_pairs = mpm.multi_proc(HALK._s_gen_train_pair,[landmarks],nx_g=self.nx_g,auto_concat=True)
        train_pairs = tc.FloatTensor(train_pairs)
        # train embs with Node2Vec.


        # perform walk.
        print(f'start to perform walk at num {self.num_walks} for each node...')
        st_time = time.time()
        mpm = utils.MPManager(batch_sz=self.batch_node_sz, num_workers=self.num_workers, use_shuffle=False)
        random.shuffle(nodes)
        walks = mpm.multi_proc(HALK._s_gen_walks, [nodes], nx_g=self.nx_g,l=self.l,num_walks=self.num_walks, auto_concat=True)
        print(f'end perform walk with time {time.time()-st_time}')

        # cal freq.
        print(f'start to cal frequency.')
        st_time = time.time()
        n2freq = {nid:0 for nid in range(self.nx_g.number_of_nodes())}
        for walk in walks:
            for nid in walk:
                n2freq[int(nid)] += 1
        walks = list(sorted(walks,key=lambda w: - n2freq[int(w[0])]))
        print(f'end cal frequency with time {time.time() - st_time}')

        # train emb.
        print('{} start to train embeddings with HALK...'.format(self))
        st_time = time.time()
        init_sz = int(len(walks)*self.init_fraction)
        w2v = gs.models.Word2Vec(sentences=walks[:init_sz], vector_size=self.emb_sz, window=3, min_count=0, workers=self.num_workers, sg=1, epochs=5)
        pnt = init_sz
        while pnt < len(walks):
            cur_walks = walks[pnt:min(pnt,len(walks))]
            pnt += self.batch_walk_sz
            w2v.build_vocab(cur_walks, keep_raw_vocab=True, trim_rule=None, progress_per=10000, update=True)
            w2v.train(cur_walks, epochs=5, total_examples=len(cur_walks))
        embs = np.zeros(shape=(self.nx_g.number_of_nodes(),self.emb_sz))
        for word in w2v.wv.key_to_index.keys():
            embs[int(word),:] = np.array(w2v.wv[word])
        self.embs = tc.as_tensor(tc.from_numpy(embs),dtype=tc.float32)
        print('{} train embeddings finished with time {:.4f}s'.format(self,time.time()-st_time))

        # train NN.
        print('{} start to train NN Distance Decoder...'.format(self))
        st_time = time.time()
        self.model = DistDecoder_HALK(emb_sz=self.emb_sz)
        loss = nn.MSELoss(reduction='sum')
        optim = tc.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(device)
        pair_idx = list(range(len(train_pairs)))
        self.model.train()
        for e_iter in range(self.iters):
            random.shuffle(pair_idx)
            train_pairs = train_pairs[pair_idx]
            train_batch_sz = 128
            pnt = 0
            train_loss = 0.
            cur_len = 0
            while pnt < len(train_pairs):
                optim.zero_grad()
                batch_idx = pnt,min(pnt + train_batch_sz,len(train_pairs))
                pnt += batch_idx[1] - batch_idx[0]
                batch_in = train_pairs[batch_idx[0]:batch_idx[1]]
                srcs = batch_in[:,0].long()
                dsts = batch_in[:,1].long()
                dists = batch_in[:,2].float()
                emb_srcs = self.embs[srcs]
                emb_dsts = self.embs[dsts]
                dists = dists.to(device)
                emb_srcs = emb_srcs.to(device)
                emb_dsts = emb_dsts.to(device)
                pred_dists = self.model(emb_srcs,emb_dsts)
                batch_loss = loss(pred_dists,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                print('\titer {} | batch {}/{} | loss:{:.5f}'.format(e_iter,pnt,len(train_pairs),train_loss / pnt))
            print('iter {} finished | loss:{:.5f}'.format(e_iter,train_loss / pnt))
        print('{} train NN Distance Decoder finished with time {:.4f}s'.format(self,time.time() - st_time))
        self._save_param_pickle()

        ed_time = time.time()

        # anal mem.
        self.train_var_lst.append(self.embs)
        self.train_var_lst.append(list(self.nx_g.edges()))
        self.train_var_lst.append(train_pairs)
        self.train_var_lst.append(landmarks)
        for walk in walks:
            self.train_var_lst.append(walk)
        for p in self.model.parameters():
            self.train_var_lst.append(p) # inner torch parameters.
        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        emb_srcs = self.embs[srcs].to(device)
        emb_dsts = self.embs[dsts].to(device)
        dists = self.model(emb_srcs, emb_dsts)
        for idx in range(srcs.shape[0]):
            if int(srcs[idx]) == int(dsts[idx]):
                dists[idx] = 0.
            elif int(dsts[idx]) in set(self.nx_g.neighbors(int(srcs[idx]))):
                dists[idx] = 1.
        ed_time = time.time()

        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        for p in self.model.parameters():
            self.query_var_lst.append(p) # inner torch parameters.
        self.query_var_lst.append(list(self.nx_g.edges()))

        return dists.view(-1),ed_time

    def _save_param(self):
        save_dict = {'embs_path': self.pwd() + '.embs.npy','model_path':self.pwd()+'.model'}
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(save_dict, f)
        np.save(self.pwd() + '.embs.npy', self.embs.numpy())
        tc.save(self.model,self.pwd()+'.model')

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)
            pk.dump(self.model, f)
            pk.dump(self.nx_g,f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)
            self.model = pk.load(f)
            self.nx_g = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.embs = tc.from_numpy(np.load(load_dict['embs_path']))
        self.model = tc.load(load_dict['model_path'],map_location=device)

class DistDecoder_Path2Vec(nn.Module):
    def __init__(self, emb_sz=16):
        super(DistDecoder_Path2Vec, self).__init__()
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
        self.lin2 = nn.Linear(emb_sz, 1)

    def forward(self, src, dst):
        out = self.lin1(tc.cat((src, dst), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class Path2VecModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Path2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        embed1 = self.embeddings(inputs[0])
        embed2 = self.embeddings(inputs[1])
        # normalize the vectors before the dot product so that dot product is the cosine proximity between the two vectors
        embed1 = embed1 / embed1.norm(2, 2, True).clamp(min=1e-12).expand_as(embed1)
        embed2 = embed2 / embed2.norm(2, 2, True).clamp(min=1e-12).expand_as(embed2)
        out = tc.sum(embed1 * embed2, dim=2)

        return out

class Path2Vec(DistanceQueryModel):
    def __init__(self,emb_sz=16,landmark_sz=100,lr=0.01,iters=15,num_workers=8,batch_landmark_sz=5,fix_seed=False,neg=3,nei_fst_coef=0.01,nei_snd_coef=0.01,use_neighbors=True,regularize=False,**kwargs):
        '''
        :param emb_sz: model embedding size
        :param landmark_sz: landmark size for training node pair
        :param lr: learning rate in NN.
        :param iters: total iterations in NN.
        :param kwargs:
        '''
        super(Path2Vec, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.landmark_sz = landmark_sz
        self.lr = lr
        self.iters = iters
        self.num_workers=num_workers
        self.batch_landmark_sz = batch_landmark_sz
        self.fix_seed = fix_seed
        self.neg = neg
        self.use_neighbors = use_neighbors
        self.nei_fst_coef = nei_fst_coef
        self.nei_snd_coef = nei_snd_coef
        self.regularize = regularize
        self.storage_lst = [self.pwd() + '.pkl']

    def __str__(self):
        return 'Path2Vec:' + self.model_name

    @staticmethod
    def _s_gen_train_pair(landmarks,nx_g,**kwargs):
        # nx_g = nx.Graph()
        pid = kwargs['__PID__']
        train_pairs = []
        for landmark in landmarks:
            search_lst = [landmark]
            dist_map = {landmark:0}
            while len(search_lst) != 0:
                cur_nid = search_lst.pop(0)
                for nnid in nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        search_lst.append(nnid)
            for nid in dist_map:
                if dist_map[nid] >= 1:
                    train_pairs.append([landmark,nid,dist_map[nid]])
        return train_pairs

    def build_vocabulary(self,pairs):
        """
        Generates vocabulary from the sentences
        Counts the total number of training pairs
        Outputs this number, vocabulary and inverted vocabulary
        """
        vocabulary = {}
        train_pairs = 0
        for pair in pairs:
            (word0, word1, similarity) = pair
            train_pairs += 1
            for word in [word0, word1]:
                vocabulary[word] = 0
        print('Vocabulary size = %d' % len(vocabulary), file=sys.stderr)
        print('Total word pairs in the training set = %d' % train_pairs, file=sys.stderr)
        inv_vocab = sorted(vocabulary.keys())
        inv_vocab.insert(0, 'UNK')
        for word in inv_vocab:
            vocabulary[word] = inv_vocab.index(word)
        return train_pairs, vocabulary, inv_vocab

    def get_negative_samples(self,current_word_index, context_word_index, vocab_size, nsize):
        # Generate random negative samples, by default the same number as positive samples
        neg_samples = skipgrams([current_word_index, context_word_index], vocab_size, window_size=1,
                                negative_samples=nsize)
        return neg_samples

    def batch_generator_2(self,pairs, vocabulary, vocab_size, nsize, batch_size):
        """
        Generates training batches
        """
        global current_pos_samples
        timing = False  # Whether to print out batch generation time

        samples_per_pair = 2 + 2 * nsize  # How many training instances we get from each pair
        # How many samples will be there in each batch?
        samples_per_batch = samples_per_pair * batch_size

        inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                       np.zeros((samples_per_batch, 1), dtype=int)]

        # Batch should be a tuple of inputs and targets. First we create it empty:
        batch = (inputs_list, np.zeros((samples_per_batch, 1)))
        inst_counter = 0
        start = time.time()
        for pair in pairs:
            # split the line on tabs
            sequence = pair
            words = sequence[:2]
            if words[0] not in vocabulary or words[1] not in vocabulary:
                continue
            sim = np.float64(sequence[2])

            # Convert real words to indexes
            sent_seq = [vocabulary[word] for word in words]

            current_word_index = sent_seq[0]
            context_word_index = sent_seq[1]

            current_pos_samples[0].append(current_word_index)
            current_pos_samples[1].append(context_word_index)

            # get negative samples for the current pair
            neg_samples = self.get_negative_samples(
                current_word_index, context_word_index, vocab_size, nsize)

            # Adding two positive examples and the corresponding negative samples to the current batch
            for i in range(samples_per_pair):
                batch[0][0][inst_counter] = neg_samples[0][i][0]
                batch[0][1][inst_counter] = neg_samples[0][i][1]

                pred_sim = neg_samples[1][i]
                # if this is a positive example, replace 1 with the real similarity from the file:
                if pred_sim != 0:
                    pred_sim = sim
                batch[1][inst_counter] = pred_sim
                inst_counter += 1
            if inst_counter == samples_per_batch:
                yield batch
                end = time.time()
                if timing:
                    print('Batch generation took', end - start, file=sys.stderr)
                inst_counter = 0
                inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                               np.zeros((samples_per_batch, 1), dtype=int)]

                batch = (inputs_list, np.zeros((samples_per_batch, 1)))
                current_pos_samples = [[], []]
                start = time.time()

        # return the remaining samples
        yield batch

    def custom_loss(self,y_pred, y_true, reg_1_output, reg_2_output, use_neighbors, beta=0.01, gamma=0.01):
        if use_neighbors:
            alpha = 1 - (beta + gamma)
            m_loss = alpha * F.mse_loss(y_pred, y_true, reduction='elementwise_mean')

            m_loss -= beta * reg_1_output
            m_loss -= gamma * reg_2_output
        else:
            m_loss = F.mse_loss(y_pred, y_true, reduction='elementwise_mean')

        return m_loss

    def save_word2vec_format(self,fname, vocab, vectors, binary=False):
        """Store the input-hidden weight matrix in the same format used by the original
            C word2vec-tool, for compatibility.
            Parameters
            ----------
            fname : str
                The file path used to save the vectors in
            vocab : dict
                The vocabulary of words with their ranks
            vectors : numpy.array
                The vectors to be stored
            binary : bool
                If True, the data wil be saved in binary word2vec format, else in plain text.
            """
        if not (vocab or vectors):
            raise RuntimeError('no input')
        total_vec = len(vocab)
        vector_size = vectors.shape[1]
        print('storing %dx%d projection weights into %s' % (total_vec, vector_size, fname))
        assert (len(vocab), vector_size) == vectors.shape
        # with utils.smart_open(fname, 'wb') as fout:
        with open(fname, 'wb') as fout:
            fout.write(gs.utils.to_utf8('%s %s\n' % (total_vec, vector_size)))
            position = 0
            for element in sorted(vocab, key=lambda word: vocab[word]):
                row = vectors[position]
                if binary:
                    row = row.astype(real)
                    fout.write(gs.utils.to_utf8(element) + b" " + row.tostring())
                else:
                    fout.write(gs.utils.to_utf8('%s %s\n' % (element, ' '.join(repr(val) for val in row))))
                position += 1

    def save_embeddings(self,filename, model, vocab_dict):
        # Saving the resulting vectors
        embeddings = model.state_dict()['embeddings.weight']
        # if tc.cuda.is_available():
        #     embeddings = embeddings.cpu()
        self.save_word2vec_format(filename, vocab_dict, embeddings.numpy())

    def _generate(self):
        global neighbors_dict
        global current_pos_samples

        # self.nx_g = nx.DiGraph()
        # gen train pairs.
        mpm = utils.MPManager(batch_sz=self.batch_landmark_sz, num_workers=self.num_workers, use_shuffle=False)
        nodes = list(self.nx_g.nodes())
        random.shuffle(nodes)
        landmarks = nodes[:self.landmark_sz]
        train_pairs = mpm.multi_proc(Path2Vec._s_gen_train_pair,[landmarks],nx_g=self.nx_g,auto_concat=True)

        # train embs with path2vec.
        print('{} start to train embeddings with path2vec...'.format(self))
        st_time = time.time()

        if self.fix_seed:
            # fix seeds for repeatability of experiments
            np.random.seed(42)
            random.seed(12345)
            tc.manual_seed(1)
            if tc.cuda.is_available():
                tc.cuda.manual_seed(1)

        print('Retreiving neighbors of training samples...')
        # helpers.build_connections(vocab_dict)
        no_train_pairs, vocab_dict, inverted_vocabulary = self.build_vocabulary(train_pairs)

        vocab_size = len(vocab_dict)

        # instantiate the model
        model_p2v = Path2VecModel(vocab_size, self.emb_sz)

        # # use GPU if available
        # if torch.cuda.is_available():
        #     model.cuda()
        #     torch.cuda.manual_seed(1)
        #     print("Using GPU...")

        optimizer = tc.optim.Adam(model_p2v.parameters(), lr=0.001)

        print('Model name and layers:')
        print(model_p2v)

        # begin the training..
        for epoch in range(10):
            print('Epoch #', epoch + 1)
            total_loss, n_batches = 0, 0
            batchGenerator = self.batch_generator_2(train_pairs, vocab_dict, vocab_size, self.neg, batch_size=100)
            for batch in batchGenerator:
                n_batches += 1
                l1_reg_term = 0
                inputs, targets = batch
                target_tensor = tc.from_numpy(targets).float()

                input_var = tc.Tensor([inputs[0], inputs[1]]).long()
                # if tc.cuda.is_available():
                #     input_var = input_var.cuda()
                #     target_tensor = target_tensor.cuda()

                model_p2v.zero_grad()
                # do the forward pass
                similarity_pred = model_p2v(input_var)

                reg1_output = 0.
                reg2_output = 0.

                if self.use_neighbors:
                    # get only the positive samples because the batch variable contains the generated negatives as well
                    positive_samples = current_pos_samples
                    inputs_list = [[], []]
                    for word_idx in positive_samples[0]:
                        neighbors = neighbors_dict[word_idx]
                        for neighbor in neighbors:
                            inputs_list[0].append([word_idx])
                            inputs_list[1].append([neighbor])

                    input_var = tc.Tensor(inputs_list).long()
                    # if tc.cuda.is_available():
                    #     input_var = input_var.cuda()

                    reg1_dot_prod = model_p2v(input_var)
                    reg1_output = tc.sum(reg1_dot_prod) / len(reg1_dot_prod)

                    inputs_list = [[], []]
                    for word_idx in positive_samples[1]:  # context words
                        neighbors = neighbors_dict[word_idx]
                        for neighbor in neighbors:
                            inputs_list[0].append([word_idx])
                            inputs_list[1].append([neighbor])

                    input_var = tc.Tensor(inputs_list).long()
                    if tc.cuda.is_available():
                        input_var = input_var.cuda()

                    reg2_dot_prod = model_p2v(input_var)
                    reg2_output = tc.sum(reg2_dot_prod) / len(reg2_dot_prod)

                # Compute the loss function.
                loss = self.custom_loss(similarity_pred, target_tensor, reg1_output, reg2_output, self.use_neighbors, self.nei_fst_coef, self.nei_snd_coef)
                if self.regularize:
                    for param in model_p2v.parameters():
                        l1_reg_term += tc.norm(param, 1)
                    l1_factor = 1e-10
                    loss += l1_factor * l1_reg_term

                # Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # normalize the loss per batch size
                total_loss += loss.item() / len(inputs[0])

            print('Total loss = ', total_loss / n_batches)
        # time_id = time.time()
        file_name = self.pwd()+'.tmp'
        self.save_embeddings(file_name, model_p2v, vocab_dict)
        with open(file_name,'r') as f:
            n,m = str(f.readline()).strip().split()
            embs = np.zeros(shape=(int(n) - 1,int(m)))
            assert (int(n) - 1) == self.nx_g.number_of_nodes() and int(m) == self.emb_sz, print(f'n={n},m={m},num_nodes={self.nx_g.number_of_nodes()},emb_sz={self.emb_sz}')
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    lst = line.split()
                    if str(lst[0]) == 'UNK':
                        continue
                    embs[int(lst[0]),:] = np.array([float(ele) for ele in lst[1:]])
        self.embs = tc.as_tensor(tc.from_numpy(embs),dtype=tc.float32)
        print('{} train embeddings finished with time {:.4f}s'.format(self,time.time()-st_time))

        # train NN.
        print('{} start to train NN Distance Decoder...'.format(self))
        train_pairs = tc.FloatTensor(train_pairs)
        st_time = time.time()
        self.model = DistDecoder_Path2Vec(emb_sz=self.emb_sz)
        loss = nn.MSELoss(reduction='sum')
        optim = tc.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(device)
        pair_idx = list(range(len(train_pairs)))
        self.model.train()
        for e_iter in range(self.iters):
            random.shuffle(pair_idx)
            train_pairs = train_pairs[pair_idx]
            train_batch_sz = 128
            pnt = 0
            train_loss = 0.
            cur_len = 0
            while pnt < len(train_pairs):
                optim.zero_grad()
                batch_idx = pnt,min(pnt + train_batch_sz,len(train_pairs))
                pnt += batch_idx[1] - batch_idx[0]
                batch_in = train_pairs[batch_idx[0]:batch_idx[1]]
                srcs = batch_in[:,0].long()
                dsts = batch_in[:,1].long()
                dists = batch_in[:,2].float()
                emb_srcs = self.embs[srcs]
                emb_dsts = self.embs[dsts]
                dists = dists.to(device)
                emb_srcs = emb_srcs.to(device)
                emb_dsts = emb_dsts.to(device)
                pred_dists = self.model(emb_srcs,emb_dsts)
                batch_loss = loss(pred_dists,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                print('\titer {} | batch {}/{} | loss:{:.5f}'.format(e_iter,pnt,len(train_pairs),train_loss / pnt))
            print('iter {} finished | loss:{:.5f}'.format(e_iter,train_loss / pnt))
        print('{} train NN Distance Decoder finished with time {:.4f}s'.format(self,time.time() - st_time))
        self._save_param_pickle()

        ed_time = time.time()

        # anal mem.
        self.train_var_lst.append(self.embs)
        self.train_var_lst.append(list(self.nx_g.edges()))
        self.train_var_lst.append(train_pairs)
        self.train_var_lst.append(landmarks)
        self.train_var_lst.append(vocab_dict)
        for v in vocab_dict:
            self.train_var_lst.append(vocab_dict[v])
        for p in self.model.parameters():
            self.train_var_lst.append(p) # inner torch parameters.
        for p in model_p2v.parameters():
            self.query_var_lst.append(p)  # inner torch parameters.

        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        emb_srcs = self.embs[srcs].to(device)
        emb_dsts = self.embs[dsts].to(device)
        dists = self.model(emb_srcs, emb_dsts)
        for idx in range(srcs.shape[0]):
            if int(srcs[idx]) == int(dsts[idx]):
                dists[idx] = 0.
            elif int(dsts[idx]) in set(self.nx_g.neighbors(int(srcs[idx]))):
                dists[idx] = 1.
        ed_time = time.time()

        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        for p in self.model.parameters():
            self.query_var_lst.append(p) # inner torch parameters.
        self.query_var_lst.append(list(self.nx_g.edges()))

        return dists.view(-1),ed_time

    def _save_param(self):
        save_dict = {'embs_path': self.pwd() + '.embs.npy','model_path':self.pwd()+'.model'}
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(save_dict, f)
        np.save(self.pwd() + '.embs.npy', self.embs.numpy())
        tc.save(self.model,self.pwd()+'.model')

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)
            pk.dump(self.model, f)
            pk.dump(self.nx_g,f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)
            self.model = pk.load(f)
            self.nx_g = pk.load(f)

    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.embs = tc.from_numpy(np.load(load_dict['embs_path']))
        self.model = tc.load(load_dict['model_path'],map_location=device)

class DistDecoder_Vdist2Vec(nn.Module):
    def __init__(self, num_nodes,emb_sz,num_hiddens):
        super(DistDecoder_Vdist2Vec, self).__init__()
        self.num_nodes = num_nodes
        self.emb_sz = emb_sz
        self.num_hiddens = num_hiddens
        self.emb = nn.Embedding(self.num_nodes,self.emb_sz)

        self.lin1 = nn.Linear(emb_sz * 2, num_hiddens[0])
        self.lin2 = nn.Linear(num_hiddens[0], num_hiddens[1])

    def forward(self, nids):
        emb = self.emb(nids)  # [2 × emb_sz]
        out = self.lin1()
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class DiffDecoder_Vdist2Vec(nn.Module):
    def __init__(self,emb_sz,num_hiddens):
        super(DiffDecoder_Vdist2Vec, self).__init__()
        self.emb_sz=emb_sz
        self.num_hiddens = num_hiddens

class Vdist2Vec(DistanceQueryModel):
    def __init__(self,num_nodes,landmark_sz=16,emb_ratio=0.05,num_hiddens=[20,100],**kwargs):
        super(Vdist2Vec, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.emb_ratio = emb_ratio
        self.emb_sz = int(math.ceil(self.num_nodes * emb_ratio)) # use ceiling for emb dim.
        assert len(num_hiddens) == 2, print(f'require two hidden layers but got {len(num_hiddens)} layers')
        self.num_hiddens = num_hiddens
        self.landmark_sz = landmark_sz
        self.mlp_dist = DistDecoder_Vdist2Vec(num_nodes=self.num_nodes,emb_sz=self.emb_sz,num_hiddens=self.num_hiddens)
        self.mlp_diff = DiffDecoder_Vdist2Vec(emb_sz=self.emb_sz,num_hiddens=self.num_hiddens)

    def __str__(self):
        return 'Vdist2Vec:' + self.model_name

    def _generate(self):
        print('simulate cluster.')
        nodes = list(self.nx_g.nodes())
        random.shuffle(nodes)
        landmarks = nodes[:self.landmark_sz]
        cls_map = {landmark:landmark for landmark in landmarks}
        diff_map = {landmark:0 for landmark in landmarks}
        search_lst = landmarks.copy()
        while len(search_lst) > 0:
            cur_nid = search_lst.pop(0)
            for nnid in self.nx_g.neighbors(cur_nid):
                if nnid not in cls_map:
                    cls_map[nnid] = cls_map[cur_nid]
                    diff_map[nnid] = diff_map[cur_nid] + 1
                    search_lst.append(nnid)

        print('train distance predictor on landmarks.')

        print('train differential predictor on other nodes.')






    def _query(self,srcs,dsts):
        pass

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)
            pk.dump(self.model,f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)
            self.model = pk.load(f)

class DistDecoder_BCDR(nn.Module):
    def __init__(self, emb_sz=16):
        super(DistDecoder_BCDR, self).__init__()
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
        self.lin2 = nn.Linear(emb_sz, 1)

    def forward(self, src, dst):
        out = self.lin1(tc.cat((src, dst), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class BCDR(DistanceQueryModel):
    def __init__(self,emb_sz=16,landmark_sz=100,lr=0.01,iters=15,l=80,num_walks=10,num_workers=8,batch_landmark_sz=5,batch_root_sz=20,bc_decay=10,dist_decay=0.98,out_walks=40,out_l=10,use_sel='rnd',fast_query=False,**kwargs):
        super(BCDR, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.landmark_sz = landmark_sz
        self.lr = lr
        self.iters = iters
        self.l = l
        self.num_walks = num_walks
        self.num_workers = num_workers
        self.batch_landmark_sz = batch_landmark_sz
        self.batch_root_sz = batch_root_sz
        self.bc_decay = bc_decay
        self.dist_decay = dist_decay
        self.out_walks = out_walks
        self.out_l = out_l
        self.use_sel = use_sel
        self.fast_query = fast_query
        # self.storage_lst = [self.pwd() + '.json', self.pwd() + '.embs.npy',self.pwd()+'.model']
        self.storage_lst = [self.pwd() + '.pkl']

    def __str__(self):
        return 'BCDR:' + self.model_name

    @staticmethod
    def _s_gen_bc_pair(landmarks,nx_g,n,**kwargs):
        # nx_g = nx.Graph()
        # pid = kwargs['__PID__']
        train_pairs = []
        bcs = np.zeros(shape=(n,))
        for landmark in landmarks:
            search_lst = [landmark]
            dist_map = {landmark:0}
            bcs[landmark] += 1
            while len(search_lst) != 0:
                cur_nid = search_lst.pop(0)
                for nnid in nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        bcs[nnid] += 1 / dist_map[nnid]
                        search_lst.append(nnid)
            for nid in dist_map:
                if dist_map[nid] >= 1:
                    train_pairs.append([landmark,nid,dist_map[nid]])
        return train_pairs,bcs

    @staticmethod
    def _s_gen_bc_walk(roots,nx_g,num_walks,l,bcs,bc_decay,out_walks,out_l,dist_decay, **kwargs):
        # nx_g = nx.Graph()
        # pid = kwargs['__PID__']
        total_walks = []
        for root in roots:
            cnt_map = {}
            dist_map = {root:0}
            for _ in range(num_walks):
                search_set = {root}
                cur_nid = root
                cur_l = 0
                while cur_l < l:
                    cur_cands = []
                    cur_probs = []
                    # update distance.
                    for nnid in nx_g.neighbors(cur_nid):
                        if nnid not in search_set:
                            cur_cands.append(nnid)
                            cur_probs.append(bcs[nnid]*(2 - math.tanh(bc_decay - cnt_map.get(nnid,0)))) # bc decay.
                            if nnid in dist_map: # bc random tree.
                                dist_map[nnid] = min(dist_map[cur_nid] + 1,dist_map[nnid])
                            else:
                                dist_map[nnid] = dist_map[cur_nid] + 1
                    l_cur_cands = len(cur_cands)
                    # update next nid.
                    if  l_cur_cands >= 2:
                        cur_probs = np.array(cur_probs)
                        cur_probs /= np.sum(cur_probs)
                        cur_nid = int(np.random.choice(a=cur_cands,size=1,p=cur_probs)[0])
                    elif l_cur_cands == 1:
                        cur_nid = cur_cands[0]
                    else:
                        cnt_map[cur_nid] = 100
                        break
                    search_set.add(cur_nid)
                    cnt_map[cur_nid] = cnt_map.get(cur_nid,0) + 1
                    cur_l += 1
            dist_map.pop(root)
            total_cands = list(dist_map.keys())
            total_probs = np.power(dist_decay,np.array(list(dist_map.values()))) * bcs[total_cands]
            total_probs /= np.sum(total_probs)
            walks = np.random.choice(a=total_cands, size=(out_walks, out_l), replace=True,p=total_probs) # TODO:replace is False?
            root_cols = np.array([root] * out_walks).reshape(-1,1)
            walks = np.concatenate([root_cols,walks],axis=1)
            total_walks.append(walks)

        return np.concatenate(total_walks,axis=0)

    def _generate(self):
        # self.nx_g = nx.DiGraph()

        # gen train pairs & centrality score.
        mpm = utils.MPManager(batch_sz=self.batch_landmark_sz,num_workers=self.num_workers,use_shuffle=False)
        nodes = list(self.nx_g.nodes())
        n = len(nodes)
        if self.use_sel == 'rnd':
            random.shuffle(nodes)  # TODO: big deg landmarks?
            landmarks = nodes[:self.landmark_sz]
        elif self.use_sel == 'deg':
            deg_dict = dict(self.nx_g.degree)
            landmarks = np.argsort([deg_dict[nid] for nid in range(n)])[-self.landmark_sz:] # since nid just == idx.
            # landmarks = np.array(nodes)[idx_deg]
            # landmarks = sorted([deg_dict[nid] for nid in range(n)],reverse=True)[:self.landmark_sz]
        else:
            raise ValueError
        ret_dict = mpm.multi_proc(BCDR._s_gen_bc_pair,[landmarks],nx_g=self.nx_g,n=n,auto_concat=False)
        train_pairs = []
        bcs = np.zeros(shape=(n,))
        for k, v in ret_dict.items():
            train_pairs.extend(v[0])
            bcs += v[1]
        train_pairs = tc.FloatTensor(train_pairs)

        # run bc random walk tree.
        mpm = utils.MPManager(batch_sz=self.batch_root_sz, num_workers=self.num_workers, use_shuffle=False)
        random.shuffle(nodes)
        ret_dict = mpm.multi_proc(BCDR._s_gen_bc_walk,[nodes],nx_g=self.nx_g,num_walks=self.num_walks,l=self.l,bcs=bcs,bc_decay=self.bc_decay,out_walks=self.out_walks,out_l=self.out_l,dist_decay=self.dist_decay,auto_concat=False)
        walks = []
        [walks.append(ele) for ele in ret_dict.values()]
        walks = np.concatenate(walks,axis=0)
        ret_walks = []
        for walk in walks:
            ret_walks.append([str(ele) for ele in walk])

        # train embs with Word2Vec.
        print('{} start to train embeddings with Word2Vec...'.format(self))
        st_time = time.time()
        model = gs.models.Word2Vec(ret_walks, vector_size=self.emb_sz, window=walks.shape[1], min_count=0, sg=1,workers=self.num_workers)
        # model = gs.models.Word2Vec(walks, vector_size=self.emb_sz, window=walks.shape[0], min_count=0, sg=1, hs=0, negative=5,workers=self.num_workers)
        self.embs = tc.FloatTensor([model.wv[str(ele)] for ele in range(n)])
        print('{} train embeddings finished with time {:.4f}s'.format(self,time.time()-st_time))

        # train NN.
        print('{} start to train NN Distance Decoder...'.format(self))
        st_time = time.time()
        self.model = DistDecoder_BCDR(emb_sz=self.emb_sz)
        loss = nn.MSELoss(reduction='sum')
        optim = tc.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(device)
        pair_idx = list(range(len(train_pairs)))
        self.model.train()
        for e_iter in range(self.iters):
            random.shuffle(pair_idx)
            train_pairs = train_pairs[pair_idx]
            train_batch_sz = 128
            pnt = 0
            train_loss = 0.
            cur_len = 0
            while pnt < len(train_pairs):
                optim.zero_grad()
                batch_idx = pnt,min(pnt + train_batch_sz,len(train_pairs))
                pnt += batch_idx[1] - batch_idx[0]
                batch_in = train_pairs[batch_idx[0]:batch_idx[1]]
                srcs = batch_in[:,0].long()
                dsts = batch_in[:,1].long()
                dists = batch_in[:,2].float()
                emb_srcs = self.embs[srcs]
                emb_dsts = self.embs[dsts]
                dists = dists.to(device)
                emb_srcs = emb_srcs.to(device)
                emb_dsts = emb_dsts.to(device)
                pred_dists = self.model(emb_srcs,emb_dsts)
                batch_loss = loss(pred_dists,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                print('\titer {} | batch {}/{} | loss:{:.5f}'.format(e_iter,pnt,len(train_pairs),train_loss / pnt))
            print('iter {} finished | loss:{:.5f}'.format(e_iter,train_loss / pnt))
        print('{} train NN Distance Decoder finished with time {:.4f}s'.format(self,time.time() - st_time))
        self._save_param_pickle()

        ed_time = time.time()

        # anal mem.
        self.train_var_lst.append(self.embs)
        self.train_var_lst.append(list(self.nx_g.edges()))
        self.train_var_lst.append(train_pairs)
        self.train_var_lst.append(landmarks)
        for p in self.model.parameters():
            self.train_var_lst.append(p) # inner torch parameters.

        return ed_time

    def _load(self):
        if not os.path.exists(self.pwd()+'.pkl'):
            return False
        self._load_param_pickle()
        return True

    def _query(self,srcs,dsts):
        emb_srcs = self.embs[srcs].to(device)
        emb_dsts = self.embs[dsts].to(device)
        dists = self.model(emb_srcs, emb_dsts)
        if not self.fast_query:
            # verified by neighborhoods.
            for idx in range(srcs.shape[0]):
                if int(srcs[idx]) == int(dsts[idx]):
                    dists[idx] = 0.
                elif int(dsts[idx]) in set(self.nx_g.neighbors(int(srcs[idx]))):
                    dists[idx] = 1.

        ed_time = time.time()

        self.query_var_lst = []
        self.query_var_lst.append(self.embs)
        self.query_var_lst.append(self.model)
        for p in self.model.parameters():
            self.query_var_lst.append(p) # inner torch parameters.
        if not self.fast_query:
            self.query_var_lst.append(list(self.nx_g.edges()))

        return dists.view(-1),ed_time

    def _save_param(self):
        save_dict = {'embs_path': self.pwd() + '.embs.npy','model_path':self.pwd()+'.model'}
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(save_dict, f)
        np.save(self.pwd() + '.embs.npy', self.embs.numpy())
        tc.save(self.model,self.pwd()+'.model')

    def _save_param_pickle(self):
        with open(self.pwd() + '.pkl', 'wb') as f:
            pk.dump(self.embs, f)
            pk.dump(self.model,f)

    def _load_param_pickle(self):
        with open(self.pwd() + '.pkl', 'rb') as f:
            self.embs = pk.load(f)
            self.model = pk.load(f)


    def _load_param(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('cur path:{}'.format(self.pwd()))
        self.embs = tc.from_numpy(np.load(load_dict['embs_path']))
        self.model = tc.load(load_dict['model_path'],map_location=device)

if __name__ == '__main__':
    print('hello dqm.')
    # orion = Orion()
    # orion.optim_dsim()







