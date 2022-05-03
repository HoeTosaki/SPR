import os
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.functional as gnF
import multiprocessing
import threadpool
import math
import time
import random
import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import m_encoder
import m_generator
import m_selector
from utils import *
import m_logger
import m_deepwalk
import fileinput
import m_dage_test

class DistanceEmbeddingLayer:
    def __init__(self,g,emb_sz=128,workers=8,out_dir=CFG['TMP_PATH'],out_file='dist-emb.graph',num_walks=10,walk_lens=6,window_sz=3,in_bc_file=None):
        self.g = g
        self.emb_sz = emb_sz
        self.workers = workers
        self.out_dir = out_dir
        self.out_file = out_file
        self.num_walks = num_walks
        self.walk_lens = walk_lens
        self.window_sz = window_sz
        self.in_bc_file = in_bc_file
    def transform(self,is_close=False,force = False):
        if not force and self._check_file():
            print('cache distance emb checked.')
            gs,_ = dgl.load_graphs(os.path.join(self.out_dir,self.out_file))
            self.g = gs[0]
            return self.g

        if is_close:
            dwe = m_encoder.DeepWalkEncoder(g=self.g,
                                      emb_sz=self.emb_sz,
                                      workers=self.workers,
                                      out_dir=self.out_dir,
                                      out_file=self.out_file,
                                      force=force,
                                      num_walks=self.num_walks,
                                      walk_lens=self.walk_lens,
                                      window_sz=self.window_sz)
            print('distance emb start to train...')
            dwe.train()
            print('distance emb trained, load from disk...')
            self.g = dwe.load()
            print('load succ, save new cache.')
            dgl.save_graphs(os.path.join(self.out_dir,self.out_file),[self.g])
            return self.g
        else:
            dre = DistanceResamplingEncoder(g=self.g,
                                            in_bc_file=self.in_bc_file,
                                            emb_sz=self.emb_sz,
                                            out_dir=self.out_dir,
                                            out_file=self.out_file,
                                            force=force,
                                            num_walks=self.num_walks,
                                            walk_lens=self.walk_lens)
            print('distance resampling emb start to train...')
            dre.train()
            print('distance resampling trained, load from disk...')
            self.g = dre.load()
            print('load succ, save new cache.')
            dgl.save_graphs(os.path.join(self.out_dir, self.out_file), [self.g])
            return self.g

    def _check_file(self):
        return os.path.exists(os.path.join(self.out_dir,self.out_file))

class DistanceResamplingEncoder:
    def __init__(self,g,in_bc_file=None,emb_sz=128,out_dir=CFG['TMP_PATH'],out_file='dremb',force=False,
                 num_walks=80,input_len=40,output_len=60,alpha=0.98,input_exps=1,output_exps=5,neg_sz=5,prod_workers=8):
        self.g = g
        self.in_bc_file = in_bc_file
        self.emb_sz = emb_sz
        self.out_dir = out_dir
        self.out_file = out_file
        self.force = force
        self.num_walks = num_walks
        self.input_len=input_len
        self.output_len=output_len
        self.alpha=alpha
        self.input_exps=input_exps
        self.output_exps = output_exps
        self.neg_sz = neg_sz

        self.thPool = None
        self.prod_workers = prod_workers

    def train(self):
        # cal. appr. bc
        if self.in_bc_file is None:

            bc_emb = self.g.ndata['bc']
            print('graph bc checked intrinsic.')
            assert bc_emb is not None and bc_emb.shape[0] == self.g.num_nodes(),print('failed bc emb {}'.format(bc_emb))
            bc_emb += 1
            bc_emb = tc.sigmoid(bc_emb)
            bc_emb = bc_emb / tc.sum(bc_emb)
            self.g.ndata['bc'] = bc_emb
            print('bc emb:',bc_emb[:50])
        else:
            self._read_bc_file()
        # gen resampled path.
        # self._gen_path('bcdr',self.g.nodes())

        self._gen_path_p('bcdr-walks')

        self.cur_emb = self.train_emb('bcdr-walks')


    def load(self):
        assert self.g.num_nodes() == self.cur_emb.shape[0],print('load failed as g node sz{} != cur_emb sz{}'.format(g.num_nodes(),self.cur_emb.shape[0]))
        self.g.ndata['emb'] = self.cur_emb

    def _read_bc_file(self):
        # % % from 0 to node sz.
        with open(self.in_bc_file,'r') as f:
            ret = ''
            for line in f.readlines():
                line = line.strip()
                ret = ret + line + ' '
            ret = ret.strip()
            lst = [float(val) for val in ret.split(' ')]
            bcs = tc.Tensor(lst).view(-1)
            bcs = bcs / tc.sum(bcs)
            self.g.ndata['bc'] = bcs
            # print(self.g.ndata['bc'])
    def _gen_path_p(self,filename):
        if os.path.exists(os.path.join(self.out_dir,self.out_file+'.'+filename+'~0')):
            print('bcdr:checked walks.')
            return
        self.per_worker_nodes = math.ceil(self.g.num_nodes() / self.prod_workers)

        procs = []
        for i in range(self.prod_workers):
            proc = multiprocessing.Process(target=DistanceResamplingEncoder._th_gen_path, args=(self, i,filename))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
    def _th_gen_path(self,pid,filename):
        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,self.g.num_nodes())

        print('DistanceResamplingEncoder_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        self._gen_path(filename+'~' + str(pid),range(st_nid,ed_nid),is_print=True,interval=10,print_head='\tDistanceResamplingEncoder_' + str(pid))

    def _gen_path(self,filename,nodes,is_print=False,interval=10,print_head=''):
        with open(os.path.join(self.out_dir, self.out_file + '.' + filename), 'w') as f:
            num_nodes = len(nodes)
            for idx,src in enumerate(nodes):
                if is_print:
                    if idx % interval == 0:
                        print('{}:{}/{}'.format(print_head,idx,num_nodes))
                src = int(src)
                d_dist = {src:0}
                # g = dgl.DGLGraph()
                for walk in range(self.num_walks*self.input_exps):
                    cur_sgn = {src}
                    cur_len = 0
                    cur_real_len = cur_len
                    cur_node = src
                    while cur_len <= self.input_len:
                        raw_succ_nodes = self.g.successors(cur_node).tolist()
                        succ_nodes = []
                        for nid in raw_succ_nodes:
                            if nid not in cur_sgn:
                                succ_nodes.append(nid)
                        if len(succ_nodes) == 0:
                            # print('early terminated at len = {}'.format(cur_len))
                            break
                        bcs = self.g.ndata['bc'][succ_nodes].view(-1)
                        bcs = [float(ele) for ele in bcs]
                        sum_bc = sum(bcs)
                        if sum_bc == 0:
                            bcs = [1] *len(bcs)
                            sum_bc = sum(bcs)
                        bcs = [ele / sum_bc for ele in bcs]

                        succ = int(np.random.choice(a=succ_nodes,size=1,p=bcs)[0])

                        if succ in d_dist:
                            if d_dist[succ] > cur_real_len + 1:
                                d_dist[succ] = cur_real_len + 1
                            else:
                                cur_real_len = d_dist[succ] - 1
                        else:
                            d_dist[succ] = cur_real_len + 1
                        cur_node = succ
                        cur_sgn.add(cur_node)
                        cur_len += 1
                        cur_real_len += 1
                nodes = d_dist.keys()
                nodes = list(nodes)
                assert src in nodes
                nodes.remove(src)
                # nodes = nodes[1:]
                probs = []
                for nid in nodes:
                    prob = math.pow(self.alpha,d_dist[nid]) * self.g.ndata['bc'][nid]
                    probs.append(float(prob))
                sum_probs = sum(probs)
                probs = [ele / sum_probs for ele in probs]
                walks = np.random.choice(a=nodes, size=(self.num_walks * self.output_exps, self.output_len),replace=True, p=probs)
                roots = np.array([src] * (self.num_walks * self.output_exps))
                new_walks = np.concatenate([roots.reshape(-1, 1), walks], axis=1)
                for walk in new_walks:
                    walk_str = [str(ele) for ele in walk]
                    f.write(' '.join(walk_str)+'\n')
                f.flush()
    def train_emb(self,filename):
        '''
            TODO: need be optimized for very large graph.
        '''

        if os.path.exists(os.path.join(self.out_dir, self.out_file+'-'+filename+'.embedding')):
            print('bcdr: emb checked.')
            emb = None
            for cnt, line in enumerate(fileinput.input(files=[os.path.join(self.out_dir,self.out_file+'-'+filename+'.embedding')])):
                if cnt == 0:
                    lst = line.strip().split()
                    assert len(lst) == 2, print('lst:{}'.format(lst))
                    print('node_sz:{}, emb_sz:{}'.format(lst[0], lst[1]))
                    emb = tc.zeros(int(lst[0]), int(lst[1]))
                else:
                    lst = line.strip().split()
                    assert len(lst) >= 2
                    nid = int(lst[0])
                    lst1 = lst[1:]
                    nemb = tc.tensor([float(ele) for ele in lst1])
                    emb[nid] = nemb
            return emb
        walks = []

        if os.path.exists(os.path.join(self.out_dir, self.out_file + '.' + filename + '~0')):
            print('bcdr: use para dumped walks.')

            in_file_lst = []
            for root, dirs, files in os.walk(self.out_dir):
                for file in files:
                    if not file.startswith(self.out_file):
                        continue
                    meta_lst = file.strip().split('~')
                    if len(meta_lst) >= 2:
                        in_file_lst.append(file)
            for file in in_file_lst:
                with open(os.path.join(self.out_dir,file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        walks.append(line.split(' '))
        else:
            in_file = os.path.join(self.out_dir, self.out_file +'.'+ filename)
            with open(in_file,'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    walks.append(line.split(' '))

        print('walks dumped checked, start to train emb...')
        st_time = time.time()
        emb = self.walk2embed(walks, emb_sz=self.emb_sz, window_sz=len(walks[0]), neg_sz=self.neg_sz,out_dir=self.out_dir, name=self.out_file +'-'+ filename)
        print('emb train finished with {:.2f}'.format(time.time() - st_time))
        return emb


    def walk2embed(self,walks, emb_sz=16, window_sz=None, neg_sz=5, out_dir='../tmp', name=''):
        if window_sz is None:
            window_sz = len(walks[0]) // 4
        model = m_deepwalk.Word2Vec(walks, size=emb_sz, window=window_sz, min_count=0, sg=1, hs=0, negative=neg_sz,
                                    workers=8)
        model.wv.save_word2vec_format(os.path.join(out_dir, name + '.embedding'))

        emb = None
        for cnt, line in enumerate(fileinput.input(files=[os.path.join(out_dir, name + '.embedding')])):
            if cnt == 0:
                lst = line.strip().split()
                assert len(lst) == 2, print('lst:{}'.format(lst))
                print('node_sz:{}, emb_sz:{}'.format(lst[0], lst[1]))
                emb = tc.zeros(int(lst[0]), int(lst[1]))
            else:
                lst = line.strip().split()
                assert len(lst) >= 2
                nid = int(lst[0])
                lst1 = lst[1:]
                nemb = tc.tensor([float(ele) for ele in lst1])
                emb[nid] = nemb
        return emb


class DistanceAttention(nn.Module):
    def __init__(self,g,emb_sz,out_dir=CFG['TMP_PATH'],out_file='dist-att',workers=8):
        super(DistanceAttention, self).__init__()
        self.g = g
        self.emb_sz = emb_sz
        self.out_dir = out_dir
        self.out_file = out_file
        self.neighbor_trans = nn.Linear(self.emb_sz * 2, self.emb_sz // 2)
        self.remote_trans = nn.Linear(self.emb_sz * 2, self.emb_sz // 2)
        self.dist_trans = nn.Linear(self.emb_sz , 3)
        self.workers = workers

        self.per_worker_nodes = -1
        self.landmark_nodes = None
        self.thPool = None

    def save_model(self):
        tc.save(self.state_dict(),os.path.join(self.out_dir,self.out_file))

    def load_model(self):
        return self.load_state_dict(tc.load(os.path.join(self.out_dir,self.out_file)))

    def check_file(self):
        return os.path.exists(os.path.join(self.out_dir,self.out_file))

    def gen_dist_tuple(self,landmark_sz,force=False):
        if not force and self.check_file():
            print('dist-att: dist att model checked.')
            return

        if not force and self.check_traversal_file():
            print('dist-att: dist tuple checked.')
            return

        ds = m_selector.DegreeSelector(g=self.g)
        self.landmark_nodes = ds.perform(cnt=landmark_sz, action='max')

        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.workers)

        procs = []
        for i in range(self.workers):
            proc = multiprocessing.Process(target=DistanceAttention._th_gen_dist_tuple, args=(self, i,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def read_dist_tuple_random(self,batch_sz,meta_batch_sz = 50):
        if not self.check_traversal_file():
            raise FileNotFoundError
        # 读入文件
        file_lst = []
        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if not file.startswith(self.out_file+'.trv'):
                    continue
                meta_lst = file.strip().split('~')
                if len(meta_lst) >= 2:
                    file_lst.append(file)
        if len(file_lst) == 0:
            # not lst, check if has only one file.
            cur_file = os.path.join(self.out_dir,self.out_file)
            if os.path.exists(cur_file):
                file_lst.append(cur_file)
        assert len(file_lst) >= 1
        file_lst = [os.path.join(self.out_dir,file) for file in file_lst]

        if not self.thPool:
            self.thPool = threadpool.ThreadPool(self.workers)
        reqs = threadpool.makeRequests(DistanceAttention._th_shuffle_file, file_lst)
        [self.thPool.putRequest(req) for req in reqs]
        self.thPool.wait()

        file_loaders = [DistanceAttention._file_loader(file_lst=[file], batch_sz=meta_batch_sz) for file in file_lst]
        ret = []
        while True:
            while len(ret) < batch_sz and len(file_loaders) > 0:
                file_loader = random.choice(file_loaders)
                meta_batch = next(file_loader, 'end')
                if meta_batch == 'end' or len(meta_batch) == 0:
                    file_loaders.remove(file_loader)
                    continue
                ret.extend(meta_batch)
            if len(ret) == batch_sz:
                yield ret
                ret = []
            if len(ret) > batch_sz:
                yield ret[:batch_sz]
                ret = ret[batch_sz:]
            if len(file_loaders) <= 0:
                if len(ret) > 0:
                    yield ret
                # has no next batch..
                break

    def _file_loader(file_lst,batch_sz):
        for file in file_lst:
            with open(file,'r') as f:
                cnt = 0
                ret = []
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    lst = line.strip().split(',')
                    if len(lst) != 4:
                        break
                    assert len(lst) == 4
                    nid = int(lst[0])
                    cid = int(lst[1])
                    rid = int(lst[2])
                    score = int(lst[3])

                    ret.append((nid,cid,rid,score))
                    cnt += 1
                    if cnt == batch_sz:
                        yield ret
                        cnt = 0
                        ret = []
                if cnt > 0:
                    ret = ret[:cnt]
                    yield ret

    def _th_shuffle_file(file):
        if not os.path.exists(file):
            return False
        lins = None
        with open(file,'r') as f:
            lins = f.readlines()
            random.shuffle(lins)
        os.remove(file)
        with open(file,'w') as f:
            f.writelines(lins)
        return True

    def check_traversal_file(self):
        return os.path.exists(os.path.join(self.out_dir,self.out_file+'.trv'+'~0'))

    def _th_gen_dist_tuple(self,pid):

        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,len(self.landmark_nodes))

        print('DistanceAttention{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'.trv'+'~'+str(pid)),'w') as f:
            for i in range(st_nid,ed_nid):
                st_time = time.time()
                lid = self.landmark_nodes[i]

                neighbor_pairs = []

                dic_id2signal = [-1] * self.g.num_nodes()
                search_lst = [lid]
                dic_id2signal[lid] = 0

                while len(search_lst) != 0:
                    nid = search_lst.pop(0)
                    lst = self.g.successors(nid)
                    for nnid in lst:
                        if dic_id2signal[nnid] < 0:
                            search_lst.append(nnid)
                            dic_id2signal[nnid] = dic_id2signal[nid] + 1
                            neighbor_pairs.append((nnid,nid,lid,0))
                        elif math.fabs(dic_id2signal[nnid] - dic_id2signal[nid]) < 1e-3:
                            neighbor_pairs.append((nnid,nid,lid,1))
                        else:
                            neighbor_pairs.append((nnid, nid,lid, 2))

                [f.write(str(int(nid))+','+str(int(cid))+','+str(int(rid))+','+str(int(score))+'\n') for nid,cid,rid,score in neighbor_pairs]
                print('DistanceAttention{}:{}/{} time consume:{:.2f}'.format(pid,i - st_nid, ed_nid - st_nid, (time.time()-st_time) ))


    def forward(self,neighbor,current,remote):
        neighbor_cat =  tc.cat([neighbor,current],dim=1)
        remote_cat = tc.cat([remote,current],dim=1)

        neighbor_cat = self.neighbor_trans(neighbor_cat)
        neighbor_cat = F.relu(neighbor_cat)

        remote_cat = self.remote_trans(remote_cat)
        remote_cat = F.relu(remote_cat)

        dist_cat = tc.cat([neighbor_cat,remote_cat],dim=1)
        dist_cat = self.dist_trans(dist_cat)
        return dist_cat

class DifferentialLayer(nn.Module):
    def __init__(self,emb_sz):
        super(DifferentialLayer, self).__init__()
        self.emb_sz = emb_sz

    def forward(self,mfg,src_emb,e_att,is_close=False):
        # mfg = dgl.DGLGraph()
        with mfg.local_scope():
            # print('mfg:',mfg)
            mfg.srcdata['emb_o'] = src_emb
            # print('src:', src_emb.shape)
            # print('mfg nodes:',mfg.srcdata[dgl.NID].tolist())
            if is_close:
                # mfg.update_all(fn.copy_u(u='emb_o', out='msg'), fn.sum(msg='msg', out='emb_n'))
                mfg.push(range(mfg.num_src_nodes()), fn.copy_u('emb_o', 'msg'), fn.sum('msg', 'emb_n'))
            else:
                mfg.edata['att'] = e_att
                # mfg.update_all(fn.src_mul_edge(src='emb_o',edge='att', out='msg'), fn.sum(msg='msg', out='emb_n'))
                mfg.push(range(mfg.num_src_nodes()), fn.src_mul_edge('emb_o','att', 'msg'), fn.sum('msg', 'emb_n'))
            return mfg.dstdata['emb_n']

class PredictLayer(nn.Module):
    def __init__(self, emb_sz):
        super(PredictLayer, self).__init__()
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz // 4)
        self.lin2 = nn.Linear(emb_sz // 4, 1)

    def forward(self,src_emb, dst_emb):
        out = self.lin1(tc.cat((src_emb,dst_emb), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        return out

class DAGE(nn.Module):
    # TODO:
    #  1. try big emb_sz initial.
    #  2. multi-head attention.
    #  3. emb vary with diff layer.
    def __init__(self,g,emb_sz,out_dir=CFG['OUT_PATH'],out_file='dage.model',close_emb=False,close_diff=False,close_gcn=False):
        super(DAGE, self).__init__()
        self.g = g
        self.emb_sz = emb_sz
        self.close_emb = close_emb
        self.close_diff = close_diff
        self.close_gcn = close_gcn
        self.out_dir = out_dir
        self.out_file = out_file
        self.emb_layer = DistanceEmbeddingLayer(g=self.g,emb_sz=self.emb_sz,out_dir=self.out_dir,out_file=self.out_file+'.emb')
        self.dist_att = DistanceAttention(g=self.g,emb_sz=self.emb_sz,out_dir=self.out_dir,out_file=self.out_file+'.att')

        if not close_gcn:
            self.diff_layer1 = DifferentialLayer(emb_sz=self.emb_sz)
            self.lin1 = nn.Linear(self.emb_sz,self.emb_sz)
            self.diff_layer2 = DifferentialLayer(emb_sz=self.emb_sz)
            self.lin2 = nn.Linear(self.emb_sz, self.emb_sz)
            self.diff_layer3 = DifferentialLayer(emb_sz=self.emb_sz)

        self.pred_layer = PredictLayer(emb_sz=self.emb_sz)

    def save_model(self,extra_sign = 'default'):
        # self.dist_att.save_model()
        tc.save(self.state_dict(),os.path.join(self.out_dir,self.out_file+'.'+extra_sign))

    def load_model(self,extra_sign = 'default'):
        self.load_state_dict(tc.load(os.path.join(self.out_dir,self.out_file + '.' + extra_sign)))
        self.dist_att.save_model()

    def preprocess(self,landmark_sz = 100,force=False):
        self._gen_emb(force=force)
        if not self.close_diff:
            self._gen_dist_att(landmark_sz=landmark_sz,force=force)
            return self.g,self.dist_att
        else:
            return self.g,None
    def _gen_emb(self,force=False):
        self.g = self.emb_layer.transform(is_close=self.close_emb,force=force)

    def _gen_dist_att(self,landmark_sz,force=False):
        if not force and self.dist_att.check_file():
            print('dage: dist att model checked.')
            self.dist_att.load_model()
            return

        self.dist_att.gen_dist_tuple(landmark_sz,force)

        optim = tc.optim.Adam(self.dist_att.parameters(),lr=0.01,betas=(0.9,0.999),eps=1e-8)
        lsoft = nn.LogSoftmax(dim=1)
        nll = nn.NLLLoss()
        epoches = 1500
        batch_sz = 2056

        print('dage: start train att model...')
        self.dist_att.train()
        st_time = time.time()
        for epoch in range(epoches):
            self.dist_att.train()
            train_loss = 0.
            samples_cnt = 0
            for idx,samples in  tqdm.tqdm(enumerate(self.dist_att.read_dist_tuple_random(batch_sz=batch_sz))):
                nids = []
                cids = []
                rids = []
                scores = []
                samples_cnt += len(samples)
                for nid,cid,rid,score in samples:
                    nids.append(nid)
                    cids.append(cid)
                    rids.append(rid)
                    scores.append(score)
                nids = tc.LongTensor(nids)
                cids = tc.LongTensor(cids)
                rids = tc.LongTensor(rids)
                scores = tc.LongTensor(scores)

                optim.zero_grad()
                preds = self.dist_att(self.g.ndata['emb'][nids],self.g.ndata['emb'][cids],self.g.ndata['emb'][rids])
                nll_loss = nll(lsoft(preds),scores)
                nll_loss.backward()
                optim.step()
                train_loss += nll_loss.item()*len(samples)
                # print('\tdage: dist-att model train, epoch = {}-{}, train loss = {:.4f}'.format(epoch,idx,train_loss / samples_cnt))

            # self.dist_att.eval()
            val_acc_cnt = 0
            val_samples_cnt = 0

            for idx,samples in  tqdm.tqdm(enumerate(self.dist_att.read_dist_tuple_random(batch_sz=batch_sz))):
                nids = []
                cids = []
                rids = []
                scores = []
                val_samples_cnt += len(samples)
                for nid,cid,rid,score in samples:
                    nids.append(nid)
                    cids.append(cid)
                    rids.append(rid)
                    scores.append(score)
                nids = tc.LongTensor(nids)
                cids = tc.LongTensor(cids)
                rids = tc.LongTensor(rids)
                scores = tc.LongTensor(scores)

                preds = self.dist_att(self.g.ndata['emb'][nids],self.g.ndata['emb'][cids],self.g.ndata['emb'][rids])

                val_acc_cnt += (tc.argmax(preds,dim=1) == scores).sum()

            print('dage: dist-att model train, epoch = {}, train loss = {:.4f}, val_acc = {:.4f}, time = {:.2f}'.format(epoch, train_loss / samples_cnt, val_acc_cnt / val_samples_cnt,(time.time()-st_time)))
            st_time = time.time()

        self.dist_att.save_model()

    def forward(self,mfgs,src,dst,debug = False):
        if not self.close_gcn:
            e_att = None

            if not self.close_diff:
                mfg = mfgs[0]
                e_srcs,e_dsts = mfg.edges()
                # self.dist_att.eval()
                edata = self.dist_att(mfg.srcdata['emb'][e_srcs],mfg.srcdata['emb'][e_dsts], mfg.srcdata['emb'][0].view(1,-1).repeat(mfg.num_edges(),1))
                eidx = tc.argmax(edata,dim = 1) # shape: n*3
                R = tc.IntTensor([-1, 0 , 1])

                edge_data = R[eidx].view(-1, 1).type_as(tc.FloatTensor())
                e_att = mfg.edata['att'] = gnF.edge_softmax(mfg,edge_data)

            src_emb = mfgs[0].srcdata['emb']
            dst_emb = self.diff_layer1(mfgs[0],src_emb,e_att,self.close_diff)
            dst_emb = self.lin1(dst_emb)
            if debug:
                debug_dst_emb1 = dst_emb
            dst_emb = F.relu(dst_emb)

            # print('1:src_emb={},dst_emb={}'.format(src_emb.shape,dst_emb.shape))

            if not self.close_diff:
                mfg = mfgs[1]
                e_srcs,e_dsts = mfg.edges()
                # self.dist_att.eval()
                edata = self.dist_att(mfg.srcdata['emb'][e_srcs],mfg.srcdata['emb'][e_dsts], mfg.srcdata['emb'][0].view(1,-1).repeat(mfg.num_edges(),1))
                eidx = tc.argmax(edata,dim = 1) # shape: n*3
                R = tc.IntTensor([-1, 0 , 1])

                edge_data = R[eidx].view(-1, 1).type_as(tc.FloatTensor())
                e_att = mfg.edata['att'] = gnF.edge_softmax(mfg,edge_data)

            src_emb = dst_emb
            dst_emb = self.diff_layer2(mfgs[1],src_emb,e_att,self.close_diff)
            dst_emb = self.lin2(dst_emb)
            if debug:
                debug_dst_emb2 = dst_emb
            dst_emb = F.relu(dst_emb)

            # print('2:src_emb={},dst_emb={}'.format(src_emb.shape, dst_emb.shape))

            if not self.close_diff:
                mfg = mfgs[2]
                e_srcs,e_dsts = mfg.edges()
                # self.dist_att.eval()
                # print('src',mfg.srcdata[dgl.NID])
                # print('dsts',mfg.dstdata[dgl.NID])
                # print('dst',dst)
                left = mfg.srcdata['emb'][e_srcs],mfg.srcdata['emb'][e_dsts]
                right = mfg.srcdata['emb'][0].view(1, -1).repeat(mfg.num_edges(), 1)
                edata = self.dist_att(*left, right)
                eidx = tc.argmax(edata,dim = 1) # shape: n*3
                R = tc.IntTensor([-1, 0 , 1])

                edge_data = R[eidx].view(-1, 1).type_as(tc.FloatTensor())
                e_att = mfg.edata['att'] = gnF.edge_softmax(mfg,edge_data)

            src_emb = dst_emb
            dst_emb = self.diff_layer3(mfgs[2],src_emb,e_att,self.close_diff)

            dist = self.pred_layer(dst_emb[1:],dst_emb[0].view(1,-1).repeat(dst_emb.shape[0] - 1,1))
            if debug:
                return dist, debug_dst_emb1,debug_dst_emb2,dst_emb
            return dist
        else:
            src_emb = mfgs[0].srcdata['emb'][1:src.shape[0]]
            dst_emb = mfgs[0].srcdata['emb'][0]
            dist = self.pred_layer(src_emb, dst_emb.view(1,-1).repeat(src_emb.shape[0],1))
            if debug:
                return dist,dst_emb,src_emb
            return dist

class DstPackingGenerator(m_generator.Generator):
    '''
        fast random centering at dst with a wide range of src.
        directly implemented in random & parallel ver.
    '''
    def __init__(self,g,scheme=m_generator.BFS(None),workers=8,out_dir='../outputs',out_file='gen-file',force=False,dst_sz=100,each_src_sz=60):
        super(DstPackingGenerator, self).__init__(g=g,scheme=scheme,workers=workers,
                                                  out_dir=out_dir,out_file=out_file,is_random=False,is_parallel=True,file_sz=100,force=force)
        self.dst_sz = dst_sz
        self.each_src_sz = each_src_sz
        self.thPool = None
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    def loader(self,meta_batch_sz=30):
        '''
            each package led by unique dst for training benefits with batch of src.
        '''
        if not self.check_file():
            raise FileNotFoundError
        # 读入文件
        file_lst = []
        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if not file.startswith(self.out_file):
                    continue
                meta_lst = file.strip().split('~')
                if len(meta_lst) >= 2:
                    file_lst.append(file)
        if len(file_lst) == 0:
            # not lst, check if has only one file.
            cur_file = os.path.join(self.out_dir,self.out_file)
            if os.path.exists(cur_file):
                file_lst.append(cur_file)
        assert len(file_lst) >= 1
        file_lst = [os.path.join(self.out_dir,file) for file in file_lst]

        if not self.thPool:
            self.thPool = threadpool.ThreadPool(self.workers)
        reqs = threadpool.makeRequests(DstPackingGenerator._th_shuffle_file,file_lst)
        [self.thPool.putRequest(req) for req in reqs]
        self.thPool.wait()

        file_loaders = [m_generator.Generator._file_loader(file_lst = [file],batch_sz=meta_batch_sz) for file in file_lst]
        ret = []
        while len(file_loaders) > 0:
            file_loader = random.choice(file_loaders)
            meta_batch = next(file_loader,'end')
            if meta_batch == 'end' or len(meta_batch) == 0:
                file_loaders.remove(file_loader)
                continue
            ret.extend(meta_batch)
            if len(ret) > 0:
                yield ret
            ret = []

    def _th_shuffle_file(file):
        if not os.path.exists(file):
            return False
        lins = None
        with open(file,'r') as f:
            lins = f.readlines()
            random.shuffle(lins)
        os.remove(file)
        with open(file,'w') as f:
            f.writelines(lins)
        return True

    def gen_to_disk(self):
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            return

        ds = m_selector.DegreeSelector(g=self.g)
        self.dst_nodes = ds.perform(cnt=self.dst_sz, action='max') # prefer to large-deg node as dst.

        random.shuffle(self.dst_nodes)

        self.per_worker_nodes = math.ceil(len(self.dst_nodes) / self.workers)
        self.lst_nodes = self.g.nodes().tolist().copy()

        procs = []
        for i in range(self.workers):
            proc = multiprocessing.Process(target=DstPackingGenerator._th_gen, args=(self, i,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def _th_gen(self,pid):
        st_nid = max(pid * self.per_worker_nodes, 0)
        ed_nid = min((pid + 1) * self.per_worker_nodes, len(self.dst_nodes))

        print('DstPackingGenerator{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        for i in range(st_nid, ed_nid):
            with open(os.path.join(self.out_dir,self.out_file+'~'+str(i)),'w') as f:
                nid = self.dst_nodes[i]
                random.shuffle(self.lst_nodes)
                src_lst = self.lst_nodes[:self.each_src_sz]
                dists = self.scheme.dist_one_to_other(src=nid,dst_set=src_lst).tolist()
                assert self.each_src_sz == len(dists)
                [f.write(str(int(sid))+','+str(int(nid))+','+str(float(dist))+'\n') for sid, dist in zip(src_lst, dists)]
                print('DstPackingGenerator{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))

def train(dataset_config,model_config,lr,epoches):
    # logger
    tblogger = m_logger.TrainBasicLogger(out_dir=None)

    # gen dist DataLoader.
    g = dataset_config.get_graph()
    dataset_config.get_config(type='train')
    dpg = DstPackingGenerator(g=g,scheme=m_generator.BFS(g),workers=8,out_dir=os.path.join(CFG['TMP_PATH'],dataset_config.filename()),out_file=dataset_config.filename(),force=False,dst_sz=dataset_config.dst_sz,each_src_sz=dataset_config.each_src_sz)
    dataset_config.get_config(type='val')
    val_dpg = DstPackingGenerator(g=g,scheme=m_generator.BFS(g),workers=8,out_dir=os.path.join(CFG['TMP_PATH'],dataset_config.filename()),out_file=dataset_config.filename(),force=False,dst_sz=dataset_config.dst_sz,each_src_sz=dataset_config.each_src_sz)

    dpg.gen_to_disk()
    val_dpg.gen_to_disk()

    print('gen dist dataLoader finished.')

    dage = DAGE(g=g,emb_sz=model_config.emb_sz,out_dir=CFG['TMP_PATH'],out_file=model_config.filename(),close_emb=model_config.close_emb,close_diff=model_config.close_diff,close_gcn=model_config.close_gcn)

    if model_config.load_model_sign is not None:
        dage.load_model(extra_sign = model_config.load_model_sign)


    g,_ = dage.preprocess(landmark_sz=model_config.landmark_sz,force=False)

    optim = tc.optim.Adam(dage.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8)
    loss = nn.MSELoss(reduction='sum')

    for epoch in range(epoches):
        train_loss = 0.
        sample_cnt = 0
        dage.train()
        # dage.dist_att.eval()
        st_time = time.time()
        # for idx, samples in tqdm.tqdm(enumerate(dpg.loader(meta_batch_sz=30))):
        for idx, samples in enumerate(dpg.loader(meta_batch_sz=1)):
            srcs = [samples[0][1]]
            dst = samples[0][1]
            dists = []
            sample_cnt += len(samples)
            for e_src,e_dst,e_dist in samples:
                srcs.append(e_src)
                assert dst == e_dst
                dists.append(e_dist)
            srcs = tc.LongTensor(srcs)
            dst = tc.LongTensor([dst])
            dists = tc.FloatTensor(dists)

            sampler = dgl.dataloading.MultiLayerNeighborSampler([10,20,30])
            dataloader = dgl.dataloading.NodeDataLoader(g=g,nids=srcs,block_sampler=sampler,device=CFG['DEVICE']
                                                        ,batch_size=len(srcs),shuffle=False,drop_last=False,num_workers=0)
            for in_nodes,out_nodes,mfgs in dataloader:
                optim.zero_grad()
                pred = dage(mfgs,srcs,dst)
                batch_loss = loss(pred,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
            print('\ttrain-dage: epoch:{}-{}, train_loss={:.4f}'.format(epoch,idx,train_loss / sample_cnt))
        print('train-dage: epoch:{}, train_loss={:.4f},time={:.2f}'.format(epoch,train_loss / sample_cnt,time.time()-st_time),end='')

        # val_loss = 0.
        # val_sample_cnt = 0.
        # dage.eval()
        # st_time = time.time()
        # # for idx, samples in tqdm.tqdm(enumerate(val_dpg.loader(meta_batch_sz=20))):
        # for idx, samples in enumerate(val_dpg.loader(meta_batch_sz=1)):
        #     if val_sample_cnt > 8000:
        #         continue
        #     srcs = [samples[0][1]]
        #     dst = samples[0][1]
        #     dists = []
        #     val_sample_cnt += len(samples)
        #     for e_src, e_dst, e_dist in samples:
        #         srcs.append(e_src)
        #         assert dst == e_dst
        #         dists.append(e_dist)
        #     srcs = tc.LongTensor(srcs)
        #     dst = tc.LongTensor([dst])
        #     dists = tc.FloatTensor(dists)
        #
        #     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3) # eval use full neighbor.
        #     dataloader = dgl.dataloading.NodeDataLoader(g=g, nids=srcs, block_sampler=sampler, device=CFG['DEVICE']
        #                                                 , batch_size=len(srcs), shuffle=False, drop_last=False,
        #                                                 num_workers=0)
        #     for in_nodes, out_nodes, mfgs in dataloader:
        #         pred = dage(mfgs, srcs, dst)
        #         batch_loss = loss(pred, dists.view(-1,1))
        #         val_loss += batch_loss.item()
        # print('\tval-dage: epoch:{}, val_loss={:.4f},time={:.2f}'.format(epoch, val_loss / val_sample_cnt,time.time()-st_time))
        # tblogger.logging({'epoch': epoch, 'train_loss': train_loss / sample_cnt, 'val_loss': val_loss / val_sample_cnt})
        if epoch >= model_config.save_after_epoch:
            if (epoch - model_config.save_after_epoch) % model_config.save_between_epoch == 0:
                dage.save_model(extra_sign=str(epoch))

    # logger
    if model_config.load_model_sign is None:
        tblogger.save_log(os.path.join(CFG['LOG_PATH'],model_config.filename()+'.log'))
    else:
        tblogger.save_log(os.path.join(CFG['LOG_PATH'],model_config.filename()+'.'+ model_config.load_model_sign +'.log'))

    print('training routine finished.')


def test(dataset_config,model_config):

    assert model_config.load_model_sign is not None

    # gen dist DataLoader.
    g = dataset_config.get_graph()
    dataset_config.get_config(type='test')
    dpg = DstPackingGenerator(g=g,scheme=m_generator.BFS(g),workers=8,out_dir=CFG['TMP_PATH'],out_file=dataset_config.filename(),force=False,dst_sz=dataset_config.dst_sz,each_src_sz=dataset_config.each_src_sz)

    dpg.gen_to_disk()

    print('gen dist dataLoader finished.')

    dage = DAGE(g=g,emb_sz=model_config.emb_sz,out_dir=CFG['TMP_PATH'],out_file=model_config.filename(),close_emb=model_config.close_emb,close_diff=model_config.close_diff,close_gcn=model_config.close_gcn)

    dage.load_model(extra_sign=model_config.load_model_sign)

    g,_ = dage.preprocess(landmark_sz=model_config.landmark_sz,force=False)

    test_cnt = 0.
    test_mae = 0.
    test_mre = 0.
    seg_loss = {}

    dage.eval()
    with tc.no_grad():
        st_time = time.time()
        # for idx, samples in tqdm.tqdm(enumerate(val_dpg.loader(meta_batch_sz=20))):
        for idx, samples in enumerate(dpg.loader(meta_batch_sz=20)):
            srcs = [samples[0][1]]
            dst = samples[0][1]
            dists = []
            test_cnt += len(samples)
            for e_src, e_dst, e_dist in samples:
                srcs.append(e_src)
                assert dst == e_dst
                dists.append(e_dist)
            srcs = tc.LongTensor(srcs)
            dst = tc.LongTensor([dst])
            dists = tc.FloatTensor(dists)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3) # eval use full neighbor.
            dataloader = dgl.dataloading.NodeDataLoader(g=g, nids=srcs, block_sampler=sampler, device=CFG['DEVICE']
                                                        , batch_size=len(srcs), shuffle=False, drop_last=False,
                                                        num_workers=0)
            for in_nodes, out_nodes, mfgs in dataloader:
                # execute once.
                pred = dage(mfgs, srcs, dst)
                pred = pred.view(-1)
                assert dists.shape == pred.shape,print(dists.shape,pred.shape)
                dists = dists.tolist()
                pred = pred.tolist()
                for pred, real in zip(pred, dists):
                    if real == 0:
                        continue # fooling sample.
                    if real in seg_loss:
                        seg_loss[real]['mae'] += abs(pred - real)
                        if real == 0:
                            seg_loss[real]['mre'] += 0
                        else:
                            seg_loss[real]['mre'] += abs(pred - real) / abs(real)
                        seg_loss[real]['samples'] += 1
                    else:
                        mae = abs(pred - real)
                        if real == 0:
                            mre = 0
                        else:
                            mre = abs(pred - real) / abs(real)
                        seg_loss[real] = {'mae': mae, 'mre': 0, 'samples': 1}
                    if real != 0:
                        # our total result will not count for zeros.
                        test_mae += abs(pred - real)
                        test_mre += abs(pred - real) / abs(real)
                        test_cnt += 1

    test_mae /= test_cnt
    test_mre /= test_cnt
    for key in seg_loss:
        seg_loss[key]['mae'] = seg_loss[key]['mae'] / seg_loss[key]['samples']
        seg_loss[key]['mre'] = seg_loss[key]['mre'] / seg_loss[key]['samples']
        seg_loss[key]['path_len'] = key
    loss_table = pd.DataFrame(seg_loss)
    loss_total_table = pd.DataFrame({'total_mae':[test_mae],'total_mre':[test_mre]})
    loss_table.to_csv(os.path.join(CFG['LOG_PATH'],model_config.filename()+'-seg.csv'),index=False)
    loss_total_table.to_csv(os.path.join(CFG['LOG_PATH'],model_config.filename() + '-total.csv'), index=False)

    print('testing routine finished.')

def draw_tsne(dataset_config,model_config,assign_node = 0,random_sz = 10,assign_src_set=None):

    assert model_config.load_model_sign is not None

    def transform(origin_emb):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_emb = tsne.fit_transform(origin_emb.detach())
        return low_emb

    def draw(embs2d, lbs,out_file,colors = None):
        assert embs2d.shape[1] == 2, print(embs2d.shape)
        xs,ys = embs2d[:,0].tolist(),embs2d[:,1].tolist()
        if colors is not None:
            plt.scatter(xs,ys,c=colors)
        else:
            plt.scatter(xs,ys)
        for x,y,lb in zip(xs,ys,lbs):
            plt.text(x + 5,y + 5,int(lb))
        plt.axis('off')
        plt.savefig(out_file)
        plt.show()



    g = dataset_config.get_graph()

    dage = DAGE(g=g,emb_sz=model_config.emb_sz,out_dir=CFG['TMP_PATH'],out_file=model_config.filename(),close_emb=model_config.close_emb,close_diff=model_config.close_diff,close_gcn=model_config.close_gcn)

    dage.load_model(extra_sign=model_config.load_model_sign)

    g,_ = dage.preprocess(landmark_sz=model_config.landmark_sz,force=False)

    embs = None

    if random_sz > 0:
        lst_nodes = g.nodes().tolist()
        random.shuffle(lst_nodes)
        lst_nodes = lst_nodes[:random_sz]
    elif assign_src_set is not None:
        lst_nodes = list(assign_src_set)
    else:
        lst_nodes = g.nodes().tolist()
    if  assign_node not in lst_nodes:
        lst_nodes.append(assign_node)
    embs = g.ndata['emb'][lst_nodes]
    embs = transform(embs)
    draw(embs,lst_nodes,os.path.join(CFG['LOG_PATH'],model_config.filename()+'.org.svg'))

    dst = assign_node

    lst_nodes.remove(dst)

    srcs = [dst]
    srcs.extend(lst_nodes)

    srcs = tc.LongTensor(srcs)
    dst = tc.LongTensor([dst])

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    dataloader = dgl.dataloading.NodeDataLoader(g=g, nids=srcs, block_sampler=sampler, device=CFG['DEVICE']
                                                , batch_size=len(srcs), shuffle=False, drop_last=False,
                                                num_workers=0)
    out_emb1 = []
    out_emb2 = []
    out_emb3 = []
    lbs = []
    for in_nodes, out_nodes, mfgs in dataloader:
        # execute once.
        pred, debug_dst_emb1, debug_dst_emb2, dst_emb = dage(mfgs, srcs, dst,debug = True)
        out_emb1.append(debug_dst_emb1[:out_nodes.shape[0]])
        out_emb2.append(debug_dst_emb2[:out_nodes.shape[0]])
        out_emb3.append(dst_emb)
        lbs.extend(out_nodes.tolist())

    out_emb1 = tc.cat(out_emb1,dim=0)
    out_emb2 = tc.cat(out_emb2, dim=0)
    out_emb3 = tc.cat(out_emb3, dim=0)

    out_emb1 = transform(out_emb1)
    out_emb2 = transform(out_emb2)
    out_emb3 = transform(out_emb3)


    draw(out_emb1,lbs,os.path.join(CFG['LOG_PATH'],model_config.filename()+'.emb1.svg'),colors=['coral']+['dodgerblue']*(out_emb1.shape[0]-1))
    draw(out_emb2,lbs,os.path.join(CFG['LOG_PATH'],model_config.filename()+'.emb2.svg'),colors=['coral']+['dodgerblue']*(out_emb2.shape[0]-1))
    draw(out_emb3,lbs,os.path.join(CFG['LOG_PATH'],model_config.filename()+'.emb3.svg'),colors=['coral']+['dodgerblue']*(out_emb3.shape[0]-1))



    print('draw tsne finished.')


def draw_att(dataset_config,model_config,assign_node = 0,random_sz = 10):
    assert model_config.load_model_sign is not None

    g = dataset_config.get_graph()

    dage = DAGE(g=g,emb_sz=model_config.emb_sz,out_dir=CFG['TMP_PATH'],out_file=model_config.filename(),close_emb=model_config.close_emb,close_diff=model_config.close_diff,close_gcn=model_config.close_gcn)

    dage.load_model(extra_sign=model_config.load_model_sign)

    g,_ = dage.preprocess(landmark_sz=model_config.landmark_sz,force=False)

    embs = None

    if random_sz > 0:
        lst_nodes = g.nodes().tolist()
        random.shuffle(lst_nodes)
        lst_nodes = lst_nodes[:random_sz]
    else:
        lst_nodes = g.nodes().tolist()

    G = dgl.to_networkx(dgl.node_subgraph(g,lst_nodes))

    G = nx.to_undirected(G)

    nx.draw(G,node_size=150,with_labels=True,node_color='dodgerblue')

    plt.savefig(os.path.join(CFG['LOG_PATH'],model_config.filename()+'.dist-att.org.svg'))
    plt.show()

    nG = nx.DiGraph()

    for src,dst in G.edges():
        pred = dage.dist_att(g.ndata['emb'][src].view(1,-1),g.ndata['emb'][dst].view(1,-1),g.ndata['emb'][assign_node].view(1,-1))
        idx = tc.argmax(pred,dim = 1)
        if idx >= 0:
            nG.add_edge(src,dst)
        else:
            nG.add_edge(dst, src)
    node_color = ['coral' if node == assign_node else 'dodgerblue' for node in nG.nodes()]
    nx.draw(nG,node_size=150,with_labels=True,node_color = node_color,pos=nx.spring_layout(G))

    plt.savefig(os.path.join(CFG['LOG_PATH'], model_config.filename() + '.dist-att.att.svg'))
    plt.show()

    print('draw dist-att finished.')


class DatasetConfig:
    def __init__(self,dataset_name='fb'):
        self.dataset_name = dataset_name
        self.dst_sz=None
        self.each_src_sz=None
        self.type=None

    def get_graph(self):
        datasets = ['../datasets/dst/karate',
                    '../datasets/dst/facebook',
                    '../datasets/dst/BlogCatalog-dataset',
                    '../datasets/dst/twitter',
                    '../datasets/dst/youtube']
        if self.dataset_name == 'ka':
            g, _ = dgl.load_graphs(datasets[0])
            return g[0]
        elif self.dataset_name == 'fb':
            g, _ = dgl.load_graphs(datasets[1])
            return g[0]
        elif self.dataset_name == 'bc':
            g, _ = dgl.load_graphs(datasets[2])
            return g[0]
        elif self.dataset_name == 'tw':
            g, _ = dgl.load_graphs(datasets[3])
            return g[0]
        elif self.dataset_name == 'yt':
            g, _ = dgl.load_graphs(datasets[4])
            return g[0]
        elif self.dataset_name == 'tg1':
            edges = []
            mid = range(1,21)
            for idx in mid:
                edges.append((0,idx))
                edges.append((idx,21))
            return const_graph(edges)

    def get_config(self,type='train'):
        ret = [type]
        if self.dataset_name == 'fb':
            if type == 'train':
                ret.extend([100,1600])
            elif type == 'val':
                ret.extend([800,50])
            elif type == 'test':
                ret.extend([1600, 20])
        elif self.dataset_name == 'ka':
            if type == 'train':
                ret.extend([10,6])
            elif type == 'val':
                ret.extend([30,1])
            elif type == 'test':
                ret.extend([30, 1])
        elif self.dataset_name == 'tg1':
            if type == 'train':
                ret.extend([10, 6])
            elif type == 'val':
                ret.extend([22, 1])
            elif type == 'test':
                ret.extend([22, 1])

        assert len(ret) == 3,print(ret)
        self.type,self.dst_sz,self.each_src_sz = ret
        return ret

    def filename(self):
        return '{}-{}[{},{}].dist'.format(self.dataset_name, self.type, self.dst_sz, self.each_src_sz)

class ModelConfig:
    def __init__(self,**params):
        self.dataset = params['dataset']
        self.emb_sz = params['emb_sz']
        self.landmark_sz = params['landmark_sz'] # used in att
        self.att_batch_sz = params['att_batch_sz']
        self.dage_batch_sz = params['dage_batch_sz']
        self.close_gcn = params['close_gcn']
        self.close_diff = params['close_diff']
        self.close_emb = params['close_emb']
        self.save_after_epoch = params['save_after_epoch']
        self.save_between_epoch = params['save_between_epoch']
        self.load_model_sign = params['load_model_sign']
        self.file_name=None
    @staticmethod
    def default_config():
        return ModelConfig(dataset='ka',emb_sz = 16,landmark_sz=100,att_batch_sz=256,dage_batch_sz=30,close_gcn=False,close_diff=False,close_emb=False,save_after_epoch=1000,save_between_epoch=500,load_model_sign=None)

    def filename(self):
        if self.file_name is not None:
            return self.file_name
        return 'dage-{}-emb={}{}{}{}.model'.format(self.dataset,self.emb_sz,('-no gcn-' if self.close_gcn else '') \
               ,('-no diff-' if self.close_diff else ''),('-no emb-' if self.close_emb else ''))


def combine_routine_cdgcn():
    dataset_names = ['fb']
    emb_szs = [16]
    for dataset_name in dataset_names:
        for emb_sz in emb_szs:
            dataset_config = DatasetConfig(dataset_name=dataset_name)
            model_config = ModelConfig.default_config()
            model_config.dataset = dataset_name
            model_config.close_emb = True
            model_config.close_gcn = False
            model_config.close_diff = True
            model_config.emb_sz = emb_sz
            model_config.save_after_epoch = 20000
            model_config.save_between_epoch = 10
            model_config.file_name = '{}-{}-emb={}'.format(dataset_name,'cdgcn',emb_sz)
            train(dataset_config=dataset_config,model_config=model_config,lr=0.001,epoches=200)

def combine_routine_dage():
    dataset_names = ['fb']
    emb_szs = [16]
    for dataset_name in dataset_names:
        for emb_sz in emb_szs:
            dataset_config = DatasetConfig(dataset_name=dataset_name)
            model_config = ModelConfig.default_config()
            model_config.dataset = dataset_name
            model_config.close_emb = True
            model_config.close_gcn = False
            model_config.close_diff = False
            model_config.emb_sz = emb_sz
            model_config.save_after_epoch = 20000
            model_config.save_between_epoch = 10
            model_config.file_name = '{}-{}-emb={}'.format(dataset_name,'cdgcn',emb_sz)
            train(dataset_config=dataset_config,model_config=model_config,lr=0.001,epoches=200)

if __name__ == '__main__':
    print('hello dage')
    # dataset_config = DatasetConfig(dataset_name='tg1')
    # model_config = ModelConfig.default_config()
    # model_config.dataset = 'tg1'
    # model_config.close_emb = True
    # model_config.close_gcn = False
    # model_config.close_diff = False
    # model_config.emb_sz = 2 #for test graph.
    #
    # model_config.save_after_epoch = 20
    # model_config.save_between_epoch = 10
    # train(dataset_config=dataset_config,model_config=model_config,lr=0.001,epoches=51)

    # model_config.load_model_sign = '50'
    # test(dataset_config,model_config)

    # draw_tsne(dataset_config,model_config,assign_node=0,random_sz=-1)
    # draw_tsne(dataset_config,model_config,assign_node=0,random_sz=-1,assign_src_set=[2,3,7,9,14,20,22,23,30])
    # draw_tsne(dataset_config, model_config, assign_node=5, random_sz=-1,assign_src_set=[2,8,9,15,16,26,28,32,33])
    # draw_tsne(dataset_config,model_config,assign_node=18,random_sz=-1,assign_src_set=[0,2,5,7,13,16,23,28,32])

    # draw_att(dataset_config,model_config,assign_node=0,random_sz=-1)

    # combine_routine_cdgcn()

