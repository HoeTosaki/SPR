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
import m_dage
import m_generator_parallel

'''
Model Definition
'''
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


class CDGCN(nn.Module):
    def __init__(self,g,emb_sz,out_dir=CFG['OUT_PATH'],out_file='cdgcn.model'):
        super(CDGCN, self).__init__()
        self.g = g
        self.emb_sz = emb_sz
        self.out_dir = out_dir
        self.out_file = out_file
        self.pred_layer = PredictLayer(emb_sz=self.emb_sz)

    def save_model(self,extra_sign = 'default'):
        # self.dist_att.save_model()
        tc.save(self.state_dict(),os.path.join(self.out_dir,self.out_file+'.'+extra_sign))

    def load_model(self,extra_sign = 'default'):
        self.load_state_dict(tc.load(os.path.join(self.out_dir,self.out_file + '.' + extra_sign)))
        self.dist_att.save_model()

    def forward(self,mfgs,nodes,debug = False):
        pass

'''
Utils
'''



'''
Train Routine
'''
def train(dataset_config,model_config,lr,epoches):
    # logger
    tblogger = m_logger.TrainBasicLogger(out_dir=None)

    # gen dist DataLoader.
    g = dataset_config.get_graph()

    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=dataset_config.get_landmark_cnt(), action='max')

    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=dataset_config.dataset_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=dataset_config.dataset_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=dataset_config.dataset_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)
    # train_generator.gen_to_disk()
    # landmark_generator.gen_to_disk()
    # val_generator.gen_to_disk()

    print('gen dist dataLoader finished.')

    cdgcn = CDGCN(g=g,emb_sz=model_config.emb_sz,out_file=model_config.filename())

    if model_config.load_model_sign is not None:
        cdgcn.load_model(extra_sign = model_config.load_model_sign)

    optim = tc.optim.Adam(cdgcn.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8)
    loss = nn.MSELoss(reduction='sum')

    for epoch in range(epoches):
        train_loss = 0.
        sample_cnt = 0
        cdgcn.train()
        # dage.dist_att.eval()
        st_time = time.time()
        # for idx, samples in tqdm.tqdm(enumerate(dpg.loader(meta_batch_sz=30))):
        for idx, samples in enumerate(train_generator.loader(batch_sz=500)):
            sample_cnt += len(samples)
            srcs = []
            dsts = []
            dists = []
            for src, dst, dist in samples:
                srcs.append(src)
                dsts.append(dst)
                dists.append(dist)
            dists = tc.FloatTensor(dists)
            nodes = srcs + dsts
            nodes = list(set(nodes))

            sampler = dgl.dataloading.MultiLayerNeighborSampler([10,20,30])
            dataloader = dgl.dataloading.NodeDataLoader(g=g,nids=nodes,block_sampler=sampler,device=CFG['DEVICE']
                                                        ,batch_size=len(nodes),shuffle=False,drop_last=False,num_workers=0)
            for in_nodes,out_nodes,mfgs in dataloader:
                optim.zero_grad()
                pred = cdgcn(mfgs,srcs,dsts)
                batch_loss = loss(pred,dists.view(-1,1))
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
            print('\ttrain-cdgcn: epoch:{}-{}, train_loss={:.4f}'.format(epoch,idx,train_loss / sample_cnt))
        print('train-cdgcn: epoch:{}, train_loss={:.4f},time={:.2f}'.format(epoch,train_loss / sample_cnt,time.time()-st_time),end='')

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
                cdgcn.save_model(extra_sign=str(epoch))

    # logger
    if model_config.load_model_sign is None:
        tblogger.save_log(os.path.join(CFG['LOG_PATH'],model_config.filename()+'.log'))
    else:
        tblogger.save_log(os.path.join(CFG['LOG_PATH'],model_config.filename()+'.'+ model_config.load_model_sign +'.log'))

    print('training routine finished.')

'''
Env. Config
'''
class DatasetConfig:
    '''
        which is conformed to old training pattern.
    '''
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
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
    def get_landmark_cnt(self):
        if self.dataset_name == 'ka':
            return 10
        elif self.dataset_name == 'fb':
            return 100
        elif self.dataset_name == 'bc':
            return 100
        elif self.dataset_name == 'tw':
            return 200
        elif self.dataset_name == 'yt':
            return 200


class ModelConfig:
    def __init__(self,**params):
        self.dataset = params['dataset']
        self.emb_sz = params['emb_sz']
        self.save_after_epoch = params['save_after_epoch']
        self.save_between_epoch = params['save_between_epoch']
        self.load_model_sign = params['load_model_sign']
        self.file_name=None
    @staticmethod
    def default_config():
        return m_dage.ModelConfig(dataset='ka',emb_sz = 16,landmark_sz=100,att_batch_sz=256,dage_batch_sz=30,close_gcn=False,close_diff=False,close_emb=False,save_after_epoch=1000,save_between_epoch=500,load_model_sign=None)

    def filename(self):
        if self.file_name is not None:
            return self.file_name
        return 'cdgcn-{}-emb={}.model'.format(self.dataset,self.emb_sz)



if __name__ == '__main__':
    print('hello cdgcn.')
    mc = ModelConfig.default_config()
    dc = DatasetConfig(dataset_name='fb')
    mc.save_after_epoch = 100000
    mc.save_between_epoch = 50
    train(dc,mc,lr=1e-14,epoches=200)
