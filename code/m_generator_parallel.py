import dgl
import random
import torch as tc
import threading
import threadpool
import os
import numpy as np
import time
import sys
import math

import m_deepwalk
import embedding_proc
import m_decoder
import m_evaluator
import m_logger
import m_generator
import m_manager
import multiprocessing

class ClassicalRandomGenerator_Acc:
    def __init__(self,g,scheme=m_generator.BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,data_sz_per_node=5,prod_workers=1):
        self.g = g
        self.scheme = scheme.__class__(g,out_dir)
        self.workers = workers
        self.out_dir = out_dir
        self.out_file = out_file
        self.is_random = is_random
        self.file_sz = file_sz
        self.force = force
        self.is_parallel = is_parallel
        self.thPool = None
        self.prod_workers = prod_workers
        self.data_sz_per_node = data_sz_per_node
    def gen_to_disk(self,early_break=-1):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            # 此处不读取实际的数据，因为加速器只负责写磁盘，后续完整读取功能交给真正的generator来做
            return
        self.early_break = early_break
        # self.lockInd = threading.Lock()
        self.lst_nodes = self.g.nodes().tolist().copy()
        self.per_worker_nodes = math.ceil(len(self.lst_nodes) / self.prod_workers)
        # self.workId = 0
        # self.ret_sgn = []
        random.shuffle(self.lst_nodes)

        procs = []
        for i in range(self.prod_workers):
            proc = multiprocessing.Process(target=ClassicalRandomGenerator_Acc._th_gen,args=(self,i,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

        # self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        # reqs = threadpool.makeRequests(ClassicalRandomGenerator_Acc._th_gen,[self]*self.prod_workers)
        # [self.thPool_prod.putRequest(req) for req in reqs]
        # self.thPool_prod.wait()

    def check_file(self):
        '''
            这里只检查并行存储的文件逻辑，所以一定由多个文件构成存储组
        :return: BOOL 上次数据计算缓存是否存在
        '''
        file_lst = []
        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if not file.startswith(self.out_file):
                    continue
                meta_lst = file.strip().split('~')
                if len(meta_lst) >= 2:
                    file_lst.append(file)
        if len(file_lst) > 0:
            return True
        return False

    def _th_gen(self,pid):
        #require worker id.
        # cur_id = -1
        # self.lockInd.acquire()
        # cur_id = self.workId
        # self.workId += 1
        # self.lockInd.release()

        # assert cur_id != -1

        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,len(self.lst_nodes))

        print('ClassicalRandomGenerator_Acc_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'~'+str(pid)),'w') as f:
            for i in range(st_nid,ed_nid):
                nid = self.lst_nodes[i]
                dst_nodes = random.sample(self.lst_nodes,self.data_sz_per_node)
                dists = self.scheme.dist_one_to_all(nid)[dst_nodes].tolist()
                # dists = self.scheme.dist_one_to_all(29)[dst_nodes].tolist()
                assert len(dists) == len(dst_nodes)
                [f.write(str(int(nid))+','+str(int(dst_node))+','+str(float(dist))+'\n') for dst_node, dist in zip(dst_nodes, dists)]
                if self.early_break > 0:
                    print('ClassicalRandomGenerator_Acc_{}:{}/{}'.format(pid,i-st_nid,self.early_break))
                else:
                    print('ClassicalRandomGenerator_Acc_{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))
                if self.early_break > 0 and (i - st_nid) >= self.early_break:
                    break
class FastRandomGenerator_Acc:
    def __init__(self, g, scheme=m_generator.BFS(None), workers=1, out_dir='../outputs', out_file='gen-file',is_random=False, is_parallel=False, file_sz=None, force=False, data_sz_per_node=5,landmark_nodes=None, prod_workers=5):
        self.g = g
        self.scheme = scheme.__class__(g, out_dir)
        self.workers = workers
        self.out_dir = out_dir
        self.out_file = out_file
        self.is_random = is_random
        self.file_sz = file_sz
        self.force = force
        self.is_parallel = is_parallel
        self.thPool = None
        self.data_sz_per_node = data_sz_per_node
        self.landmark_nodes = landmark_nodes
        self.prod_workers = prod_workers

    def gen_to_disk(self,early_break=-1):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            # 此处不读取实际的数据，因为加速器只负责写磁盘，后续完整读取功能交给真正的generator来做
            return

        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            # 此处不读取实际的数据，因为加速器只负责写磁盘，后续完整读取功能交给真正的generator来做
            return

        self.early_break = early_break
        assert self.landmark_nodes is not None
        random.shuffle(self.landmark_nodes)

        # self.lockInd = threading.Lock()
        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.prod_workers)
        # self.workId = 0
        self.lst_nodes = self.g.nodes().tolist().copy()

        # self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        # reqs = threadpool.makeRequests(FastRandomGenerator_Acc._th_gen, [self] * self.prod_workers)
        # [self.thPool_prod.putRequest(req) for req in reqs]
        # self.thPool_prod.wait()

        procs = []
        for i in range(self.prod_workers):
            proc = multiprocessing.Process(target=FastRandomGenerator_Acc._th_gen, args=(self, i,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def check_file(self):
        '''
            这里只检查并行存储的文件逻辑，所以一定由多个文件构成存储组
        :return: BOOL 上次数据计算缓存是否存在
        '''
        file_lst = []
        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if not file.startswith(self.out_file):
                    continue
                meta_lst = file.strip().split('~')
                if len(meta_lst) >= 2:
                    file_lst.append(file)
        if len(file_lst) > 0:
            return True
        return False

    def _th_gen(self,pid):
        # require worker id.
        # cur_id = -1
        # self.lockInd.acquire()
        # cur_id = self.workId
        # self.workId += 1
        # self.lockInd.release()
        # assert cur_id != -1

        st_nid = max(pid * self.per_worker_nodes, 0)
        ed_nid = min((pid + 1) * self.per_worker_nodes, len(self.landmark_nodes))

        print('FastRandomGenerator_Acc_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'~'+str(pid)),'w') as f:

            for i in range(st_nid, ed_nid):
                nid = self.landmark_nodes[i]
                dists = self.scheme.dist_one_to_all(nid).tolist()
                assert len(self.lst_nodes) == len(dists)
                [f.write(str(int(nid))+','+str(int(dst_node))+','+str(float(dist))+'\n') for dst_node, dist in zip(self.lst_nodes, dists)]
                if self.early_break > 0:
                    print('FastRandomGenerator_Acc_{}:{}/{}'.format(pid,i - st_nid, self.early_break))
                else:
                    print('FastRandomGenerator_Acc_{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))
                if i == self.early_break:
                    break

class LandmarkInnerGenerator_Acc:
    def __init__(self, g, scheme=m_generator.BFS(None), workers=1, out_dir='../outputs', out_file='gen-file',is_random=False, is_parallel=False, file_sz=None, force=False, landmark_nodes=None,prod_workers=5):
        self.g = g
        self.scheme = scheme.__class__(g, out_dir)
        self.workers = workers
        self.out_dir = out_dir
        self.out_file = out_file
        self.is_random = is_random
        self.file_sz = file_sz
        self.force = force
        self.is_parallel = is_parallel
        self.thPool = None
        self.landmark_nodes = landmark_nodes
        self.prod_workers = prod_workers
    def gen_to_disk(self,early_break=-1):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            # 此处不读取实际的数据，因为加速器只负责写磁盘，后续完整读取功能交给真正的generator来做
            return

        self.early_break = early_break
        assert self.landmark_nodes is not None
        random.shuffle(self.landmark_nodes)

        # self.ret = []
        # self.lockInd = threading.Lock()
        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.prod_workers)
        # self.workId = 0

        # self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        # reqs = threadpool.makeRequests(LandmarkInnerGenerator_Acc._th_gen,[self]*self.prod_workers)
        # [self.thPool_prod.putRequest(req) for req in reqs]
        # self.thPool_prod.wait()

        procs = []
        for i in range(self.prod_workers):
            proc = multiprocessing.Process(target=LandmarkInnerGenerator_Acc._th_gen, args=(self, i,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()


    def check_file(self):
        '''
            这里只检查并行存储的文件逻辑，所以一定由多个文件构成存储组
        :return: BOOL 上次数据计算缓存是否存在
        '''
        file_lst = []
        for root, dirs, files in os.walk(self.out_dir):
            for file in files:
                if not file.startswith(self.out_file):
                    continue
                meta_lst = file.strip().split('~')
                if len(meta_lst) >= 2:
                    file_lst.append(file)
        if len(file_lst) > 0:
            return True
        return False

    def _th_gen(self,pid):
        #require worker id.
        # cur_id = -1
        # self.lockInd.acquire()
        # cur_id = self.workId
        # self.workId += 1
        # self.lockInd.release()
        # assert cur_id != -1

        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,len(self.landmark_nodes))

        print('LandmarkInnerGenerator_Acc_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'~'+str(pid)),'w') as f:
            for i in range(st_nid,ed_nid):
                nid = self.landmark_nodes[i]
                dists = self.scheme.dist_one_to_all(nid)[self.landmark_nodes].tolist()
                assert len(self.landmark_nodes) == len(dists)
                [f.write(str(int(nid))+','+str(int(dst_node))+','+str(float(dist))+'\n') for dst_node,dist in zip(self.landmark_nodes,dists)]
                if self.early_break > 0:
                    print('LandmarkInnerGenerator_Acc_{}:{}/{}'.format(pid,i - st_nid, self.early_break))
                else:
                    print('LandmarkInnerGenerator_Acc_{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))
                if i ==self.early_break:
                    break

class NodeRangGenerator_Acc(m_generator.NodeRangeGenerator_p):
    pass

class StochasticNodeRangeGenerator_Acc(m_generator.StochasticNodeRangeGenerator_p):
    pass

class FastNodeRangeGenerator_Acc(m_generator.FastNodeRangeGenerator_p):
    pass



if __name__ == '__main__':
    print('hello manager.')
    g, _ = dgl.load_graphs('../datasets/dst/facebook')
    g = g[0]
    emb_sz = 128
    workers = 16
    landmark_nodes = random.sample(g.nodes().tolist(),100)
    # ged = m_manager.GEDManager(workers=16)

    train_generator = FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp1',out_file='facebook-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)
    # val_generator = ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file='facebook-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=50,force=False,prod_workers=4)
    # landmark_generator = LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file='facebook-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)
    # node_range_generator = NodeRangGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file='fb-noderange',is_random=True,is_parallel=True,file_sz=10000,force=False,pair_sz=4)
    # stochastic_node_range_generator = StochasticNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=10, out_dir='../tmp',out_file='fb-snoderange=2000', is_random=True, is_parallel=True,file_sz=10000, force=False, pair_sz=2000,proximity_sz=20)

    # fast_node_range_generator = FastNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=8,
    #                                                                    out_dir='../tmp', out_file='fb-fnoderange=8',
    #                                                                    is_random=True, is_parallel=True, file_sz=10000,
    #                                                                    force=False, pair_sz=8*10, per_node_dst_sz=100,proximity_sz=20)

    train_generator.gen_to_disk(early_break=-1)
    # val_generator.gen_to_disk(early_break=30)
    # landmark_generator.gen_to_disk(early_break=-1)
    # node_range_generator.gen_to_disk(early_break=-1)
    # stochastic_node_range_generator.gen_to_disk(early_break=-1)
    # fast_node_range_generator.gen_to_disk(early_break=-1)
