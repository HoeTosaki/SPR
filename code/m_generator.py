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
import multiprocessing
import m_selector

# 回环？可。 边有权？边权负？不行
# TODO demo校验，以及扩展计算：如快速SILC，兼容有权图

## Acc.
use_file_acc = True
##

class SearchScheme:
    def __init__(self,g,out_dir='../outputs'):
        self.g = g
        self.out_dir = out_dir
        self.is_preprocessed = False
        self.preprocess()
    def preprocess(self):
        self.is_preprocessed = True
    def dist_between(self, src, dst):
        if not self.is_preprocessed:
            self.preprocess()

    def dist_one_to_all(self, src):
        if not self.is_preprocessed:
            self.preprocess()

    # @optional
    def dist_group_to_all(self,src_set,force=False):
        if not self.is_preprocessed:
            self.preprocess()
        # force will be & only be used in sub-class which implements this function.
        if not force:
            raise NotImplementedError

    # @optional
    def dist_one_to_other(self,src,dst_set,is_selected=True,force=False):
        if not self.is_preprocessed:
            self.preprocess()
        # force will be & only be used in sub-class which implements this function.
        if not force:
            raise NotImplementedError

    # @optional
    def path_one_to_other(self,src,dst_set,force=True):
        if not self.is_preprocessed:
            self.preprocess()
        # force will be & only be used in sub-class which implements this function.
        if not force:
            raise NotImplementedError


class BFS(SearchScheme):
    def __init__(self,g,out_dir='../outputs'):
        super(BFS, self).__init__(g,out_dir)

    def dist_between(self, src, dst):
        super(BFS, self).dist_between(src,dst)

        with self.g.local_scope():
            self.g.ndata['signal'] = -(tc.ones(self.g.num_nodes())).type_as(tc.IntTensor())

            # print(self.g.ndata['signal'])

            search_lst = [src]
            self.g.ndata['signal'][src] = 0
            while len(search_lst) != 0:
                # print(self.g.ndata['signal'])
                nid = search_lst.pop(0)
                if nid == dst:
                    return self.g.ndata['signal'][nid].tolist()
                lst = self.g.successors(nid)
                for nnid in lst:
                    if self.g.ndata['signal'][nnid] < 0:
                        search_lst.append(nnid)
                        self.g.ndata['signal'][nnid] = self.g.ndata['signal'][nid].tolist() + 1
        return -1 # 不相连通
    def dist_one_to_all(self, src):
        super(BFS, self).dist_one_to_all(src)

        dic_id2signal = [-1]*self.g.num_nodes()
        search_lst = [src]
        dic_id2signal[src] = 0
        while len(search_lst) != 0:
            nid = search_lst.pop(0)
            lst = self.g.successors(nid)
            for nnid in lst:
                if dic_id2signal[nnid] < 0:
                    search_lst.append(nnid)
                    dic_id2signal[nnid] = dic_id2signal[nid] + 1
        return tc.IntTensor(dic_id2signal)

    # @optional
    # only used in unweighted graph
    def dist_one_to_other(self,src,dst_set,is_selected=True,force=False):
        super(BFS, self).dist_one_to_other(src=src,dst_set=dst_set,is_selected=True,force=True)
        max_cnt = len(dst_set)
        cur_cnt = 0
        dic_id2signal = [-1]*self.g.num_nodes()
        search_lst = [src]
        dic_id2signal[src] = 0
        while len(search_lst) != 0:
            nid = search_lst.pop(0)
            lst = self.g.successors(nid).tolist()
            for nnid in lst:
                if dic_id2signal[nnid] < 0:
                    search_lst.append(nnid)
                    dic_id2signal[nnid] = dic_id2signal[nid] + 1
                    if nnid in dst_set:
                        cur_cnt += 1
                    if cur_cnt >= max_cnt:
                        break
            if cur_cnt >= max_cnt:
                break
        return tc.IntTensor(dic_id2signal)[list(dst_set)] if is_selected else tc.IntTensor(dic_id2signal)

    def path_one_to_other(self,src,dst_set,force=False):
        super(BFS, self).path_one_to_other(src=src, dst_set=dst_set,force=True)
        max_cnt = len(dst_set)
        if src in dst_set:
            max_cnt -= 1 # exclude src.
        cur_cnt = 0
        dic_id2signal = [-1] * self.g.num_nodes()
        search_lst = [src]
        dic_id2signal[src] = 0

        trace_map = {src:-1}
        while len(search_lst) != 0:
            nid = search_lst.pop(0)
            lst = self.g.successors(nid).tolist()
            for nnid in lst:
                nnid = int(nnid)
                if dic_id2signal[nnid] < 0:
                    search_lst.append(nnid)
                    dic_id2signal[nnid] = dic_id2signal[nid] + 1
                    if nnid in dst_set:
                        cur_cnt += 1
                    trace_map[nnid] = nid
                    if cur_cnt >= max_cnt:
                        break
            if cur_cnt >= max_cnt:
                break
        return trace_map,dic_id2signal


# class GroupBFS(BFS):
#     def __init__(self,g,out_dir='../outputs'):
#         super(GroupBFS, self).__init__(g,out_dir)
#
#     def dist_group_to_all(self,src_set,force=False):
#         super(GroupBFS, self).dist_group_to_all(src_set=src_set,force=True)
#         dic_id2signal = [-1] * self.g.num_nodes()
#         search_lst = [src]
#         dic_id2signal[src] = 0
#         while len(search_lst) != 0:
#             nid = search_lst.pop(0)
#             lst = self.g.successors(nid)
#             for nnid in lst:
#                 if dic_id2signal[nnid] < 0:
#                     search_lst.append(nnid)
#                     dic_id2signal[nnid] = dic_id2signal[nid] + 1
#         return tc.IntTensor(dic_id2signal)


class Floyd(SearchScheme):
    def __init__(self,g,out_dir='../outputs',out_file = 'floyd'):
        super(Floyd, self).__init__(g,out_dir)
        self.out_file = out_file
    def preprocess(self):
        super(Floyd, self).preprocess()

        if os.path.exists(os.path.join(self.out_dir,self.out_file+'.npy')):
            com_dict = np.load(os.path.join(self.out_dir,self.out_file+'.npy')).item()
            indices = com_dict['indices']
            values = com_dict['values']
            size = com_dict['size']
            # print(indices)
            # print(values)
            # print(size)
            self.dist = tc.sparse.FloatTensor(indices,values,size)
            self.ind_lst = com_dict['ind_lst']
            print('floyd read file succ.')
            return
        # self.g = dgl.DGLGraph()

        adj = self.g.adj()
        n = adj.shape[0]
        dist = tc.sparse.FloatTensor(size=adj.shape)
        # dist = tc.sparse_coo_tensor(size=adj.shape)

        # 对角线置0
        dist.add_(tc.sparse.IntTensor(tc.LongTensor([range(n),range(n)]),tc.zeros(n,dtype=tc.float32),adj.shape))

        # 加入邻接边
        dist.add_(adj)
        if not dist.is_coalesced():
            dist = dist.coalesce()
        self.tmp_lst = None
        self.tmp_is_changed = True
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.tmp_is_changed:
                        lst1,lst2 = dist.indices().tolist()
                        self.tmp_lst = [(x,y) for x,y in zip(lst1,lst2)]
                    if (i,k) in self.tmp_lst:
                        self.tmp_a = dist[i][k]
                    else:
                        self.tmp_a = -1
                    if (k, j) in self.tmp_lst:
                        self.tmp_b = dist[k][j]
                    else:
                        self.tmp_b = -1
                    if (i, j) in self.tmp_lst:
                        self.tmp_c = dist[i][j]
                    else:
                        self.tmp_c = -1
                    self.tmp_is_changed = False
                    if self.tmp_a != -1 and self.tmp_b != -1:
                        if self.tmp_c == -1:
                            self.tmp_is_changed = True
                        elif self.tmp_c > self.tmp_a + self.tmp_b:
                            self.tmp_is_changed = True
                    if self.tmp_is_changed:
                        new_dist = self.tmp_a + self.tmp_b
                        dist.add_(tc.sparse.IntTensor(tc.LongTensor([[i],[j]]), tc.FloatTensor([-dist[i][j].tolist() + new_dist]),
                                                      adj.shape))
                        if not dist.is_coalesced():
                            dist = dist.coalesce()
        self.dist = dist
        lst1,lst2 = dist.indices().tolist()
        self.ind_lst = [(x,y) for x,y in zip(lst1,lst2)]

        np.save(os.path.join(self.out_dir,self.out_file+'.npy'),{'indices':self.dist.indices(),'values':self.dist.values(),'size':self.dist.size(),'ind_lst':self.ind_lst})
    def dist_between(self, src, dst):
        super(Floyd, self).dist_between(src,dst)
        if (src,dst) in self.ind_lst:
            return self.dist[src][dst]
        else:
            return -1
    def dist_one_to_all(self, src):
        super(Floyd, self).dist_one_to_all(src)
        return [self.dist_between(src,nid) for nid in self.g.nodes().tolist()]

class SILC(SearchScheme):
    def __init__(self,g,out_dir='../outputs'):
        super(SILC, self).__init__(g,out_dir)
    def preprocess(self):
        super(SILC, self).preprocess()

        self.g = dgl.DGLGraph()
        nodes_dict = {}
        for src in self.g.nodes().tolist():
            nodes_dict[src] = {}
            with self.g.local_scope():
                self.g.ndata['signal'] = tc.zeros(self.g.num_nodes(),dtype=tc.bool)
                search_lst = [src]
                self.g.ndata['signal'][src] = True
                while len(search_lst) != 0:
                    nid = search_lst.pop(0)
                    lst = self.g.successors(nid)
                    for nnid in lst:
                        if not self.g.ndata['signal'][nnid]:
                            search_lst.append(nnid)
                            self.g.ndata['signal'][nnid] = True

                return self.g.ndata['signal']

    def dist_between(self, src, dst):
        super(SILC, self).dist_between(src,dst)
        with self.g.local_scope():
            self.g.ndata['signal'] = -(tc.ones(self.g.num_nodes())).type_as(tc.IntTensor())

            # print(self.g.ndata['signal'])

            search_lst = [src]
            self.g.ndata['signal'][src] = 0
            while len(search_lst) != 0:
                # print(self.g.ndata['signal'])
                nid = search_lst.pop(0)
                if nid == dst:
                    return self.g.ndata['signal'][nid].tolist()
                lst = self.g.successors(nid)
                for nnid in lst:
                    if self.g.ndata['signal'][nnid] < 0:
                        search_lst.append(nnid)
                        self.g.ndata['signal'][nnid] = self.g.ndata['signal'][nid].tolist() + 1
        return -1 # 不相连通
    def dist_one_to_all(self, src):
        super(SILC, self).dist_one_to_all(src)

        # self.g = dgl.DGLGraph()
        with self.g.local_scope():
            self.g.ndata['signal'] = -(tc.ones(self.g.num_nodes())).type_as(tc.IntTensor())

            # print(self.g.ndata['signal'])

            search_lst = [src]
            self.g.ndata['signal'][src] = 0
            while len(search_lst) != 0:
                nid = search_lst.pop(0)
                lst = self.g.successors(nid)
                for nnid in lst:
                    if self.g.ndata['signal'][nnid] < 0:
                        search_lst.append(nnid)
                        self.g.ndata['signal'][nnid] = self.g.ndata['signal'][nid].tolist() + 1
            return self.g.ndata['signal']
        # 异常错误
        return None

class Generator:
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False):
        '''
        目前仅在并行处理时使用小文件组、随机读取。
        :param g:
        :param scheme:
        :param workers:
        :param out_dir:
        :param out_file:
        :param is_random: 指定读入是否自动打乱顺序，仅在小文件组有效
        :param is_parallel:使用并行处理
        :param file_sz: 使用小文件组存储gen时指定
        :param force: 复用文件
        '''
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
        self.rerand_cnt = 0
        self.rerand_max = 20
    def check_file(self):
        if self.file_sz:
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
        else:
            if os.path.exists(os.path.join(self.out_dir, self.out_file)):
                return True
        return False

    def loader(self,batch_sz=3,meta_batch_sz=10):
        '''
        随机读取需要存储时设定合适的file_sz，否则内排效率会变差
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
        # print('cur file_lst=',file_lst)
        # print('gen shuffle files...')
        st_time = time.time()
        if self.is_random and self.rerand_cnt == 0:
            if not self.thPool:
                self.thPool = threadpool.ThreadPool(self.workers)
            reqs = threadpool.makeRequests(Generator._th_shuffle_file,file_lst)
            [self.thPool.putRequest(req) for req in reqs]
            self.thPool.wait()
        # print('gen shuffle files finished ,time:{:.2f}s'.format(time.time()-st_time))

        # since we have a first random selection for training sequences. delay one time to perform this.
        self.rerand_cnt = (self.rerand_cnt + 1) % self.rerand_max

        if self.is_random:
            file_loaders = [Generator._file_loader(file_lst = [file],batch_sz=meta_batch_sz) for file in file_lst]
            ret = []
            while True:
                while len(ret) < batch_sz and len(file_loaders) > 0:
                    file_loader = random.choice(file_loaders)
                    meta_batch = next(file_loader,'end')
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
                    #has no next batch..
                    break
        else:
             self._file_loader(file_lst=file_lst,batch_sz=batch_sz)

    def gen_to_disk(self):
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            return
        print('gen_to_disk start to work...')
        st_time = time.time()
        if self.is_parallel:
            # parallel request usage of file group.
            assert self.file_sz
            self.lockFile = threading.Lock()
            self.lockIter = threading.Lock()
            self.file_cnt = 0
            self.gen_iter_instance = self.gen_iter()
            self.file_lst = []

            if not self.thPool:
                self.thPool = threadpool.ThreadPool(self.workers)

            reqs = threadpool.makeRequests(Generator._th_gen_to_disk,[self]*self.workers)
            [self.thPool.putRequest(req) for req in reqs]
            self.thPool.wait()
            return self.file_lst
        else:
            if self.file_sz:
                file_cnt = 0
                file_lst = []
                gen_iter_instance = self.gen_iter()
                while True:
                    cur_out_file = self.out_file + '~' + str(file_cnt)
                    file_cnt += 1
                    file_lst.append(cur_out_file)
                    is_mod = False
                    file_sz_cnt = 0
                    with open(os.path.join(self.out_dir,cur_out_file),'w') as f:
                        while True:
                            batch = next(gen_iter_instance, 'stop')
                            if batch == 'stop' or len(batch) == 0:
                                break
                            # batch not empty.
                            is_mod = True
                            for src, dst, dist in batch:
                                f.writelines('{},{},{}\n'.format(src, dst, dist))
                            file_sz_cnt += len(batch)
                            if file_sz_cnt > self.file_sz:
                                break
                    if not is_mod:
                        os.remove(os.path.join(self.out_dir,cur_out_file))
                        file_lst.remove(cur_out_file)
                        break
            else:
                with open(os.path.join(self.out_dir,self.out_file),'w') as f:
                    for batch in self.gen_iter():
                        for src,dst,dist in batch:
                            f.writelines('{},{},{}\n'.format(src,dst,dist))

    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        pass

    def _th_gen_to_disk(self):
        while True:
            is_mod = False
            self.lockFile.acquire()
            file_id = self.file_cnt
            self.file_cnt += 1
            cur_gen_file = self.out_file+'~'+str(file_id)
            self.file_lst.append(cur_gen_file)
            self.lockFile.release()
            file_sz_cnt = 0
            with open(os.path.join(self.out_dir,cur_gen_file),'w') as f:
                while True:
                    self.lockIter.acquire()
                    batch = next(self.gen_iter_instance,'stop')
                    self.lockIter.release()
                    if batch == 'stop' or len(batch) == 0:
                        break
                    # batch not empty.
                    is_mod = True
                    for src,dst,dist in batch:
                        f.writelines('{},{},{}\n'.format(src,dst,dist))
                    file_sz_cnt += len(batch)
                    if file_sz_cnt > self.file_sz:
                        break
            if not is_mod:
                #cur file is empty.
                os.remove(os.path.join(self.out_dir,cur_gen_file))
                self.file_lst.remove(cur_gen_file)
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
                    if len(lst) != 3:
                        break
                    assert len(lst) == 3
                    if lst[0].startswith('tensor'):
                        src = int(lst[0].split('(')[1][:-1])
                    else:
                        src = int(lst[0])
                    if lst[1].startswith('tensor'):
                        dst = int(lst[1].split('(')[1][:-1])
                    else:
                        dst = int(lst[1])
                    dist = float(lst[2])

                    ret.append((src,dst,dist))
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

class ClassicalRandomGenerator(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,data_sz_per_node=5):
        super(ClassicalRandomGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.data_sz_per_node = data_sz_per_node
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # g = dgl.DGLGraph()
        lst_nodes = self.g.nodes().tolist().copy()
        random.shuffle(lst_nodes)
        for node in lst_nodes:
            dst_nodes = random.sample(lst_nodes,min(len(lst_nodes),self.data_sz_per_node))
            dists = self.scheme.dist_one_to_all(node)[dst_nodes].tolist()
            assert len(dists) == len(dst_nodes)
            yield [(node,dst_node,dist) for dst_node,dist in zip(dst_nodes,dists)]

class ClassicalRandomGenerator_p(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,data_sz_per_node=5,prod_workers=1):
        super(ClassicalRandomGenerator_p, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.prod_workers = prod_workers
        self.data_sz_per_node = data_sz_per_node
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        self.ret = []
        self.lockProd = threading.Lock()
        self.lockInd = threading.Lock()
        # g = dgl.DGLGraph()
        self.lst_nodes = self.g.nodes().tolist().copy()
        self.per_worker_nodes = math.ceil(len(self.lst_nodes) / self.prod_workers)
        self.workId = 0
        self.ret_sgn = []
        random.shuffle(self.lst_nodes)

        self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        reqs = threadpool.makeRequests(ClassicalRandomGenerator_p._th_gen,[self]*self.prod_workers)
        [self.thPool_prod.putRequest(req) for req in reqs]
        while len(self.ret_sgn) < self.prod_workers:
            batch_ret = []
            self.lockProd.acquire()
            # 按生产者权限接入，查询有多少可以yield
            if len(self.ret) != 0:
                batch_ret.extend(self.ret)
                self.ret = []
            self.lockProd.release()
            if len(batch_ret) > 0:
                yield batch_ret
            time.sleep(2)
        self.thPool_prod.wait()

    def _th_gen(self):
        #require worker id.
        cur_id = -1
        self.lockInd.acquire()
        cur_id = self.workId
        self.workId += 1
        self.lockInd.release()

        assert cur_id != -1

        st_nid = max(cur_id*self.per_worker_nodes,0)
        ed_nid = min((cur_id+1) * self.per_worker_nodes,len(self.lst_nodes))

        print(cur_id,' start to work for [{},{})'.format(st_nid,ed_nid))

        prod_batch_sz = 10
        prod_batch = []
        for i in range(st_nid,ed_nid):
            nid = self.lst_nodes[i]
            dst_nodes = random.sample(self.lst_nodes,self.data_sz_per_node)
            dists = self.scheme.dist_one_to_all(nid)[dst_nodes].tolist()
            assert len(dists) == len(dst_nodes)
            prod_batch.extend([(nid,dst_node,dist) for dst_node,dist in zip(dst_nodes,dists)])
            if len(prod_batch) > prod_batch_sz:
                self.lockProd.acquire()
                self.ret.extend(prod_batch)
                prod_batch = []
                self.lockProd.release()
        if len(prod_batch) > 0:
            self.lockProd.acquire()
            self.ret.extend(prod_batch)
            self.lockProd.release()

        self.lockInd.acquire()
        self.ret_sgn.append(True)
        self.lockInd.release()


class FastRandomGenerator(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,data_sz_per_node=5,landmark_nodes=None):
        super(FastRandomGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.data_sz_per_node = data_sz_per_node
        self.landmark_nodes = landmark_nodes
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # g = dgl.DGLGraph()
        assert self.landmark_nodes is not None

        landmark_nodes = self.landmark_nodes.copy()
        random.shuffle(landmark_nodes)

        ret = []
        for node in self.landmark_nodes:
            dst_nodes = self.g.nodes().tolist()
            dists = self.scheme.dist_one_to_all(node).tolist()
            assert len(dists) == len(dst_nodes)
            yield [(node,dst_node,dist) for dst_node,dist in zip(dst_nodes,dists)] #此处暂不排除src = dst


class FastRandomGenerator_p(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,data_sz_per_node=5,landmark_nodes=None,prod_workers=5):
        super(FastRandomGenerator_p, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.data_sz_per_node = data_sz_per_node
        self.landmark_nodes = landmark_nodes
        self.prod_workers = prod_workers
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        assert self.landmark_nodes is not None
        random.shuffle(self.landmark_nodes)

        self.ret = []
        self.lockProd = threading.Lock()
        self.lockInd = threading.Lock()
        # g = dgl.DGLGraph()
        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.prod_workers)
        self.workId = 0
        self.ret_sgn = []
        self.lst_nodes = self.g.nodes().tolist()

        self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        reqs = threadpool.makeRequests(FastRandomGenerator_p._th_gen,[self]*self.prod_workers)
        [self.thPool_prod.putRequest(req) for req in reqs]
        while len(self.ret_sgn) < self.prod_workers:
            batch_ret = []
            self.lockProd.acquire()
            # 按生产者权限接入，查询有多少可以yield
            if len(self.ret) != 0:
                batch_ret.extend(self.ret)
                self.ret = []
            self.lockProd.release()
            if len(batch_ret) > 0:
                yield batch_ret
            time.sleep(2)
        self.thPool_prod.wait()

    def _th_gen(self):
        #require worker id.
        cur_id = -1
        self.lockInd.acquire()
        cur_id = self.workId
        self.workId += 1
        self.lockInd.release()

        assert cur_id != -1

        st_nid = max(cur_id*self.per_worker_nodes,0)
        ed_nid = min((cur_id+1) * self.per_worker_nodes,len(self.landmark_nodes))

        print(cur_id,' start to work for [{},{})'.format(st_nid,ed_nid))

        prod_batch_sz = 10
        prod_batch = []
        for i in range(st_nid,ed_nid):
            nid = self.landmark_nodes[i]
            dists = self.scheme.dist_one_to_all(nid).tolist()
            assert len(self.lst_nodes) == len(dists)
            prod_batch.extend([(nid,dst_node,dist) for dst_node,dist in zip(self.lst_nodes,dists)])
            if len(prod_batch) > prod_batch_sz:
                self.lockProd.acquire()
                self.ret.extend(prod_batch)
                prod_batch = []
                self.lockProd.release()
        if len(prod_batch) > 0:
            self.lockProd.acquire()
            self.ret.extend(prod_batch)
            self.lockProd.release()

        self.lockInd.acquire()
        self.ret_sgn.append(True)
        self.lockInd.release()


class LandmarkInnerGenerator(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,landmark_nodes=None):
        super(LandmarkInnerGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.landmark_nodes = landmark_nodes

    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # g = dgl.DGLGraph()
        assert self.landmark_nodes is not None

        landmark_nodes = self.landmark_nodes.copy()
        random.shuffle(landmark_nodes)

        ret = []
        for node in self.landmark_nodes:
            dst_nodes = self.landmark_nodes.copy()
            random.shuffle(dst_nodes)
            dists = self.scheme.dist_one_to_all(node)[dst_nodes].tolist()
            assert len(dists) == len(dst_nodes)
            yield [(node, dst_node, dist) for dst_node, dist in zip(dst_nodes, dists) if node != dst_node]

class LandmarkInnerGenerator_p(Generator):
    def __init__(self, g, scheme=BFS(None), workers=1, out_dir='../outputs', out_file='gen-file',is_random=False, is_parallel=False, file_sz=None, force=False, landmark_nodes=None,prod_workers=5):
        super(LandmarkInnerGenerator_p, self).__init__(g=g, scheme=scheme, workers=workers, out_dir=out_dir,out_file=out_file, is_random=is_random,is_parallel=is_parallel, file_sz=file_sz, force=force)
        self.landmark_nodes = landmark_nodes
        self.prod_workers = prod_workers
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        assert self.landmark_nodes is not None
        random.shuffle(self.landmark_nodes)

        self.ret = []
        self.lockProd = threading.Lock()
        self.lockInd = threading.Lock()
        # g = dgl.DGLGraph()
        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.prod_workers)
        self.workId = 0
        self.ret_sgn = []

        self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        reqs = threadpool.makeRequests(LandmarkInnerGenerator_p._th_gen,[self]*self.prod_workers)
        [self.thPool_prod.putRequest(req) for req in reqs]
        while len(self.ret_sgn) < self.prod_workers:
            batch_ret = []
            self.lockProd.acquire()
            # 按生产者权限接入，查询有多少可以yield
            if len(self.ret) != 0:
                batch_ret.extend(self.ret)
                self.ret = []
            self.lockProd.release()
            if len(batch_ret) > 0:
                yield batch_ret
            time.sleep(2)
        self.thPool_prod.wait()

    def _th_gen(self):
        #require worker id.
        cur_id = -1
        self.lockInd.acquire()
        cur_id = self.workId
        self.workId += 1
        self.lockInd.release()

        assert cur_id != -1

        st_nid = max(cur_id*self.per_worker_nodes,0)
        ed_nid = min((cur_id+1) * self.per_worker_nodes,len(self.landmark_nodes))

        print(cur_id,' start to work for [{},{})'.format(st_nid,ed_nid))

        prod_batch_sz = 10
        prod_batch = []
        for i in range(st_nid,ed_nid):
            nid = self.landmark_nodes[i]
            dists = self.scheme.dist_one_to_all(nid)[self.landmark_nodes].tolist()
            assert len(self.landmark_nodes) == len(dists)
            prod_batch.extend([(nid,dst_node,dist) for dst_node,dist in zip(self.landmark_nodes,dists)])
            if len(prod_batch) > prod_batch_sz:
                self.lockProd.acquire()
                self.ret.extend(prod_batch)
                prod_batch = []
                self.lockProd.release()
        if len(prod_batch) > 0:
            self.lockProd.acquire()
            self.ret.extend(prod_batch)
            self.lockProd.release()

        self.lockInd.acquire()
        self.ret_sgn.append(True)
        self.lockInd.release()


class LandmarkInnerGenerator_p(Generator):
    def __init__(self, g, scheme=BFS(None), workers=1, out_dir='../outputs', out_file='gen-file',is_random=False, is_parallel=False, file_sz=None, force=False, landmark_nodes=None,prod_workers=5):
        super(LandmarkInnerGenerator_p, self).__init__(g=g, scheme=scheme, workers=workers, out_dir=out_dir,out_file=out_file, is_random=is_random,is_parallel=is_parallel, file_sz=file_sz, force=force)
        self.landmark_nodes = landmark_nodes
        self.prod_workers = prod_workers
    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        assert self.landmark_nodes is not None
        random.shuffle(self.landmark_nodes)

        self.ret = []
        self.lockProd = threading.Lock()
        self.lockInd = threading.Lock()
        # g = dgl.DGLGraph()
        self.per_worker_nodes = math.ceil(len(self.landmark_nodes) / self.prod_workers)
        self.workId = 0
        self.ret_sgn = []

        self.thPool_prod = threadpool.ThreadPool(num_workers=self.prod_workers)
        reqs = threadpool.makeRequests(LandmarkInnerGenerator_p._th_gen,[self]*self.prod_workers)
        [self.thPool_prod.putRequest(req) for req in reqs]
        while len(self.ret_sgn) < self.prod_workers:
            batch_ret = []
            self.lockProd.acquire()
            # 按生产者权限接入，查询有多少可以yield
            if len(self.ret) != 0:
                batch_ret.extend(self.ret)
                self.ret = []
            self.lockProd.release()
            if len(batch_ret) > 0:
                yield batch_ret
            time.sleep(2)
        self.thPool_prod.wait()

    def _th_gen(self):
        #require worker id.
        cur_id = -1
        self.lockInd.acquire()
        cur_id = self.workId
        self.workId += 1
        self.lockInd.release()

        assert cur_id != -1

        st_nid = max(cur_id*self.per_worker_nodes,0)
        ed_nid = min((cur_id+1) * self.per_worker_nodes,len(self.landmark_nodes))

        print(cur_id,' start to work for [{},{})'.format(st_nid,ed_nid))

        prod_batch_sz = 10
        prod_batch = []
        for i in range(st_nid,ed_nid):
            nid = self.landmark_nodes[i]
            dists = self.scheme.dist_one_to_all(nid)[self.landmark_nodes].tolist()
            assert len(self.landmark_nodes) == len(dists)
            prod_batch.extend([(nid,dst_node,dist) for dst_node,dist in zip(self.landmark_nodes,dists)])
            if len(prod_batch) > prod_batch_sz:
                self.lockProd.acquire()
                self.ret.extend(prod_batch)
                prod_batch = []
                self.lockProd.release()
        if len(prod_batch) > 0:
            self.lockProd.acquire()
            self.ret.extend(prod_batch)
            self.lockProd.release()

        self.lockInd.acquire()
        self.ret_sgn.append(True)
        self.lockInd.release()

class NodeRangeGenerator(Generator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,pair_sz=1000):
        super(NodeRangeGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force)
        self.pair_sz = pair_sz

    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # self.g = dgl.DGLGraph()
        assert self.pair_sz is not None
        assert self.pair_sz < self.g.num_nodes()

        nodes = self.g.nodes().copy().tolist()
        random.shuffle(nodes)
        src_nodes = nodes[:self.pair_sz]
        dst_nodes = nodes[-self.pair_sz:]
        for src,dst in zip(src_nodes,dst_nodes):
            src_set = self.g.successors(src).tolist() + self.g.predecessors(src).tolist()
            dst_set = self.g.successors(dst).tolist() + self.g.predecessors(dst).tolist()

            src_set.append(src)
            dst_set.append(dst)

            dist_dic = {}
            dist_dic[src] = {}
            dist_dic[dst] = {}

            dists_src = self.scheme.dist_one_to_all(src)[dst_set]
            dists_dst = self.scheme.dist_one_to_all(dst)[src_set]

            for e_dist, e_dst in zip(dists_src, dst_set):
                dist_dic[src][e_dst] = e_dist
            for e_dist, e_src in zip(dists_dst, src_set):
                dist_dic[dst][e_src] = e_dist

            # dist_dic = {}
            # for e_src in src_set:
            #     dist_dic[e_src] = {}
            # for e_src in src_set:
            #     dists = self.scheme.dist_one_to_all(e_src)
            #     for e_dst in dst_set:
            #         dist_dic[e_src][e_dst] = dists[e_dst]
            yield [(src, dst, dist_dic)] # once at one time.

    def gen_to_disk(self):
        with open(os.path.join(self.out_dir, self.out_file), 'w') as f:
            for batch in self.gen_iter():
                for src, dst, dist_dic in batch:
                    dist_str = ''
                    for e_src in dist_dic.keys():
                        for e_dst in dist_dic[e_src].keys():
                            dist_str += str(e_src)+'&'+str(e_dst)+'&' + str(dist_dic[e_src][e_dst]) + '@'
                    f.writelines('{},{},{}\n'.format(src, dst, dist_str))

    def loader(self, batch_sz=3, meta_batch_sz=10):
        '''
        随机读取需要存储时设定合适的file_sz，否则内排效率会变差
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
            cur_file = os.path.join(self.out_dir, self.out_file)
            if os.path.exists(cur_file):
                file_lst.append(cur_file)
        assert len(file_lst) >= 1
        file_lst = [os.path.join(self.out_dir, file) for file in file_lst]
        # print('cur file_lst=',file_lst)
        # print('gen shuffle files...')
        st_time = time.time()
        if self.is_random:
            if not self.thPool:
                self.thPool = threadpool.ThreadPool(self.workers)
            reqs = threadpool.makeRequests(Generator._th_shuffle_file, file_lst)
            [self.thPool.putRequest(req) for req in reqs]
            self.thPool.wait()
        # print('gen shuffle files finished ,time:{:.2f}s'.format(time.time()-st_time))

        if self.is_random:
            file_loaders = [NodeRangeGenerator._file_loader(file_lst=[file], batch_sz=meta_batch_sz) for file in file_lst]
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
        else:
            self._file_loader(file_lst=file_lst, batch_sz=batch_sz)

    def _th_gen_to_disk(self):
        raise NotImplementedError

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
                    if len(lst) != 3:
                        break
                    assert len(lst) == 3
                    if lst[0].startswith('tensor'):
                        src = int(lst[0].split('(')[1][:-1])
                    else:
                        src = int(lst[0])
                    if lst[1].startswith('tensor'):
                        dst = int(lst[1].split('(')[1][:-1])
                    else:
                        dst = int(lst[1])
                    pair_lst = lst[2].strip().split('@')
                    dist_dic = {}
                    for pair in pair_lst:
                        if pair.strip() == "":
                            continue
                        node_pair = pair.strip().split('&')
                        assert len(node_pair) == 3
                        e_src = int(node_pair[0])
                        e_dst = int(node_pair[1])
                        e_dist = int(node_pair[2])
                        if e_src not in dist_dic:
                            dist_dic[e_src] = {}
                        dist_dic[e_src][e_dst] = e_dist
                    ret.append((src,dst,dist_dic))
                    cnt += 1
                    if cnt == batch_sz:
                        yield ret
                        cnt = 0
                        ret = []
                if cnt > 0:
                    ret = ret[:cnt]
                    yield ret

class NodeRangeGenerator_p(NodeRangeGenerator):
    def __init__(self, g, scheme=BFS(None), workers=1, out_dir='../outputs', out_file='gen-file', is_random=False,is_parallel=False, file_sz=None, force=False, pair_sz=1000):
        super(NodeRangeGenerator_p, self).__init__(g=g, scheme=scheme, workers=workers, out_dir=out_dir,
                                                 out_file=out_file, is_random=is_random, is_parallel=is_parallel,
                                                 file_sz=file_sz, force=force,pair_sz=pair_sz)
    def gen_to_disk(self,early_break=-1):
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            return
        print('gen_to_disk start to work...')
        st_time = time.time()
        assert self.file_sz
        self.early_break = early_break
        # if not self.thPool:
        #     self.thPool = multiprocessing.Pool(self.workers)
        nodes = self.g.nodes().tolist().copy()
        random.shuffle(nodes)
        self.src_nodes = nodes[:self.pair_sz]
        self.dst_nodes = nodes[-self.pair_sz:]
        self.per_worker_nodes = math.ceil(self.pair_sz / self.workers)
        procs = []
        for i in range(self.workers):
            proc = multiprocessing.Process(target=NodeRangeGenerator_p._th_gen,args=(self,i,))
            proc.start()
            # self.thPool.apply_async(NodeRangeGenerator_p._th_gen,(i,))
        # self.thPool.close()
        # self.thPool.join()
        for proc in procs:
            proc.join()
        print('gen finish, time consume:{:.2f}'.format(time.time()-st_time))

    def _th_gen(self,pid):

        assert pid != -1

        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,len(self.src_nodes))

        print('NodeRangeGenerator_p_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'~'+str(pid)),'w') as f:
            for i in range(st_nid,ed_nid):
                src = self.src_nodes[i]
                dst = self.dst_nodes[i]

                src_set = self.g.successors(src).tolist() + self.g.predecessors(src).tolist()
                dst_set = self.g.successors(dst).tolist() + self.g.predecessors(dst).tolist()

                src_set.append(src)
                dst_set.append(dst)

                dist_dic = {}
                dist_dic[src] = {}
                dist_dic[dst] = {}

                dists_src = self.scheme.dist_one_to_all(src)[dst_set]
                dists_dst = self.scheme.dist_one_to_all(dst)[src_set]

                for e_dist, e_dst in zip(dists_src, dst_set):
                    dist_dic[src][e_dst] = e_dist
                for e_dist, e_src in zip(dists_dst, src_set):
                    dist_dic[dst][e_src] = e_dist

                # dist_dic = {}
                # for e_src in src_set:
                #     dist_dic[e_src] = {}
                # for e_src in src_set:
                #     dists = self.scheme.dist_one_to_all(e_src)
                #     for e_dst in dst_set:
                #         dist_dic[e_src][e_dst] = dists[e_dst]
                dist_str = ''
                for e_src in dist_dic.keys():
                    for e_dst in dist_dic[e_src].keys():
                        dist_str += str(int(e_src)) + '&' + str(int(e_dst)) + '&' + str(int(dist_dic[e_src][e_dst])) + '@'
                f.writelines('{},{},{}\n'.format(src, dst, dist_str))
                if self.early_break > 0:
                    print('NodeRangeGenerator_p_{}:{}/{}'.format(pid,i-st_nid,self.early_break))
                else:
                    print('NodeRangeGenerator_p_{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))
                if self.early_break > 0 and (i - st_nid) >= self.early_break:
                    break

class StochasticNodeRangeGenerator(NodeRangeGenerator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,pair_sz=1000,proximity_sz = 10):
        super(StochasticNodeRangeGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force,pair_sz=pair_sz)
        self.proximity_sz = proximity_sz

    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # self.g = dgl.DGLGraph()
        assert self.pair_sz is not None
        assert self.pair_sz < self.g.num_nodes()

        nodes = self.g.nodes().copy().tolist()
        random.shuffle(nodes)
        src_nodes = nodes[:self.pair_sz]
        dst_nodes = nodes[-self.pair_sz:]
        for src,dst in zip(src_nodes,dst_nodes):
            src_set = self.g.successors(src).tolist() + self.g.predecessors(src).tolist()
            dst_set = self.g.successors(dst).tolist() + self.g.predecessors(dst).tolist()

            if len(src_set) > self.proximity_sz:
                random.shuffle(src_set)
                src_set = src_set[:self.proximity_sz]
            if len(dst_set) > self.proximity_sz:
                random.shuffle(dst_set)
                dst_set = dst_set[:self.proximity_sz]

            src_set.append(src)
            dst_set.append(dst)

            dist_dic = {}
            for e_src in src_set:
                dist_dic[e_src] = {}
            for e_src in src_set:
                dists = self.scheme.dist_one_to_all(e_src)
                for e_dst in dst_set:
                    dist_dic[e_src][e_dst] = dists[e_dst]
            yield [(src, dst, dist_dic)] # once at one time.


class StochasticNodeRangeGenerator_p(StochasticNodeRangeGenerator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,pair_sz=1000,proximity_sz = 10):
        super(StochasticNodeRangeGenerator_p, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force,pair_sz=pair_sz,proximity_sz=proximity_sz)

    def gen_to_disk(self,early_break=-1):
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            return
        print('gen_to_disk start to work...')
        st_time = time.time()
        assert self.file_sz
        self.early_break = early_break
        # if not self.thPool:
        #     self.thPool = multiprocessing.Pool(self.workers)
        nodes = self.g.nodes().tolist().copy()
        random.shuffle(nodes)
        self.src_nodes = nodes[:self.pair_sz]
        self.dst_nodes = nodes[-self.pair_sz:]
        self.per_worker_nodes = math.ceil(self.pair_sz / self.workers)
        procs = []
        for i in range(self.workers):
            proc = multiprocessing.Process(target=StochasticNodeRangeGenerator_p._th_gen,args=(self,i,))
            proc.start()
            procs.append(proc)
            # self.thPool.apply_async(NodeRangeGenerator_p._th_gen,(i,))
        # self.thPool.close()
        # self.thPool.join()
        for proc in procs:
            proc.join()
        print('gen finish, time consume:{:.2f}'.format(time.time()-st_time))

    def _th_gen(self,pid):

        assert pid != -1

        st_nid = max(pid*self.per_worker_nodes,0)
        ed_nid = min((pid+1) * self.per_worker_nodes,len(self.src_nodes))

        print('StochasticNodeRangeGenerator_p_{}: start to work for [{},{})\n'.format(pid,st_nid,ed_nid),end='')

        with open(os.path.join(self.out_dir,self.out_file+'~'+str(pid)),'w') as f:
            for i in range(st_nid,ed_nid):
                src = self.src_nodes[i]
                dst = self.dst_nodes[i]

                src_set = self.g.successors(src).tolist() + self.g.predecessors(src).tolist()
                dst_set = self.g.successors(dst).tolist() + self.g.predecessors(dst).tolist()

                if len(src_set) > self.proximity_sz:
                    random.shuffle(src_set)
                    src_set = src_set[:self.proximity_sz]
                if len(dst_set) > self.proximity_sz:
                    random.shuffle(dst_set)
                    dst_set = dst_set[:self.proximity_sz]

                src_set.append(src)
                dst_set.append(dst)

                dist_dic = {}
                dist_dic[src] = {}
                dist_dic[dst] = {}

                dists_src = self.scheme.dist_one_to_all(src)[dst_set]
                dists_dst = self.scheme.dist_one_to_all(dst)[src_set]

                for e_dist,e_dst in zip(dists_src,dst_set):
                    dist_dic[src][e_dst] = e_dist
                for e_dist, e_src in zip(dists_dst, src_set):
                    dist_dic[dst][e_src] = e_dist

                # for e_src in src_set:
                #     dist_dic[e_src] = {}
                # for e_src in src_set:
                #     dists = self.scheme.dist_one_to_all(e_src)
                #     for e_dst in dst_set:
                #         dist_dic[e_src][e_dst] = dists[e_dst]


                dist_str = ''
                for e_src in dist_dic.keys():
                    for e_dst in dist_dic[e_src].keys():
                        dist_str += str(int(e_src)) + '&' + str(int(e_dst)) + '&' + str(int(dist_dic[e_src][e_dst])) + '@'
                f.writelines('{},{},{}\n'.format(src, dst, dist_str))
                if self.early_break > 0:
                    print('StochasticNodeRangeGenerator_p_{}:{}/{}'.format(pid,i-st_nid,self.early_break))
                else:
                    print('StochasticNodeRangeGenerator_p_{}:{}/{}'.format(pid,i - st_nid, ed_nid - st_nid))
                if self.early_break > 0 and (i - st_nid) >= self.early_break:
                    break


class FastNodeRangeGenerator(NodeRangeGenerator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,pair_sz=1000,per_node_dst_sz=100,proximity_sz=20):
        super(FastNodeRangeGenerator, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force,pair_sz=pair_sz)
        self.per_node_dst_sz = per_node_dst_sz
        self.proximity_sz = proximity_sz

    def gen_iter(self):
        '''
        子类自定义最适的一次生成样本数
        :return:[(src, dst, dist),...]
        '''
        # self.g = dgl.DGLGraph()
        assert self.pair_sz is not None
        assert self.pair_sz < self.g.num_nodes()

        # sort and get a descending list by deg of nodes.
        nodes = self.g.nodes().tolist().copy()
        ds = m_selector.DegreeSelector(g=self.g)
        src_nodes = ds.perform(cnt=self.pair_sz, action='max')
        for src in src_nodes:
            dst_nodes = random.shuffle(nodes)[:self.per_node_dst_sz]

            # generate random proximity of dst node (but at one src node, it must be fixed.).
            dict_dst_node2prox_set = {}
            all_dst_related_node = []
            for dst_node in dst_nodes:
                lst_of_prox = self.g.successors(dst_node).tolist().copy()
                if len(lst_of_prox) > self.proximity_sz:
                    dict_dst_node2prox_set[dst_node] = random.shuffle(lst_of_prox)[:self.proximity_sz]
                else:
                    dict_dst_node2prox_set[dst_node] = lst_of_prox
                dict_dst_node2prox_set[dst_node].append(dst_node) # add self
                all_dst_related_node.extend(dict_dst_node2prox_set[dst_node])
            # strongly assert the graph is undirected, and we use the successors to represent node range neighborhood.
            src_set = set()
            lst_of_src_prox = self.g.successors(src).tolist().copy()
            if len(lst_of_src_prox) > self.proximity_sz:
                random.shuffle(lst_of_src_prox)
                lst_of_src_prox = lst_of_src_prox[:self.proximity_sz]
            src_set.update(lst_of_src_prox)
            src_set.add(src)
            dict_src_prox2dist = {}
            for src_prox in src_set:
                dict_src_prox2dist[src_prox] = self.scheme.dist_one_to_other(src,all_dst_related_node,is_selected=False)
            lst_of_ret_dic = []
            for dst_node in dst_nodes:
                dist_dic = {}
                dist_dic[src] = {}
                dist_dic[dst_node] = {}
                dists_src = dict_src_prox2dist[src][dict_dst_node2prox_set[dst_node]]
                dists_dst = []
                for src_prox in src_set:
                    dists_dst.append(dict_src_prox2dist[src_prox][dst_node])
                for e_dist, e_dst in zip(dists_src, dict_dst_node2prox_set[dst_node]):
                    dist_dic[src][e_dst] = e_dist
                for e_dist, e_src in zip(dists_dst, src_set):
                    dist_dic[dst][e_src] = e_dist
                lst_of_ret_dic.extend([(src,dst_node,dist_dic)])
            yield lst_of_ret_dic

class FastNodeRangeGenerator_p(FastNodeRangeGenerator):
    def __init__(self,g,scheme=BFS(None),workers=1,out_dir='../outputs',out_file='gen-file',is_random=False,is_parallel=False,file_sz=None,force=False,pair_sz=1000,per_node_dst_sz=100,proximity_sz=20):
        super(FastNodeRangeGenerator_p, self).__init__(g=g,scheme=scheme,workers=workers,out_dir=out_dir,out_file=out_file,is_random=is_random,is_parallel=is_parallel,file_sz=file_sz,force=force,pair_sz=pair_sz,per_node_dst_sz=per_node_dst_sz,proximity_sz=proximity_sz)

    def gen_to_disk(self, early_break=-1):
        if not self.force and self.check_file():
            print('gen_to_disk done & checked.')
            return
        print('gen_to_disk start to work...')
        st_time = time.time()
        assert self.file_sz
        self.early_break = early_break

        ds = m_selector.DegreeSelector(g=self.g)
        self.src_nodes = ds.perform(cnt=self.pair_sz, action='max')
        self.per_worker_nodes = math.ceil(self.pair_sz / self.workers)
        procs = []
        for i in range(self.workers):
            proc = multiprocessing.Process(target=FastNodeRangeGenerator_p._th_gen, args=(self, i,))
            proc.start()
            procs.append(proc)
            # self.thPool.apply_async(NodeRangeGenerator_p._th_gen,(i,))
        # self.thPool.close()
        # self.thPool.join()
        for proc in procs:
            proc.join()
        print('gen finish, time consume:{:.2f}'.format(time.time() - st_time))

    def _th_gen(self, pid):

        assert pid != -1

        st_nid = max(pid * self.per_worker_nodes, 0)
        ed_nid = min((pid + 1) * self.per_worker_nodes, len(self.src_nodes))

        print('FastNodeRangeGenerator_p_{}: start to work for [{},{})\n'.format(pid, st_nid, ed_nid), end='')

        with open(os.path.join(self.out_dir, self.out_file + '~' + str(pid)), 'w') as f:
            for i in range(st_nid, ed_nid):
                src = self.src_nodes[i]
                nodes = self.g.nodes().tolist().copy()
                random.shuffle(nodes)
                dst_nodes = nodes[:self.per_node_dst_sz]

                # avoid dst set containing src node.
                if src in dst_nodes:
                    dst_nodes.remove(src)

                # generate random proximity of dst node (but at one src node, it must be fixed.).
                dict_dst_node2prox_set = {}
                all_dst_related_node = []
                for dst_node in dst_nodes:
                    lst_of_prox = self.g.successors(dst_node).tolist().copy()
                    if len(lst_of_prox) > self.proximity_sz:
                        random.shuffle(lst_of_prox)
                        dict_dst_node2prox_set[dst_node] = lst_of_prox[:self.proximity_sz]
                    else:
                        dict_dst_node2prox_set[dst_node] = lst_of_prox
                    dict_dst_node2prox_set[dst_node].append(dst_node)  # add self
                    all_dst_related_node.extend(dict_dst_node2prox_set[dst_node])
                # strongly assert the graph is undirected, and we use the successors to represent node range neighborhood.
                src_set = set()
                lst_of_src_prox = self.g.successors(src).tolist().copy()
                if len(lst_of_src_prox) > self.proximity_sz:
                    random.shuffle(lst_of_src_prox)
                    lst_of_src_prox = lst_of_src_prox[:self.proximity_sz]
                src_set.update(lst_of_src_prox)
                src_set.add(src)
                dict_src_prox2dist = {}
                for src_prox in src_set:
                    dict_src_prox2dist[src_prox] = self.scheme.dist_one_to_other(src, all_dst_related_node,
                                                                                 is_selected=False)
                lst_of_ret_dic = []
                for dst_node in dst_nodes:
                    dist_dic = {}
                    dist_dic[src] = {}
                    dist_dic[dst_node] = {}
                    dists_src = dict_src_prox2dist[src][dict_dst_node2prox_set[dst_node]]
                    dists_dst = []
                    for src_prox in src_set:
                        dists_dst.append(dict_src_prox2dist[src_prox][dst_node])
                    for e_dist, e_dst in zip(dists_src, dict_dst_node2prox_set[dst_node]):
                        dist_dic[src][e_dst] = e_dist
                    for e_dist, e_src in zip(dists_dst, src_set):
                        dist_dic[dst_node][e_src] = e_dist
                    lst_of_ret_dic.extend([(src, dst_node, dist_dic)])

                for v_src,v_dst,v_dist_dic in lst_of_ret_dic:
                    dist_str = ''
                    for e_src in v_dist_dic.keys():
                        for e_dst in v_dist_dic[e_src].keys():
                            dist_str += str(int(e_src)) + '&' + str(int(e_dst)) + '&' + str(
                                int(v_dist_dic[e_src][e_dst])) + '@'
                    f.writelines('{},{},{}\n'.format(v_src, v_dst, dist_str))
                    if self.early_break > 0:
                        print('FastNodeRangeGenerator_p_{}:{}/{}'.format(pid, i - st_nid, self.early_break))
                    else:
                        print('FastNodeRangeGenerator_p_{}:{}/{}'.format(pid, i - st_nid, ed_nid - st_nid))
                    if self.early_break > 0 and (i - st_nid) >= self.early_break:
                        break
                if self.early_break > 0 and (i - st_nid) >= self.early_break:
                    break


class BFSGenerator_d:
    def __init__(self, g):
        self.g = g

    def random_loader(file, num_batch=3):
        with open(file+'.rand','r') as f:
            cnt = 0
            ret = tc.zeros(num_batch,3).type_as(tc.IntTensor())
            while True:
                line = f.readline()
                if line == "":
                    break
                lst = line.strip().split(',')
                assert len(lst) == 3
                src, dst, dist = int(lst[0]), int(lst[1]), int(lst[2])
                ret[cnt] = tc.LongTensor([src,dst,dist])
                cnt += 1
                if cnt == num_batch:
                    yield ret
                    cnt = 0
                    ret = tc.zeros(num_batch, 3).type_as(tc.IntTensor())
            if cnt > 0:
                ret = ret[:cnt]
                yield ret

    def landmark_intra_loader(file, num_batch=4):
        with open(file+'.intra','r') as f:
            cnt = 0
            ret = tc.zeros(num_batch, 3).type_as(tc.IntTensor())
            while True:
                line = f.readline()
                if line == "":
                    break
                lst = line.strip().split(',')
                assert len(lst) == 3
                src, dst, dist = int(lst[0]), int(lst[1]), int(lst[2])
                ret[cnt] = tc.LongTensor([src, dst, dist])
                cnt += 1
                if cnt == num_batch:
                    yield ret
                    cnt = 0
                    ret = tc.zeros(num_batch, 3).type_as(tc.IntTensor())
            if cnt > 0:
                ret = ret[:cnt]
                yield ret

    def landmark_inter_loader(file, num_batch=7):
        with open(file+'.inter','r') as f:
            cnt = 0
            ret = tc.zeros(num_batch, 3).type_as(tc.IntTensor())
            while True:
                line = f.readline()
                if line == "":
                    break
                lst = line.strip().split(',')
                assert len(lst) == 3
                src, dst, dist = int(lst[0]), int(lst[1]), int(lst[2])
                ret[cnt] = tc.LongTensor([src, dst, dist])
                cnt += 1
                if cnt == num_batch:
                    yield ret
                    cnt = 0
                    ret = tc.zeros(num_batch, 3).type_as(tc.IntTensor())
            if cnt > 0:
                ret = ret[:cnt]
                yield ret
    def fast_generate_random_disk(self,file,worker=1,num = -1):
        if num == -1 or num < worker:
            print('current fast generate function NOT support, jump to normal generation process.')
            self.generate_random_disk(file,num)
            return
        threads = [None]*worker
        for idx in range(len(threads)):
            threads[idx] = threading.Thread(target=BFSGenerator.generate_random_disk,args=(self,file+'.\{{}\}'.format(idx),num // 4))
            threads[idx].start()
        for thd in threads:
            thd.join()
        print('pair generation finish.')

    def generate_random_disk(self, file, num=-1):
        with open(file+'.rand','w') as f:
            for src,dst,dist in self.generate_random_iter(num=num):
                f.writelines('{},{},{}\n'.format(src,dst,dist))

    def generate_landmark_disk(self, file, num=-1):
        with open(file+'.intra', 'w') as f1:
            with open(file+'.inter', 'w') as f2:
                isIntra = True
                for src, dst, dist in self.generate_landmark_iter(num_landmark=num):
                    if src == -1 and dst == -1 and dist == -1:
                        isIntra = False
                        continue
                    if isIntra:
                        f1.writelines('{},{},{}\n'.format(src, dst, dist))
                    else:
                        f2.writelines('{},{},{}\n'.format(src, dst, dist))
    def generate_random_iter(self, num=-1):
        # self.g = dgl.DGLGraph()
        lst = self.g.nodes().tolist().copy()

        random.shuffle(lst)

        if num == -1:
            lst_src = lst[:len(lst) // 2]
            lst_dst = lst[len(lst) // 2:]
        else:
            assert num*2 <= len(lst)
            lst_src = lst[:num]
            lst_dst = lst[num:num * 2]

        for src, dst in zip(lst_src, lst_dst):
            yield (src, dst, self.dist_between(src, dst))

    def generate_landmark_iter(self, num_landmark=100):
        # self.g = dgl.DGLGraph()
        lst = self.g.nodes().tolist().copy()

        random.shuffle(lst)

        landmarks = lst[:num_landmark]
        dist_dict = {}
        for landmark in landmarks:
            dist_dict[landmark] = self.dist_one_to_all(landmark)
        for lm1 in landmarks:
            for lm2 in landmarks:
                if lm2 != lm1:
                    yield (lm1,lm2,dist_dict[lm1][lm2].tolist())

        yield (-1,-1,-1)

        for lm in landmarks:
            for nid in self.g.nodes().tolist():
                if nid not in landmarks:
                    yield (lm,nid,dist_dict[lm][nid].tolist())
                else:
                    # print('{} in {}!'.format(nid,landmarks)) # 保险证明
                    pass

    def generate_random(self, num=-1):
        # self.g = dgl.DGLGraph()
        lst = self.g.nodes().tolist().copy()

        random.shuffle(lst)

        if num == -1:
            lst_src = lst[:len(lst) // 2]
            lst_dst = lst[len(lst) // 2:]
        else:
            assert num*2 <= len(lst)
            lst_src = lst[:num]
            lst_dst = lst[num:num * 2]

        ret = []
        for src, dst in zip(lst_src, lst_dst):
            ret.append((src, dst, self.dist_between(src, dst)))
        return ret

    def generate_landmark(self, num_landmark=100):
        # self.g = dgl.DGLGraph()
        lst = self.g.nodes().tolist().copy()

        random.shuffle(lst)

        landmarks = lst[:num_landmark]
        dist_dict = {}
        for landmark in landmarks:
            dist_dict[landmark] = self.dist_one_to_all(landmark)
        intra_landmark = []
        inter_landmark = []
        for lm1 in landmarks:
            for lm2 in landmarks:
                if lm2 != lm1:
                    intra_landmark.append((lm1,lm2,dist_dict[lm1][lm2].tolist()))
        for lm in landmarks:
            for nid in self.g.nodes().tolist():
                if nid not in landmarks:
                    inter_landmark.append((lm,nid,dist_dict[lm][nid].tolist()))
                else:
                    # print('{} in {}!'.format(nid,landmarks)) # 保险证明
                    pass
        return intra_landmark, inter_landmark


class PathGenerator:
    def __init__(self,g,scheme=BFS(None),workers=8,out_dir='../outputs',out_file='path-gen-file',force=False,src_sz=100):
        self.g = g
        self.scheme = scheme.__class__(g, out_dir)
        self.out_dir = out_dir
        self.out_file = out_file
        self.force = force


if __name__ == '__main__':
    print('hello generator.')
    edges = [(3, 0), (3, 1), (3, 2), (3, 8), (3, 7), (1, 4), (4, 5), (4, 7), (7, 6), (6, 5), (8, 7), (8, 9), (9, 10), (8, 10)]
    src = []
    dst = []
    for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])
    g = dgl.DGLGraph((src,dst))
    g = dgl.to_bidirected(g)

    # f = Floyd(g=g)
    f = BFS(g=g)
    # print(f.dist_between(1,5))
    # print(f.dist_between(3,7))
    # print(f.dist_between(3,9))
    print(f.dist_one_to_all(3))
    print(f.dist_one_to_other(3,set([0,1,4,6]),is_selected=False))

    # gen = BFSGenerator(g)
    # print(gen.dist_between(4,10))

    # print(gen.generate_random(num = 5))

    # intra, inter = gen.generate_landmark(num_landmark=3)
    # for ele in intra:
    #     print(ele)
    # print('~~~~~~~~~~~~~~~~~')
    # for ele in inter:
    #     print(ele)

    # gen.generate_random_disk('why',5)
    # gen.generate_landmark_disk('why',3)

    # for ten in BFSGenerator.random_loader('why'):
    #     print(ten)
    #
    # print('~~~~~~~~~~~~~~~~~')
    #
    # for ten in BFSGenerator.landmark_intra_loader('why'):
    #     print(ten)
    #
    # print('~~~~~~~~~~~~~~~~~')
    #
    # for ten in BFSGenerator.landmark_inter_loader('why'):
    #     print(ten)

