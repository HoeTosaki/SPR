import m_dqm
import os
import torch as tc
import time
import memory_profiler as mem_profile
import random
import numpy as np
import utils
import pickle as pk
import tqdm
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

glb_cs = ['coral', 'dodgerblue', 'darkcyan', 'darkviolet', 'olive', 'cyan', 'black', 'firebrick',
              'gainsboro']

class DistanceQueryEval:
    def __init__(self,nx_g=None,dqms=None,generator=None,eval_name='dq-eval',tmp_dir='../log',force=False,seed=None):
        self.nx_g = nx_g
        self.dqms = dqms
        self.generator = generator
        self.eval_name=eval_name
        self.tmp_dir = tmp_dir
        self.force = force
        self.work_dic = {}
        self.seed = seed

    def evaluate(self):
        for dqm in self.dqms:
            print('[{}] - [{}]'.format(self,dqm))
            dqm.force = self.force
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            self.work_dic[dqm.model_name] = self._evaluate(dqm)
        return self.work_dic

    def _evaluate(self,dqm):
        raise NotImplementedError

    def pwd(self):
        return os.path.join(self.tmp_dir,self.eval_name)

    def __str__(self):
        return 'Eval:' + self.eval_name

class DistOnlineEval(DistanceQueryEval):
    def __init__(self,test_sz=10000,gen_batch_sz=200,report_interval=3,**kwargs):
        super(DistOnlineEval, self).__init__(**kwargs)
        self.gen_batch_sz = gen_batch_sz
        self.test_sz = test_sz
        self.report_interval = report_interval

    def __str__(self):
        return 'DistOnlineEval:' + self.eval_name

    def _evaluate(self,dqm):
        # dqm = m_dqm.DistanceQueryModel()

        gen_loader = self.generator.loader(batch_sz=self.gen_batch_sz, meta_batch_sz=10)
        cur_sample_len = 0
        total_mae = 0.
        total_mre = 0.
        total_time = 0.
        total_mem = []
        # path_mae

        for idx,queries in enumerate(gen_loader):
            queries = tc.Tensor(queries)
            cur_sample_len += queries.shape[0]
            if cur_sample_len >= self.test_sz:
                delay_break = True # break later in case that tail samples are processed.
                queries = queries[:queries.shape[0] - (cur_sample_len - self.test_sz)]
            srcs = queries[:, 0]
            dsts = queries[:, 1]
            dists = queries[:, 2]
            dists[dists < 0] = 20
            srcs = srcs.type_as(tc.LongTensor())
            dsts = dsts.type_as(tc.LongTensor())

            cur_st_time = time.time()
            pred_dists,ed_time = dqm.query(srcs=srcs,dsts=dsts)
            total_time += ed_time - cur_st_time
            if type(pred_dists) is not list:
                pred_dists = pred_dists.tolist()
            pred_dists = tc.Tensor([round(ele) for ele in pred_dists])

            sub = tc.abs(pred_dists.view(-1) - dists)
            sub = sub[dists > 0]
            dists = dists[dists > 0]
            # if tc.sum(sub).item() > 0:
            #     pred_dists = dqm.query(srcs=srcs, dsts=dsts)
            total_mae += tc.sum(sub).item()
            sub = sub / dists
            total_mre += tc.sum(sub).item()

            total_mem.append(dqm.get_mem_usage(is_train=False))
            if idx % self.report_interval == 0:
                print('dqm {} | iter {} | mae={:.4f} | mre={:.4f} | query_time={:.5f} | mem={:5f}MB'.format(dqm,idx,total_mae/cur_sample_len,total_mre/cur_sample_len,total_time/cur_sample_len,sum(total_mem) / len(total_mem)))
        if cur_sample_len != self.test_sz:
            print('Warning[{}]: test_sz {} != cur_sample_len {}'.format(self,self.test_sz,cur_sample_len))
        return {
            'mae':total_mae / cur_sample_len,
            'mre':total_mre / cur_sample_len,
            'query_time':total_time / cur_sample_len,
            'query_mem':sum(total_mem) / len(total_mem),
        }

class DistOfflineEval(DistanceQueryEval):
    def __init__(self,**kwargs):
        super(DistOfflineEval, self).__init__(**kwargs)

    def __str__(self):
        return 'DistOfflineEval:' + self.eval_name

    def _evaluate(self,dqm):
        # dqm = m_dqm.DistanceQueryModel()

        total_time = time.time()
        ed_time = dqm.generate()
        total_time = ed_time - total_time

        total_mem = dqm.get_mem_usage(is_train=True)
        total_storage = dqm.get_disk_usage()
        print('dqm {} | gen_time={:.5f} | mem={:.5f} | storage={:.5f}'.format(dqm,total_time,total_mem,total_storage))
        return {
            'gen_time': total_time,
            'train_mem': total_mem,
            'storage': total_storage,
        }

class BNTestManager:
    def __init__(self,cache_name='../log/BNTest.cache'):
        self.data_dict = {}
        self.cache_name = cache_name
        # nid - pid - {g_sz,pll_ind,pll_qry,bc_ind,bc_qry}
        if os.path.exists(cache_name):
            self.pull_log()
            print(f'load cache file, and data_dict len = {len(self.data_dict)}')
        else:
            print('cache file not found, refresh.')
            self.push_log()

    def add_log(self,nid,pid,g_sz=None,pll_ind=None,pll_qry=None,bc_ind=None,bc_qry=None):
        if nid not in self.data_dict:
            self.data_dict[nid] = {}

        if pid not in self.data_dict[nid]:
            self.data_dict[nid][pid] = {}

        cur_dict = self.data_dict[nid][pid]
        if g_sz is not None:
            cur_dict['g_sz'] = g_sz

        if pll_ind is not None:
            cur_dict['pll_ind'] = pll_ind

        if pll_qry is not None:
            cur_dict['pll_qry'] = pll_qry

        if bc_ind is not None:
            cur_dict['bc_ind'] = bc_ind

        if bc_qry is not None:
            cur_dict['bc_qry'] = bc_qry

    def push_log(self):
        with open(self.cache_name,'wb') as f:
            pk.dump(self.data_dict,f)

    def pull_log(self):
        with open(self.cache_name,'rb') as f:
            self.data_dict = pk.load(f)

    def auto_add_g_sz(self,print_detail=False):
        self.pull_log()
        add_cnt = 0
        for home,dirs,files in os.walk('../tmp/'):
            for file in files:
                if file.endswith('.edgelist'):
                    g_sz = os.stat(os.path.join(home,file)).st_size/64/1024/1024
                    lst = file[:-len('.edgelist')].split('_')
                    self.add_log(nid=str(lst[4]),pid=str(lst[2]),g_sz=str(g_sz))
                    add_cnt += 1
                    if print_detail:
                        print(f'[nid={lst[4]} pid={lst[2]}] add g_sz={g_sz}')
        print(f'[auto-add g_sz]find total record:{add_cnt}')
        self.push_log()

    def auto_add_pll_ind(self,print_detail=False):
        self.pull_log()
        add_cnt = 0
        for home,dirs,files in os.walk('../tmp/'):
            for file in files:
                if file.endswith('.edgelist.ind'):
                    pll_ind = os.stat(os.path.join(home,file)).st_size/1024/1024
                    lst = file[:-len('.edgelist.ind')].split('_')
                    self.add_log(nid=str(lst[4]),pid=str(lst[2]),pll_ind=str(pll_ind))
                    add_cnt += 1
                    if print_detail:
                        print(f'[nid={lst[4]} pid={lst[2]}] add pll_ind={pll_ind}')
        print(f'[auto-add pll_ind]find total record:{add_cnt}')
        self.push_log()

    def auto_add_pll_qry(self,print_detail=False):
        self.pull_log()
        add_cnt = 0
        for home,dirs,files in os.walk('../tmp/'):
            for file in files:
                if file.endswith('.edgelist.query.txt'):
                    with open(os.path.join(home,file), 'r') as f:
                        pll_qry = float(f.readline())*1000
                    lst = file[:-len('.edgelist.query.txt')].split('_')
                    self.add_log(nid=str(lst[4]),pid=str(lst[2]),pll_qry=str(pll_qry))
                    add_cnt += 1
                    if print_detail:
                        print(f'[nid={lst[4]} pid={lst[2]}] add pll_qry={pll_qry}')
        print(f'[auto-add pll_qry]find total record:{add_cnt}')
        self.push_log()

    def auto_add_bc_ind(self,print_detail=False):
        self.pull_log()
        add_cnt = 0
        for home,dirs,files in os.walk('../log/'):
            for file in files:
                if file.startswith('bn_emb_'): # bn_emb_nsz_{}_psz_{}.log
                    bc_ind = os.stat(os.path.join(home, file)).st_size / 1024 / 1024
                    lst = file[:-len('.log')].split('_')
                    self.add_log(nid=str(lst[3]),pid=str(lst[5]),bc_ind=str(bc_ind))
                    add_cnt += 1
                    if print_detail:
                        print(f'[nid={lst[3]} pid={lst[5]}] add bc_ind={bc_ind}')
        print(f'[auto-add bc_ind]find total record:{add_cnt}')
        self.push_log()

    def auto_add_bc_qry(self,print_detail=False):
        self.pull_log()
        if not os.path.exists('../log/bn_bcdr_query.log'):
            print('[auto-add bc_qry] not found file.')
            return

        with open('../log/bn_bcdr_query.log', 'rb') as f:
            bc_qry = pk.load(f)
            nids = pk.load(f)
            pids = pk.load(f)
        bc_qry = np.array(bc_qry) * 1000
        add_cnt = 0
        for idx,nid in enumerate(nids):
            for idy,pid in enumerate(pids):
                self.add_log(nid=str(nid),pid=str(pid),bc_qry=str(bc_qry[idx][idy]))
                if print_detail:
                    print(f'[nid={nid} pid={pid}] add bc_qry={bc_qry[nid][pid]}')
                add_cnt += 1
        print(f'[auto-add bc_qry]find total record:{add_cnt}')
        self.push_log()

    def print_stat(self,print_detail=False):
        self.pull_log()
        print(self.data_dict)

    def query_meta(self,nid=None,pid=None):
        return self.data_dict[str(nid)][str(pid)]

def _th_gen_bn_graph(node_seq,p,node_sz,**kwargs):
    ret_lst = []
    for nid in node_seq:
        adj_arr = np.random.random(size=(node_sz,)) < p
        for idx,ele in enumerate(adj_arr):
            if ele == True and idx != nid:
                ret_lst.append(f'{nid}\t{idx}\n')
    return ret_lst

def gen_bn_graph(graph_name,node_sz,p=0.5,num_workers=8):
    batch_gen_node = max(min(2048,node_sz),int(node_sz//64))
    batch_wrt_node = batch_gen_node * num_workers
    all_node_list = list(range(node_sz))
    cur_wrt_pnt = 0
    mpm = utils.MPManager(batch_sz=batch_gen_node, num_workers=num_workers, use_shuffle=False)
    if os.path.exists(f'../tmp/{graph_name}.edgelist'):
        print(f'found processed graph file {graph_name}')
        return
    while cur_wrt_pnt < len(all_node_list):
        ret_dict = mpm.multi_proc(_th_gen_bn_graph, [all_node_list[cur_wrt_pnt:min(len(all_node_list),cur_wrt_pnt+batch_wrt_node)]], p=p, node_sz=node_sz, auto_concat=False)
        with open(f'../tmp/{graph_name}.edgelist', 'a') as f:
            for k, v in ret_dict.items():
                f.writelines(v)
            f.flush()
        cur_wrt_pnt += batch_wrt_node

def gen_fake_emb_query(node_szs,ps,query_sz=1000000):
    lsts = []
    for node_sz in tqdm.tqdm(node_szs):
        cur_lst = []
        for p in ps:
            srcs = tc.from_numpy(np.random.randint(low=0,high=node_sz,size=(query_sz,))).long()
            dsts = tc.from_numpy(np.random.randint(low=0, high=node_sz, size=(query_sz,))).long()
            embs = tc.from_numpy(np.random.random(size=(node_sz,16))).float()
            model = m_dqm.DistDecoder_BCDR()
            test_val = 0
            for _ in range(5):
                st_time = time.time()
                emb_srcs = embs[srcs]
                emb_dsts = embs[dsts]
                dists = model(emb_srcs, emb_dsts)
                ed_time = time.time()
                test_val += (ed_time - st_time) * (1000000 / query_sz)
            cur_lst.append(test_val / 5)
        lsts.append(cur_lst)
    with open('../log/bn_bcdr_query.log','wb') as f:
        pk.dump(lsts,f)
        pk.dump(node_szs,f)
        pk.dump(ps,f)

def gen_fake_bcdr(node_szs,ps=None):
    for node_sz in node_szs:
        embs = tc.from_numpy(np.random.random(size=(node_sz,16)))
        if ps is None:
            with open(f'../log/bngraph/bn_emb_nsz_{node_sz}.log','wb') as f:
                pk.dump(embs,f)
        else:
            for p in ps:
                with open(f'../log/bn_emb_nsz_{node_sz}_psz_{p}.log', 'wb') as f:
                    pk.dump(embs, f)

glb_node_max=12
def eval_bn_storage():
    p_dir = "../log/bngraph"
    ps = [(ele + 1) / 20 for ele in range(20)]
    node_szs = [int(math.pow(2, ele + 1)) for ele in range(glb_node_max)]
    s_graph = []
    s_ind = []
    s_emb = []
    for node_sz in node_szs:
        cur_s_graph = []
        cur_s_ind = []
        cur_emb = []
        for p in ps:
            cur_s_graph.append(os.stat(os.path.join(p_dir,f'bn_p_{p}_nsz_{node_sz}.edgelist')).st_size/64/1024/1024)
            cur_s_ind.append(os.stat(os.path.join(p_dir, f'bn_p_{p}_nsz_{node_sz}.edgelist.ind')).st_size / 1024 / 1024)
            cur_emb.append(os.stat(os.path.join(p_dir, f'bn_emb_nsz_{node_sz}.log')).st_size / 1024 / 1024)
        s_graph.append(cur_s_graph)
        s_ind.append(cur_s_ind)
        s_emb.append(cur_emb)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # cs = ['coral', 'dodgerblue', 'darkcyan', 'darkviolet', 'olive', 'cyan', 'black', 'firebrick',
    #       'gainsboro']

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection='3d')


    x = np.array(ps)
    y = np.array([ele+1 for ele in range(glb_node_max)])
    x,y = np.meshgrid(x,y)
    z1 = np.array(s_graph)
    z2 = np.array(s_ind)
    z3 = np.array(s_emb)

    # c: 颜色
    # ‘b’ blue 蓝色、g’ green 绿色、‘r’ red 红色、‘c’ cyan 兰青色
    # ‘m’ magenta 紫色、‘y’ yellow 黄色、‘k’ black 黑色、‘w’white 白色
    ax.plot_wireframe(x, y, z1, color='k',label='graph size')
    ax.plot_wireframe(x, y, z2, color='coral',label='exact representation')
    ax.plot_wireframe(x, y, z3, color='dodgerblue',label='approximate representation')

    ax.set_xlabel('p',fontsize=20)
    ax.set_ylabel('log |V|',fontsize=20)
    ax.set_zlabel('storage cost (MB)',fontsize=20)

    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_storage.pdf')
    plt.show()

def eval_bn_query():
    p_dir = "../log/bngraph"
    ps = [(ele + 1) / 20 for ele in range(20)]
    node_szs = [int(math.pow(2, ele + 1)) for ele in range(glb_node_max)]
    q_ind = []
    q_emb = []
    with open('../log/bngraph/bn_bcdr_query.log','rb') as f:
        q_emb = pk.load(f)
        q_emb = np.array(q_emb)[:glb_node_max,:] * 1000
    for node_sz in node_szs:
        cur_q_ind = []
        for p in ps:
            with open(f'../log/bngraph/bn_p_{p}_nsz_{node_sz}.edgelist.query.txt','r') as f:
                cur_q_ind.append(float(f.readline())*1000)
        q_ind.append(cur_q_ind)
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection='3d')
    x = np.array(ps)
    y = np.array([ele+1 for ele in range(glb_node_max)])
    x,y = np.meshgrid(x,y)
    z1 = np.array(q_ind)
    z2 = np.array(q_emb)

    # c: 颜色
    # ‘b’ blue 蓝色、g’ green 绿色、‘r’ red 红色、‘c’ cyan 兰青色
    # ‘m’ magenta 紫色、‘y’ yellow 黄色、‘k’ black 黑色、‘w’white 白色
    ax.plot_wireframe(x, y, z1, color='coral',label='exact representation')
    ax.plot_wireframe(x, y, z2, color='dodgerblue',label='approximate representation')

    ax.set_xlabel('p',fontsize=20)
    ax.set_ylabel('log |V|',fontsize=20)
    ax.set_zlabel('response time (ns)',fontsize=20)

    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_query.pdf')
    plt.show()

glb_node_max=14
def eval_bn_storage1():
    bntm = BNTestManager()
    ps = [(ele + 1) / 20 for ele in range(20)]
    node_szs = [int(math.pow(2, ele + 1)) for ele in range(glb_node_max)]
    s_graph = []
    s_ind = []
    s_emb = []
    for node_sz in node_szs:
        cur_s_graph = []
        cur_s_ind = []
        cur_emb = []
        for p in ps:
            cur_s_graph.append(float(bntm.query_meta(nid=node_sz,pid=p)['g_sz']))
            cur_s_ind.append(float(bntm.query_meta(nid=node_sz,pid=p)['pll_ind']))
            cur_emb.append(float(bntm.query_meta(nid=node_sz,pid=p)['bc_ind']))
        s_graph.append(cur_s_graph)
        s_ind.append(cur_s_ind)
        s_emb.append(cur_emb)

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection='3d')


    x = np.array(ps)
    y = np.array([ele+1 for ele in range(glb_node_max)])
    x,y = np.meshgrid(x,y)
    z1 = np.array(s_graph)
    z2 = np.array(s_ind)
    z3 = np.array(s_emb)

    # c: 颜色
    # ‘b’ blue 蓝色、g’ green 绿色、‘r’ red 红色、‘c’ cyan 兰青色
    # ‘m’ magenta 紫色、‘y’ yellow 黄色、‘k’ black 黑色、‘w’white 白色
    ax.plot_wireframe(x, y, z1, color='k',label='graph size')
    ax.plot_wireframe(x, y, z2, color='coral',label='exact representation')
    ax.plot_wireframe(x, y, z3, color='dodgerblue',label='approximate representation')

    ax.set_xlabel('p',fontsize=20)
    ax.set_ylabel('log |V|',fontsize=20)
    ax.set_zlabel('storage cost (MB)',fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_storage.pdf')
    plt.show()

def eval_bn_query1():
    bntm = BNTestManager()
    ps = [(ele + 1) / 20 for ele in range(20)]
    node_szs = [int(math.pow(2, ele + 1)) for ele in range(glb_node_max)]
    q_ind = []
    q_emb = []
    for node_sz in node_szs:
        cur_q_ind = []
        cur_q_emb = []
        for p in ps:
            cur_q_ind.append(float(bntm.query_meta(nid=node_sz,pid=p)['pll_qry']))
            cur_q_emb.append(float(bntm.query_meta(nid=node_sz,pid=p)['bc_qry']))
        q_ind.append(cur_q_ind)
        q_emb.append(cur_q_emb)
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection='3d')
    x = np.array(ps)
    y = np.array([ele+1 for ele in range(glb_node_max)])
    x,y = np.meshgrid(x,y)
    z1 = np.array(q_ind)
    z2 = np.array(q_emb)

    # c: 颜色
    # ‘b’ blue 蓝色、g’ green 绿色、‘r’ red 红色、‘c’ cyan 兰青色
    # ‘m’ magenta 紫色、‘y’ yellow 黄色、‘k’ black 黑色、‘w’white 白色
    ax.plot_wireframe(x, y, z1, color='coral',label='exact representation')
    ax.plot_wireframe(x, y, z2, color='dodgerblue',label='approximate representation')

    ax.set_xlabel('p',fontsize=20)
    ax.set_ylabel('log |V|',fontsize=20)
    ax.set_zlabel('response time (ns)',fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_query.pdf')
    plt.show()

def eval_bn_storage_fixed_p(ps = [0.05,0.2],ns=[1,12]):
    bntm = BNTestManager()
    node_szs = [int(math.pow(2, ele)) for ele in range(*ns)]
    s_graph = []
    s_ind = []
    s_emb = []
    for p in ps:
        cur_s_graph = []
        cur_s_ind = []
        cur_emb = []
        for node_sz in node_szs:
            cur_s_graph.append(float(bntm.query_meta(nid=node_sz,pid=p)['g_sz']))
            cur_s_ind.append(float(bntm.query_meta(nid=node_sz,pid=p)['pll_ind']))
            cur_emb.append(float(bntm.query_meta(nid=node_sz,pid=p)['bc_ind']))
        s_graph.append(cur_s_graph)
        s_ind.append(cur_s_ind)
        s_emb.append(cur_emb)

    plt.figure(figsize=(8,8))
    lss = ['-',':','-.']
    for idx,(p,ls) in enumerate(zip(ps,lss)):
        plt.plot(range(*ns),s_graph[idx],color='k',linestyle=ls,marker='o',label=f'p={p}, graph size')
        plt.plot(range(*ns),s_ind[idx],color='coral',linestyle=ls,label=f'p={p}, exact representation')
        plt.plot(range(*ns),s_emb[idx],color='dodgerblue',linestyle=ls,label=f'p={p}, approximate representation')

    plt.xlabel('log |V|',fontsize=20)
    plt.ylabel('storage cost (MB)',fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_storage_fixed_p.pdf')
    plt.show()

def eval_bn_query_fixed_p(ps = [0.05,0.2],ns=[1,12]):
    bntm = BNTestManager()
    node_szs = [int(math.pow(2, ele)) for ele in range(*ns)]
    q_ind = []
    q_emb = []
    for p in ps:
        cur_q_ind = []
        cur_q_emb = []
        for node_sz in node_szs:
            cur_q_ind.append(float(bntm.query_meta(nid=node_sz,pid=p)['pll_qry']))
            cur_q_emb.append(float(bntm.query_meta(nid=node_sz,pid=p)['bc_qry']))
        q_ind.append(cur_q_ind)
        q_emb.append(cur_q_emb)

    plt.figure(figsize=(8,8))
    lss = ['-',':','-.']
    for idx,(p,ls) in enumerate(zip(ps,lss)):
        plt.plot(range(*ns),q_ind[idx],color='coral',linestyle=ls,label=f'p={p}, exact representation')
        plt.plot(range(*ns),q_emb[idx],color='dodgerblue',linestyle=ls,label=f'p={p}, approximate representation')

    plt.xlabel('log |V|',fontsize=20)
    plt.ylabel('response time (ns)',fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.gcf().subplots_adjust(right=0.95)
    plt.savefig('../fig/bn_query_fixed_p.pdf')
    plt.show()


if __name__ == '__main__':
    # a = np.random.random(size=(10,)) < 0.5
    # print(a)
    # print(a[1]==False)
    # eval_bn_storage()
    # eval_bn_query()

    # eval_bn_storage_fixed_p(ps=[0.05])
    eval_bn_query_fixed_p(ps = [0.05],ns=[1,15])
    # eval_bn_storage1()
    # eval_bn_query1()

    # bntm = BNTestManager()
    #
    # bntm.auto_add_g_sz()
    # bntm.auto_add_bc_ind()
    # bntm.auto_add_bc_qry()
    # bntm.auto_add_pll_qry()
    # bntm.auto_add_pll_ind()
    #
    # bntm.print_stat()

    pass


