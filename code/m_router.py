import dgl
import torch as tc
import numpy
import os
import m_evaluator
import utils
import m_generator
import random
from line_profiler import LineProfiler
import numpy as np
class Router:
    def __init__(self,out_dir='../outputs',model_name='fb-dwext-emb=64',is_embed_model=False):
        self.out_dir = out_dir
        self.model_name = model_name

        model,g = m_evaluator.BasicEvaluator.load_model(out_dir=self.out_dir,out_file=self.model_name)
        if g is None or model is None:
            print('wrong path for model: {}'.format(self.model_name))
            raise EnvironmentError
        self.model = model
        self.g = g
    def query_path(self,src,dst,prox_m=1,top_k=3):
        if prox_m != 1:
            raise NotImplementedError
        if src == dst:
            return [src],0
        with self.g.local_scope():
            self.g.ndata['path_sgn_src'] = -(tc.ones(self.g.num_nodes())).type_as(tc.IntTensor())
            self.g.ndata['path_sgn_dst'] = -(tc.ones(self.g.num_nodes())).type_as(tc.IntTensor())
            src_map = {src:0}
            dst_map = {dst:0}
            self.g.ndata['path_sgn_src'][src] = 0
            self.g.ndata['path_sgn_dst'][dst] = 0
            trace_src_map = {src:-1}
            trace_dst_map = {dst:-1}
            while len(set(src_map.keys()).intersection(set(dst_map.keys()))) == 0 or len(src_map.keys()) == 0 or len(dst_map.keys()) == 0:
                top_k_que_val = [9999]
                top_k_que_idx = [-0.5]
                new_src_map = {}
                new_dst_map = {}
                for e_src in src_map:
                    for e_dst in dst_map:
                        cur_path_len_pred = -1
                        # print('\t',e_src,e_dst)
                        if e_src == e_dst:
                            cur_path_len_pred = src_map[e_src] + dst_map[e_dst]
                            return Router._get_path_from_trace(e_src=e_src,e_dst=e_dst,trace_src_map=trace_src_map,trace_dst_map=trace_dst_map),cur_path_len_pred
                        elif self.g.has_edge_between(e_src,e_dst):
                            cur_path_len_pred = src_map[e_src] + dst_map[e_dst] + 1
                            return Router._get_path_from_trace(e_src=e_src, e_dst=e_dst, trace_src_map=trace_src_map,
                                                               trace_dst_map=trace_dst_map), cur_path_len_pred
                        else:
                            mutual_pred = self.model(self.g.ndata['emb'][e_src].view(1,-1),self.g.ndata['emb'][e_dst].view(1,-1)).item()
                            cur_path_len_pred = src_map[e_src] + dst_map[e_dst] + mutual_pred
                        if cur_path_len_pred < top_k_que_val[-1]:
                            top_k_que_val.append(cur_path_len_pred)
                            top_k_que_idx.append((e_src,e_dst))
                            if len(top_k_que_val) > top_k + 1: # since we have 1 dummy node.
                                top_k_que_val.pop(0)
                                top_k_que_idx.pop(0)
                if top_k_que_val[0] == 9999:
                    top_k_que_val.pop(0)
                    top_k_que_idx.pop(0)

                top_k_que_val_r = top_k_que_val[::-1]
                top_k_que_idx_r = top_k_que_idx[::-1]
                for e_val,e_idx in zip(top_k_que_val_r,top_k_que_idx_r):
                    src,dst = e_idx
                    for src_succ in self.g.successors(src):
                        src_succ = int(src_succ.item())
                        if self.g.ndata['path_sgn_src'][src_succ] < 0: # which means a new node.
                            self.g.ndata['path_sgn_src'][src_succ] = 0
                            new_src_map[src_succ] = src_map[src] + 1
                            trace_src_map[src_succ] = src
                    for dst_succ in self.g.successors(dst):
                        dst_succ = int(dst_succ)
                        if self.g.ndata['path_sgn_dst'][dst_succ] < 0:
                            self.g.ndata['path_sgn_dst'][dst_succ] = 0
                            new_dst_map[dst_succ] = dst_map[dst] + 1
                            trace_dst_map[dst_succ] = dst
                best_idx = None
                best_len = 9999

                # first we consider src->new_dst dst->new_src
                nsrc_in_dst = set(new_src_map.keys()).intersection(set(dst_map.keys()))
                for nid in nsrc_in_dst:
                    cur_len = src_map[trace_src_map[nid]] + dst_map[nid] + 1
                    if cur_len < best_len:
                        best_idx = trace_src_map[nid],nid
                        best_len = cur_len

                src_in_ndst = set(src_map.keys()).intersection(set(new_dst_map.keys()))
                for nid in src_in_ndst:
                    cur_len = src_map[nid] + dst_map[trace_dst_map[nid]] + 1
                    if cur_len < best_len:
                        best_idx = nid, trace_dst_map[nid]
                        best_len = cur_len

                nsrc_in_ndst = set(new_src_map.keys()).intersection(set(new_dst_map.keys()))
                for nid in nsrc_in_ndst:
                    cur_len = src_map[trace_src_map[nid]] + dst_map[trace_dst_map[nid]] + 2
                    if cur_len < best_len:
                        best_idx = nid, trace_dst_map[nid]
                        best_len = cur_len
                if best_idx != None:
                    return Router._get_path_from_trace(e_src=best_idx[0],e_dst=best_idx[1],trace_src_map=trace_src_map,
                                                                   trace_dst_map=trace_dst_map),best_len
                # then for new_dst ∩ new_src

                # for e_val,e_idx in zip(top_k_que_val,top_k_que_idx):
                #     src,dst = e_idx
                #     for src_succ in self.g.successors(src):
                #         src_succ = int(src_succ.item())
                #         if self.g.ndata['path_sgn_src'][src_succ] < 0: # which means a new node.
                #             self.g.ndata['path_sgn_src'][src_succ] = 0
                #             new_src_map[src_succ] = src_map[src] + 1
                #             trace_src_map[src_succ] = src
                #         if self.g.ndata['path_sgn_dst'][src_succ] >= 0: # only in case that this node in dst_map.
                #             if src_succ in dst_map:
                #                 cur_path_len_pred = src_map[src] + dst_map[src_succ] + 1
                #                 return Router._get_path_from_trace(e_src=src,e_dst=src_succ,trace_src_map=trace_src_map,
                #                                                    trace_dst_map=trace_dst_map),cur_path_len_pred
                #             else:
                #                 if src_succ not in new_dst_map:
                #                     print('--path debug:',src,dst)
                #                 assert src_succ in new_dst_map
                #                 cur_path_len_pred = src_map[src] + dst_map[dst] + 2
                #                 return Router._get_path_from_trace(e_src=src, e_dst=src_succ, trace_src_map=trace_src_map,
                #                                                    trace_dst_map=trace_dst_map), cur_path_len_pred
                #     for dst_succ in self.g.successors(dst):
                #         dst_succ = int(dst_succ)
                #         if self.g.ndata['path_sgn_dst'][dst_succ] < 0:
                #             self.g.ndata['path_sgn_dst'][dst_succ] = 0
                #             new_dst_map[dst_succ] = dst_map[dst] + 1
                #             trace_dst_map[dst_succ] = dst
                #         if self.g.ndata['path_sgn_src'][dst_succ] >= 0: # in case this node in src_map or in new_src_map.
                #             if dst_succ in src_map:
                #                 cur_path_len_pred = src_map[dst_succ] + dst_map[dst] + 1
                #                 return Router._get_path_from_trace(e_src=dst_succ, e_dst=dst, trace_src_map=trace_src_map,
                #                                                    trace_dst_map=trace_dst_map), cur_path_len_pred
                #             else:
                #                 assert dst_succ in new_src_map
                #                 cur_path_len_pred = src_map[src] + dst_map[dst] + 2
                #                 return Router._get_path_from_trace(e_src=dst_succ, e_dst=dst, trace_src_map=trace_src_map,
                #                                                    trace_dst_map=trace_dst_map), cur_path_len_pred
                src_map.clear()
                dst_map.clear()
                src_map.update(new_src_map)
                dst_map.update(new_dst_map)
            if len(src_map.keys()) == 0 or len(dst_map.keys()) == 0:
                print('pred path fail to converge, please increase k.')
                return None
            for e_src in src_map:
                for e_dst in dst_map:
                    if e_src == e_dst:
                        cur_path_len_pred = src_map[e_src] + dst_map[e_dst]
                        return Router._get_path_from_trace(e_src=e_src,e_dst=e_dst,trace_src_map=trace_src_map,trace_dst_map=trace_dst_map),cur_path_len_pred

        print('''here won't be exec.''')
        assert False
    def gen_test_sample_basic(self):
        pass

    def _get_path_from_trace(e_src,e_dst,trace_src_map,trace_dst_map):
        assert (e_src != -1) and (e_dst != -1)
        trace_src = []
        trace_dst = []

        cur_src = e_src
        while cur_src != -1:
            trace_src.append(cur_src)
            cur_src = trace_src_map[cur_src]

        cur_dst = e_dst
        while cur_dst != -1:
            trace_dst.append(cur_dst)
            cur_dst = trace_dst_map[cur_dst]
        if e_src == e_dst:
            trace_src = trace_src[1:][::-1]
        else:
            trace_src = trace_src[::-1]
        return trace_src + trace_dst

class ParaRouter(Router):
    def __init__(self,**kwargs):
        super(ParaRouter, self).__init__(**kwargs)
        self.cur_query_times = 0
        self.scheme = m_generator.BFS(g=self.g)
        self.is_failed = False
    def query_single_path(self,src,dst,cap=1):
        self.is_failed = False
        self.cur_query_times = 0
        if src == dst:
            return [src]

        sgn_s,sgn_d = tc.zeros(self.g.num_nodes()).type_as(tc.BoolTensor()),tc.zeros(self.g.num_nodes()).type_as(tc.BoolTensor())
        off_s,off_d = {src:0},{dst:0}
        prec_s,prec_d = {src:-1},{dst:-1}
        sgn_s[src],sgn_d[dst] = True,True

        while len(off_s.keys()) != 0 and len(off_d.keys()) != 0:
            q = utils.MinMaxHeap(reserve=max(200,len(off_s.keys()) * len(off_d.keys())))
            noff_s,noff_d = {},{}

            e_srcs = []
            e_dsts = []
            for e_src in off_s:
                for e_dst in off_d:
                    # print('\t',e_src,e_dst)
                    # if e_src == e_dst or self.g.has_edge_between(e_src,e_dst):
                    if e_src == e_dst:
                        return Router._get_path_from_trace(e_src=e_src,e_dst=e_dst,trace_src_map=prec_s,trace_dst_map=prec_d)
                    else:
                        e_srcs.append(e_src)
                        e_dsts.append(e_dst)
            # print(e_srcs)
            # print(e_dsts)
            e_dists = self.model(self.g.ndata['emb'][e_srcs].view(len(e_srcs),-1),self.g.ndata['emb'][e_dsts].view(len(e_dsts),-1)).view(-1).tolist()
            self.cur_query_times += len(e_dists)
            for e_src,e_dst,e_dist in zip(e_srcs,e_dsts,e_dists):
                len_e = off_s[e_src] + off_d[e_dst] + e_dist
                # if q.size >= cap and len_e > q.peekmax()[0]:
                #     continue
                # q.insert((len_e,(e_src,e_dst)))
                if q.size == 0 or len_e < q.peekmax()[0]:
                    q.insert((len_e, (e_src, e_dst)))
                    if q.size > cap:
                        q.popmax()

            assert q.size > 0,print(off_s,'::',off_d)

            while q.size > 0:
                _, (e_src,e_dst) = q.popmin()
                for src_succ in self.g.successors(e_src):
                    src_succ = int(src_succ.item())
                    if not sgn_s[src_succ]: # which means a new node.
                        sgn_s[src_succ] = True
                        noff_s[src_succ] = off_s[e_src] + 1
                        prec_s[src_succ] = e_src
                for dst_succ in self.g.successors(e_dst):
                    dst_succ = int(dst_succ)
                    if not sgn_d[dst_succ]:
                        sgn_d[dst_succ] = True
                        noff_d[dst_succ] = off_d[e_dst] + 1
                        prec_d[dst_succ] = e_dst

            best_idx = None
            best_len = 9999

            # first we consider src->new_dst dst->new_src
            nsrc_in_dst = set(noff_s.keys()).intersection(set(off_d.keys()))
            for nid in nsrc_in_dst:
                len_e = off_s[prec_s[nid]] + off_d[nid] + 1
                if len_e < best_len:
                    best_idx = prec_s[nid],nid
                    best_len = len_e

            src_in_ndst = set(off_s.keys()).intersection(set(noff_d.keys()))
            for nid in src_in_ndst:
                len_e = off_s[nid] + off_d[prec_d[nid]] + 1
                if len_e < best_len:
                    best_idx = nid, prec_d[nid]
                    best_len = len_e

            nsrc_in_ndst = set(noff_s.keys()).intersection(set(noff_d.keys()))
            for nid in nsrc_in_ndst:
                len_e = off_s[prec_s[nid]] + off_d[prec_d[nid]] + 2
                if len_e < best_len:
                    best_idx = nid, prec_d[nid]
                    best_len = len_e
            if best_idx != None:
                return Router._get_path_from_trace(e_src=best_idx[0],e_dst=best_idx[1],trace_src_map=prec_s,trace_dst_map=prec_d)

            off_s.clear()
            off_d.clear()
            off_s.update(noff_s)
            off_d.update(noff_d)

        print('fail to figure out path from {} to {}'.format(src,dst))
        self.is_failed = True
        prec_s, _ = self.scheme.path_one_to_other(src=src, dst_set=[dst], force=False)
        path = []
        nid = dst
        # print('src',src,'dst',dst)
        while nid != -1:
            path.append(nid)
            nid = prec_s[nid]
        return path

    def peek_query_times(self):
        return self.cur_query_times

    def peek_is_failed(self):
        return self.is_failed

class BFSRouter(ParaRouter):
    def __init__(self,**kwargs):
        super(BFSRouter, self).__init__(**kwargs)

    def query_single_path(self,src,dst,cap=-1):
        assert cap == -1, print('BFS has no support for any cap adjustment(like cap={}).'.format(cap))
        self.is_failed = False
        self.cur_query_times = 0

        if dst == src:
            return [src]
        prec_s,_ = self.scheme.path_one_to_other(src=src,dst_set=[dst],force=False)
        path = []
        nid = dst
        # print('src',src,'dst',dst)
        while nid != -1:
            path.append(nid)
            nid = prec_s[nid]
        return path

class RigelRouter(ParaRouter):
    def __init__(self,**kwargs):
        super(RigelRouter, self).__init__(**kwargs)

    def trace(self,e_src,e_dst,e_prec_s):
        if e_dst == e_src:
            return [e_src]
        nid = e_dst
        path = []
        while nid != -1:
            path.append(nid)
            nid = e_prec_s[nid]
        return path

    def query_single_path(self,src,dst,cap=200,factor=2.0):
        self.is_failed=False
        self.cur_query_times = 0
        if src == dst:
            return [src]

        sgn_s = tc.zeros(self.g.num_nodes()).type_as(tc.BoolTensor())
        prec_s = {src:-1}
        n2d = {}
        search_lst = [src]
        search_heap = utils.MinMaxHeap(reserve=cap)
        sgn_s[src] = True

        dist_src = self.model(self.g.ndata['emb'][src].view(1, -1),
                           self.g.ndata['emb'][dst].view(1, -1)).view(-1).item()
        self.cur_query_times +=1
        search_heap.insert((src,dist_src))
        n2d[src] = dist_src
        while search_heap.size > 0:
            nid, cur_dist = search_heap.popmin()
            nnids = []
            for nnid in self.g.successors(nid):
                if sgn_s[nnid]:
                    continue
                nnid = int(nnid)
                nnids.append(nnid)
                if nnid == dst:
                    prec_s[nnid] = nid
                    return self.trace(src,dst,prec_s)
                if self.g.has_edge_between(nnid,dst):
                    prec_s[dst] = nnid
                    prec_s[nnid] = nid
                    return self.trace(src,dst,prec_s)
            if len(nnids) == 0:
                continue
            dsts = [dst] * len(nnids)

            dists = self.model(self.g.ndata['emb'][nnids].view(len(nnids), -1),
                                 self.g.ndata['emb'][dsts].view(len(dsts), -1)).view(-1).tolist()
            # cands = []
            self.cur_query_times += len(dists)

            for i in range(len(dists)):
                if i == len(dists) - 1:
                    break
                # if n2d[nid] - 1 - factor < dists[i] < n2d[nid] - 1 + factor:
                if dists[i] < n2d[nid] - 1 + factor:
                    # cands.append(nnids[i])
                    sgn_s[nnids[i]] = True
                    n2d[nnids[i]] = dists[i]
                    prec_s[nnids[i]] = nid
                    search_heap.insert((nnids[i],dists[i]))
            # random.shuffle(cands)
            while search_heap.size > cap:
                search_heap.popmax()
            # search_lst = cands + search_lst
            # search_lst = search_lst[:cap]

        print('fail to figure out path from {} to {}'.format(src,dst))

        self.is_failed = True
        prec_s, _ = self.scheme.path_one_to_other(src=src, dst_set=[dst], force=False)
        path = []
        nid = dst
        # print('src',src,'dst',dst)
        while nid != -1:
            path.append(nid)
            nid = prec_s[nid]
        return path

def Xtest_query_bdd():
    test_sz = 1000
    srcs,dsts = [],[]
    for _ in range(test_sz):
        srcs.append(random.randint(1,2200))
        dsts.append(random.randint(1,2200))
    rt = ParaRouter(model_name='cr-dw-emb=16-par=0~450')
    cap = 3
    for idx,(src,dst) in enumerate(zip(srcs,dsts)):
        rt.cur_query_times = 0
        if src == dst:
            continue
        is_over = False
        sgn_s, sgn_d = tc.zeros(rt.g.num_nodes()).type_as(tc.BoolTensor()), tc.zeros(rt.g.num_nodes()).type_as(
            tc.BoolTensor())
        off_s, off_d = {src: 0}, {dst: 0}
        prec_s, prec_d = {src: -1}, {dst: -1}
        sgn_s[src], sgn_d[dst] = True, True
        while len(off_s.keys()) != 0 and len(off_d.keys()) != 0:
            q = utils.MinMaxHeap(reserve=max(200, len(off_s.keys()) * len(off_d.keys())))
            noff_s, noff_d = {}, {}

            e_srcs = []
            e_dsts = []
            for e_src in off_s:
                for e_dst in off_d:
                    # print('\t',e_src,e_dst)
                    # if e_src == e_dst or rt.g.has_edge_between(e_src, e_dst):
                    if e_src == e_dst:
                        Router._get_path_from_trace(e_src=e_src, e_dst=e_dst, trace_src_map=prec_s,
                                                           trace_dst_map=prec_d)
                        is_over = True
                        break
                    else:
                        e_srcs.append(e_src)
                        e_dsts.append(e_dst)
                if is_over:
                    break
            if is_over:
                break
            # print(e_srcs)
            # print(e_dsts)
            e_dists = rt.model(rt.g.ndata['emb'][e_srcs].view(len(e_srcs), -1),
                                 rt.g.ndata['emb'][e_dsts].view(len(e_dsts), -1)).view(-1).tolist()
            rt.cur_query_times += len(e_dists)
            for e_src, e_dst, e_dist in zip(e_srcs, e_dsts, e_dists):
                len_e = off_s[e_src] + off_d[e_dst] + e_dist
                # if q.size >= cap and len_e > q.peekmax()[0]:
                #     continue
                if q.size == 0 or len_e < q.peekmax()[0]:
                    q.insert((len_e, (e_src, e_dst)))
                    if q.size > cap:
                        q.popmax()


            assert q.size > 0, print(off_s, '::', off_d)

            while q.size > 0:
                _, (e_src, e_dst) = q.popmin()
                for src_succ in rt.g.successors(e_src):
                    src_succ = int(src_succ.item())
                    if not sgn_s[src_succ]:  # which means a new node.
                        sgn_s[src_succ] = True
                        noff_s[src_succ] = off_s[e_src] + 1
                        prec_s[src_succ] = e_src
                for dst_succ in rt.g.successors(e_dst):
                    dst_succ = int(dst_succ)
                    if not sgn_d[dst_succ]:
                        sgn_d[dst_succ] = True
                        noff_d[dst_succ] = off_d[e_dst] + 1
                        prec_d[dst_succ] = e_dst

            best_idx = None
            best_len = 9999

            # first we consider src->new_dst dst->new_src
            nsrc_in_dst = set(noff_s.keys()).intersection(set(off_d.keys()))
            for nid in nsrc_in_dst:
                len_e = off_s[prec_s[nid]] + off_d[nid] + 1
                if len_e < best_len:
                    best_idx = prec_s[nid], nid
                    best_len = len_e

            src_in_ndst = set(off_s.keys()).intersection(set(noff_d.keys()))
            for nid in src_in_ndst:
                len_e = off_s[nid] + off_d[prec_d[nid]] + 1
                if len_e < best_len:
                    best_idx = nid, prec_d[nid]
                    best_len = len_e

            nsrc_in_ndst = set(noff_s.keys()).intersection(set(noff_d.keys()))
            for nid in nsrc_in_ndst:
                len_e = off_s[prec_s[nid]] + off_d[prec_d[nid]] + 2
                if len_e < best_len:
                    best_idx = nid, prec_d[nid]
                    best_len = len_e
            if best_idx != None:
                Router._get_path_from_trace(e_src=best_idx[0], e_dst=best_idx[1], trace_src_map=prec_s,
                                                   trace_dst_map=prec_d)
                is_over = True
                break

            off_s.clear()
            off_d.clear()
            off_s.update(noff_s)
            off_d.update(noff_d)
        print('{}/{}'.format(idx,len(srcs)))
        if is_over:
            continue

def Xtest_total_bdd():
    test_sz = 1000
    srcs, dsts = [], []
    for _ in range(test_sz):
        srcs.append(random.randint(1, 2200))
        dsts.append(random.randint(1, 2200))
    rt = ParaRouter(model_name='cr-dw-emb=16-par=0~450')
    cap = 10
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        rt.query_single_path(src=src,dst=dst,cap=cap)
        print('{}/{}'.format(idx, len(srcs)))


def Xtest_total_bfs():
    test_sz = 1000
    srcs, dsts = [], []
    for _ in range(test_sz):
        srcs.append(random.randint(1, 2200))
        dsts.append(random.randint(1, 2200))
    rt = BFSRouter(model_name='cr-dw-emb=16-par=0~450')
    for idx, (src, dst) in enumerate(zip(srcs, dsts)):
        rt.query_single_path(src=src, dst=dst, cap=-1)
        print('{}/{}'.format(idx, len(srcs)))


if __name__ == '__main__':
    print('Hello Router.')
    # rt = ParaRouter(model_name='cr-dw-emb=16-par=0~450')
    # rt = BFSRouter(model_name='cr-dw-emb=16-par=0~450')
    rt = RigelRouter(model_name='cr-dw-emb=16-par=0~450')
    path = rt.query_single_path(32,486)
    print('path',path)

    # lp = LineProfiler(Xtest_query_bdd)
    # lp.run('Xtest_query_bdd()')

    # lp = LineProfiler(Xtest_total_bdd)
    # lp.run('Xtest_total_bdd()')

    # lp = LineProfiler(Xtest_total_bfs)
    # lp.run('Xtest_total_bfs()')


    # lp.print_stats()