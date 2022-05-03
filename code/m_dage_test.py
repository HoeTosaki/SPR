import copy
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

plt.rcParams['axes.unicode_minus']=False
'''
verify BC-based walk better performed than other walk patterns.
'''

'''
global info for test acceleration.
'''
glb_g2dmat = {}
glb_node2BC = {}
glb_embed = {}

'''
generate test graph.
'''
def gen_random_graph(regular=20,node_sz=100,st_id=0,name='test',out_dir='../fig',is_print=True):
    G = nx.DiGraph()
    for i in range(st_id,st_id + node_sz):
        G.add_node(i)
    node_lst = list(G.nodes())
    for i in range(st_id,st_id+node_sz):
        cnt = 0
        while cnt < regular:
            dst = random.choice(node_lst)
            if dst != i and not G.has_edge(i,dst):
                if not G.has_edge(dst,i):
                    # avoid to generate double edges in G.edges.
                    G.add_edge(i, dst)
                cnt += 1
    if is_print:
        drawG(G)
        plt.savefig(os.path.join(out_dir,name+'-graph.pdf'))
        plt.show()
    return nx.to_undirected(G)

def gen_comm_graph(cls_sz=100,conn_sz = 2,regular=20,node_sz=100,name='test',out_dir='../fig',is_print=True):
    G = nx.DiGraph()
    for cid in range(cls_sz):
        node_lst = list(G.nodes())
        subG = gen_random_graph(regular=regular,st_id=len(node_lst),node_sz=node_sz,is_print=False)
        # G = nx.union(G, subG,rename=('','G'+str(cid)+'-'))
        G = nx.union(G, subG)
        if len(node_lst) > 0:
            random.shuffle(node_lst)
            conn_ids = node_lst[:conn_sz]
            for conn_id in conn_ids:
                # G.add_edge(conn_id,'G'+str(cid)+'-' + str(random.choice(list(subG.nodes()))))
                G.add_edge(conn_id,random.choice(list(subG.nodes())))

    drawG(G)
    plt.savefig(os.path.join(out_dir,name+'-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_circle_graph(circle_num=10,circle_sz = [5,20],conn_sz = 2,name='circle',out_dir='../fig',is_print=True):
    G = nx.DiGraph()
    def _gen_one_circle(circle_sz,st_id):
        G = nx.DiGraph()
        for i in range(st_id, st_id + circle_sz):
            G.add_node(i)
        node_lst = list(G.nodes())
        for i in range(st_id, st_id + circle_sz):
            if i != st_id + circle_sz - 1:
                G.add_edge(i,i+1)
            else:
                G.add_edge(i, st_id)
        return nx.to_undirected(G)
    for cid in range(circle_num):
        node_lst = list(G.nodes())
        subG = _gen_one_circle(random.randint(circle_sz[0],circle_sz[1]),st_id=len(node_lst))
        G = nx.union(G, subG)
        if len(node_lst) > 0:
            cur_node_lst = node_lst.copy()
            random.shuffle(cur_node_lst)
            conn_ids = cur_node_lst[:conn_sz]
            for conn_id in conn_ids:
                G.add_edge(conn_id,random.choice(list(subG.nodes())))
    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_circleX_graph(circle_num=10,circle_sz = [5,20],conn_sz = 1,cross_num=0,inner_num=0,name='circleX',out_dir='../fig',is_print=True,is_cir_cnt=True):
    cir_cnt = 0
    G = nx.DiGraph()
    def _gen_one_circle(circle_sz,st_id):
        G = nx.DiGraph()
        for i in range(st_id, st_id + circle_sz):
            G.add_node(i)
        node_lst = list(G.nodes())
        for i in range(st_id, st_id + circle_sz):
            if i != st_id + circle_sz - 1:
                G.add_edge(i,i+1)
            else:
                G.add_edge(i, st_id)
        return nx.to_undirected(G)
    cir2nods_meta = []
    for cid in range(circle_num):
        node_lst = list(G.nodes())
        cur_cir_sz = random.randint(circle_sz[0], circle_sz[1])
        subG = _gen_one_circle(cur_cir_sz,st_id=len(node_lst))
        if cur_cir_sz >= 3:
            cir_cnt += 1
        cir2nods_meta.append( (len(node_lst),len(node_lst) + len(subG.nodes())) )
        G = nx.union(G, subG)
        if len(node_lst) > 0:
            cur_node_lst = node_lst.copy()
            random.shuffle(cur_node_lst)
            conn_ids = cur_node_lst[:conn_sz]
            for conn_id in conn_ids:
                G.add_edge(conn_id,random.choice(list(subG.nodes())))
    if cross_num > 0:
        cir2nods_cross = [list(range(st, ed)) for st, ed in cir2nods_meta]
        for cur_cross in range(cross_num):
            cir2nods_cross_cp = []
            for ele in cir2nods_cross:
                if len(ele) > 0:
                    cir2nods_cross_cp.append(ele)
            cir2nods_cross = cir2nods_cross_cp
            random.shuffle(cir2nods_cross)
            cir_st,cir_ed = cir2nods_cross[:2]
            nod_st = random.choice(cir_st)
            nod_ed = random.choice(cir_ed)
            if G.has_edge(nod_st,nod_ed) or G.has_edge(nod_ed,nod_st):
                pass
            else:
                G.add_edge(nod_st,nod_ed)
                cir_cnt+=1
            cir2nods_cross[0].remove(nod_st)

    if inner_num > 0:
        cir2nods_inner = [list(range(st, ed)) for st, ed in cir2nods_meta]
        for cur_inner in range(inner_num):
            cir2nods_inner_cp = []
            for ele in cir2nods_inner:
                if len(ele) >= 4:
                    cir2nods_inner_cp.append(ele)
            cir2nods_inner = cir2nods_inner_cp
            if len(cir2nods_inner) == 0:
                print('no circle remains!')
                break
            random.shuffle(cir2nods_inner)
            cur_cir = cir2nods_inner[0]
            assert len(cur_cir) >= 2
            cur_cir_cp = copy.deepcopy(cur_cir)
            random.shuffle(cur_cir_cp)
            nod_st,nod_ed = cur_cir_cp[:2]
            if G.has_edge(nod_st, nod_ed) or G.has_edge(nod_ed, nod_st):
                pass
            else:
                G.add_edge(nod_st, nod_ed)
                cir_cnt += 1
            cir2nods_inner[0].remove(nod_st)
            cir2nods_inner[0].remove(nod_ed)

    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    if is_cir_cnt:
        return nx.to_undirected(G),cir_cnt
    else:
        return nx.to_undirected(G)

def gen_triangle_graph(triangle_sz = [3,6],conn_sz=1,traingle_num=100,name='triangle',out_dir='../fig',is_print=True):
    G = nx.DiGraph()

    for cid in range(traingle_num):
        node_lst = list(G.nodes())
        cur_triangle_sz = random.randint(triangle_sz[0], triangle_sz[1])
        if cid == 0:
            new_nodes = list(range(cur_triangle_sz))
            new_triangle_nodes = new_nodes
        else:
            new_nodes = list(range(len(node_lst),len(node_lst) + cur_triangle_sz - conn_sz))
            new_triangle_nodes = list(range(len(node_lst) - conn_sz,len(node_lst) + cur_triangle_sz - conn_sz))
        for nn in new_nodes:
            G.add_node(nn)
        for idx,i in enumerate(new_triangle_nodes):
            for j in new_triangle_nodes[idx + 1:]:
                G.add_edge(i,j)
    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_tricircle_graph(circle_num=10,triangle_sz = [3,6],conn_sz=1,traingle_num=20,name='tricircle',out_dir='../fig',is_print=True):
    G = nx.DiGraph()

    def _gen_one_circle(st_id,triangle_sz,conn_sz,traingle_num):
        G = nx.DiGraph()
        for cid in range(traingle_num):
            node_lst = list(G.nodes())
            node_lst = [ele - st_id for ele in node_lst]
            cur_triangle_sz = random.randint(triangle_sz[0], triangle_sz[1])
            if cid == 0:
                new_nodes = list(range(cur_triangle_sz))
                new_triangle_nodes = new_nodes
            elif cid == traingle_num - 1:
                new_nodes = list(range(len(node_lst), len(node_lst) + cur_triangle_sz - 2 * conn_sz))
                new_triangle_nodes = list(range(len(node_lst) - conn_sz, len(node_lst) + cur_triangle_sz - 2 * conn_sz))
                new_triangle_nodes.extend(list(range(conn_sz)))
            else:
                new_nodes = list(range(len(node_lst), len(node_lst) + cur_triangle_sz - conn_sz))
                new_triangle_nodes = list(range(len(node_lst) - conn_sz, len(node_lst) + cur_triangle_sz - conn_sz))

            for nn in new_nodes:
                G.add_node(nn + st_id)
            for idx, i in enumerate(new_triangle_nodes):
                for j in new_triangle_nodes[idx + 1:]:
                    G.add_edge(i+st_id, j+st_id)

        return nx.to_undirected(G)
    for cid in range(circle_num):
        node_lst = list(G.nodes())
        subG = _gen_one_circle(st_id=len(node_lst),triangle_sz=triangle_sz,conn_sz=conn_sz,traingle_num=traingle_num)
        G = nx.union(G, subG)
        if len(node_lst) > 0:
            cur_node_lst = node_lst.copy()
            random.shuffle(cur_node_lst)
            conn_ids = cur_node_lst[:conn_sz]
            for conn_id in conn_ids:
                G.add_edge(conn_id,random.choice(list(subG.nodes())))

    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_tree_graph(child_sz=[1,5],extend_decay = 0.9,max_depth=10,name='tree',out_dir='../fig',is_print=True):
    G = nx.DiGraph()
    G.add_node(0)
    open_lst = [(0,extend_decay)]
    min_prob = math.pow(extend_decay,max_depth)
    while len(open_lst) != 0:
        cur_root,cur_prob = open_lst.pop(0)
        if random.uniform(0,1) < cur_prob and cur_prob > min_prob:
            node_lst = list(G.nodes())
            child_nodes = list(range(len(node_lst),len(node_lst) + random.randint(child_sz[0],child_sz[1])))
            for child_node in child_nodes:
                G.add_node(child_node)
                G.add_edge(cur_root,child_node)
                open_lst.append((child_node,cur_prob*extend_decay))

    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_spiral_graph(circle_expend=10,st_circle_sz=10,conn_sz = 5,name='circle',out_dir='../fig',is_print=True):
    G = nx.DiGraph()

    def _gen_one_line(line_sz,st_id):
        G = nx.DiGraph()
        for i in range(st_id, st_id + line_sz):
            G.add_node(i)
        node_lst = list(G.nodes())
        for i in range(st_id, st_id + line_sz-1):
            G.add_edge(i,i+1)
        return nx.to_undirected(G)

    circle_ints = []
    for cid in range(circle_expend):
        node_lst = list(G.nodes())
        subG = _gen_one_line(st_circle_sz + circle_expend * cid,st_id=len(node_lst))
        G = nx.union(G, subG)
        circle_ints.append((len(node_lst),len(node_lst) + st_circle_sz + circle_expend * cid))
        # if len(node_lst) > 0:
        #     cur_node_lst = node_lst.copy()
        #     random.shuffle(cur_node_lst)
        #     conn_ids = cur_node_lst[:conn_sz]
        #     for conn_id in conn_ids:
        #         G.add_edge(conn_id,random.choice(list(subG.nodes())))
    node_lst = list(G.nodes())
    for idx in range(1,len(circle_ints)):
        n_st,n_ed = circle_ints[idx]
        c_st,c_ed = circle_ints[idx-1]
        conn_lst1 = node_lst[c_st:c_ed].copy()
        conn_lst2 = node_lst[n_st:n_ed].copy()
        random.shuffle(conn_lst1)
        random.shuffle(conn_lst2)
        conn_lst1 = list(sorted(conn_lst1[:conn_sz]))
        conn_lst2 = list(sorted(conn_lst2[:conn_sz]))
        for n1,n2 in zip(conn_lst1,conn_lst2):
            G.add_edge(n1,n2)

    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_net_graph(net_shape=[100,200],edge_prob=0.99,name='net',out_dir='../fig',is_print=True):
    G = nx.DiGraph()
    m = net_shape[0]
    n = net_shape[1]
    for i in range(m):
        for j in range(n):
            G.add_node(n*i+j)
    for i in range(m):
        for j in range(n):
            if j >= 1 and random.uniform(0,1) < edge_prob:
                G.add_edge(i*n + j,i*n+j-1)
            if i >= 1 and random.uniform(0, 1) < edge_prob:
                G.add_edge(i * n + j, (i-1) * n + j)
    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    return nx.to_undirected(G)

def gen_disk_like_graph(circle_num=10,node_sz=200,name='disk',out_dir='../fig',is_print=True,is_cir_cnt = True):
    assert circle_num >= 1
    cir_cnt = 1
    G = nx.DiGraph()
    for i in range(node_sz):
        G.add_node(i)
    for i in range(node_sz-1):
        G.add_edge(i,i+1)
    G.add_edge(node_sz-1,0)
    nodes = copy.deepcopy(list(G.nodes()))
    for j in range(circle_num-1):
        if len(nodes) <= 1:
            print('Caution:current circle number {} > node sz {}!'.format(circle_num,node_sz))
            break
        random.shuffle(nodes)
        cur_st,cur_ed = nodes[:2]
        if G.has_edge(cur_st,cur_ed) or G.has_edge(cur_ed,cur_st):
            pass
        else:
            G.add_edge(cur_st,cur_ed)
            cir_cnt += 1
        nodes = nodes[1:]
    drawG(G)
    plt.savefig(os.path.join(out_dir, name + '-graph.pdf'))
    if is_print:
        plt.show()
    if is_cir_cnt:
        return nx.to_undirected(G),cir_cnt
    else:
        return nx.to_undirected(G)


def gen_disk_graph(name):
    # useful for large graph.
    out_dir = '../datasets/dst'
    g, _ = dgl.load_graphs(os.path.join(out_dir,name))
    g = g[0]
    # bc_file = os.path.join(out_dir, name + '-BC.txt')
    # if os.path.exists(bc_file):
    #     g.ndata['bc'] = tc.zeros(g.num_nodes(),1)
    #     with open(bc_file) as f:
    #         cnt = 0
    #         for line in f:
    #             cnt += 1
    #             line = line.strip()
    #             nid,bc = line.split('-')
    #             nid = int(nid)
    #             bc = float(bc)
    #             g.ndata['bc'][nid] = bc
    #         assert cnt >= g.num_nodes(),print('g bc cnt {} < node cnt {}'.format(cnt,g.num_nodes()))
    return g

def Dtest_direct(G):
    G = nx.to_undirected(G)
    node_lst = list(G.nodes())
    for nid in node_lst:
        for nnid in G.neighbors(nid):
            node_n_lst = list(G.neighbors(nnid))
            if nid not in node_n_lst:
                print('not conformed for {} and {}'.format(nid,nnid))
    print('check finished.')

def Dtest_uniconn(G,is_fast=False):
    lst_conn = []
    set_sgn = set()
    num_node = G.num_nodes()
    for idx,nid in enumerate(G.nodes()):
        nid = int(nid)
        if idx % 200 == 0:
            print('graph_conj:{}/{}'.format(idx,num_node))
        if nid not in set_sgn:
            new_group = set() # corresponding to close set of A*��
            set_sgn.add(nid)
            new_group.add(nid)
            que_open = [nid]
            while len(que_open) > 0:
                cur = que_open.pop(0)
                for nnid in G.successors(cur):
                    nnid = int(nnid)
                    if nnid not in set_sgn:
                        que_open.append(nnid)
                        set_sgn.add(nnid)
                        new_group.add(nnid)
            lst_conn.append(new_group)
            if is_fast:
                print('fast checking...')
                if len(lst_conn) > 1:
                    print('graph with more than one group.')
                    return False
    if len(lst_conn) > 1:
        with open('../fig/test-uniconn.log','w') as f:
            [f.write('-'.join([str(eele) for eele in ele])+'\n') for ele in lst_conn]
            f.flush()
        print('graph with {} connected groups'.format(len(lst_conn)))
        return False
    else:
        print('graph has been checked uni-connected.')
        return True



'''
graph statistics.
'''
def anal_G_basic(Gs,names,out_dir='../fig',name=''):
    # assert undirected G.
    title = ['names','nodes','edges','e/n','avg_deg','max_deg','min_deg']
    data = []
    for G,name in zip(Gs,names):
        G = nx.to_undirected(G)
        ns = G.number_of_nodes()
        es = G.number_of_edges()
        max_deg = -1
        min_deg = ns + 1
        sum_deg = -1
        for node in G.nodes():
            nbs_sz = len(list(G.neighbors(node)))
            sum_deg += nbs_sz
            max_deg = max(max_deg,nbs_sz)
            min_deg = min(min_deg,nbs_sz)
        row = [name,ns,es,es/ns,sum_deg/ns,max_deg,min_deg]
        data.append(row)
    df = pd.DataFrame(columns=title,data=data)
    df.to_csv(os.path.join(out_dir,name + 'anal-basic.csv'),index=False)
    return df

def anal_G_complex(Gs,names,out_dir='../fig',name='',is_print=True):
    # assert undirected G.
    title = ['names', 'diameter','eccentricity','avg_BC','max_BC','min_BC']
    data = []
    for G, name in zip(Gs, names):
        G = nx.to_undirected(G)
        nodes = list(G.nodes())
        node2BC = {}
        sp_cnt = 0
        ecc = len(nodes)
        diameter = 0
        is_connected = True
        dist_mat = tc.zeros(size=(len(nodes),len(nodes)),dtype=tc.int16)
        for idx,i in enumerate(nodes):
            if idx == len(nodes):
                break
            for j in nodes[idx + 1:]:
                if j == i:
                    continue
                try:
                    for path in nx.all_shortest_paths(G,source=i,target=j):
                        sp_cnt += 1
                        for nid in path:
                            if nid in node2BC:
                                node2BC[nid] += 1
                            else:
                                node2BC[nid] = 1
                        path_len = len(path) - 1
                        diameter = max(diameter,path_len)

                        dist_mat[i,j] = path_len
                        dist_mat[j,i] = path_len
                except nx.exception.NetworkXNoPath:
                    print('{} and {} occupy no path!'.format(i,j))
                    is_connected = False
                    continue
            if idx % 10 == 0:
                print('complex finish {}'.format(idx))

        global glb_g2dmat
        glb_g2dmat[name] = dist_mat

        max_BC = -1
        min_BC = sp_cnt + 1
        sum_BC = 0.

        for i in node2BC:
            node2BC[i] /= sp_cnt
            sum_BC += node2BC[i]
            max_BC = max(max_BC,node2BC[i])
            min_BC = min(min_BC,node2BC[i])

        global glb_node2BC
        glb_node2BC[name] = node2BC

        if is_connected:
            for idx,i in enumerate(nodes):
                if i == len(nodes) - 1:
                    break
                ecc_at_j = 0
                for j in nodes[idx + 1:]:
                    ecc_at_j = max(ecc_at_j,dist_mat[i,j].item())
                ecc = min(ecc,ecc_at_j)
        else:
            ecc = -1
            diameter = -1
        row = [name,diameter,ecc,sum_BC / len(nodes),max_BC,min_BC]
        data.append(row)

        # draw BC distribution.
        xs = nodes
        ys = [node2BC[nid] for nid in nodes]
        plt.bar(x=xs,height=ys,width=0.05)
        plt.title(name)
        plt.ylabel('BC value of nodes')
        plt.xlabel('nodes')
        plt.savefig(os.path.join(out_dir,name+'-BCDist.pdf'))
        if is_print:
            plt.show()
    df = pd.DataFrame(columns=title, data=data)
    df.to_csv(os.path.join(out_dir,name + 'anal-complex.csv'),index=False)
    return df

def anal_BC_fast(G_file='../datasets/dst/current.graph',num_worker=1,out_dir='../tmp',name='bc'):
    # %%
    # import dgl
    # G = dgl.DGLGraph()
    # G.add_edge(0,2)
    # G.add_edge(0, 1)
    # G.add_edge(0, 3)
    # G.add_edge(3, 4)
    # G.add_edge(3, 5)
    # G.add_edge(5, 4)
    # G.add_edge(5, 6)
    # G = dgl.to_simple(G)
    # G = dgl.to_bidirected(G)
    # dgl.save_graphs('../datasets/dst/current.graph',[G])
    # %%

    G,_ = dgl.load_graphs(G_file)
    G = G[0]

    node_lst = copy.deepcopy(list(G.nodes()))
    # random.shuffle(node_lst)

    # range2id = {}
    node_ranges = []
    node_per_workers = math.ceil(len(node_lst) / num_worker)
    for wid in range(num_worker):
        node_ranges.append(node_lst[wid*node_per_workers:min((wid+1)*node_per_workers,len(node_lst))])
    len_chk = 0
    for node_range in node_ranges:
        len_chk += len(node_range)
        print(len(node_range))
    print('new len {} == old len {}'.format(len_chk,len(node_lst)))

    for i in range(num_worker):
        with open(os.path.join(out_dir,name)+'.index~{}'.format(i),'w') as f:
            f.write(' '.join([str(int(ele)) for ele in node_ranges[i]]))
            f.flush()
    print('main thread indices constructed.')

    procs = []
    for i in range(num_worker):
        proc = multiprocessing.Process(target=_anal_BC_fast, args=(G_file,os.path.join(out_dir,name),i))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
    print('all proc dumped.')
    node2BC_all = {}
    for i in range(num_worker):
        patch_file_name = '{}.bcpatch~{}'.format(os.path.join(out_dir,name),i)
        with open(patch_file_name,'r') as f:
            for line in f.readlines():
                if line is not None:
                    line = line.strip()
                    if line != '':
                        lst = line.split('-')
                        assert len(lst) == 2, print(lst)
                        nid = int(lst[0])
                        bc = int(lst[1])
                        if nid in node2BC_all:
                            node2BC_all[nid] += bc
                        else:
                            node2BC_all[nid] = bc
    print('main thread write BC result.')
    with open(os.path.join(out_dir,name+'.txt'),'w') as f:
        for nid in node2BC_all:
            f.writelines('{}-{}\n'.format(nid,node2BC_all[nid]))
        f.flush()
    print('main thread write finished.')

def _anal_BC_fast(G_file,out_file,pid):
    print('start proc {} nodes...'.format(pid))
    G, _ = dgl.load_graphs(G_file)
    G = G[0]
    node_range = []
    with open(out_file+'.index~{}'.format(pid),'r') as f:
        data = ''.join(f.readlines()).strip()
        lst = data.split(' ')
        node_range.extend([int(ele) for ele in lst])
    print('Proc {}: readed node range'.format(pid))
    # node_lst = list(G.nodes())
    node2BC = {}
    for idx,src in enumerate(node_range):
        dic_id2pre = [-1] * G.num_nodes()
        search_lst = [src]
        dic_id2pre[src] = None # self
        while len(search_lst) != 0:
            nid = search_lst.pop(0)
            lst = G.successors(nid)
            for nnid in lst:
                nnid = int(nnid)
                if dic_id2pre[nnid] is not None and dic_id2pre[nnid] < 0:
                    search_lst.append(nnid)
                    dic_id2pre[nnid] = nid
        print('Proc {}: constructed index {}/{}'.format(pid,idx,len(node_range)))

        for nid in G.nodes():
            p = dic_id2pre[nid]
            if p is None:
                continue
            while dic_id2pre[p] is not None:
                assert p != -1, print('construct index failed at index {} back to {}'.format(nid,src))
                if p in node2BC:
                    node2BC[p] += 1
                else:
                    node2BC[p] = 1
                p = dic_id2pre[p]
        print('Proc {}: finished {}/{}'.format(pid, idx, len(node_range)))
    with open('{}.bcpatch~{}'.format(out_file,pid),'w') as f:
        for nid in node2BC:
            f.writelines('{}-{}\n'.format(nid,node2BC[nid]))
        f.flush()
    print('Proc {}: all dumped {}/{}'.format(pid, idx, len(node_range)))


'''
implements of random walks
'''
def walk_general(root,G,G_name,length,batch_sz):
    global glb_g2dmat
    assert G_name in glb_g2dmat
    dist_mat = glb_g2dmat[G_name]

    paths = []
    for bid in range(batch_sz):
        cur_nid = root
        path = [cur_nid]
        while len(path) < length:
            next_lst = []
            for nid in G.neighbors(cur_nid):
                if nid in path:
                    continue
                next_lst.append(nid)
            if len(next_lst) == 0:
                break  # since there's no extra transition.
            next_nid = random.choice(next_lst)
            path.append(next_nid)
            cur_nid = next_nid
        paths.append(path)
    return paths

def walk_node2vec(root,G,G_name,length,batch_sz,p=1,q=1):
    global glb_g2dmat
    assert G_name in glb_g2dmat
    dist_mat = glb_g2dmat[G_name]

    paths = []
    for bid in range(batch_sz):
        pre_nid = -1
        cur_nid = root
        path = [cur_nid]
        while len(path) < length:
            next_lst = []
            next_lst_prob = []
            for nid in G.neighbors(cur_nid):
                if nid in path:
                    continue
                next_lst.append(nid)
                if pre_nid != -1:
                    assert dist_mat[nid][pre_nid] <= 2,print('nid:{},pre_nid:{},dist:{}-'.format(nid,pre_nid,dist_mat[nid][pre_nid],dist_mat[pre_nid][nid]))
                    if pre_nid == nid:
                        next_lst_prob.append(1 / p)
                    elif dist_mat[nid][pre_nid] == 1:
                        next_lst_prob.append(1)
                    else:
                        assert dist_mat[nid][pre_nid] == 2,print('nid:{},pre_nid:{},dist:{}-'.format(nid,pre_nid,dist_mat[nid][pre_nid],dist_mat[pre_nid][nid]))
                        next_lst_prob.append(1 / q)
                else:
                    next_lst_prob.append(1)
            if len(next_lst) == 0:
                break  # since there's no extra transition.
            sum_next_lst_prob = sum(next_lst_prob)
            next_lst_prob = [ele / sum_next_lst_prob for ele in next_lst_prob]
            next_nid = int(np.random.choice(a=next_lst, size=1, p=next_lst_prob)[0])
            path.append(next_nid)
            pre_nid = cur_nid
            cur_nid = next_nid
        paths.append(path)
    return paths

def walk_surf(root,G,G_name,length,batch_sz,alpha=0.98):
    global glb_g2dmat
    assert G_name in glb_g2dmat
    dist_mat = glb_g2dmat[G_name]

    paths = []
    for bid in range(batch_sz):
        cur_nid = root
        path = [cur_nid]
        while len(path) < length:
            next_lst = []
            for nid in G.neighbors(cur_nid):
                next_lst.append(nid)
            if len(next_lst) == 0:
                if cur_nid == root:
                    break # since there's no extra transition.
                else:
                    cur_nid = root
                    continue
            else:
                next_nid = random.choice(next_lst)
                path.append(next_nid)
                chs_lst = [True,False]
                assert 0 <= alpha <= 1
                chs_prob = [1-alpha,alpha]
                chs = bool(np.random.choice(a=chs_lst, size=1, p=chs_prob)[0])
                if chs:
                    cur_nid = root
                else:
                    cur_nid = next_nid
        paths.append(path)
    return paths

def walk_bc(root,G,G_name,length,batch_sz,tau=True):
    global glb_node2BC
    assert G_name in glb_node2BC
    node2BC = glb_node2BC[G_name]

    paths = []
    for bid in range(batch_sz):
        pre_nid = -1
        cur_nid = root
        path = [cur_nid]
        while len(path) < length:
            next_lst = []
            next_lst_prob = []
            for nid in G.neighbors(cur_nid):
                if nid in path:
                    continue
                next_lst.append(nid)
                next_lst_prob.append(node2BC[nid])
            if len(next_lst) == 0:
                break  # since there's no extra transition.
            if not tau:
                sum_next_lst_prob = sum(next_lst_prob)
                next_lst_prob = [ele / sum_next_lst_prob for ele in next_lst_prob]
            else:
                next_lst_prob = np.array(next_lst_prob)
                next_lst_prob = np.log(next_lst_prob + 1)
                next_lst_prob = next_lst_prob / next_lst_prob.sum()
                next_lst_prob = next_lst_prob.tolist()
            next_nid = int(np.random.choice(a=next_lst, size=1, p=next_lst_prob)[0])
            path.append(next_nid)
            pre_nid = cur_nid
            cur_nid = next_nid
        paths.append(path)
    return paths

# keep |input set| ? keep |output set|
# keep input len ? output len
# most desirable �� keep output len & output set for embedding time.
# others �� keep input len & set for traversal time.
def walk_dr(root,G,G_name,input_len,output_len,batch_sz,alpha=0.98,input_exps=5,output_exps=5):
    walks = []
    global glb_node2BC
    global glb_g2dmat
    node2BC = glb_node2BC[G_name]
    dist_mat = glb_g2dmat[G_name]
    cur_walks = walk_bc(root=root, G=G, G_name=G_name, length=input_len, batch_sz=batch_sz*input_exps)
    sample_node_lst = set()
    for walk in cur_walks:
        for nid in walk[1:]:
            sample_node_lst.add(nid)
    sample_node_lst = list(sample_node_lst)
    if len(sample_node_lst) == 0:
        print('empty node lst!')
    sample_prob = []
    for node in sample_node_lst:
        sample_prob.append(node2BC[node] * math.pow(alpha, int(dist_mat[root][node])))
    sum_sample_prob = sum(sample_prob)
    sample_prob = [ele / sum_sample_prob for ele in sample_prob]
    samples = np.random.choice(a=sample_node_lst, size=(len(cur_walks) * output_exps, output_len), p=sample_prob)
    roots = np.array([root] * len(cur_walks) * output_exps)
    new_walks = np.concatenate([roots.reshape(-1, 1), samples], axis=1)
    return new_walks





'''
analyze & compare of random walks
'''
def anal_walks_basic(Gs,G_names,walk_len=5,test_src_sz=10,test_batch_sz=30,walk_types=['gen','n2v','surf'],out_dir='../fig',is_print=True):
    assert len(Gs) == len(G_names)
    for G,G_name in zip(Gs,G_names):
        assert G_name in glb_g2dmat,'not found {}'.format(G_name)
        node_lst = list(G.nodes())
        test_src_sz = min(test_src_sz,len(node_lst))
        test_batch_sz = min(test_batch_sz,len(node_lst))

        # generate nodes.
        random.shuffle(node_lst)
        srcs = node_lst[:test_src_sz]
        score_dict = {}
        for walk_type in walk_types:
            if walk_type == 'gen':
                score = []
                for i in range(walk_len):
                    score.append([])
                for src in srcs:
                    paths = walk_general(root=src,G=G,G_name=G_name,length=walk_len,batch_sz=test_batch_sz)
                    for path in paths:
                        assert len(path) <= walk_len,print('path len = {}, walk_len = {}'.format(len(path),walk_len))
                        for idx,pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G,source=src,target=pid))
                score_dict[walk_type] = score
            if walk_type == 'n2v':
                score = []
                for i in range(walk_len):
                    score.append([])
                for src in srcs:
                    paths = walk_node2vec(root=src, G=G, G_name=G_name, length=walk_len, batch_sz=test_batch_sz)
                    for path in paths:
                        assert len(path) <= walk_len, print('path len = {}, walk_len = {}'.format(len(path), walk_len))
                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
            if walk_type == 'bfs':
                score = []
                for i in range(walk_len):
                    score.append([])
                for src in srcs:
                    paths = walk_node2vec(root=src, G=G, G_name=G_name, length=walk_len, batch_sz=test_batch_sz,p=10,q = 0.1)
                    for path in paths:
                        assert len(path) <= walk_len, print('path len = {}, walk_len = {}'.format(len(path), walk_len))
                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
            if walk_type == 'surf':
                score = []
                for i in range(walk_len):
                    score.append([])
                for src in srcs:
                    paths = walk_node2vec(root=src, G=G, G_name=G_name, length=walk_len, batch_sz=test_batch_sz)
                    for path in paths:
                        assert len(path) <= walk_len, print('path len = {}, walk_len = {}'.format(len(path), walk_len))
                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
            if walk_type == 'bc':
                score = []
                for i in range(walk_len):
                    score.append([])
                for src in srcs:
                    paths = walk_bc(root=src, G=G, G_name=G_name, length=walk_len, batch_sz=test_batch_sz)
                    for path in paths:
                        assert len(path) <= walk_len, print('path len = {}, walk_len = {}'.format(len(path), walk_len))
                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
            if walk_type == 'dr-in': # which means only input data is scaled.
                exp_coef = 5
                score = []
                for i in range(walk_len + 1):
                    score.append([])
                for src in srcs:
                    paths = walk_dr(root=src, G=G, G_name=G_name, input_len=walk_len*exp_coef,output_len=walk_len, batch_sz=test_batch_sz,input_exps=exp_coef,output_exps=1)
                    for path in paths:
                        # since there's a root at the first of output walks.
                        assert len(path) <= walk_len + 1, print('path len = {}, walk_len = {}'.format(len(path), walk_len))

                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
            if walk_type == 'dr-out': # which means only output data is scaled.
                exp_coef = 5
                score = []
                for i in range(walk_len*exp_coef + 1):
                    score.append([])
                for src in srcs:
                    paths = walk_dr(root=src, G=G, G_name=G_name, input_len=walk_len,output_len=walk_len*exp_coef, batch_sz=test_batch_sz,input_exps=1,output_exps=exp_coef)
                    for path in paths:
                        # since there's a root at the first of output walks.
                        assert len(path) <= walk_len*exp_coef + 1, print('path len = {}, walk_len = {}'.format(len(path), walk_len))

                        for idx, pid in enumerate(path):
                            score[idx].append(nx.shortest_path_length(G=G, source=src, target=pid))
                score_dict[walk_type] = score
        if not is_print:
            return score_dict
        for name in score_dict:
            xs = []
            ys = []
            for idx,scores in enumerate(score_dict[name]):
                xs.extend([idx]*len(scores))
                ys.extend(scores)
            plt.scatter(x=xs,y=ys,c='dodgerblue',marker=None,alpha=0.01)
            # plt.title(name)
            plt.ylim(0,walk_len)
            plt.xlabel('Walk length')
            plt.ylabel('Probability of Explored Distance')
            plt.savefig(os.path.join(out_dir,G_name+'-'+name+'-walk.pdf'))
            if is_print:
                plt.show()


'''
max-likelihood optim for node embedding.
'''
def embed_general(G,G_name,emb_sz,walk_len,walks_per_node,window_sz=None,neg_sz=5):
    walks = []
    for node in G.nodes():
        walks.extend(walk_general(root=node,G=G,G_name=G_name,length=walk_len,batch_sz=walks_per_node))
    walks = walk2str(walks)
    emb = walk2embed(walks,emb_sz=emb_sz,window_sz=window_sz,neg_sz=neg_sz,name=G_name+'-gen')
    return emb

def embed_node2vec(G,G_name,emb_sz,walk_len,walks_per_node,window_sz=None,neg_sz=5):
    walks = []
    for node in G.nodes():
        walks.extend(walk_node2vec(root=node,G=G,G_name=G_name,length=walk_len,batch_sz=walks_per_node))
    walks = walk2str(walks)
    emb = walk2embed(walks,emb_sz=emb_sz,window_sz=window_sz,neg_sz=neg_sz,name=G_name+'-gen')
    return emb

def embed_surf(G,G_name,emb_sz,walk_len,walks_per_node,window_sz=None,neg_sz=5):
    walks = []
    for node in G.nodes():
        walks.extend(walk_surf(root=node,G=G,G_name=G_name,length=walk_len,batch_sz=walks_per_node))
    walks = walk2str(walks)
    emb = walk2embed(walks,emb_sz=emb_sz,window_sz=window_sz,neg_sz=neg_sz,name=G_name+'-gen')
    return emb

def embed_bcdr(G,G_name,emb_sz,input_len,output_len,walks_per_node,window_sz=None,neg_sz=5,alpha=0.98,input_exps=5,output_exps=5):
    assert window_sz is None,'bcdr NOT support customized window sz.'
    walks = []
    global glb_node2BC
    global glb_g2dmat
    node2BC = glb_node2BC[G_name]
    dist_mat = glb_g2dmat[G_name]
    for root in G.nodes():
        cur_walks = walk_bc(root=root,G=G,G_name=G_name,length=input_len,batch_sz=walks_per_node*input_exps)
        sample_node_lst = set()
        for walk in cur_walks:
            for nid in walk[1:]:
                sample_node_lst.add(nid)
        sample_node_lst = list(sample_node_lst)
        if len(sample_node_lst) == 0:
            print('empty node lst!')
        sample_prob = []
        for node in sample_node_lst:
            sample_prob.append(node2BC[node] * math.pow(alpha,int(dist_mat[root][node])))
        sum_sample_prob = sum(sample_prob)
        sample_prob = [ele / sum_sample_prob for ele in sample_prob]
        samples = np.random.choice(a=sample_node_lst,size=(len(cur_walks)*output_exps,output_len),p=sample_prob)
        roots = np.array([root]*len(cur_walks)*output_exps)
        new_walks = np.concatenate([roots.reshape(-1,1),samples],axis=1)
        walks.extend(new_walks.tolist())
        # walks.extend(cur_walks)
    walks = walk2str(walks)
    # assert len(walks[0]) == walk_len + 1
    emb = walk2embed(walks,emb_sz=emb_sz,window_sz=len(walks[0]),neg_sz=neg_sz,name=G_name+'-gen')
    return emb


'''
anal embedding space.
'''
def anal_embed(Gs,G_names,emb_sz,walk_len=40,walks_per_node=20,test_src_sz=20,test_batch_sz=100,walk_types=['gen','n2v','surf'],out_dir='../fig',is_print=True):
    global glb_g2dmat
    assert len(Gs) == len(G_names)
    for G, G_name in zip(Gs, G_names):
        assert G_name in glb_g2dmat, 'not found {}'.format(G_name)
        dist_mat = glb_g2dmat[G_name]
        node_lst = list(G.nodes())
        test_src_sz = min(test_src_sz,len(node_lst))
        test_batch_sz = min(test_batch_sz,len(node_lst))
        emb_dict = {}
        for walk_type in walk_types:
            if walk_type == 'gen':
                emb = embed_general(G=G,G_name=G_name,emb_sz=emb_sz,walk_len=walk_len,walks_per_node=walks_per_node)
                emb_dict[walk_type] = emb
            if walk_type == 'n2v':
                emb = embed_node2vec(G=G, G_name=G_name, emb_sz=emb_sz, walk_len=walk_len, walks_per_node=walks_per_node)
                emb_dict[walk_type] = emb
            if walk_type == 'surf':
                emb = embed_surf(G=G, G_name=G_name, emb_sz=emb_sz, walk_len=walk_len, walks_per_node=walks_per_node)
                emb_dict[walk_type] = emb
            if walk_type == 'bcdr':
                exps_coef = 5
                emb = embed_bcdr(G=G, G_name=G_name, emb_sz=emb_sz, input_len=walk_len,output_len=walk_len*exps_coef, walks_per_node=walks_per_node,input_exps=1,output_exps=exps_coef)
                emb_dict[walk_type] = emb
        glb_embed[G_name] = emb_dict
        # generate test nodes.
        random.shuffle(node_lst)
        srcs = node_lst[:test_src_sz]

        score_dict = {}
        for name in emb_dict:
            cur_embs = emb_dict[name]
            score = []
            for i in range(100): # max for graph diameter.
                score.append([])
            for src in srcs:
                src_emb = cur_embs[src]
                random.shuffle(node_lst)
                dsts = node_lst[:test_batch_sz]
                dsts_embs = cur_embs[dsts]
                dsts_dist = dist_mat[dsts,src]
                assert dsts_dist.shape[0] == len(dsts) == dsts_embs.shape[0],print('dst:{}, dst emb:{},dst dist:{}'.format(len(dsts),dsts_embs.shape[0],dsts_dist.shape[0]))
                for i in range(len(dsts)):
                    cur_score = tc.matmul(src_emb,dsts_embs[i])
                    assert int(dsts_dist[i]) < len(score), print('dist:{}, MAX len:{}'.format(int(dsts_dist[i]),len(score)))
                    score[int(dsts_dist[i])].append(cur_score)
            score_dict[name] = score
        for name in emb_dict:
            cur_score = score_dict[name]
            if len(cur_score) == 0 or cur_score is None:
                continue
            xs = []
            ys = []
            for idx,scores in enumerate(cur_score):
                for score in scores:
                    xs.append(idx)
                    ys.append(score)
            plt.scatter(x=xs,y=ys,c='dodgerblue',alpha=0.05)
            plt.ylim(0,15)
            plt.xlabel('shortest distance between nodes on the graph')
            plt.ylabel('distance measured on the embedding space')
            # plt.title(name)
            plt.savefig(os.path.join(out_dir,G_name+'-'+name+'-embed.pdf'))
            if is_print:
                plt.show()

def anal_embed_rate(Gs,G_names,test_src_sz=20,test_batch_sz=100,out_dir='../fig',is_print=True):
    global glb_embed
    global glb_g2dmat
    assert len(Gs) == len(G_names)
    data = {}
    for G, G_name in zip(Gs, G_names):
        assert G_name in glb_embed, 'not found {}'.format(G_name)
        embed_dict = glb_embed[G_name]
        dist_mat = glb_g2dmat[G_name]
        data[G_name] = {}
        for name in embed_dict:
            node_lst = list(G.nodes())
            test_src_sz = min(test_src_sz,len(node_lst))
            test_batch_sz = min(test_src_sz,len(node_lst))

            random.shuffle(node_lst)
            srcs = node_lst[:test_src_sz]
            lgs = []
            succ_num = 0
            for src in srcs:
                random.shuffle(node_lst)
                test_bs = node_lst[:test_batch_sz]
                random.shuffle(node_lst)
                test_cs = node_lst[:test_batch_sz]
                for test_b,test_c in zip(test_bs,test_cs):
                    lg = (dist_mat[src][test_b] - dist_mat[src][test_c])*(tc.matmul(embed_dict[name][src],embed_dict[name][test_b]) - tc.matmul(embed_dict[name][src],embed_dict[name][test_c]))
                    lgs.append(lg)
                    if lg <= 0:
                        succ_num += 1
            colors_chs = ['coral','dodgerblue']
            cs = [colors_chs[int(lg <= 0)] for lg in lgs]
            idx_pos = -1
            idx_neg = -1
            for idx,c in enumerate(cs):
                if c == 'coral':
                    idx_pos = idx
                else:
                    idx_neg = idx
                if idx_pos != -1 and idx_neg != -1:
                    break

            plt.scatter(x=[idx_pos],y=[lgs[idx_pos]],c='coral',label='violated')
            plt.scatter(x=[idx_neg],y=[lgs[idx_neg]],c='dodgerblue',label='preserved')

            plt.scatter(x=range(len(lgs)),y=lgs,c=cs)

            plt.legend(loc='upper right')
            # plt.title(name)
            plt.ylabel('value of distance expression')
            plt.xlabel('sampled node triples')
            plt.savefig(os.path.join(out_dir, G_name + '-' + name + '-ineq.pdf'))
            if is_print:
                plt.show()
            data[G_name][name] = succ_num / len(lgs)
    assert len(data.values()) != 0
    cols = list(data.values())[0].keys()
    rows = data.keys()
    df = pd.DataFrame(data=None,columns=cols,index=rows)
    for G_name in data:
        for walk_name in data[G_name]:
            df.loc[G_name,walk_name] = data[G_name][walk_name]
    print(df)
    df.to_csv(os.path.join(out_dir,'anal_embed_rate.csv'),index=True)
    return df

'''
utils.
'''
def drawG(G,c='dodgerblue',is_undirected=True):
    if is_undirected:
        G = nx.to_undirected(G)
    nx.draw(G, node_size=5, with_labels=False, node_color=c)

def walk2embed(walks,emb_sz=16,window_sz=None,neg_sz=5,out_dir='../tmp',name=''):
    if window_sz is None:
        window_sz = len(walks[0]) // 4
    model = m_deepwalk.Word2Vec(walks, size=emb_sz, window=window_sz, min_count=0, sg=1, hs=0,negative=neg_sz,workers=8)
    model.wv.save_word2vec_format(os.path.join(out_dir, 'test-' + name + '.embed'))

    emb = None
    for cnt, line in enumerate(fileinput.input(files=[os.path.join(out_dir, 'test-' + name + '.embed')])):
        if cnt == 0:
            lst = line.strip().split()
            assert len(lst) == 2,print('lst:{}'.format(lst))
            print('node_sz:{}, emb_sz:{}'.format(lst[0],lst[1]))
            emb = tc.zeros(int(lst[0]),int(lst[1]))
        else:
            lst = line.strip().split()
            assert len(lst) >= 2
            nid = int(lst[0])
            lst1 = lst[1:]
            nemb = tc.tensor([float(ele) for ele in lst1])
            emb[nid] = nemb
    return emb

def walk2str(walks):
    ret_walks = []
    for walk in walks:
        ret_walks.append([str(ele) for ele in walk])
    return ret_walks

def graph_conj(G):
    lst_conn = []
    max_group = None
    set_sgn = set()
    num_node = G.num_nodes()
    for idx,nid in enumerate(G.nodes()):
        nid = int(nid)
        if idx % 200 == 0:
            print('graph_conj:{}/{}'.format(idx,num_node))
        if nid not in set_sgn:
            new_group = set() # corresponding to close set of A*��
            set_sgn.add(nid)
            new_group.add(nid)
            que_open = [nid]
            while len(que_open) > 0:
                cur = que_open.pop(0)
                for nnid in G.successors(cur):
                    nnid = int(nnid)
                    if nnid not in set_sgn:
                        que_open.append(nnid)
                        set_sgn.add(nnid)
                        new_group.add(nnid)
            lst_conn.append(new_group)
            if max_group is None or len(new_group) > len(max_group):
                max_group = new_group
    assert max_group is not None, print('max group:{}'.format(max_group[:20]))
    assert max_group in lst_conn, print('lst conn sz:{}'.format(len(lst_conn)))
    lst_conn.remove(max_group)
    max_group = list(max_group)
    for e_group in lst_conn:
        nid = random.choice(list(e_group))
        nnid = random.choice(max_group)
        G.add_edge(nid,nnid)
        G.add_edge(nnid,nid)
    return G


def graph_relabel(in_file,out_dict_file):
    '''
    process on src graph which is organized as txt.
    :return:
    '''
    id_dict = {}
    id_cnt = 0
    G = nx.DiGraph()
    with open(in_file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            lst = line.split('\t')
            assert len(lst), print('lst={}'.format(lst))
            src, dst = int(lst[0]), int(lst[1])
            if src not in id_dict:
                id_dict[src] = id_cnt
                id_cnt += 1
            if dst not in id_dict:
                id_dict[dst] = id_cnt
                id_cnt += 1
            if not G.has_node(id_dict[src]):
                G.add_node(id_dict[src])
            if not G.has_node(id_dict[dst]):
                G.add_node(id_dict[dst])
            if not G.has_edge(id_dict[src],id_dict[dst]) and not G.has_edge(id_dict[dst],id_dict[src]):
                G.add_edge(id_dict[src],id_dict[dst])

    with open(out_dict_file,'w') as f:
        for r_nid in id_dict:
            f.write('{}-{}\n'.format(r_nid,id_dict[r_nid]))

    G = G.to_undirected()

    g = dgl.from_networkx(G)
    return g

def circle_path_test(G,root_sz=3,length=10,batch_sz=5,walk_types = ['gen','n2v','surf','bfs','bc'],name='cirpath',is_print=True):
    global glb_g2dmat
    global glb_embed
    global glb_node2BC
    G = nx.to_undirected(G)
    node_sz = [glb_node2BC[name][nod] for nod in G.nodes()]
    sum_node_sz = sum(node_sz)
    node_sz = [ele/sum_node_sz*len(node_sz)*5 for ele in node_sz]
    plt.clf()
    nx.draw(G, node_size=node_sz, with_labels=False, node_color='dodgerblue')
    plt.savefig('../fig/{}-BCGraph.pdf'.format(name))
    if is_print:
        plt.show()
    node_lst = copy.deepcopy(list(G.nodes()))
    random.shuffle(node_lst)
    for walk_type in walk_types:
        for root_idx in range(root_sz):
            root_node = node_lst[root_idx]
            if walk_type == 'gen':
                paths = walk_general(root=root_node,G=G,G_name=name,length=length,batch_sz=batch_sz)
            elif walk_type == 'n2v':
                paths = walk_node2vec(root=root_node, G=G, G_name=name, length=length, batch_sz=batch_sz,p=1,q=1)
            elif walk_type == 'surf':
                paths = walk_surf(root=root_node, G=G, G_name=name, length=length, batch_sz=batch_sz, alpha=0.98)
            elif walk_type == 'bfs':
                paths = walk_node2vec(root=root_node, G=G, G_name=name, length=length, batch_sz=batch_sz, p=10, q=0.1)
            elif walk_type == 'bc':
                paths = walk_bc(root=root_node, G=G, G_name=name, length=length, batch_sz=batch_sz, tau=True)
            node_cs = []
            for node in G.nodes():
                node_cs.append('dodgerblue')
            for path in paths:
                for node in path:
                    node_cs[node] = 'red'
            node_cs[root_node] = 'black'
            plt.clf()
            nx.draw(G, node_size=node_sz, with_labels=False, node_color=node_cs)
            plt.savefig('../fig/{}-{}-root={}.pdf'.format(name,walk_type,root_node))
            if is_print:
                plt.show()


'''
routine
'''
def rt_acc_embed_rate(test_sz=20):
    comm_configs = [(10,1,3,10),(20,1,2,10)]
    # comm_configs = [(3,1,3,10),(3,1,3,5)]
    ret_df = None
    for comm_config in comm_configs:
        cur_df = None
        for i in range(test_sz):
            cls_sz,conn_sz,regular,node_sz = comm_config
            G4 = gen_comm_graph(cls_sz=cls_sz, conn_sz=conn_sz, regular=regular, node_sz=node_sz, name='Graph4',is_print=False)
            print(anal_G_basic([G4],names=['Graph4']))
            print(anal_G_complex([G4],names=['Graph4'],is_print=False))
            anal_embed([G4],['Graph4'],emb_sz=16,walk_len=10,walks_per_node=20,test_src_sz=1,test_batch_sz=1,walk_types=['gen','n2v','surf','bcdr'],is_print=False)
            df = anal_embed_rate([G4],['Graph4'],test_src_sz=100,test_batch_sz=500,is_print=False)
            # anal_embed([G4], ['Graph4'], emb_sz=16, walk_len=6, walks_per_node=10, test_src_sz=1, test_batch_sz=1,walk_types=['gen', 'n2v', 'surf', 'bcdr'], is_print=False)
            # df = anal_embed_rate([G4], ['Graph4'], test_src_sz=5, test_batch_sz=10, is_print=False)
            # print(i,':',df)
            if cur_df is None:
                cur_df = df
            else:
                cur_df = pd.concat([cur_df,df],axis=0)
        assert cur_df is not None
        print(comm_config,':',cur_df)
        new_df = pd.concat([cur_df.mean(),cur_df.std()],axis=1).T
        if ret_df is None:
            ret_df = new_df
        else:
            ret_df = pd.concat([ret_df, new_df], axis=0)
    print('~~~~~~~~~')
    print(ret_df)
    ret_df.to_csv(os.path.join('../fig','anal_embed_rate_ex.csv'),index=True)
    return ret_df

def rt_real_graph_anal():
    data_names = ['facebook']
    for data_name in data_names:
        g = gen_disk_graph(data_name)
        G_cr = dgl.to_networkx(g)
        G_cr = nx.to_undirected(G_cr)
        # drawG(G_cr)
        # plt.savefig(os.path.join('../fig/visual-facebook.pdf'))
        # plt.show()
        print('anal {}...'.format(data_name))
        print(anal_G_basic([G_cr], names=[data_name]))
        print(anal_G_complex([G_cr], names=[data_name]))

        anal_walks_basic([G_cr],G_names=[data_name],walk_len=8,test_src_sz=20,test_batch_sz=10,walk_types=['gen','n2v','surf','dr-in','dr-out','bc'])

def rt_proc_graph_multiconn():
    # oG = gen_disk_graph('cora')
    # nG = graph_conj(oG)
    # # dG = dgl.to_networkx(nG)
    # # dG = dG.to_undirected()
    # # drawG(dG)
    # # plt.savefig('../fig/cora-conn.pdf')
    # # plt.show()
    # dgl.save_graphs('../fig/cora',[nG])
    # assert test_uniconn(nG)

    oG = gen_disk_graph('GrQc')
    nG = graph_conj(oG)
    # dG = dgl.to_networkx(nG)
    # dG = dG.to_undirected()
    # drawG(dG)
    # plt.savefig('../fig/grqc-conn.pdf')
    # plt.show()
    dgl.save_graphs('../fig/GrQc',[nG])
    assert test_uniconn(nG)


def rt_general():
    # G1 = gen_random_graph(regular=2, node_sz=300,name='Graph1')
    # G2 = gen_random_graph(regular=7, node_sz=50,name='Graph2')
    # G3 = gen_random_graph(regular=3, node_sz=100,name='Graph3')
    G4 = gen_comm_graph(cls_sz=20,conn_sz=1,regular=3,node_sz=10,name='Graph4')
    # print(anal_G_basic([G1,G2,G3],names=['Graph1','Graph2','Graph3']))
    # print(anal_G_complex([G1,G2,G3],names=['Graph1','Graph2','Graph3']))

    # print(anal_G_basic([G1],names=['Graph1']))
    # print(anal_G_complex([G1],names=['Graph1']))

    # test_direct(G4)

    print(anal_G_basic([G4],names=['Graph4']))
    print(anal_G_complex([G4],names=['Graph4']))

    # anal_walks_basic([G4],G_names=['Graph4'],walk_len=10,test_src_sz=20,test_batch_sz=10,walk_types=['gen','n2v','surf','bc'])

    anal_embed([G4],['Graph4'],emb_sz=16,walk_len=10,walks_per_node=20,test_src_sz=20,test_batch_sz=100,walk_types=['gen','n2v','surf','bcdr'])
    for i in range(10):
        anal_embed_rate([G4],['Graph4'],test_src_sz=100,test_batch_sz=500)

    # x = [1, 2, 3, 5]  # ���ݼ�
    # plt.boxplot(x)  # ��ֱ��ʾ����ͼ
    # plt.show()  # ��ʾ��ͼ

def rt_super_walk_test():
    # G_cir1 = gen_circle_graph(circle_num=20, circle_sz=[5, 20], conn_sz=1, name='G_Circle')
    # G_tri1 = gen_triangle_graph(triangle_sz=[3,6],conn_sz=1,traingle_num=100,name='G_Triangle')
    # G_tc1 = gen_tricircle_graph(circle_num=10,triangle_sz=[3,6],conn_sz=1,traingle_num=7,name='G_Tri-circle')
    # G_tree1 = gen_tree_graph(child_sz=[1,3],extend_decay=0.95,max_depth=10,name='G_Tree')
    # G_sp1 = gen_spiral_graph(circle_expend=10,st_circle_sz=10,conn_sz=5,name='G_Spiral')
    G_net1 = gen_net_graph(net_shape=[10,10],edge_prob=0.99,name='G_Net')

    G = G_net1
    G_name = 'G_Net'
    print(anal_G_basic([G],names=[G_name]))
    print(anal_G_complex([G],names=[G_name]))
    anal_walks_basic([G],G_names=[G_name],walk_len=15,test_src_sz=20,test_batch_sz=10,walk_types=['gen','n2v','surf','bfs','bc'])

def rt_circle_walk_test_independent():
    # circle_rates = np.linspace(0.05,0.9,3)
    circle_sz_means = range(3,33,3)
    G_cirs = []
    G_cir_crates = []
    exp_ratio_arrs_dict = {}
    for idx,circle_sz_mean in enumerate(circle_sz_means):
        # nod_mean = 1 / crate
        # nod_var = 1
        # cir_sz_min = max(1, int(nod_mean - nod_var))
        # cir_sz_max = max(cir_sz_min+1,int(nod_mean + nod_var))
        G_name = 'G_Independent_{}'.format(idx)
        # G_cir,cir_cnt = gen_circleX_graph(circle_num=15, circle_sz=[cir_sz_min,cir_sz_max], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)
        G_cir,cir_cnt = gen_circleX_graph(circle_num=8, circle_sz=[max(3,circle_sz_mean-1),max(4,circle_sz_mean+5)], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)

        G_cirs.append(G_cir)
        G_cir_crates.append(cir_cnt / len(G_cir.nodes()))
        print(anal_G_basic([G_cir], names=[G_name]))
        df = anal_G_complex([G_cir], names=[G_name])
        print(df)
        cur_d = int(df['diameter'][0])
        score_dict = anal_walks_basic([G_cir], G_names=[G_name], walk_len=math.ceil(cur_d*0.5), test_src_sz=20, test_batch_sz=5,walk_types=['gen', 'n2v', 'surf', 'bfs', 'bc'],is_print=False)
        for name in score_dict:
            if name not in exp_ratio_arrs_dict:
                exp_ratio_arrs_dict[name] = []
            cur_exp_ratio_arr = []
            for e_arr in score_dict[name]:
                # cur_exp_ratio_arr.append(max(e_arr) / cur_d)
                cur_exp_ratio_arr.extend([ele / math.ceil(cur_d*0.5) for ele in e_arr])
            # cur_exp_ratio_arr = list(reversed(sorted(cur_exp_ratio_arr)))
            random.shuffle(cur_exp_ratio_arr)
            cur_exp_ratio_arr = cur_exp_ratio_arr[:math.ceil(cur_d*0.5)*1]
            exp_ratio_arrs_dict[name].append(cur_exp_ratio_arr)

    for name in exp_ratio_arrs_dict:
        plt.clf()
        for crate,exp_ratio_arr in zip(G_cir_crates,exp_ratio_arrs_dict[name]):
            print('len of cur :{}'.format(len(cur_exp_ratio_arr)))
            plt.scatter(x=[crate]*len(exp_ratio_arr),y=exp_ratio_arr,c='dodgerblue',alpha=min(1,10 / len(cur_exp_ratio_arr)))
        plt.plot([0, max(G_cir_crates)], [1, 1], ':', c='black', label='diameter')

        plt.xlabel('Circles / Nodes')
        plt.ylabel('Exploration Distance Ratio')
        plt.ylim(0, 1.2)
        plt.savefig('../fig/CW_test_independent_{}.pdf'.format(name))
        plt.show()

def rt_circle_walk_test_cross():
    circle_rates = np.linspace(0.05,0.9,15)
    # circle_sz_means = range(3,33,10)
    G_cirs = []
    G_cir_crates = []
    exp_ratio_arrs_dict = {}
    for idx,crate in enumerate(circle_rates):
        # nod_mean = 1 / crate
        # nod_var = 1
        # cir_sz_min = max(1, int(nod_mean - nod_var))
        # cir_sz_max = max(cir_sz_min+1,int(nod_mean + nod_var))
        G_name = 'G_Cross_{}'.format(idx)
        # G_cir,cir_cnt = gen_circleX_graph(circle_num=15, circle_sz=[cir_sz_min,cir_sz_max], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)
        G_cir,cir_cnt = gen_circleX_graph(circle_num=40, circle_sz=[3,10], conn_sz=1,cross_num=max(0,int(4*40*crate)),inner_num=0, name=G_name,is_cir_cnt=True)

        G_cirs.append(G_cir)
        G_cir_crates.append(cir_cnt / len(G_cir.nodes()))
        print(anal_G_basic([G_cir], names=[G_name]))
        df = anal_G_complex([G_cir], names=[G_name])
        print(df)
        cur_d = int(df['diameter'][0])
        score_dict = anal_walks_basic([G_cir], G_names=[G_name], walk_len=math.ceil(cur_d*0.5), test_src_sz=20, test_batch_sz=5,walk_types=['gen', 'n2v', 'surf', 'bfs', 'bc'],is_print=False)
        for name in score_dict:
            if name not in exp_ratio_arrs_dict:
                exp_ratio_arrs_dict[name] = []
            cur_exp_ratio_arr = []
            for e_arr in score_dict[name]:
                # cur_exp_ratio_arr.append(max(e_arr) / cur_d)
                cur_exp_ratio_arr.extend([ele / math.ceil(cur_d*0.5) for ele in e_arr])
            # cur_exp_ratio_arr = list(reversed(sorted(cur_exp_ratio_arr)))
            random.shuffle(cur_exp_ratio_arr)
            cur_exp_ratio_arr = cur_exp_ratio_arr[:math.ceil(cur_d * 0.5) * 1]
            exp_ratio_arrs_dict[name].append(cur_exp_ratio_arr)

    for name in exp_ratio_arrs_dict:
        plt.clf()
        for crate,exp_ratio_arr in zip(G_cir_crates,exp_ratio_arrs_dict[name]):
            plt.scatter(x=[crate]*len(exp_ratio_arr),y=exp_ratio_arr,c='dodgerblue',alpha=min(1,2 / len(cur_exp_ratio_arr)))
        plt.plot([0, max(G_cir_crates)], [1, 1], ':', c='black', label='diameter')

        plt.xlabel('Circles / Nodes')
        plt.ylabel('Exploration Distance Ratio')
        plt.ylim(0, 1.2)
        plt.savefig('../fig/CW_test_cross_{}.pdf'.format(name))
        plt.show()

def rt_circle_walk_test_inner():
    circle_rates = np.linspace(0.05,0.9,15)
    # circle_sz_means = range(3,33,10)
    G_cirs = []
    G_cir_crates = []
    exp_ratio_arrs_dict = {}
    for idx,crate in enumerate(circle_rates):
        # nod_mean = 1 / crate
        # nod_var = 1
        # cir_sz_min = max(1, int(nod_mean - nod_var))
        # cir_sz_max = max(cir_sz_min+1,int(nod_mean + nod_var))
        G_name = 'G_Inner_{}'.format(idx)
        # G_cir,cir_cnt = gen_circleX_graph(circle_num=15, circle_sz=[cir_sz_min,cir_sz_max], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)
        G_cir,cir_cnt = gen_circleX_graph(circle_num=20, circle_sz=[10,20], conn_sz=1,cross_num=0,inner_num=max(0,int(4*40*crate)), name=G_name,is_cir_cnt=True)

        G_cirs.append(G_cir)
        G_cir_crates.append(cir_cnt / len(G_cir.nodes()))
        print(anal_G_basic([G_cir], names=[G_name]))
        df = anal_G_complex([G_cir], names=[G_name])
        print(df)
        cur_d = int(df['diameter'][0])
        score_dict = anal_walks_basic([G_cir], G_names=[G_name], walk_len=math.ceil(cur_d*0.5), test_src_sz=20, test_batch_sz=5,walk_types=['gen', 'n2v', 'surf', 'bfs', 'bc'],is_print=False)
        for name in score_dict:
            if name not in exp_ratio_arrs_dict:
                exp_ratio_arrs_dict[name] = []
            cur_exp_ratio_arr = []
            for e_arr in score_dict[name]:
                # cur_exp_ratio_arr.append(max(e_arr) / cur_d)
                cur_exp_ratio_arr.extend([ele / math.ceil(cur_d*0.5) for ele in e_arr])
            # cur_exp_ratio_arr = list(reversed(sorted(cur_exp_ratio_arr)))
            random.shuffle(cur_exp_ratio_arr)
            cur_exp_ratio_arr = cur_exp_ratio_arr[:math.ceil(cur_d * 0.5) * 1]
            exp_ratio_arrs_dict[name].append(cur_exp_ratio_arr)

    for name in exp_ratio_arrs_dict:
        plt.clf()
        for crate,exp_ratio_arr in zip(G_cir_crates,exp_ratio_arrs_dict[name]):
            plt.scatter(x=[crate]*len(exp_ratio_arr),y=exp_ratio_arr,c='dodgerblue',alpha=min(1,4 / len(cur_exp_ratio_arr)))
        plt.plot([0, max(G_cir_crates)], [1, 1], ':', c='black', label='diameter')

        plt.xlabel('Circles / Nodes')
        plt.ylabel('Exploration Distance Ratio')
        plt.ylim(0,1.2)
        plt.savefig('../fig/CW_test_inner_{}.pdf'.format(name))
        plt.show()

def rt_circle_walk_test_disk():
    circle_rates = np.linspace(0.05,0.9,15)
    # circle_sz_means = range(3,33,10)
    G_cirs = []
    G_cir_crates = []
    exp_ratio_arrs_dict = {}
    for idx,crate in enumerate(circle_rates):
        # nod_mean = 1 / crate
        # nod_var = 1
        # cir_sz_min = max(1, int(nod_mean - nod_var))
        # cir_sz_max = max(cir_sz_min+1,int(nod_mean + nod_var))
        G_name = 'G_Disk_{}'.format(idx)
        # G_cir,cir_cnt = gen_circleX_graph(circle_num=15, circle_sz=[cir_sz_min,cir_sz_max], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)
        G_cir,cir_cnt = gen_disk_like_graph(circle_num=max(1,int(crate*250)),node_sz=250,name=G_name,is_print=True,is_cir_cnt=True)

        G_cirs.append(G_cir)
        G_cir_crates.append(cir_cnt / len(G_cir.nodes()))
        print(anal_G_basic([G_cir], names=[G_name]))
        df = anal_G_complex([G_cir], names=[G_name])
        print(df)
        cur_d = int(df['diameter'][0])
        score_dict = anal_walks_basic([G_cir], G_names=[G_name], walk_len=math.ceil(cur_d*0.5), test_src_sz=20, test_batch_sz=5,walk_types=['gen', 'n2v', 'surf', 'bfs', 'bc'],is_print=False)
        for name in score_dict:
            if name not in exp_ratio_arrs_dict:
                exp_ratio_arrs_dict[name] = []
            cur_exp_ratio_arr = []
            for e_arr in score_dict[name]:
                # cur_exp_ratio_arr.append(max(e_arr) / cur_d)
                cur_exp_ratio_arr.extend([ele / math.ceil(cur_d*0.5) for ele in e_arr])
            # cur_exp_ratio_arr = list(reversed(sorted(cur_exp_ratio_arr)))
            random.shuffle(cur_exp_ratio_arr)
            cur_exp_ratio_arr = cur_exp_ratio_arr[:math.ceil(cur_d * 0.5) * 1]
            exp_ratio_arrs_dict[name].append(cur_exp_ratio_arr)

    for name in exp_ratio_arrs_dict:
        plt.clf()
        for crate,exp_ratio_arr in zip(G_cir_crates,exp_ratio_arrs_dict[name]):
            plt.scatter(x=[crate]*len(exp_ratio_arr),y=exp_ratio_arr,c='dodgerblue',alpha=min(1,2.5 / len(cur_exp_ratio_arr)))
        plt.plot([0, max(G_cir_crates)], [1, 1], ':', c='black', label='diameter')

        plt.xlabel('Circles / Nodes')
        plt.ylabel('Exploration Distance Ratio')
        plt.ylim(0, 1.2)
        plt.savefig('../fig/CW_test_disk_{}.pdf'.format(name))
        plt.show()

def rt_circle_path_visual():
    circle_sz_mean = 5
    crate=0.2
    # G_name = 'CP-independent'
    # G,cir_cnt = gen_circleX_graph(circle_num=8, circle_sz=[max(3,circle_sz_mean-1),max(4,circle_sz_mean+1)], conn_sz=1,cross_num=0,inner_num=0, name=G_name,is_cir_cnt=True)

    # G_name = 'CP-disk'
    # G, cir_cnt = gen_disk_like_graph(circle_num=max(1, int(crate * 250)), node_sz=250, name=G_name, is_print=True,
    #                                      is_cir_cnt=True)
    #
    # G_name = 'CP-Cross'
    # G, cir_cnt = gen_circleX_graph(circle_num=40, circle_sz=[3, 10], conn_sz=1,
    #                                    cross_num=max(0, int(4 * 40 * crate)), inner_num=0, name=G_name, is_cir_cnt=True)
    #
    # G_name = 'CP-Inner'
    # G, cir_cnt = gen_circleX_graph(circle_num=20, circle_sz=[10, 20], conn_sz=1, cross_num=0,
    #                                    inner_num=max(0, int(4 * 40 * crate)), name=G_name, is_cir_cnt=True)
    #
    G_name = 'CP-Disk'
    G, cir_cnt = gen_disk_like_graph(circle_num=max(1, int(crate * 250)), node_sz=250, name=G_name, is_print=True,
                                         is_cir_cnt=True)

    print(anal_G_basic([G], names=[G_name]))
    df = anal_G_complex([G], names=[G_name])
    print(df)
    circle_path_test(G,root_sz=3,length=10,batch_sz=3,walk_types=['gen','n2v','surf','bfs','bc'],name=G_name,is_print=True)

'''
parallel routine.
'''
def prt_acc_embed_rate(test_sz=20,workers=8):
    pass

if __name__ == '__main__':

    # G1 = gen_comm_graph(cls_sz=20, conn_sz=1, regular=3, node_sz=10, name='Graph1')
    # G2 = gen_comm_graph(cls_sz=20, conn_sz=1, regular=3, node_sz=10, name='Graph2')
    # G3 = gen_comm_graph(cls_sz=20, conn_sz=1, regular=3, node_sz=10, name='Graph3')

    # G_cir1 = gen_circle_graph(circle_num=20,circle_sz=[5,20], conn_sz=1, name='G_cir1')
    # G_tri1 = gen_triangle_graph(triangle_sz=[3,6],conn_sz=1,traingle_num=100,name='G_tri')
    # G_tc1 = gen_tricircle_graph(circle_num=10,triangle_sz=[3,6],conn_sz=1,traingle_num=7,name='G_tc1')
    # G_tree1 = gen_tree_graph(child_sz=[1,5],extend_decay=0.9,max_depth=7,name='G_tree1')
    # G_sp1 = gen_spiral_graph(circle_expend=10,st_circle_sz=10,conn_sz=5,name='G_sp1')
    # G_net1 = gen_net_graph(net_shape=[20,10],edge_prob=0.99,name='G_net1')

    # rt_general()
    # rt_acc_embed_rate(test_sz=10)
    # rt_real_graph_anal()
    # rt_proc_graph_multiconn()

    # rt_super_walk_test()
    # rt_circle_walk_test_independent()
    # rt_circle_walk_test_cross()
    # rt_circle_walk_test_inner()
    # rt_circle_walk_test_disk()
    #
    # rt_circle_path_visual()

    # crate = 0.9
    # G_cir, cir_cnt = gen_disk_like_graph(circle_num=max(1, int(crate * 250)), node_sz=250, is_print=True,
    #                                      is_cir_cnt=True)
    # print(len(G_cir.nodes()),cir_cnt)

    # graph_relabel('../datasets/src/GrQc/GrQc.txt','../datasets/src/GrQc/GrQc-ReL.txt')

    # g,_ = dgl.load_graphs('../datasets/dst/DBLP')
    # g= g[0]
    # print(g.nodes())
    # test_uniconn(g,is_fast=True)

    # rt_real_graph_anal()
    # G,_ = dgl.load_graphs('../datasets/dst/DBLP')
    # G = G[0]
    anal_BC_fast(G_file='../datasets/dst/DBLP',name='DBLP-bc',num_worker=8)