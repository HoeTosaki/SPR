import dgl
import m_selector
import math
import os
import time
import m_encoder
import m_decoder
import torch.nn as nn
from torch.autograd import Variable
import dgl.nn as dglnn
import torch.nn.functional as F
import torch as tc
import embedding_proc
# entities
class OracleGraph:
    def __init__(self,g,depth=-1,cluster_sz=10,cluster_num=1000,proto_sz=-1,expand_speed=1,need_remove_dup=True,out_file='og'):
        '''
        :param g: 输入原始图
        :param depth: OracleGraph的深度，可以省略，此时深度由最上级图的规模proto_sz确定
        :param cluster_sz: 一次聚类的节点数建议大小
        :param cluster_num: 一次聚类的总类别数建议大小
        :param proto_sz: OracleGraph最上级图的规模，可以省略，此时将由OracleGraph的深度确定
        :param expand_speed:默认每个cluster一次只扩展一个节点，提高可以加快此构造进程，但在cluster_sz / graph_sz较小时误差较大
        :param need_remove_dup:是否对有一阶proximity的cluster center进行去重，会导致实际筛选的cluster sz比设定值略小，且每层节点数量会有波动。
        '''
        self.org = g
        self.depth = depth
        self.cluster_sz = cluster_sz
        self.cluster_num = cluster_num
        self.proto_sz = proto_sz
        self.gs = [self.org]
        self.need_remove_dup = need_remove_dup
        self.out_file=out_file
        assert self.cluster_sz == -1 or self.cluster_num == -1
        assert not (self.cluster_sz == -1 and self.cluster_num == -1)
        # assert self.depth == -1 or self.proto_sz == -1
        assert not (self.depth == -1 and self.proto_sz == -1)
        self.upper_maps = [] # 从当前层的中心节点到上层节点的双射
        self.lower_maps = [] # 从当前层节点到下层中心节点的双射（指针向上跃进一层，以简化代码逻辑）
        self.inner_maps = [] # 从当前层的任意节点到中心节点的满射
        self.outer_maps = [] # 从当前层的中心节点到从属点集的映射
    def gen_oracle(self,force=False):
        if self._check_file(self.out_file):
            self.load(self.out_file)
            print('load oracle graph succeed.')
            return
        print('start to gen oracle...')
        st_time = time.time()
        cur_g = self.org
        # cur_g = dgl.DGLGraph()
        cur_depth = 0
        while True:
            if self.depth != -1 and cur_depth >= self.depth:
                print('construction break as cur depth={}'.format(cur_depth + 1))
                break
            if self.proto_sz != -1 and cur_g.num_nodes() <= self.proto_sz:
                print('construction break as cur graph ndoe sz:{} <= proto_sz:{}'.format(cur_g.num_nodes(),self.proto_sz))
                break
            cur_g,upper_map,lower_map,inner_map,outer_map = self.gen_next_layer(cur_g=cur_g)
            self.gs.append(cur_g)
            self.upper_maps.append(upper_map)
            self.lower_maps.append(lower_map)
            self.inner_maps.append(inner_map)
            self.outer_maps.append(outer_map)
            cur_depth += 1
            print('depth',cur_depth,'g',cur_g)
        print('gen oracle graph finished by {:.2f}s'.format(time.time()-st_time))
        self.save(self.out_file)
    def gen_next_layer(self,cur_g):
        # cur_g = dgl.DGLGraph()

        next_g = dgl.DGLGraph()
        upper_map = {} # from cur_g to next_g
        lower_map = {} # from next_g to cur_g
        inner_map = {} # among cur_g
        outer_map = {} # among cur_g

        # sample seed nodes & avoid first proximity relation among seeds. (large-deg node will be fine.)
        ds = m_selector.DegreeSelector(g=cur_g)
        tmp_lst_nodes = []
        if self.cluster_sz != -1:
            tmp_lst_nodes = ds.perform(cnt=cur_g.num_nodes() // self.cluster_sz, action='max').tolist()
        else:
            tmp_lst_nodes = ds.perform(cnt=math.floor(cur_g.num_nodes() * self.cluster_num), action='max').tolist()
        lst_nodes = []
        assert type(tmp_lst_nodes) == list
        if self.need_remove_dup:
            dup_set = set()
            for node in tmp_lst_nodes:
                if node in dup_set:
                    continue
                dup_set.add(node)
                dup_set.update(cur_g.successors(node).tolist())
                lst_nodes.append(node)
        else:
            lst_nodes = tmp_lst_nodes
        print('\tcur layer sampled {} seeds factually.'.format(len(lst_nodes)))
        next_g = dgl.DGLGraph()
        next_g.add_nodes(len(lst_nodes))
        assert next_g.num_nodes() == len(lst_nodes)
        for cur_node,next_node in zip(lst_nodes,next_g.nodes().tolist()):
            upper_map[cur_node] = next_node
            lower_map[next_node] = cur_node
            inner_map[cur_node] = cur_node
            outer_map[cur_node] = set()
            outer_map[cur_node].add(cur_node)

        dic_query_dic = {}
        need_stop = False
        while not need_stop:
            need_stop = True
            cnt_lst = len(lst_nodes)
            for idx,node in enumerate(lst_nodes):
                if self.expand_once(cur_node=node,cur_g=cur_g,next_g=next_g,upper_map=upper_map,lower_map=lower_map,inner_map=inner_map,outer_map=outer_map,dic_query_dic=dic_query_dic):
                    need_stop=False
                if idx %100 == 0:
                    print('{}/{}'.format(idx,cnt_lst))
        next_g = dgl.to_simple(next_g)
        next_g = dgl.to_bidirected(next_g)
        return next_g,upper_map,lower_map,inner_map,outer_map
    def expand_once(self,cur_node,cur_g,next_g,upper_map,lower_map,inner_map,outer_map,dic_query_dic):
        if cur_node not in dic_query_dic:
            # first expand
            dic_query_dic[cur_node] = {}
            for node in cur_g.successors(cur_node).tolist():
                dic_query_dic[cur_node][node] = 2 - len(cur_g.successors(node).tolist())
        # find the most strongly connected node in query list.
        isEmptyLoop = True
        max_score = -1
        max_idx = -1
        later_removed_key = []
        for node in dic_query_dic[cur_node]:
            # exlude node which was occupied.
            if node in inner_map:
                # assert inner_map[node] != cur_node
                if inner_map[node] != cur_node:
                    later_removed_key.append(node)
                    cur_center = upper_map[cur_node]
                    other_center = upper_map[inner_map[node]]
                    if not next_g.has_edges_between(cur_center,other_center):
                        next_g.add_edges(cur_center,other_center)
                continue
            cur_score = dic_query_dic[cur_node][node]
            if isEmptyLoop:
                max_score = cur_score
                max_idx = node
                isEmptyLoop = False
            elif cur_score > max_score:
                max_score = cur_score
                max_idx = node
        for key in later_removed_key:
            del dic_query_dic[cur_node][key]
        if isEmptyLoop:
            #all queried nodes are occupied by other cluster.
            # print('cluster',cur_node,'expand finished.')
            return False
        inner_map[max_idx] = cur_node
        outer_map[cur_node].add(max_idx)
        for node in cur_g.successors(max_idx).tolist():
            if node in inner_map:
                if inner_map[node] == cur_node:
                    # self node.
                    continue
                else:
                    # other cluster's node.
                    # add it, causing next_g edges updated.
                    pass
            if node in dic_query_dic[cur_node]:
                dic_query_dic[cur_node][node] += 1
            else:
                dic_query_dic[cur_node][node] = 2 - len(cur_g.successors(node).tolist())
        # print('cluster',cur_node,'expands',max_idx,'by',max_score)
        del dic_query_dic[cur_node][max_idx]
        return True

    def save(self,file):
        print('start to save oracle graphs...')
        st_time = time.time()
        dgl.save_graphs(file+'.graphs', self.gs)
        with open(file+'.maps','w') as f:
            data = ''
            for idx,map in enumerate(self.upper_maps):
                data += self._archive_dict('upper_map'+str(idx),map)
            for idx, map in enumerate(self.lower_maps):
                data += self._archive_dict('lower_map' + str(idx), map)
            for idx, map in enumerate(self.inner_maps):
                data += self._archive_dict('inner_map' + str(idx), map)
            for idx, map in enumerate(self.outer_maps):
                data += self._archive_dict('outer_map' + str(idx), map,is_2nd_set=True)
            f.write(data)
        print('save oracle graphs finished by {:.2f}s'.format(time.time()-st_time))

    def load(self,file):
        print('start to load oracle graphs...')
        st_time = time.time()

        gs,_ = dgl.load_graphs(file+'.graphs')
        self.gs = gs
        with open(file+'.maps','r') as f:
            lst = f.readlines()
            # assert len(lst) == 1
            if len(lst) == 1:
                data_dict = self._unarchive_dict(lst[0])
                assert len(data_dict.keys()) == (len(self.gs)-1) * 4
                self.upper_maps = []
                self.lower_maps = []
                self.inner_maps = []
                self.outer_maps = []
                for i in range(len(self.gs) - 1):
                    self.upper_maps.append(data_dict['upper_map' + str(i)])
                    self.lower_maps.append(data_dict['lower_map' + str(i)])
                    self.inner_maps.append(data_dict['inner_map' + str(i)])
                    self.outer_maps.append(data_dict['outer_map' + str(i)])
            else:
                print('Warning: read map illegal.')
                self.upper_maps = []
                self.lower_maps = []
                self.inner_maps = []
                self.outer_maps = []
        print('load oracle graphs finished by {:.2f}s'.format(time.time() - st_time))

    def _check_file(self,file):
        if os.path.exists(file+'.graphs') and os.path.exists(file+'.maps'):
            return True
        return False

    def _archive_dict(self,dict_key,dict,is_2nd_set=False):
        content_str = str(dict_key)+'$'
        for key in dict:
            if is_2nd_set:
                inner_str = ''
                for inner_ele in dict[key]:
                    if type(inner_ele) != int:
                       print('not support 3rd dict / set but int.')
                       assert False
                    inner_str += str(inner_ele)+'-'
                content_str += str(key) + ':' + inner_str + ','
            else:
                content_str += str(key)+':'+str(dict[key])+','
        content_str += '#'
        return content_str
    def _unarchive_dict(self,str):
        ret_dict = {}
        lst1 = str.strip().split('#')
        for str2 in lst1:
            #test if is_2nd_set
            is_2nd_set = False
            if str2.__contains__('-'):
                is_2nd_set = True
            if str2.strip() == '':
                continue
            lst2 = str2.strip().split('$')
            assert len(lst2) == 2
            dict_key = lst2[0]
            dict_content = {}
            lst3 = lst2[1].strip().split(',')
            for str4 in lst3:
                if str4.strip() == '':
                    continue
                lst4 = str4.strip().split(':')
                assert len(lst4) == 2
                if is_2nd_set:
                    lst5 = lst4[1].strip().split('-')
                    inner_set = set()
                    for inner_ele in lst5:
                        if inner_ele.strip() == '':
                            continue
                        inner_set.add(int(inner_ele))
                    dict_content[int(lst4[0])] = inner_set
                else:
                    dict_content[int(lst4[0])] = int(lst4[1])
            ret_dict[dict_key] = dict_content
        return ret_dict

    def graph(self, idx):
        return self.gs[idx]

    def upper_map(self, depth, nid):
        return self.upper_maps[depth][nid]

    def inner_map(self, depth, nid):
        return self.inner_maps[depth][nid]

    def outer_map(self, depth, nid):
        return self.outer_maps[depth][nid]

    def lower_map(self, depth, nid):
        return self.lower_maps[depth][nid]


class SmGnnEncoder(m_encoder.Encoder):
    def __init__(self, g, emb_sz=128, workers=1, out_dir='../outputs', out_file='encoder', force=False, **oracle_params):
        super(SmGnnEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.oracle_params = oracle_params
        self.graphEncoders = []
        self.og = None
        # m_encoder.DeepWalkEncoder(g=None, emb_sz=emb_sz, workers=workers, out_dir=out_dir, out_file=out_file,
        #                           force=force, num_walks=, walk_lens=, window_sz=, max_mem=0, seed=0,
        #                           is_dense_degree=False)
    def config_OracleGraph(self,**oracle_params):
        self.oracle_params = oracle_params
    def train(self):
        # check file
        self.oracle_params['out_file'] = os.path.join(self.out_dir,self.out_file)
        self.og = OracleGraph(**self.oracle_params)
        if not self.force and self.check_file():
            print('embedding file checked.')
            return

        # calc. oracle graph
        print('start to calc oracle graph...')
        st_time = time.time()
        self.og.gen_oracle()
        print('calc oracle graph finished by {:.2f}'.format(time.time()-st_time))

        # calc. initial emb for each graph layer.
        print('start to embedding graph layer...')
        st_time = time.time()
        self.proc_deepwalk()
        print('start to embedding graph layer finished by {:2f}'.format(time.time()-st_time))

        #save embedding.
        self.save()

    def proc_deepwalk(self):
        # self.og = OracleGraph()
        for idx,g in enumerate(self.og.gs):
            # g = dgl.DGLGraph()
            is_dense = False
            # if g.num_edges() > g.num_nodes() * 3:
            #     print('\tgraph {} has more edges({}) to nodes({}), use dense matrix'.format(idx,g.num_edges(),g.num_nodes()))
            #     is_dense = True
            num_walks = 80
            walk_lens = 40
            window_sz = 20
            dw = m_encoder.DeepWalkEncoder(g=g, emb_sz=self.emb_sz, workers=self.workers, out_dir=self.out_dir, out_file=self.out_file+'-'+str(idx),force=self.force, num_walks=num_walks, walk_lens=walk_lens, window_sz=window_sz, max_mem=0, seed=0,is_dense_degree=is_dense)
            print('\tstart train for graph {} ...'.format(idx))
            st_time = time.time()
            dw.train()
            self.graphEncoders.append(dw)
            print('\ttrain for graph {} finished by {:2f}'.format(idx,time.time()-st_time))

    def save(self):
        self.og.save(self.og.out_file)

    def check_file(self):
        has_og = self.og._check_file(self.og.out_file)
        if has_og:
            self.og.gen_oracle()
            print('oracle graph loaded, check emb for g_0',self.og.graph(0))
            return 'emb' in self.og.graph(0).ndata
        return False
    def _load_emb(self):
        for i in range(len(self.og.gs)):
            emb = embedding_proc.Embedding(os.path.join(self.out_dir, self.out_file+'-'+str(i)))
            if not emb.check_with_graph(self.og.graph(i)):
                print('warning: cur emb not match oracle graph at {}'.format(i))
                emb.debug_with_graph(self.og.graph(i))
            else:
                emb.add_to_graph(self.og.graph(i))
    def get_embedded_og(self):
        self.og.gen_oracle()
        self._load_emb()
        return self.og

class SmGnnDecoder(m_decoder.Decoder):
    def __init__(self,sub_type='condonly',**kwargs):
        super(SmGnnDecoder, self).__init__()
        if sub_type == 'condonly':
            self.decoder = SmGnnDecoder_condonly(**kwargs)
        elif sub_type == 'single':
            self.decoder = SmGnnDecoder_single(**kwargs)
        elif sub_type == 'range':
            self.decoder = SmGnnDecoder_range(**kwargs)
        else:
            assert NotImplementedError
    def forward(self, **inputs):
        return self.decoder.forward(**inputs)

class SmGnnDecoder_single(m_decoder.Decoder):
    def __init__(self, emb_sz,og, deep_dict={}):
        super(SmGnnDecoder_single, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.og = og
        # self.og = OracleGraph()
        self.convs = self.create_gnn_layers()
        self.lin_embs = [nn.Linear(emb_sz*2, emb_sz // 4) for i in range(len(self.og.gs))]
        self.lin_output1 = nn.Linear(emb_sz*len(self.og.gs)*(emb_sz//4),emb_sz) # solely or duplicate?
        self.lin_output2 = nn.Linear(emb_sz, emb_sz // 4)
        self.lin_output3 = nn.Linear(emb_sz // 4, 1)

    def create_gnn_layers(self):
        layers = [{} for i in range(len(self.og.gs))]
        for idx,layer in enumerate(layers):
            if idx == len(layers) - 1:
                continue
            for key in self.og.outer_maps[idx]:
                layer[key] = dglnn.SAGEConv(in_feats=self.emb_sz,out_feats=self.emb_sz,aggregator_type='mean')
        layers[-1][-1] = dglnn.SAGEConv(in_feats=self.emb_sz,out_feats=self.emb_sz,aggregator_type='mean')
        return layers

    def forward(self, src_dst):
        # self.og = OracleGraph()
        cur_depth = 0
        comb_src_emb = []
        comb_dst_emb = []
        while True:
            if cur_depth >= len(self.og.gs):
                break
            if cur_depth == len(self.os.gs) - 1:
                sub_g = self.og.graph(cur_depth)
                sub_g_emb = self.convs[cur_depth][-1](sub_g, sub_g.ndata['emb'])
                src_emb = sub_g_emb[src]
                dst_emb = sub_g_emb[dst]
                comb_src_emb.append(src_emb)
                comb_dst_emb.append(dst_emb)
            elif self.og.inner_map(cur_depth,src) == self.og.inner_map(cur_depth,dst):
                # same graph at current depth.
                sub_g_cen = self.og.inner_map(cur_depth, src)
                sub_g = dgl.node_subgraph(self.og.graph(cur_depth), self.og.outer_map(cur_depth, sub_g_cen))
                sub_g_emb = self.convs[cur_depth][sub_g_cen](sub_g, sub_g.ndata['emb'])
                src_emb = sub_g_emb[src]
                dst_emb = sub_g_emb[dst]
                sub_g_cen_emb = sub_g_emb[sub_g_cen]
                src_rel_emb = self.lin_embs[cur_depth](src_emb, sub_g_cen_emb)
                dst_rel_emb = self.lin_embs[cur_depth](dst_emb,sub_g_cen_emb)
                comb_src_emb.append(src_rel_emb)
                comb_dst_emb.append(dst_rel_emb)
                break
            else:
                src_sub_g_cen = self.og.inner_map(cur_depth,src)
                src_sub_g = dgl.node_subgraph(self.og.graph(cur_depth),self.og.outer_map(cur_depth,src_sub_g_cen))
                src_sub_g_emb = self.convs[cur_depth][src_sub_g_cen](src_sub_g,src_sub_g.ndata['emb'])
                src_emb = src_sub_g_emb[src]
                src_sub_g_cen_emb = src_sub_g_emb[src_sub_g_cen]
                src_rel_emb = self.lin_embs[cur_depth](src_emb,src_sub_g_cen_emb)
                # src_rel_emb = F.relu(src_rel_emb)
                # 这里可以组合多个gnn，目前先看性能吧
                comb_src_emb.append(src_rel_emb)

                dst_sub_g_cen = self.og.inner_map(cur_depth, dst)
                dst_sub_g = dgl.node_subgraph(self.og.graph(cur_depth), self.og.outer_map(cur_depth, dst_sub_g_cen))
                dst_sub_g_emb = self.convs[cur_depth][dst_sub_g_cen](dst_sub_g, dst_sub_g.ndata['emb'])
                dst_emb = dst_sub_g_emb[dst]
                dst_sub_g_cen_emb = dst_sub_g_emb[dst_sub_g_cen]
                dst_rel_emb = self.lin_embs[cur_depth](dst_emb, dst_sub_g_cen_emb)
                comb_dst_emb.append(dst_rel_emb)
            src = self.og.inner_map(cur_depth,src)
            src = self.og.upper_map(cur_depth,src)
            dst = self.og.inner_map(cur_depth,dst)
            dst = self.og.upper_map(cur_depth,dst)
            cur_depth += 1
        ret_src_emb = tc.cat(comb_src_emb,dim=1)
        ret_dst_emb = tc.cat(comb_dst_emb,dim=1)
        ret_emb = tc.cat([ret_src_emb,ret_dst_emb],dim = 1)
        ret_emb = self.lin_output1(ret_emb)
        ret_emb = F.relu(ret_emb)
        ret_emb = self.lin_output2(ret_emb)
        ret_emb = F.relu(ret_emb)
        ret_emb = self.lin_output3(ret_emb)
        return ret_emb

class SmGnnDecoder_range(m_decoder.Decoder):
    def __init__(self, emb_sz,og, deep_dict={}):
        raise NotImplementedError
    def forward(self, inputs):
        raise NotImplementedError

class SmGnnDecoder_condonly(m_decoder.Decoder):
    def __init__(self, emb_sz,og, deep_dict={}):
        super(SmGnnDecoder_condonly, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.og = og
        # self.og = OracleGraph()
        self.convs = None
        self.lin_embs = [nn.Linear(emb_sz*2, emb_sz // 4) for i in range(len(self.og.gs))]
        self.lin_output1 = nn.Linear((len(self.og.gs)+3)*(emb_sz//4)*2,emb_sz) # solely or duplicate?
        self.lin_output2 = nn.Linear(emb_sz, emb_sz // 4)
        self.lin_output3 = nn.Linear(emb_sz // 4, 1)

        for idx,lin in enumerate(self.lin_embs):
            setattr(self,'node_embed' + str(idx),lin)

    def forward(self, srcs,dsts):
        src_lst = srcs.tolist()
        dst_lst = dsts.tolist()
        ret_emb_lst = None
        for src,dst in zip(src_lst,dst_lst):
            # self.og = OracleGraph()
            cur_depth = 0
            comb_src_emb = []
            comb_dst_emb = []
            while True:
                if cur_depth >= len(self.og.gs):
                    break
                if cur_depth == len(self.og.gs) - 1:
                    sub_g = self.og.graph(cur_depth)
                    src_emb = self.og.graph(cur_depth).ndata['emb'][src].view(1,-1)
                    dst_emb = self.og.graph(cur_depth).ndata['emb'][dst].view(1,-1)
                    comb_src_emb.append(src_emb)
                    comb_dst_emb.append(dst_emb)
                    # print('break!')
                    break
                elif self.og.inner_map(cur_depth,src) == self.og.inner_map(cur_depth,dst):
                    # same graph at current depth.
                    # print('one patch!!')
                    sub_g_cen = self.og.inner_map(cur_depth, src)
                    src_emb = self.og.graph(cur_depth).ndata['emb'][src]
                    dst_emb = self.og.graph(cur_depth).ndata['emb'][dst]
                    sub_g_cen_emb = self.og.graph(cur_depth).ndata['emb'][sub_g_cen]
                    src_rel_emb = self.lin_embs[cur_depth](tc.cat([src_emb, sub_g_cen_emb],dim=0).view(1,-1))
                    dst_rel_emb = self.lin_embs[cur_depth](tc.cat([dst_emb,sub_g_cen_emb],dim=0).view(1,-1))
                    comb_src_emb.append(src_rel_emb)
                    comb_dst_emb.append(dst_rel_emb)
                    break
                else:
                    # print('one miss!! at depth',cur_depth)
                    src_sub_g_cen = self.og.inner_map(cur_depth,src)
                    src_emb = self.og.graph(cur_depth).ndata['emb'][src]
                    src_sub_g_cen_emb = self.og.graph(cur_depth).ndata['emb'][src_sub_g_cen]
                    src_rel_emb = self.lin_embs[cur_depth](tc.cat([src_emb,src_sub_g_cen_emb],dim=0).view(1,-1))
                    comb_src_emb.append(src_rel_emb)

                    dst_sub_g_cen = self.og.inner_map(cur_depth, dst)
                    dst_emb = self.og.graph(cur_depth).ndata['emb'][dst]
                    dst_sub_g_cen_emb = self.og.graph(cur_depth).ndata['emb'][dst_sub_g_cen]
                    dst_rel_emb = self.lin_embs[cur_depth](tc.cat([dst_emb, dst_sub_g_cen_emb],dim=0).view(1,-1))
                    comb_dst_emb.append(dst_rel_emb)
                src = self.og.inner_map(cur_depth,src)
                src = self.og.upper_map(cur_depth,src)
                dst = self.og.inner_map(cur_depth,dst)
                dst = self.og.upper_map(cur_depth,dst)
                cur_depth += 1
            # print('src_emb',comb_src_emb)
            # print('dst_emb', comb_dst_emb)
            ret_src_emb = tc.cat(comb_src_emb,dim=1)
            ret_dst_emb = tc.cat(comb_dst_emb,dim=1)
            # print('src shape',ret_src_emb.shape)
            # print('dst shape', ret_dst_emb.shape)
            ret_emb = tc.cat([ret_src_emb,ret_dst_emb],dim = 1)

            limit = (len(self.og.gs) + 3) * (self.emb_sz // 4) * 2
            if ret_emb.shape[1] < limit:
                ret_emb = tc.cat([ret_emb,tc.zeros(1,limit-ret_emb.shape[1])],dim=1)
            ret_emb = self.lin_output1(ret_emb)
            ret_emb = F.relu(ret_emb)
            ret_emb = self.lin_output2(ret_emb)
            ret_emb = F.relu(ret_emb)
            ret_emb = self.lin_output3(ret_emb)
            # loss = nn.MSELoss(reduction='sum')
            # target = tc.ones(ret_emb.shape)
            # target = target.type_as(tc.FloatTensor())
            # print(type(target),target)
            # ret_loss = loss(ret_emb,target)
            # print(ret_loss)
            # ret_loss.backward()

            # ret_emb_lst.append(ret_emb)
            if ret_emb_lst is None:
                ret_emb_lst = ret_emb
            else:
                ret_emb_lst = tc.cat([ret_emb_lst,ret_emb],dim=0)

        # loss = nn.MSELoss(reduction='sum')
        # target = tc.ones(ret_emb_lst.shape)
        # target = target.type_as(tc.FloatTensor())
        # print(type(target),target)
        # ret_loss = loss(ret_emb_lst,target)
        # print(ret_loss)
        # ret_loss.backward()
        return ret_emb_lst

def test1():
    edges = [(3, 0), (3, 1), (3, 2), (3, 4), (3, 12), (4, 5), (4, 7), (4, 6), (5, 7), (5, 6), (6, 7), (4, 8), (8, 9),
             (8, 10), (8, 11), (9, 10), (8, 12), (12, 13), (14, 13), (12, 14)]
    src = []
    dst = []
    for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])
    g = dgl.DGLGraph((src, dst))
    g = dgl.to_bidirected(g)
    og = OracleGraph(g=g, depth=2, cluster_sz=-1, cluster_num=4 / 15, proto_sz=4, expand_speed=1, need_remove_dup=False,out_file='../tmp/testOG1')
    og.gen_oracle()
    print(og.gs[1].edges())
    print('upper_map',og.upper_maps[0])
    print('inner_map', og.inner_maps[0])
    print('lower_map', og.lower_maps[0])
    print('outer_map', og.outer_maps[0])


def test2():
    g, _ = dgl.load_graphs('../datasets/dst/facebook')
    g = g[0]
    og = OracleGraph(g=g, depth=6, cluster_sz=-1, cluster_num=0.5, proto_sz=-1, expand_speed=1, need_remove_dup=False,out_file='../tmp/fb-og=6')
    og.gen_oracle()
    og.gs[1].edges()
    print(og.gs[1].edges())

if __name__ == '__main__':
    test1()
    # test2()