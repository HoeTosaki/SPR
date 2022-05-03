import torch as tc
import m_evaluator
import m_router
import os
import m_generator
import dgl
import random
import m_selector
import sys

def query_dist(model_name='fb-dwext-REALFAST-emb=128',src_lst=[],dst_lst=[], **kwargs):
    if len(src_lst) == 0 or len(dst_lst) == 0 or len(src_lst) != len(dst_lst):
        print('ERR;illegal input src/dst lst.')
        return None
    target_file = ''
    if 'extend_cache_sgn' in kwargs:
        target_file = os.path.join('../tmp', 'ext-dist={}.cache'.format(kwargs['extend_cache_sgn']))
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or line == '':
                        continue
                    print(line)
            return

    model,g = m_evaluator.BasicEvaluator.load_model(out_file=model_name)
    src_lst = tc.LongTensor(src_lst)
    dst_lst = tc.LongTensor(dst_lst)
    src_emb_lst = g.ndata['emb'][src_lst]
    dst_emb_lst = g.ndata['emb'][dst_lst]
    dist_lst = model(src_emb_lst,dst_emb_lst)
    ret_lst = dist_lst.view(-1).tolist()

    ret_str = [str(ret) for ret in ret_lst]
    ret = '-'.join(ret_str)
    print(ret)
    if 'extend_cache_sgn' in kwargs:
        with open(target_file,'r') as f:
            f.write(ret)
            f.flush()
    return ret_lst

def query_path(model_name='fb-dwext-REALFAST-emb=128',src_lst=[],dst_lst=[],top_k=3, **kwargs):
    if len(src_lst) == 0 or len(dst_lst) == 0 or len(src_lst) != len(dst_lst):
        print('ERR;illegal input src/dst lst.')
        return

    target_file = ''
    if 'extend_cache_sgn' in kwargs:
        target_file = os.path.join('../tmp', 'ext-path={}.cache'.format(kwargs['extend_cache_sgn']))
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or line == '':
                        continue
                    print(line)
            return


    rt = m_router.Router(model_name=model_name)
    path_lst = []
    for src,dst in zip(src_lst,dst_lst):
        path,_ = rt.query_path(src,dst,top_k=top_k)
        path_lst.append(path)

    ret_lst = []
    for path in path_lst:
        if path is not None:
            path_str = [str(node) for node in path]
            ret_lst.append('-'.join(path_str))
        else:
            ret_lst.append('None')
    ret = '\n'.join(ret_lst)
    print(ret)
    if 'extend_cache_sgn' in kwargs:
        with open(target_file,'r') as f:
            f.write(ret)
            f.flush()
    return path_lst

def query_mode_gt_dist(dataset_name='fb',src=-1,dst_sz = 20,use_cache=True, **kwargs):
    '''
        generate a random node set relative to current source with mutual distance.
    :return: (src, [(src,e_dst,dist)])
    '''
    dataset_names = ['fb','bc','tw','yt']
    if dataset_name not in dataset_names:
        print('ERR;not a valid pre-trained dataset.')
        return

    target_file = os.path.join('../tmp', '{}-gt-dist-src={}.cache'.format(dataset_name, src))
    if use_cache == True and os.path.exists(target_file):
        with open(target_file,'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                print(line)
        return
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter','../datasets/dst/youtube']
    g, _ = dgl.load_graphs(datasets[dataset_names.index(dataset_name)])
    g = g[0]

    if src < 0:
        ds = m_selector.DegreeSelector(g=g)
        preferred_srcs = list(ds.perform(cnt=10, action='max'))
        src = random.choice(preferred_srcs)
        src = int(src)

    assert src >= 0

    node_lst = g.nodes().tolist().copy()
    random.shuffle(node_lst)
    dst_nodes = node_lst[:dst_sz]
    bfs = m_generator.BFS(g)
    dst_dists = bfs.dist_one_to_other(src,dst_set=dst_nodes).tolist()

    # make str.
    dst_nodes = [str(dst_node) for dst_node in dst_nodes]
    dst_dists = [str(dst_dist) for dst_dist in dst_dists]

    txt_head = '#'.join([str(src),'-'.join(dst_nodes)])
    txt_cont_lst = []

    for dst_node,dst_dist in zip(dst_nodes,dst_dists):
        txt_cont_lst.append('-'.join([str(src),dst_node,dst_dist]))
    txt_cont = '\n'.join(txt_cont_lst)

    print(txt_head)
    print(txt_cont)

    with open(target_file,'w') as f:
        f.write(txt_head+'\n')
        f.write(txt_cont)
        f.flush()
    return

def query_mode_gt_path(dataset_name='fb',src=-1,prox_depth=4,dst_sz =20,per_layer_sz=-1,use_cache=True, **kwargs):
    '''
        generate a random node set relative to current source with path.
    :return: (src, [(src,e_dst,dist)])
    '''
    dataset_names = ['fb', 'bc', 'tw', 'yt']
    if dataset_name not in dataset_names:
        print('ERR;not a valid pre-trained dataset.')
        return

    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
                '../datasets/dst/youtube']
    g, _ = dgl.load_graphs(datasets[dataset_names.index(dataset_name)])
    g = g[0]

    if src < 0:
        ds = m_selector.DegreeSelector(g=g)
        preferred_srcs = list(ds.perform(cnt=10, action='max'))
        src = random.choice(preferred_srcs)
        src = int(src)
    assert src >= 0

    target_file = os.path.join('../tmp', '{}-gt-path-src={}.cache'.format(dataset_name, src))
    if use_cache == True and os.path.exists(target_file):
        with open(target_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                print(line)
        return

    dic_id2signal = [-1] * g.num_nodes()
    search_lst = [src]
    dic_id2signal[src] = 0
    dst_close_set = set()
    dst_layer_close_set = [set() for ele in range(prox_depth)]

    trace_map = {src:-1}
    for idx in range(prox_depth):
        if len(search_lst) == 0 or search_lst is None:
            break
        new_search_lst = []
        while len(search_lst) > 0:
            nid = search_lst.pop(0)
            lst = g.successors(nid).tolist()
            for nnid in lst:
                if dic_id2signal[nnid] < 0:
                    new_search_lst.append(nnid)
                    dic_id2signal[nnid] = dic_id2signal[nid] + 1
                    trace_map[nnid] = nid
                    dst_close_set.add(nnid)
                    dst_layer_close_set[idx].add(nnid)
        search_lst = new_search_lst
    # print('dst_sz',dst_sz,'per_layer_sz',per_layer_sz)

    dst_nodes = []
    if dst_sz > 0:
        assert per_layer_sz == -1
        # print('dst_sz',dst_sz,'per_layer_sz',per_layer_sz)
        # use uniform sample for dst nodes.
        dst_nodes.extend(list(dst_close_set))
        random.shuffle(dst_nodes)
        dst_nodes = dst_nodes[:dst_sz]
    else:
        assert per_layer_sz > 0
        # use layer sample for dst nodes with diff proximity.
        for idx in range(prox_depth):
            layer_nodes = list(dst_layer_close_set[idx])
            random.shuffle(layer_nodes)
            dst_nodes.extend(layer_nodes[:per_layer_sz])

    path_lst = []
    for dst_node in dst_nodes:
        path = [dst_node]
        cur_node = dst_node
        while cur_node in trace_map:
            cur_node = trace_map[cur_node]
            if cur_node != -1:
                path.append(cur_node)
            else:
                break
        path = path[::-1]
        path_lst.append(path)

    # make str.
    dst_nodes = [str(dst_node) for dst_node in dst_nodes]


    txt_head = '#'.join([str(src), '-'.join(dst_nodes)])
    txt_cont_lst = []

    for dst_node, path in zip(dst_nodes, path_lst):
        assert int(dst_node) == path[-1]
        assert src == path[0]
        txt_cont_lst.append('-'.join([str(meta_node) for meta_node in path]))
    txt_cont = '\n'.join(txt_cont_lst)

    print(txt_head)
    print(txt_cont)

    with open(target_file, 'w') as f:
        f.write(txt_head + '\n')
        f.write(txt_cont)
        f.flush()
    return


def query_mode_gt_graph(dataset_name='fb',node_sz=100,use_cache=True, **kwargs):
    '''
        generate a subgraph which demands reconstruced by spq model.
    :return: (src, [(src,e_dst,dist)])
    '''
    dataset_names = ['fb', 'bc', 'tw', 'yt']
    if dataset_name not in dataset_names:
        print('ERR;not a valid pre-trained dataset.')
        return


    target_file = os.path.join('../tmp', '{}-gt-graph-node={}.cache'.format(dataset_name,node_sz))
    if use_cache == True and os.path.exists(target_file):
        with open(target_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                print(line)
        return
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
                '../datasets/dst/youtube']
    g, _ = dgl.load_graphs(datasets[dataset_names.index(dataset_name)])
    g = g[0]

    lst_nodes = g.nodes().tolist().copy()
    random.shuffle(lst_nodes)

    lst_nodes = lst_nodes[:node_sz]

    if 'nodes' in kwargs:
        lst_nodes = kwargs['nodes']
    bfs = m_generator.BFS(g)
    dic_src2trace = {}
    for src in lst_nodes:
        trace_map, _ = bfs.path_one_to_other(src,dst_set=lst_nodes)
        dic_src2trace[src] = trace_map

    # construct subgraph by selecting nodes & edges.
    node_set = set()
    edge_set = set()
    for src in lst_nodes:
        cur_trace_map = dic_src2trace[src]
        for dst in lst_nodes:
            if dst == src:
                continue
            cur_node = dst
            node_set.add(cur_node)
            while cur_node in cur_trace_map:
                lst_node = cur_node
                cur_node = cur_trace_map[cur_node]
                if cur_node != -1:
                    node_set.add(cur_node)
                    if (lst_node,cur_node) not in edge_set:
                        edge_set.add((cur_node,lst_node))
                else:
                    break


    # make str.
    txt_head = '#'.join([str(node) for node in lst_nodes])
    txt_cont = '\n'.join([str(src)+'-'+str(dst) for (src,dst) in edge_set])

    print(txt_head)
    print(txt_cont)

    with open(target_file, 'w') as f:
        f.write(txt_head + '\n')
        f.write(txt_cont)
        f.flush()
    return

if __name__ == '__main__':
    # print('hello interface.')
    # print(query_dist(src_lst=[1,2,3],dst_lst=[660,801,904]))
    # print(query_path(src_lst=[1], dst_lst=[801],top_k=5000))
    # print(query_mode_gt_dist(dataset_name='fb',src=-1,dst_sz=20,use_cache=False))
    # print(query_mode_gt_path(dataset_name='fb',src=10,prox_depth=5,dst_sz=-1,per_layer_sz=1,use_cache=False))
    # print(query_mode_gt_graph(dataset_name='fb',node_sz=100,use_cache=True))
    # print(sys.argv)

    # print(query_mode_gt_graph(dataset_name='fb',node_sz=1000,use_cache=False,nodes=[3456,1185,832,3,4,3257]))

    argv = sys.argv
    if len(sys.argv) >= 2:
        argv = argv[2:]
    else:
        print("ERR;please check arg list as {}".format(sys.argv))
    arg_dict = {}
    for arg in argv:
        arg = arg.strip()
        k,v = arg.split('@=@')
        if v.startswith('['):
            # int lst.
            v = v[1:]
            v = v[:-1]
            # print(v)
            raw_lst = v.split(',')
            ret_lst = []
            for ele in raw_lst:
                ret_lst.append(int(ele))
            arg_dict[k] = ret_lst
        elif v.startswith("@int@"):
            # int.
            v = v[5:]
            arg_dict[k] = int(v)
        elif v.startswith("@bool@"):
            # bool.
            v = v[6:]
            v = v.strip()
            if v == 'false':
                arg_dict[k] = False
            else:
                arg_dict[k] = True
        else:
            # others for str.
            arg_dict[k] = v

    if arg_dict['method'] == 'query_dist':
        query_dist(**arg_dict)
    elif arg_dict['method'] == 'query_path':
        query_path(**arg_dict)
    elif arg_dict['method'] == 'query_mode_gt_dist':
        query_mode_gt_dist(**arg_dict)
    elif arg_dict['method'] == 'query_mode_gt_path':
        query_mode_gt_path(**arg_dict)
    elif arg_dict['method'] == 'query_mode_gt_graph':
        query_mode_gt_graph(**arg_dict)
    else:
        print("ERR;not match method.")