import dgl
import torch as tc
import random
def components_of_graph(g,has_detail = False):
    with g.local_scope():
        g.ndata['s'] = tc.zeros(g.num_nodes(), dtype=tc.bool)
        ret_comps = []
        while True:
            rest_nodes = g.nodes()[g.ndata['s'] == False]
            if len(rest_nodes) == 0:
                break
            nid = random.choice(rest_nodes).tolist()
            work_queue = [nid]
            cur_nodes = [nid]
            cur_cnt = 1
            g.ndata['s'][nid] = True  # first node has been visited.
            while len(work_queue) > 0:
                cur_nid = work_queue.pop(0)
                succ = g.successors(cur_nid)
                succ = succ[g.ndata['s'][succ] == False]
                if has_detail:
                    cur_nodes.extend(succ.tolist())
                else:
                    cur_cnt += succ.shape[0]
                g.ndata['s'][succ] = True
                work_queue.extend(succ.tolist())
            if has_detail:
                ret_comps.append({"num_nodes":len(cur_nodes),"nodes_lst":cur_nodes})
            else:
                ret_comps.append(cur_cnt)
        return ret_comps

if __name__ == '__main__':
    # print('hello extend_dgl')
    # edges = [(0,1),(1,2),(0,2),(3,4),(4,5),(5,6),(6,3),(7,8),(8,9),(8,10),(8,11)]
    # src = []
    # dst = []
    # for edge in edges:
    #     src.append(edge[0])
    #     dst.append(edge[1])
    # g = dgl.DGLGraph((src, dst))
    # g = dgl.to_bidirected(g)
    # print('ret:',components_of_graph(g))

    g,_ = dgl.load_graphs('../datasets/dst/facebook')
    g = g[0]
    print('facebook',components_of_graph(g))

    g,_ = dgl.load_graphs('../datasets/dst/BlogCatalog-dataset')
    g = g[0]
    print('blogcatalog',components_of_graph(g))

    g,_ = dgl.load_graphs('../datasets/dst/twitter')
    g = g[0]
    print('twitter',components_of_graph(g))

    g,_ = dgl.load_graphs('../datasets/dst/youtube')
    g = g[0]
    print('youtube',components_of_graph(g))

'''
    facebook [3959]
    blogcatalog [10312]
    twitter [76250]
    youtube [1134890]
'''