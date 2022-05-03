import torch as tc
import torch.nn  as nn
import dgl
import dgl.nn as gnn
import dgl.function as gF
import m_generator

def showDataInfo():
    # datasets = ['../datasets/dst/karate','../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/NotreDame','../datasets/dst/DBLP', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
    #             '../datasets/dst/youtube']
    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/NotreDame','../datasets/dst/DBLP']

    for dataset in datasets:
        g, _ = dgl.load_graphs(dataset)
        g = g[0]
        max_BC = -1
        min_BC = 100000
        n_cnt = 0
        sum_BC = 0
        n_sz = g.num_nodes()
        with open(dataset+'-BC.txt','r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                # print(line)
                src,dst = line.split('-',1)
                src = int(src)
                dst = float(dst)
                dst /= n_sz
                max_BC = max(max_BC,dst)
                min_BC = min(min_BC,dst)
                n_cnt += 1
                sum_BC += dst
        print(dataset,'node_sz',g.num_nodes(),',edge_sz',g.num_edges(),'e/n ratio',g.num_edges() / g.num_nodes(),'RoBC',max_BC-min_BC,'mBC',sum_BC / n_cnt)
        assert n_sz == n_cnt, print('n_sz:{}, n_cnt:{}'.format(n_sz,n_cnt))

def showGenInfo():
    dataset = '../datasets/dst/facebook'
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file='fb-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=None,
                                            force=False, prod_workers=4)
    lst = list(train_generator.loader(10,10))
    print(len(lst))



if __name__ == '__main__':
    # input = tc.FloatTensor([1,2,3,4])
    # param = nn.Linear(4,2)
    # output = param(input)
    # target = tc.FloatTensor([1,2])
    # loss = nn.MSELoss(reduction='sum')
    # ret_loss = loss(output,target)
    # print(ret_loss)
    # ret_loss.backward()
    # print(input.grad)

    showDataInfo()
    # showGenInfo()



