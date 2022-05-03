import torch as tc
import torch.nn  as nn
import dgl
import dgl.nn as gnn
import dgl.function as gF

def testDGLGraph():
    edges = [[0,2],[1,2],[3,2],[2,4],[4,5],[5,6],[6,7],[7,8],[8,5]]
    src = []
    dst = []
    for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])
    g = dgl.DGLGraph((src, dst))
    g = dgl.to_bidirected(g)

    g.ndata['emb'] = tc.ones(g.num_nodes(),1)
    g.ndata['att'] = tc.zeros(g.num_nodes(),1)
    g.ndata['emb'][[0,1,2,3,4]] = tc.Tensor([0,1,2,3,4]).view(-1,1)
    g.ndata['att'][[0,1,2,3,4]] = tc.Tensor([0.1,0.2,1.2,2.5,0.6]).view(-1,1)

    # print(g.ndata['emb'])
    # g = dgl.DGLGraph()
    # g.apply_edges(gF.copy_u('att','att'))
    # dgl.nn.functional.edge_softmax(g,g.edata['att'])


    # print(g.ndata['emb'])
    print(g.successors(3))
# def testTensor():
#     a = tc.Tensor([[1,2,3],[4,5,6]])
#     print(a)
#     b = a.repeat(1,2)
#     print(b)
#
# import math
# def testABS():
#     print(math.fabs(1.0 - 1.000001312) < 1e-3)
#     print(math.fabs(0 - 1.000001312) < 1e-3)


# def testNLL():
#     x = tc.Tensor([[0.2,0.1,0.7]])
#     y = tc.IntTensor([2])
#     loss = nn.NLLLoss()
#     print(x,y)
#     l = loss(x,y)
#     print(l.item())

if __name__ == '__main__':
    # testNLL()
    # testTensor()
    testDGLGraph()
