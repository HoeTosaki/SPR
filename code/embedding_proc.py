import sys
import fileinput
import os
import dgl
import torch as tc


class Embedding:
    def __init__(self,file):
        self.file = file
        self.node_sz = 0
        self.emb_sz = 0
        with open(self.file, 'r') as f:
            line = f.readline()  # 取第一行
            lst = line.strip().split()
            assert len(lst) == 2
            self.node_sz = int(lst[0])
            self.emb_sz = int(lst[1])
        print('embedding readed. nodes_sz = {}, emb_sz = {}'.format(self.node_sz,self.emb_sz))

    def check_with_graph(self,g):
        return g.num_nodes() == self.node_sz

    def debug_with_graph(self,g,attr='emb'):
        self.attr = attr
        g.ndata[self.attr] = tc.rand(g.num_nodes(), self.emb_sz)
        g.ndata['debug'] = -tc.ones(g.num_nodes(),dtype=tc.int32)
        print('debug nodes',g.ndata['debug'])
        for cnt, line in enumerate(fileinput.input(files=[self.file])):
            if cnt == 0:
                lst = line.strip().split()
                assert len(lst) == 2
                # self.node_sz = lst[0]
                # self.emb_sz = lst[1]
            else:
                lst = line.strip().split()
                assert len(lst) >= 2
                nid = int(lst[0])
                lst1 = lst[1:]
                nemb = tc.tensor([float(ele) for ele in lst1])
                g.ndata[attr][nid] = nemb
                g.ndata['debug'][nid] = 1
        print('nodes with no prior embedding are : ',g.nodes()[g.ndata['debug'] < 0])
        self.node_sz = g.num_nodes()

    def add_to_graph(self,g,attr = 'emb'):
        self.attr = attr
        g.ndata[self.attr] = tc.rand(self.node_sz,self.emb_sz)
        for cnt, line in enumerate(fileinput.input(files=[self.file])):
            if cnt == 0:
                lst = line.strip().split()
                assert len(lst) == 2
                # self.node_sz = lst[0]
                # self.emb_sz = lst[1]
            else:
                lst = line.strip().split()
                assert len(lst) >= 2
                nid = int(lst[0])
                lst1 = lst[1:]
                nemb = tc.tensor([float(ele) for ele in lst1])
                g.ndata[attr][nid] = nemb


if __name__ == '__main__':
    g, _ = dgl.load_graphs('../datasets/dst/BlogCatalog-dataset')
    g = g[0]
    emb = Embedding('../outputs/dw-blogcatalog')
    print('g:',g)

    print(emb.check_with_graph(g))

    emb.add_to_graph(g)
    print('g:',g)

    print(g.ndata['emb'])