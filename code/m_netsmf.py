import numpy as np
import dgl
import pandas as pd
import os

def gen_edge_list(datasets,data_names):
    for dataset,data_name in zip(datasets,data_names):
        # assert dataset in ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc', '../datasets/dst/NotreDame','']
        g, _ = dgl.load_graphs(dataset)
        g = g[0]
        with open('../datasets/other/'+data_name+'.edgelist','w') as f:
            for edge in np.stack(g.edges()).T:
                f.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')
            f.flush()

if __name__ == '__main__':
    print('hello netsmf.')
    gen_edge_list(datasets=['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc'],data_names = ['cr','fb','gq'])
