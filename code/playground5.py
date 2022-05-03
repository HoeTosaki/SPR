import dgl
import hyperbolic as hyper
import drawSvg

import utils
import time

def routine_eclid():
    hyper.euclid.shapes.Line(0,0,1,1)

class A:
    def proc(lst,bias,**kwargs):
        # print('instance {} start with {}.'.format(kwargs['__PID__'],lst))
        # print(lst)
        for ele in lst:
            time.sleep(0.2)
            # print(ele+100)
        # print('instance {} finished.'.format(kwargs['__PID__']))
        return [ele + bias for ele in lst]


def routine_parallel():
    mpm = utils.MPManager(batch_sz=5,num_workers=4)
    data = list(range(50))
    ret = mpm.multi_proc(func=A.proc,seq_args=[data],auto_concat=True,bias=200)
    print('final ret:',ret)


if __name__ == '__main__':
    routine_parallel()

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