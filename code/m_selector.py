import dgl

class DegreeSelector:
    def __init__(self,g):
        self.g = g
    def perform(self,cnt=-1,action='max'):
        if cnt == -1:
            cnt = self.g.num_nodes()
        idx = self.g.out_degrees(self.g.nodes()).argsort(descending = (action=='max'))
        return self.g.nodes()[idx[:cnt]]


if __name__ == '__main__':
    edges = [(3, 0), (3, 1), (3, 2), (3, 8), (3, 7), (1, 4), (4, 5), (4, 7), (7, 6), (6, 5), (8, 7), (8, 9), (9, 10), (8, 10)]
    src = []
    dst = []
    for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])
    g = dgl.DGLGraph((src,dst))
    g = dgl.to_bidirected(g)
    ds = DegreeSelector(g=g)
    nod_lst = ds.perform(cnt=-1,action='min')
    print(nod_lst)


