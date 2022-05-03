import numpy as np
import pandas as pd
import m_encoder
import dgl
import os
def gen_edge_csv(datasets,data_names):
    for dataset,data_name in zip(datasets,data_names):
        # assert dataset in ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc', '../datasets/dst/NotreDame','']
        g, _ = dgl.load_graphs(dataset)
        g = g[0]
        pd.DataFrame(np.stack(g.edges()).T).to_csv('../datasets/other/edge-'+data_name+'.csv', index=False)




class GraRepEncoder(m_encoder.Encoder):
    def __init__(self, g,emb_sz=128,workers=1,out_dir='../outputs',out_file='encoder',force=False,order=5,random_seed=42,svd_iters=20):
        super(GraRepEncoder, self).__init__(g=g,emb_sz=emb_sz,workers=workers,out_dir=out_dir,out_file=out_file,force=force)
        self.order=order
        self.random_seed = random_seed
        self.svd_iters = svd_iters

    def train(self):
        # g = dgl.DGLGraph()
        # if not self.force and os.path.exists(os.path.join(self.out_dir,'edges-' + self.out_file)):
        #     print('encoder graph edges file checked.')
        # else:
        #     pd.DataFrame(np.stack(g.edges()).T).to_csv(os.path.join(self.out_dir,'edges-'+self.out_file),index=False)
        if not self.force and self.check_file():
            print('encoder cache file checked')
            return
        if os.path.exists(os.path.join(self.out_dir,'csv-'+self.out_file)):
            data_csv = pd.read_csv(os.path.join(self.out_dir,'csv-'+self.out_file))
            with open(os.path.join(self.out_dir,self.out_file),'w') as f:
                f.write('{} {}\n'.format(int(data_csv.values.shape[0]),int(data_csv.values.shape[1]-1)))
                for line in data_csv.values:
                    f.write(' '.join([str(ele) if idx > 0 else str(int(ele))  for idx,ele in enumerate(line)]) + '\n')
            return
        raise NotImplementedError

    def save(self):
        pass




if __name__ == '__main__':
    print('hello grarep.')
    gen_edge_csv(datasets=['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc'],data_names = ['cr','fb','gq'])
