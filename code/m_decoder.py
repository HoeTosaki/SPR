import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dgl
import dgl.function as fn
import dgl.nn.functional as gnF

class Decoder(nn.Module):
    def __init__(self,deep_dict = {}):
        super(Decoder, self).__init__()
        self.deep_dict = deep_dict

    def forward(self, inputs):
        raise NotImplementedError

    def get_external_param(self):
        return self.deep_dict


class LinearReg(Decoder):
    def __init__(self,emb_sz,deep_dict={}):
        super(LinearReg, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.lin = nn.Linear(emb_sz*2, 1)

    def forward(self, src,dst):
        # print('emb',self.emb_sz)
        # print('emb_input:',src,dst)
        return self.lin(tc.cat((src,dst),dim = 1))


class MLP(Decoder):
    def __init__(self,emb_sz,deep_dict={}):
        super(MLP, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz*2, emb_sz // 4)
        self.lin2 = nn.Linear(emb_sz // 4, 1)

    def forward(self, src,dst):
        # print('emb',self.emb_sz)
        # print('emb_input:',src,dst)
        out = self.lin1(tc.cat((src,dst),dim = 1))
        out = F.relu(out)
        out = self.lin2(out)
        return out

class DADL(Decoder):
    def __init__(self, emb_sz, deep_dict={}):
        super(DADL, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
        self.lin2 = nn.Linear(emb_sz, 1)

    def forward(self, src, dst):
        # print('emb',self.emb_sz)
        # print('emb_input:',src,dst)
        out = self.lin1(tc.cat((src, dst), dim=1))
        out = F.relu(out)
        out = self.lin2(out)
        out = F.softplus(out)
        return out

class NNcls(Decoder):
    def __init__(self, emb_sz,cls_num=10, deep_dict={}):
        super(NNcls, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.cls_num = cls_num
        self.lin1 = nn.Linear(emb_sz, emb_sz)
        self.lin2 = nn.Linear(emb_sz, cls_num)

    def forward(self, nemb):
        # print('emb',self.emb_sz)
        # print('emb_input:',src,dst)
        out = self.lin1(nemb)
        out = F.leaky_relu(out)
        out = self.lin2(out)
        out = F.log_softmax(out,dim=1)
        return out

# class CDGCN(Decoder):
#     def __init__(self, emb_sz, deep_dict={}):
#         super(CDGCN, self).__init__(deep_dict=deep_dict)
#         self.emb_sz = emb_sz
#         self.g = deep_dict['g']
#         self.lin1 = nn.Linear(emb_sz * 2, emb_sz)
#         self.lin2 = nn.Linear(emb_sz, 1)
#
#     def forward(self, src, dst):
#         # print('emb',self.emb_sz)
#         # print('emb_input:',src,dst)
#         self.g = dgl.DGLGraph()
#         with self.g.local_scope():
#
#         out = self.lin1(tc.cat((src, dst), dim=1))
#         out = F.relu(out)
#         out = self.lin2(out)
#         out = F.softplus(out)
#         return out
#
#         with mfg.local_scope():
#             # print('mfg:',mfg)
#             mfg.srcdata['emb_o'] = src_emb
#             # print('src:', src_emb.shape)
#             # print('mfg nodes:',mfg.srcdata[dgl.NID].tolist())
#             if is_close:
#                 # mfg.update_all(fn.copy_u(u='emb_o', out='msg'), fn.sum(msg='msg', out='emb_n'))
#                 mfg.push(range(mfg.num_src_nodes()), fn.copy_u('emb_o', 'msg'), fn.sum('msg', 'emb_n'))
#             else:
#                 mfg.edata['att'] = e_att
#                 # mfg.update_all(fn.src_mul_edge(src='emb_o',edge='att', out='msg'), fn.sum(msg='msg', out='emb_n'))
#                 mfg.push(range(mfg.num_src_nodes()), fn.src_mul_edge('emb_o', 'att', 'msg'), fn.sum('msg', 'emb_n'))
#             return mfg.dstdata['emb_n']


class EmbedModel(Decoder):
    '''
        EmbedModel训练图上不同节点的embedding，而不仅仅使用外部参数对先验embedding做推断
    '''
    def __init__(self,emb_sz,g,deep_dict={}):
        super(EmbedModel, self).__init__(deep_dict=deep_dict)
        self.emb_sz = emb_sz
        self.g = g
        # if 'emb' in self.g.ndata:
        #     self.node_embed = Variable(self.g.ndata['emb'], requires_grad=True)
    def load_param(self,g):
        self.g = g
        # self.register_parameter(name='node_embed', param=self.g.ndata['emb'])
        self.node_embed = tc.nn.Parameter(self.g.ndata['emb'])
    def forward(self,src,dst):
        return ((self.node_embed[src] - self.node_embed[dst]) ** 2).sum(dim=1)


