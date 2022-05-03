import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import dgl
import m_encoder
import m_lle
import m_le
import m_lpca
import m_grarep
import m_gf
import m_dage
import m_decoder
import scipy.io as sio
import scipy.sparse as sp
import operator
import math

from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer


'''
res loading.
'''
def load_data_(data_name='cr',data_type='dgl'):
    '''
    :param data_name: in ['cr','bc']
    :param data_type: in ['mat','dgl']
    :return: graph or adj mat
    '''
    if data_type == 'mat':
        return sio.loadmat(os.path.join('../cls/', '{}.mat'.format(data_name if data_name == 'cr' else 'blogcatalog')))
    if data_name == 'cr':
        return dgl.data.CoraGraphDataset()[0]
    elif data_name == 'bc':
        # data_path = '../datasets/dst/blogcatalog.mat'
        if os.path.exists('../datasets/dst/blogcatalog'):
            g,_ = dgl.load_graphs('../datasets/dst/blogcatalog')
            return g[0]
        data_path = '../datasets/src/BlogCatalog-dataset/data'

        g = dgl.DGLGraph()
        with open(os.path.join(data_path,'edges.csv'),'r') as f:
            srcs = []
            dsts = []
            for line in f.readlines():
                if line != None:
                    line = line.strip()
                    lst = line.split(',')
                    assert len(lst) == 2
                    srcs.append(int(lst[0])-1)
                    dsts.append(int(lst[1])-1)
            srcs = np.array(srcs)
            dsts = np.array(dsts)
            g.add_edges(srcs,dsts)
            g = dgl.to_bidirected(g)
            g = dgl.to_simple_graph(g)

            # dgl.save_graphs('../datasets/dst/blogcatalog',[g])
        with open(os.path.join(data_path, 'group-edges.csv'), 'r') as f:
            g.ndata['label'] = tc.zeros(size=(g.num_nodes(),))
            for line in f.readlines():
                if line != None:
                    line = line.strip()
                    lst = line.split(',')
                    assert len(lst) == 2
                    g.ndata['label'][int(lst[0])-1] = int(lst[1])-1
            # for default test & stub mask.
            g.ndata['train_mask'] = tc.BoolTensor([True] * 140 + [False] * (g.ndata['label'].shape[0] - 140))
            g.ndata['val_mask'] = tc.BoolTensor([False] * 140 + [True] * 70 + [False] * (g.ndata['label'].shape[0] - 210))

        return g

    assert False

def load_data(data_name='cr',data_type='dgl',bc_file=None):
    '''
    :param data_name: in ['cr','bc']
    :param data_type: in ['mat','dgl']
    :param bc_file: betweenness precal file, if None, take default config.
    :return: graph or adj mat
    '''
    g = load_data_(data_name=data_name,data_type=data_type)
    if data_type == 'mat':
        # unable to attach bc.
        return g
    if bc_file is None:
        bc_file = '{}-BC.txt'.format(data_name)
    g.ndata['bc'] = tc.ones(size=(g.num_nodes(),))
    if bc_file:
        check_num = 0
        with open(os.path.join('../cls/data/',bc_file)) as f:
            for line in f.readlines():
                if line is not None:
                    line = line.strip()
                    if line != '':
                        lst = line.split('-')
                        assert len(lst) == 2
                        g.ndata['bc'][int(lst[0])] = float(lst[1])
                        check_num += 1
        assert check_num == g.num_nodes(), print('check num {} != g nodes {} '.format(check_num,g.num_nodes()))
    return g

def check_graph_consistency(g):
    print(g)
    g = dgl.DGLGraph()
    # is_connected = True
    # is_compressed = True
    is_undirected = True
    for nid in g.nodes():
        for nnid in g.successors(nid):
            if not g.has_edge_between(nnid,nid):
                is_undirected = False
        if len(list(g.successors(nid))) == 0:
            print('nid:{} has no edge!'.format(nid))
    if not is_undirected:
        print('directed graph.')

def load_embs(data_name,g,encoder_name,emb_sz,out_dir='../cls/emb'):
    encoder = gen_emb(data_name=data_name,g=g,encoder_name=encoder_name,emb_sz=emb_sz)
    g_embedded = encoder.load()
    return g_embedded

def load_decoder(emb_sz,cls_num,decoder_name='NNcls'):
    if decoder_name == 'NNcls':
        return m_decoder.NNcls(emb_sz=emb_sz,cls_num=cls_num)
    assert False

'''
Train.
'''
def gen_emb(encoder_name='dw',g=None,data_name='cr',out_dir='../cls/emb',emb_sz=16,alpha=None):
    is_set_alpha = (alpha is not None)
    if alpha is None:
        alpha = 0.9
    if encoder_name == 'dw':
        encoder = m_encoder.DeepWalkEncoder(g=g, emb_sz=emb_sz, workers=16, out_dir=out_dir, out_file=None,
                                            force=False, num_walks=40, walk_lens=40, window_sz=20, max_mem=0, seed=0,
                                            is_dense_degree=False)
    elif encoder_name == 'n2v':
        encoder = m_encoder.Node2VecEncoder(g=g, emb_sz=emb_sz, workers=16, out_dir=out_dir, out_file=None,
                                                force=False, num_walks=40, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
                                                is_directed=False, is_weighted=False, weight_arr=None)
    elif encoder_name == 'lle':
        encoder = m_lle.LLEEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir=out_dir,out_file=None,force=False)
    elif encoder_name == 'gf':
        encoder = m_gf.GFEncoder(g=g, emb_sz=emb_sz, workers=16, out_dir=out_dir, out_file=None, force=False,
                                 iter=500, r=1.0, lr=1e-3, print_step=10)
    elif encoder_name == 'le':
        encoder = m_le.LEEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir=out_dir,out_file=None,force=False)
    elif encoder_name == 'dadl':
        encoder = m_encoder.Node2VecEncoder(g=g, emb_sz=emb_sz, workers=16, out_dir=out_dir, out_file=None,
                                                force=False, num_walks=40, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
                                                is_directed=False, is_weighted=False, weight_arr=None)
    elif encoder_name == 'grp':
        encoder = m_grarep.GraRepEncoder(g=g, emb_sz=emb_sz, workers=6, out_dir=out_dir, out_file=None,
                                             force=False)
    elif encoder_name == 'netmf':
        encoder = m_encoder.NetMFEncoder(g=g, emb_sz=emb_sz, workers=6, out_dir=out_dir, out_file=None,
                                               force=False, order=2, iteration=10, neg_sz=1, seed=42)
    elif encoder_name == 'lpca':
        encoder = m_lpca.LPCAEncoder(g=g, emb_sz=None, workers=6, out_dir=out_dir, out_file=None, force=False)
    elif encoder_name == 'vs':
        encoder = m_encoder.VerseEncoder(g=g, emb_sz=emb_sz, workers=6, out_dir=out_dir, out_file=None,
                                               force=False,force_fit_emb_sz=True)
    elif encoder_name == 'bcdr':
        encoder = m_dage.DistanceResamplingEncoder(g=g, in_bc_file=None, emb_sz=emb_sz, out_dir=out_dir,
                                                    out_file=None, num_walks=40, input_len=40, output_len=10, alpha=alpha,
                                                    input_exps=1, output_exps=4, neg_sz=5)
    encoder.out_file = '{}-{}-emb={}'.format(data_name,encoder_name,emb_sz)
    if encoder_name == 'bcdr' and is_set_alpha:
        encoder.out_file = '{}-{}-emb={}-alpha={}'.format(data_name, encoder_name, emb_sz,alpha)

    encoder.train()
    return encoder

'''
evaluation.
'''
def eval_multi_cls_nn(data_name,emb_sz,cls_num,train_mask,valid_mask,encoder_name,out_dir='../cls/model',add_name=None):
    # load graph.
    g = load_data(data_name=data_name)
    nlst = np.array(g.nodes())
    train_loader = DataLoader(nlst[train_mask],batch_size=256)
    valid_loader = DataLoader(nlst[valid_mask], batch_size=256)

    # load pre-trained model.
    embs = load_embs(data_name=data_name,g=g,encoder_name=encoder_name,emb_sz=emb_sz).ndata['emb']
    decoder = load_decoder(emb_sz=emb_sz,cls_num=cls_num,decoder_name='NNcls')
    loss = nn.NLLLoss()
    optim = tc.optim.Adam(decoder.parameters(), lr=1e-4)
    save_between_idy = 10
    epochs=1000
    for idy in range(epochs):
        train_acc = 0.
        train_loss = 0.
        valid_acc = 0.
        valid_loss = 0.
        decoder.train()
        for idx,nids in enumerate(train_loader):
            optim.zero_grad()
            pred = decoder(embs[nids])
            real = tc.LongTensor(g.ndata['label'][nids])
            batch_loss = loss(pred,real)
            batch_loss.backward()
            train_loss += batch_loss.item()
            optim.step()
            train_acc += (tc.argmax(pred,dim=1) == real).sum()
        decoder.eval()
        for idx, nids in enumerate(valid_loader):
            pred = decoder(embs[nids])
            real = g.ndata['label'][nids]
            batch_loss = loss(pred, real)
            valid_loss += batch_loss.item()
            valid_acc += (tc.argmax(pred, dim=1) == real).sum()
        print('epoch {} train acc: {:.4f} | valid loss: {:.4f} , train loss: {:.4f} | valid loss: {:.4f}'.format(idy,train_acc/len(train_mask),valid_acc/len(valid_mask),train_loss/len(valid_mask),valid_loss/len(valid_mask)))
        if idy % save_between_idy == 0 and idy != 0:
            if add_name is None:
                tc.save(decoder,os.path.join(out_dir,'{}-{}-emb={}.clsdec~{}'.format(data_name,encoder_name,emb_sz,idy)))
            else:
                tc.save(decoder,os.path.join(out_dir, '{}-{}-emb={}{}.clsdec~{}'.format(data_name, encoder_name, emb_sz,add_name, idy)))
    if add_name is None:
        tc.save(decoder, os.path.join(out_dir, '{}-{}-emb={}.clsdec'.format(data_name, encoder_name, emb_sz)))
    else:
        tc.save(decoder, os.path.join(out_dir, '{}-{}-emb={}{}.clsdec'.format(data_name, encoder_name, emb_sz,add_name)))

def eval_multi_cls_lr(data_name,emb_sz,encoder_name,add_name=None,out_dir='../cls/emb',is_print=True,max_iter=100):
    # Load Embeddings
    if add_name is None:
        model = KeyedVectors.load_word2vec_format(os.path.join(out_dir,'{}-{}-emb={}'.format(data_name, encoder_name, emb_sz)), binary=False)
    else:
        model = KeyedVectors.load_word2vec_format(os.path.join(out_dir,'{}-{}-emb={}{}'.format(data_name, encoder_name, emb_sz,add_name)), binary=False)
    # basic params.
    num_shuffles = 10


    # Load labels
    mat = load_data(data_name=data_name,data_type='mat')
    A = mat['network']
    graph = sparse2graph(A)
    labels_matrix = mat['group']
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(classes=list(range(labels_count)))

    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])

    # Shuffle, to create train/test groups
    shuffles = []
    for x in range(num_shuffles):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # to score each train/test group
    all_results = defaultdict(list)

    training_percents = [round(0.02*i,2) for i in range(1,5)] + [round(0.1*i,2) for i in range(1,10)]
    # training_percents = [0.1,0.4,0.7]

    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]

            cy = y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for _ in range(y_test_.shape[0])]

            cy = y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression(max_iter=max_iter))
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)

            all_results[train_percent].append(results)
    if is_print:
        print('Results, using embeddings of dimensionality', X.shape[1])
        print('-------------------')
        for train_percent in sorted(all_results.keys()):
            print('Train percent:', train_percent)
            for index, result in enumerate(all_results[train_percent]):
                print('Shuffle #%d:   ' % (index + 1), result)
            avg_score = defaultdict(float)
            for score_dict in all_results[train_percent]:
                for metric, score in iteritems(score_dict):
                    avg_score[metric] += score
            for metric in avg_score:
                avg_score[metric] /= len(all_results[train_percent])
            print('Average score:', dict(avg_score))
            print('-------------------')
    return all_results

'''
anal.
'''
def anal_all_enc_cls(encoder_names=['lle','gf','le','dw','n2v','grp','netmf','verse','dadl','lpca','bcdr'],out_dir='../cls/log',data_names=['cr','bc'],emb_sz=16,max_iter=100,is_comb=False):
    enc_map = {
        'lle': 'LLE',
        'le': 'LE',
        'gf': 'GF',
        'dw': 'DeepWalk',
        'n2v': 'Node2Vec',
        'dadl': 'DADL',
        'bcdr': 'BCDR(ours.)',
        'lpca': 'LPCA',
        'netmf': 'NetMF',
        'vs': 'Verse',
        'grp': 'GreRep',
    }

    for data_name in data_names:
        data = []
        check_col = []
        check_row = []
        is_first = True
        for encoder_name in encoder_names:
            print('anal on {}|{} ...'.format(encoder_name,data_name))
            check_row.append(enc_map.get(encoder_name,encoder_name))
            cur_dic = eval_multi_cls_lr(
                data_name=data_name,emb_sz=emb_sz,encoder_name=encoder_name,add_name=None,is_print=True,max_iter=max_iter)
            items = sorted(cur_dic.items(),key=operator.itemgetter(0),reverse=False)
            cur_row_lst = []
            for idx,(k,v) in enumerate(items):
                if is_first:
                    check_col.append(k)
                else:
                    if check_col[idx] != k:
                        # assert order unchaned during the same data graph.
                        assert False, print('check col error, {} != {}'.format(check_col,items))
                ele_avg_pair = {}
                for ele in v:
                    for k in ele.keys():
                        ele_avg_pair[k] = ele_avg_pair.get(k,0) + ele[k]
                for k in ele_avg_pair.keys():
                    ele_avg_pair[k] = ele_avg_pair[k] / len(v)
                cur_row_lst.append(ele_avg_pair)
            data.append(cur_row_lst)

            if is_first:
                is_first = False
                assert len(check_col)>1 # percent list len has never to be a small value.
        data_micro = [[ele['micro'] for ele in row] for row in data]
        data_macro = [[ele['macro'] for ele in row] for row in data]
        df_micro = pd.DataFrame(data_micro,columns=check_col,index=check_row)
        df_macro = pd.DataFrame(data_macro, columns=check_col, index=check_row)
        df_micro.to_csv(os.path.join(out_dir,'anal{}-{}-emb={}-cls-mi.csv'.format('-comb' if is_comb else '',data_name,emb_sz)))
        df_macro.to_csv(os.path.join(out_dir, 'anal{}-{}-emb={}-cls-ma.csv'.format('-comb' if is_comb else '',data_name,emb_sz)))

def anal_bcdr_enc_cls(alphas=[0.1,0.4,0.9],out_dir='../cls/log',data_names=['cr','bc'],emb_sz=16,max_iter=100,is_comb=False):

    for data_name in data_names:
        data = []
        check_col = []
        check_row = []
        is_first = True
        for alpha in alphas:
            print('anal on {}|{}|{}|{} ...'.format(data_name,'bcdr',emb_sz,alpha))
            check_row.append('BCDR@{}'.format(alpha))
            cur_dic = eval_multi_cls_lr(
                data_name=data_name,emb_sz=emb_sz,encoder_name='bcdr',add_name='-alpha={}-bcdr-walks.embedding'.format(alpha),is_print=True,max_iter=max_iter)
            items = sorted(cur_dic.items(),key=operator.itemgetter(0),reverse=False)
            cur_row_lst = []
            for idx,(k,v) in enumerate(items):
                if is_first:
                    check_col.append(k)
                else:
                    if check_col[idx] != k:
                        # assert order unchaned during the same data graph.
                        assert False, print('check col error, {} != {}'.format(check_col,items))
                ele_avg_pair = {}
                for ele in v:
                    for k in ele.keys():
                        ele_avg_pair[k] = ele_avg_pair.get(k,0) + ele[k]
                for k in ele_avg_pair.keys():
                    ele_avg_pair[k] = ele_avg_pair[k] / len(v)
                cur_row_lst.append(ele_avg_pair)
            data.append(cur_row_lst)

            if is_first:
                is_first = False
                assert len(check_col)>1 # percent list len has never to be a small value.
        data_micro = [[ele['micro'] for ele in row] for row in data]
        data_macro = [[ele['macro'] for ele in row] for row in data]
        df_micro = pd.DataFrame(data_micro,columns=check_col,index=check_row)
        df_macro = pd.DataFrame(data_macro, columns=check_col, index=check_row)
        df_micro.to_csv(os.path.join(out_dir,'anal{}-{}-bcdr-emb={}-cls-mi.csv'.format('-comb' if is_comb else '',data_name,emb_sz)))
        df_macro.to_csv(os.path.join(out_dir, 'anal{}-{}-bcdr-emb={}-cls-ma.csv'.format('-comb' if is_comb else '',data_name,emb_sz)))

'''
utils.
'''
def dump_edgelist(out_dir='../cls',data_name='cr'):
    g = load_data(data_name)
    with open(os.path.join(out_dir,data_name+'.edgelist'),'w') as f:
        srcs,dsts = g.edges()
        for src,dst in zip(srcs,dsts):
            f.write('{}\t{}\n'.format(int(src),int(dst)))

        f.flush()

def dump_edgelist_csv(out_dir='../cls',data_name='cr'):
    g = load_data(data_name)
    with open(os.path.join(out_dir,'edge-' + data_name + '.csv'),'w') as f:
        srcs,dsts = g.edges()
        for src,dst in zip(srcs,dsts):
            f.write('{},{}\n'.format(int(src),int(dst)))

        f.flush()


def dump_mat(out_dir='../cls',data_name='cr',cls_num=7):
    g = load_data(data_name)
    mat = g.adj(scipy_fmt='coo')
    lbs = g.ndata['label'].numpy().reshape(-1)
    indpnts = list(range(len(lbs))) + [len(lbs)]
    inds = lbs
    vals = [1]*lbs
    lb_mat = sp.csr_matrix((vals,inds,indpnts),shape=(len(lbs),cls_num))
    for i in range(lb_mat.shape[0]):
        lb_mat[i,lbs[i]] = 1

    sio.savemat(file_name=os.path.join(out_dir,data_name+'.mat'),mdict={'network':mat,'group':lb_mat})

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k, v in iteritems(G)}

def emb2rw(encoder_names=['lle'],data_names=['cr'],emb_szs = [16],in_dir='../cls/emb',out_dir='../cls/tmp'):
    for data_name in data_names:
        for encoder_name in encoder_names:
            for emb_sz in emb_szs:
                in_file = os.path.join(in_dir,'{}-{}-emb={}'.format(data_name,encoder_name,emb_sz))
                out_file = os.path.join(out_dir, '{}-{}-emb={}'.format(data_name, encoder_name, emb_sz))
                new_data = []
                with open(in_file,'r') as f:
                    is_first = True
                    is_ok = False
                    for line in f.readlines():
                        if line is None:
                            continue
                        line = line.strip()
                        if line != '':
                            lst = line.split(' ')
                            if is_first and len(lst) == 2:
                                is_ok = True
                                break
                            else:
                                lst = [0 if math.isnan(float(ele)) else float(ele) for ele in lst]
                                lst = [str(ele) for ele in lst]
                                new_data.append(' '.join(lst))
                        is_first = False
                if is_ok:
                    continue
                else:
                    with open(out_file,'w') as f:
                        f.write('{} {}\n'.format(len(new_data),emb_sz))
                        for idx,line in enumerate(new_data):
                            f.write('{} {}\n'.format(idx,line))

def emb_join(data_name = ['cr'],emb_szs = [16,112],encoder_names=['lle','bcdr'],out_dir='../cls/emb'):
    in_file1 = os.path.join(out_dir,'{}-{}-emb={}'.format(data_name,encoder_names[0],emb_szs[0]))
    in_file2 = os.path.join(out_dir, '{}-{}-emb={}'.format(data_name, encoder_names[1], emb_szs[1]))
    out_file = os.path.join(out_dir, '{}-{}@{}|{}@{}-emb={}'.format(data_name, encoder_names[0],emb_szs[0],encoder_names[1],emb_szs[1], sum(emb_szs)))
    new_data = {}
    with open(in_file1,'r') as f:
        is_first = True
        for line in f.readlines():
            if line is None:
                continue
            line = line.strip()
            if line != '':
                lst = line.split(' ')
                if is_first:
                    assert len(lst) == 2
                else:
                    new_data[int(lst[0])] = lst[1:]
            is_first = False
    with open(in_file2, 'r') as f:
        is_first = True
        for line in f.readlines():
            if line is None:
                continue
            line = line.strip()
            if line != '':
                lst = line.split(' ')
                if is_first:
                    assert len(lst) == 2
                else:
                    new_data[int(lst[0])].extend(lst[1:])
            is_first = False

    with open(out_file,'w') as f:
        f.write('{} {}\n'.format(len(list(new_data.keys())),sum(emb_szs)))
        for k in new_data.keys():
            f.write(' '.join([str(k)] + new_data[k])+'\n')

'''
routine.
'''

def routine_basic_test():
    g = load_data('bc')
    check_graph_consistency(g)

def routine_eval_test():
    g = load_data('bc')
    # eval_multi_cls1(data_name='bc',emb_sz=16,cls_num=39,train_mask=g.ndata['train_mask'],valid_mask=g.ndata['val_mask'],encoder_name='dw',add_name='test')

def routine_gen_all_direct_embs():
    encoder_names = ['lle','gf','le','dw','n2v','dadl']
    data_names = ['cr','bc']
    emb_szs = [16,112,128]
    for data_name in data_names:
        g = load_data(data_name=data_name,data_type='dgl')
        for encoder_name in encoder_names:
            for emb_sz in emb_szs:
                print('gen on {}|{}|{} ...'.format(encoder_name,data_name,emb_sz))
                gen_emb(encoder_name=encoder_name,g=g,data_name=data_name,emb_sz=emb_sz)
    print('gen all completed!')

def routine_gen_bcdr_embs():
    encoder_names = ['bcdr']
    # data_names = ['cr','bc']
    data_names = ['cr', 'bc']
    emb_szs = [16,128]
    for data_name in data_names:
        g = load_data(data_name=data_name,data_type='dgl')
        for encoder_name in encoder_names:
            for emb_sz in emb_szs:
                print('gen on {}|{}|{} ...'.format(encoder_name,data_name,emb_sz))
                gen_emb(encoder_name=encoder_name,g=g,data_name=data_name,emb_sz=emb_sz)
    print('gen completed!')

def routine_gen_bcdr_alpha_embs():
    encoder_names = ['bcdr']
    # data_names = ['cr','bc']
    data_names = ['bc']
    emb_szs = [16]
    # alphas = [0.1,0.5,0.9]
    alphas = [0.01,0.04,0.07]
    for data_name in data_names:
        g = load_data(data_name=data_name,data_type='dgl')
        for encoder_name in encoder_names:
            for emb_sz in emb_szs:
                for alpha in alphas:
                    print('gen on {}|{}|{}|{} ...'.format(encoder_name,data_name,emb_sz,alpha))
                    gen_emb(encoder_name=encoder_name,g=g,data_name=data_name,emb_sz=emb_sz,alpha=alpha)
    print('gen completed!')



def routine_gen_other_embs():
    encoder_names = ['netmf','grp','vs','lpca']
    # data_names = ['cr','bc']
    data_names = ['cr', 'bc']
    emb_szs = [16,112,128]
    for data_name in data_names:
        g = load_data(data_name=data_name,data_type='dgl')
        for encoder_name in encoder_names:
            for emb_sz in emb_szs:
                print('gen on {}|{}|{} ...'.format(encoder_name,data_name,emb_sz))
                gen_emb(encoder_name=encoder_name,g=g,data_name=data_name,emb_sz=emb_sz)
    print('gen completed!')

def routine_emb_convert_nrw():
    encoder_names = ['lle', 'gf', 'le']
    data_names = ['cr', 'bc']
    emb_szs = [16, 112, 128]
    emb2rw(encoder_names=encoder_names,data_names=data_names,emb_szs=emb_szs)
    print('convert all completed!')


def routine_emb_join_rw():
    # 112 + 16
    org_enc_names = ['lle','gf','le','dw','n2v','grp','netmf','vs','dadl','lpca']
    ins_enc_name = 'bcdr'
    data_names = ['cr','bc']
    emb_szs = [112,16]
    for data_name in data_names:
        for org_enc_name in org_enc_names:
            emb_join(data_name=data_name,emb_szs=emb_szs,encoder_names=[org_enc_name,ins_enc_name])


def routine_eval_bcdr_search():
    pass


if __name__ == '__main__':
    print('hello nodecls.')
    # routine_basic_test()
    # routine_eval_test()
    # dump_edgelist(data_name='cr')
    # dump_edgelist(data_name='bc')

    # dump_edgelist_csv(data_name='cr')
    # dump_edgelist_csv(data_name='bc')

    # dump_mat(data_name='cr')
    # mat = sio.loadmat('../cls/blogcatalog.mat')
    # print(mat)
    # routine_gen_all_direct_embs()
    # routine_gen_other_embs()
    # routine_gen_bcdr_embs()

    # anal_all_enc_cls(encoder_names=['lle','gf','le','dw','n2v','grp','netmf','vs','dadl','lpca','bcdr'],data_names=['cr','bc'],emb_sz=16,max_iter=100)
    # anal_all_enc_cls(encoder_names=['lle','gf','le','dw','n2v','grp','netmf','vs','dadl','lpca','bcdr'],data_names=['cr','bc'],emb_sz=128,max_iter=300)
    # anal_all_enc_cls(encoder_names=['lle','gf','le','dw','n2v','dadl','bcdr'],data_names=['cr'],emb_sz=128)

    # all_lst = ['lle','gf','le','dw','n2v','grp','netmf','vs','dadl','lpca','bcdr']
    # anal_all_enc_cls(encoder_names=all_lst + [ele+'@112'+'|bcdr@16' for ele in all_lst[:-1]], data_names=['cr','bc'], emb_sz=128,mmax_iter=300,is_comb=True)

    # anal_all_enc_cls(encoder_names=['lle','gf','le','dw','n2v','dadl','bcdr'],data_names=['bc'])
    # routine_emb_convert_nrw()

    # routine_emb_join_rw()

    # routine_gen_bcdr_alpha_embs()
    anal_bcdr_enc_cls(alphas=[0.01, 0.04, 0.07], out_dir='../cls/log', data_names=['bc'], emb_sz=16, max_iter=300,is_comb=False)
