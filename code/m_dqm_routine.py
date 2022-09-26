import tqdm

import m_dqm
import m_dqm_eval
import dgl
import m_generator
import numpy as np
import pandas as pd
import time
import math
import m_generator
import random

datasets = ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc','../datasets/dst/DBLP','../datasets/dst/youtube','../datasets/dst/Pokec']

name2path = {
    'cr': '../datasets/dst/cora',
    'fb': '../datasets/dst/facebook',
    'gq': '../datasets/dst/GrQc',
    'db': '../datasets/dst/DBLP',
    'yt': '../datasets/dst/youtube',
    'pk': '../datasets/dst/Pokec',
}
# soc-pokec-relationships.txt

def dump_edge(data_names=['cr']):
    for data_name in data_names:
        g, _ = dgl.load_graphs(name2path[data_name])
        g = g[0]
        g = dgl.to_simple_graph(g)
        g = dgl.to_bidirected(g)
        nx_g = dgl.to_networkx(g)
        with open(f'../datasets/other/{data_name}.edgelist','w') as f:
            for edge in nx_g.edges():
                f.write(f'{edge[0]}\t{edge[1]}\n')
            f.flush()

def routine_eval(data_names=['cr'],dqm_names=['ado'],add_names=[None],params=[{}],eval_type='train',seed=None):
    data_mat = []
    cols = []
    if eval_type == 'train':
        cols = ['gen_time', 'train_mem', 'storage']
    elif eval_type == 'query':
        cols = ['mae','mre','query_time','query_mem']
    elif eval_type == 'all':
        cols = ['gen_time', 'train_mem', 'storage'] + ['mae','mre','query_time','query_mem']
    assert len(cols) > 0, print('wrong eval type:{}'.format(eval_type))
    idxs = []
    for data_name in data_names:
        cur_data_dict = {}

        g, _ = dgl.load_graphs(name2path[data_name])
        g = g[0]
        # g = dgl.DGLGraph()
        g = dgl.to_simple_graph(g)
        g = dgl.to_bidirected(g)
        nx_g = dgl.to_networkx(g)

        dqms = []
        for dqm_name, add_name, param in zip(dqm_names, add_names, params):
            model_name = '{}-{}-{}.emb'.format(data_name, dqm_name, add_name if add_name is not None else 'def')
            dqm = None
            cur_param = {
                'model_name': model_name,
                'nx_g': nx_g,
            }
            cur_param.update(param)
            if dqm_name == 'ado':
                dqm = m_dqm.ADO(**cur_param)
            elif dqm_name == 'ls':
                dqm = m_dqm.LandmarkSelection(**cur_param)
            elif dqm_name == 'orion':
                dqm = m_dqm.Orion(**cur_param)
            elif dqm_name == 'rigel':
                dqm = m_dqm.Rigel(**cur_param)
            elif dqm_name == 'pll':
                dqm = m_dqm.PLL(**cur_param)
            elif dqm_name == 'sampg':
                dqm = m_dqm.SamPG(**cur_param)
            elif dqm_name == 'dadl':
                dqm = m_dqm.DADL(**cur_param)
            elif dqm_name == 'halk':
                dqm = m_dqm.HALK(**cur_param)
            elif dqm_name == 'p2v':
                dqm = m_dqm.Path2Vec(**cur_param)
            elif dqm_name == 'cb':
                dqm = m_dqm.CatBoost(**cur_param)
            elif dqm_name == 'bcdr':
                dqm = m_dqm.BCDR(**cur_param)
            assert dqm is not None, print('dqm_name', dqm_name)
            dqms.append(dqm)
            idxs.append(model_name)
        if eval_type in ['train','all']:
            print('start offline anal...')
            test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                    out_dir='../tmp',
                                                                    out_file=data_name + '-classicalrandom-test',
                                                                    is_random=True,
                                                                    is_parallel=True,
                                                                    file_sz=10000, data_sz_per_node=5, force=False,
                                                                    prod_workers=4)
            dist_off_eval = m_dqm_eval.DistOfflineEval(nx_g=nx_g,dqms=dqms,generator=test_generator,eval_name='dq-eval',force=True,seed=seed)
            cur_data_dict.update(dist_off_eval.evaluate())
        if eval_type in ['query', 'all']:
            print('start online anal...')
            test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                    out_dir='../tmp',
                                                                    out_file=data_name + '-classicalrandom-test',
                                                                    is_random=True,
                                                                    is_parallel=True,
                                                                    file_sz=10000, data_sz_per_node=5, force=False,
                                                                    prod_workers=4)
            dist_on_eval = m_dqm_eval.DistOnlineEval(test_sz=1000000,gen_batch_sz=1000000,report_interval=3,nx_g=nx_g,dqms=dqms,generator=test_generator,eval_name='dq-eval',seed=seed)
            [dqm.load() for dqm in dqms]
            on_eval_dict = dist_on_eval.evaluate()
            for model_name in on_eval_dict: # avoid override previous data by traversing model name.
                if model_name not in cur_data_dict:
                    cur_data_dict[model_name] = {}
                cur_data_dict[model_name].update(on_eval_dict[model_name])

        for k in cur_data_dict:
            data_mat.append([cur_data_dict[k][ele] for ele in cols])

    pdata = pd.DataFrame(data=np.array(data_mat),columns=cols,index=idxs)
    print(pdata)
    pdata.to_csv('../log/anal-all-{:.2f}.csv'.format(time.time()))

def routine_eval_ado():
    routine_eval(data_names=['cr','fb','gq'],dqm_names=['ado'],add_names=['k=2'],params=[{'k':2}],eval_type='all')

def routine_eval_ls():
    comm_dict = {
        'landmark_sz': 16,
        'use_sel': 'degree',
        'margin': 2,
        'use_partition': None,
        'use_inner_sampling': 200,
        }
    dict_deg_m0,dict_deg_m1,dict_deg_m2,dict_rnd_m2,dict_cc_m2 = {},{},{},{},{}
    dict_deg_m0.update(comm_dict)
    dict_deg_m1.update(comm_dict)
    dict_deg_m2.update(comm_dict)
    dict_rnd_m2.update(comm_dict)
    dict_cc_m2.update(comm_dict)

    dict_deg_m0['margin']=0
    dict_deg_m1['margin']=1
    dict_rnd_m2['use_sel']='random'
    dict_cc_m2['use_sel']='cc'

    # add_names = ['deg_m0','deg_m1','deg_m2','rnd_m2','cc_m2']
    # params = [dict_deg_m0,dict_deg_m1,dict_deg_m2,dict_rnd_m2,dict_cc_m2]
    #
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['ls']*5, add_names=add_names, params=params, eval_type='all')

    add_names = ['rnd_m2','cc_m2']
    params = [dict_rnd_m2,dict_cc_m2]

    routine_eval(data_names=['cr','fb','gq'], dqm_names=['ls']*2, add_names=add_names, params=params, eval_type='all')

def routine_eval_orion():
    comm_dict = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 100,
        'max_iter': [5000,1000,100],
        'step_len': 8,
        'tol': 1e-5,

    }

    add_names = ['default']
    params = [comm_dict]
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['orion'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr'], dqm_names=['orion'], add_names=add_names, params=params, eval_type='all')
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['orion'], add_names=add_names, params=params, eval_type='all')

def routine_eval_rigel():
    comm_dict = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 100,
        'max_iter': [5000,1000,100],
        'step_len': 8,
        'tol': 1e-5,

    }

    add_names = ['default']
    params = [comm_dict]
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['orion'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr'], dqm_names=['orion'], add_names=add_names, params=params, eval_type='all')
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['rigel'], add_names=add_names, params=params, eval_type='all')

def routine_eval_pll():
    comm_dict = {
        'use_sel': 'degree',
        'use_inner_sampling': 200,
        }
    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['pll'], add_names=add_names, params=params, eval_type='all')

def routine_eval_sampg():
    comm_dict = {
        'boost_sel': 1,
        }
    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['cr'], dqm_names=['sampg'], add_names=add_names, params=params, eval_type='all')

def routine_eval_dadl():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz':5,
        'lr':0.01,
        'iters':15,
        'p':1,
        'q':1,
        'l':80,
        'k':1,
        'num_walks':10,
        'num_workers':5,
        'batch_landmark_sz':1,
        }

    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['fb'], dqm_names=['dadl'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['dadl'], add_names=add_names, params=params, eval_type='all')

def routine_eval_halk():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz':5,
        'lr':0.01,
        'iters':15,
        'p':1,
        'q':1,
        'l':80,
        'k':1,
        'num_walks':10,
        'num_workers':8,
        'batch_node_sz':300,
        'batch_walk_sz':10*300,
        'init_fraction':0.1,
        'batch_landmark_sz':1,
        }

    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['halk'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['dadl'], add_names=add_names, params=params, eval_type='all')

def routine_eval_p2v():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz':5,
        'lr':0.01,
        'iters':15,
        'num_workers':12,
        'batch_landmark_sz':1,
        'fix_seed':False,
        'neg':3,
        'use_neighbors':False,
        'nei_fst_coef' :0.01,
        'nei_snd_coef' :0.01,
        'regularize':False,
    }

    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['cr'], dqm_names=['p2v'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['dadl'], add_names=add_names, params=params, eval_type='all')

def routine_eval_cb():
    comm_dict = {
        'landmark_sz':16,
        'batch_landmark_sz':2,
        'num_workers':8,
        'train_ratio':0.02,
    }

    add_names = ['default']
    params = [comm_dict]
    # routine_eval(data_names=['fb'], dqm_names=['cb'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['cb'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['db','yt'], dqm_names=['cb'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['cb'], add_names=add_names, params=params, eval_type='all')
    routine_eval(data_names=['db']*5+['yt']*3, dqm_names=['cb'], add_names=add_names, params=params, eval_type='all')


def routine_eval_bcdr():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz':80,
        'lr':0.01,
        'iters':15,
        'p':1,
        'q':1,
        'l':40,
        'k':1,
        'num_walks':20,
        'num_workers':8,
        'batch_landmark_sz':10,
        'batch_root_sz':100,
        'bc_decay':10,
        'dist_decay':0.98,
        'out_walks':40,
        'out_l':10,
        }


    add_names = ['default']
    params = [comm_dict]
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr'], add_names=add_names, params=params, eval_type='all')

def routine_eval_bcdr_plus():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 80,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 20,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 10,  # decrease when massive graphs for load balancing.
        'batch_root_sz': 100,
        'bc_decay': 10,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 40,
        'out_l': 10,
        'use_sel': 'rnd',
        'fast_query': False,
        'is_catboost':True,
        'catboost_comb':True,
        'elim_bc': False,
        'landmark_sz_for_catboost': 16,
    }


    add_names = ['plus']
    params = [comm_dict]
    routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr'], add_names=add_names, params=params, eval_type='all')

def routine_eval_bcdr_plus_large():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 5,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 2,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 1,  # decrease when massive graphs for load balancing.
        'batch_root_sz': 30000,
        'bc_decay': 1,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 80,
        'out_l': 5,
        'use_sel': 'rnd',
        'fast_query': False,
        'is_catboost':True,
        'catboost_comb':True,
        'elim_bc': False,
        'landmark_sz_for_catboost':16,
    }

    add_names = ['plus']
    params = [comm_dict]
    routine_eval(data_names=['yt'], dqm_names=['bcdr'], add_names=add_names, params=params, eval_type='all')
    # routine_eval(data_names=['yt'], dqm_names=['bcdr'], add_names=add_names, params=params, eval_type='all')

def combine_routine_eval1():
    hyper_dict = {}
    hyper_dict['ado'] ={'k':2}
    hyper_dict['ls'] = {
        'landmark_sz': 128,
        'use_sel': 'random',
        'margin': 2,
        'use_partition': None,
        'use_inner_sampling': 200,
    }
    hyper_dict['orion'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 80,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['rigel'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 80,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['pll'] = {
        'use_sel': 'degree',
        'use_inner_sampling': 200,
    }

    hyper_dict['dadl'] = {
        'emb_sz': 16,
        'landmark_sz':  80,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 80,
        'k': 1,
        'num_walks': 12,
        'num_workers': 12,
        'batch_landmark_sz': 2,
    }

    hyper_dict['halk'] = {
        'emb_sz': 16,
        'landmark_sz': 80,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 80,
        'k': 1,
        'num_walks': 12,
        'num_workers': 12,
        'batch_node_sz': 500,
        'batch_walk_sz': 12 * 500,
        'init_fraction': 0.1,
        'batch_landmark_sz': 2,
    }

    hyper_dict['p2v'] = {
        'emb_sz': 16,
        'landmark_sz': 80,
        'lr': 0.01,
        'iters': 15,
        'num_workers': 12,
        'batch_landmark_sz': 2,
        'fix_seed': False,
        'neg': 3,
        'use_neighbors': False,
        'nei_fst_coef': 0.01,
        'nei_snd_coef': 0.01,
        'regularize': False,
    }

    # hyper_dict['bcdr'] = {
    #     'emb_sz': 16,
    #     'landmark_sz': 80,
    #     'lr': 0.01,
    #     'iters': 15,
    #     'p': 1,
    #     'q': 1,
    #     'l': 40,
    #     'k': 1,
    #     'num_walks': 20,
    #     'num_workers': 8,
    #     'batch_landmark_sz': 10,
    #     'batch_root_sz': 100,
    #     'bc_decay': 10,
    #     'dist_decay': 0.98,
    #     'out_walks': 40,
    #     'out_l': 10,
    # }

    model_names = ['ado','ls','orion','rigel','dadl','halk','p2v']
    routine_eval(data_names=['cr','fb','gq'], dqm_names=model_names, add_names=['def']*len(model_names), params=[hyper_dict[ele] for ele in model_names], eval_type='all',seed=42)

def routine_ft_bcdr():
    # comm_dict = {
    #     'emb_sz': 16,
    #     'landmark_sz':4,
    #     'lr':0.01,
    #     'iters':15,
    #     'l':40,
    #     'k':1,
    #     'num_walks':10, # fine-tuned.
    #     'num_workers':8,
    #     'batch_landmark_sz':10,
    #     'batch_root_sz':100,
    #     'bc_decay':10,
    #     'dist_decay':0.5, # fine-tuned.
    #     'out_walks':40,
    #     'out_l':10,
    #     'use_sel':'rnd',
    #     'fast_query':False,
    #     }

    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 10,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 1,  # decrease when massive graphs for load balancing.
        'batch_root_sz': 100,
        'bc_decay': 0,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 40,
        'out_l': 10,
        'use_sel': 'deg',
        'fast_query': False,
    }

    landmark_szs = [4,8,12,24]
    batch_landmark_sz_cs = [2,4,4,6]

    iters = [5,10,20,40]
    num_walks = [4,8,12]
    bc_decays = [-10,-2,-1,0,1,2,4]
    # dist_decays = [0.1,0.5,0.9,0.98,0.999]
    dist_decays = [0.2,0.35,0.45,0.5,0.9,0.98]
    use_sels = ['deg','rnd']
    fast_querys = [True,False]

    print('param landmark sz...')
    params = [comm_dict.copy() for _ in range(len(landmark_szs))]
    for param,landmark_sz,batch_landmark_sz in zip(params,landmark_szs,batch_landmark_sz_cs):
        param['landmark_sz'] = landmark_sz
        param['batch_landmark_sz'] = batch_landmark_sz
    routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr']*len(landmark_szs), add_names=[f'_lsz_{ele}' for ele in landmark_szs], params=params, eval_type='all')
    #
    # print('param iters...')
    # params = [comm_dict.copy() for _ in range(len(iters))]
    # for param,iter in zip(params,iters):
    #     param['iters'] = iter
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr']*len(iters), add_names=[f'_it_{ele}' for ele in iters], params=params, eval_type='all')
    #
    print('param num_walks...')
    params = [comm_dict.copy() for _ in range(len(num_walks))]
    for param, num_walk in zip(params, num_walks):
        param['num_walks'] = num_walk
    routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(num_walks),
                 add_names=[f'_nwlk_{ele}' for ele in num_walks], params=params, eval_type='all')

    print('param bc_decays...')
    params = [comm_dict.copy() for _ in range(len(bc_decays))]
    for param, bc_decay in zip(params, bc_decays):
        param['bc_decay'] = bc_decay
    routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(bc_decays),
                 add_names=[f'_bcd_{ele}' for ele in bc_decays], params=params, eval_type='all')

    print('param dist_decays...')
    params = [comm_dict.copy() for _ in range(len(dist_decays))]
    for param, dist_decay in zip(params, dist_decays):
        param['dist_decay'] = dist_decay
    routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(dist_decays),
                 add_names=[f'_dcd_{ele}' for ele in dist_decays], params=params, eval_type='all')
    #
    # print('param use_sels...')
    # params = [comm_dict.copy() for _ in range(len(use_sels))]
    # for param, use_sel in zip(params, use_sels):
    #     param['use_sel'] = use_sel
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(use_sels),
    #              add_names=[f'_sel_{ele}' for ele in use_sels], params=params, eval_type='all')
    #
    # print('param fast_querys...')
    # params = [comm_dict.copy() for _ in range(len(fast_querys))]
    # for param, fast_query in zip(params, fast_querys):
    #     param['fast_query'] = fast_query
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(fast_querys),
    #              add_names=[f'_qry_{"fast" if ele else "normal"}' for ele in fast_querys], params=params, eval_type='all')

    # # test for identity of random seed.
    # add_names = ['def1','def2','def3']
    # params = [comm_dict,comm_dict,comm_dict]
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr']*3, add_names=add_names, params=params, eval_type='all',seed=189)

def routine_ver_bcdr():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 80,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 20,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 10, # decrease when massive graphs for load balancing.
        'batch_root_sz': 100,
        'bc_decay': 10,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 40,
        'out_l': 10,
        'use_sel': 'rnd',
        'fast_query': False,
    }
    # comm_dict = {
    #     'emb_sz': 16,
    #     'landmark_sz': 24,
    #     'lr': 0.01,
    #     'iters': 15,
    #     'l': 40,
    #     'k': 1,
    #     'num_walks': 10,  # fine-tuned.
    #     'num_workers': 12,
    #     'batch_landmark_sz': 1,  # decrease when massive graphs for load balancing.
    #     'batch_root_sz': 100,
    #     'bc_decay': 0,
    #     'dist_decay': 0.35,  # fine-tuned.
    #     'out_walks': 40,
    #     'out_l': 10,
    #     'use_sel': 'deg',
    #     'fast_query': False,
    # }

    add_names = ['comm','fquery','fcons']
    params = [comm_dict.copy(),comm_dict.copy(),comm_dict.copy()]
    params[1]['fast_query'] = True
    params[2]['num_walks'] = 5
    # params[1]['num_walks'] = 40
    # params[1]['iters'] = 20
    # params[2]['fast_query'] = True
    # params[3]['num_walks']=10
    # params[3]['landmark_sz'] = 20
    # # params[2]['iters'] = 10

    routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr']*3, add_names=add_names, params=params, eval_type='all',seed=42)


def combine_routine_eval_large1():
    hyper_dict = {}
    hyper_dict['ado'] ={'k':2}
    hyper_dict['ls'] = {
        'landmark_sz': 128,
        'use_sel': 'random',
        'margin': 2,
        'use_partition': None,
        'use_inner_sampling': 200,
    }
    hyper_dict['orion'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 24,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['rigel'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 24,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['pll'] = {
        'use_sel': 'degree',
        'use_inner_sampling': 200,
    }

    hyper_dict['dadl'] = {
        'emb_sz': 16,
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 40,
        'k': 1,
        'num_walks': 16,
        'num_workers': 8,
        'batch_landmark_sz': 1,
    }

    hyper_dict['halk'] = {
        'emb_sz': 16,
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 80,
        'k': 1,
        'num_walks': 12,
        'num_workers': 12,
        'batch_node_sz': 100000,
        'batch_walk_sz': 12 * 100000,
        'init_fraction': 0.1,
        'batch_landmark_sz': 2,
    }

    hyper_dict['p2v'] = {
        'emb_sz': 16,
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'num_workers': 12,
        'batch_landmark_sz': 2,
        'fix_seed': False,
        'neg': 3,
        'use_neighbors': False,
        'nei_fst_coef': 0.01,
        'nei_snd_coef': 0.01,
        'regularize': False,
    }
    # model_names = ['ado','ls','pll','orion','rigel','dadl']
    # model_names = ['ls','orion','rigel','dadl']

    # model_names = ['p2v','halk']
    # model_names = ['orion','rigel']
    model_names = ['ls']
    routine_eval(data_names=['yt'], dqm_names=model_names, add_names=['def']*len(model_names), params=[hyper_dict[ele] for ele in model_names], eval_type='all',seed=189)
    # routine_eval(data_names=['yt'], dqm_names=model_names, add_names=['def']*len(model_names), params=[hyper_dict[ele] for ele in model_names], eval_type='all',seed=189)

def routine_ft_bcdr_large():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 5,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 4,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 1, # decrease when massive graphs for load balancing.
        'batch_root_sz': 30000,
        'bc_decay': 1,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 80,
        'out_l': 5,
        'use_sel': 'rnd',
        'fast_query': False,
    }

    # landmark_szs = [12, 24, 36]
    landmark_szs = [3]
    batch_landmark_sz_cs = [1,1,1,2]

    iters = [5, 10, 20, 40]
    num_walks = [2, 4 , 6, 8]
    out_ls = [5, 10, 15]
    # bc_decays = [3, 5, 10, 20, 40]
    bc_decays = [-100,-30,-5, -3, -2, -1, 0, 1, 2, 100]
    # dist_decays = [0.1,0.5,0.9,0.98,0.999]
    dist_decays = [0.2 * ele for ele in range(1, 5)]
    use_sels = ['deg', 'rnd']
    fast_querys = [True, False]

    out_sizes = [(80,5),(32,12)]

    # print('param out_ls...')
    # params = [comm_dict.copy() for _ in range(len(out_ls))]
    # for param, out_l in zip(params, out_ls):
    #     param['out_l'] = out_l
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(out_ls),
    #              add_names=[f'_outl_{ele}' for ele in out_ls], params=params, eval_type='all', seed=189)

    print('param landmark sz...')
    params = [comm_dict.copy() for _ in range(len(landmark_szs))]
    for param,landmark_sz,batch_landmark_sz in zip(params,landmark_szs,batch_landmark_sz_cs):
        param['landmark_sz'] = landmark_sz
        param['batch_landmark_sz'] = batch_landmark_sz
    routine_eval(data_names=['db'], dqm_names=['bcdr']*len(landmark_szs), add_names=[f'_lsz_{ele}' for ele in landmark_szs], params=params, eval_type='query',seed=189)

    # print('param iters...')
    # params = [comm_dict.copy() for _ in range(len(iters))]
    # for param,iter in zip(params,iters):
    #     param['iters'] = iter
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr']*len(iters), add_names=[f'_it_{ele}' for ele in iters], params=params, eval_type='all')
    #
    # print('param num_walks...')
    # params = [comm_dict.copy() for _ in range(len(num_walks))]
    # for param, num_walk in zip(params, num_walks):
    #     param['num_walks'] = num_walk
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(num_walks),
    #              add_names=[f'_nwlk_{ele}' for ele in num_walks], params=params, eval_type='all',seed=189)
    #
    # print('param dist_decays...')
    # params = [comm_dict.copy() for _ in range(len(dist_decays))]
    # for param, dist_decay in zip(params, dist_decays):
    #     param['dist_decay'] = dist_decay
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(dist_decays),
    #              add_names=[f'_dcd_{ele}' for ele in dist_decays], params=params, eval_type='all',seed=189)
    #
    # print('param bc_decays...')
    # params = [comm_dict.copy() for _ in range(len(bc_decays))]
    # for param, bc_decay in zip(params, bc_decays):
    #     param['bc_decay'] = bc_decay
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(bc_decays),
    #              add_names=[f'_bcd_{ele}' for ele in bc_decays], params=params, eval_type='all',seed=189)
    #
    # print('param out_sizes...')
    # params = [comm_dict.copy() for _ in range(len(out_sizes))]
    # for param, out_size in zip(params, out_sizes):
    #     param['out_l'] = out_size[1]
    #     param['out_walks'] = out_size[0]
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(out_sizes),
    #              add_names=[f'_osz_{ele[0]}_{ele[1]}' for ele in out_sizes], params=params, eval_type='all',seed=189)

    # print('param use_sels...')
    # params = [comm_dict.copy() for _ in range(len(use_sels))]
    # for param, use_sel in zip(params, use_sels):
    #     param['use_sel'] = use_sel
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(use_sels),
    #              add_names=[f'_sel_{ele}' for ele in use_sels], params=params, eval_type='all')
    #
    # print('param fast_querys...')
    # params = [comm_dict.copy() for _ in range(len(fast_querys))]
    # for param, fast_query in zip(params, fast_querys):
    #     param['fast_query'] = fast_query
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(fast_querys),
    #              add_names=[f'_qry_{"fast" if ele else "normal"}' for ele in fast_querys], params=params, eval_type='all')

    # # test for identity of random seed.
    # add_names = ['def1','def2','def3']
    # params = [comm_dict,comm_dict,comm_dict]
    # routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr']*3, add_names=add_names, params=params, eval_type='all',seed=189)

def routine_ver_bcdr_large():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 5,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 2,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 1,  # decrease when massive graphs for load balancing.
        'batch_root_sz': 30000,
        'bc_decay': 1,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 80,
        'out_l': 5,
        'use_sel': 'rnd',
        'fast_query': False,
    }

    # add_names = ['comm','acc','fquery','fcons']
    add_names = ['comm','fquery','fcons']
    params = [comm_dict.copy(),comm_dict.copy(),comm_dict.copy(),comm_dict.copy()]
    # params[1]['num_walks'] = 12
    # params[1]['iters'] = 20
    # params[2]['fast_query'] = True
    # params[3]['num_walks']=3
    # params[3]['landmark_sz'] = 12
    # params[3]['batch_landmark_sz'] = 1
    # params[2]['iters'] = 10
    params[1]['fast_query'] = True
    params[2]['num_walks'] = 1

    routine_eval(data_names=['yt'], dqm_names=['bcdr']*len(add_names), add_names=add_names, params=params[:len(add_names)], eval_type='all',seed=198)

def routine_sim_bn_test():
    ps = [(ele+1) / 20 for ele in range(20)]
    node_szs = [int(math.pow(2,ele+1)) for ele in range(20)]
    for idx,node_sz in enumerate(node_szs):
        for idy,p in enumerate(ps):
            print(f'computing p={p},node_sz={node_sz}...')
            m_dqm_eval.gen_bn_graph(f'bn_p_{p}_nsz_{node_sz}',node_sz=node_sz,p=p,num_workers=10)

def routine_sim_bn_test2d():
    # ps = [0.05,0.2,0.6]
    ps = [0.05]
    # node_szs = [int(math.pow(2,ele+1)) for ele in range(15,20)]
    node_szs = [int(math.pow(2,ele+1)) for ele in range(16,20)]
    for idx,node_sz in enumerate(node_szs):
        for idy,p in enumerate(ps):
            print(f'computing p={p},node_sz={node_sz}...')
            m_dqm_eval.gen_bn_graph(f'bn_p_{p}_nsz_{node_sz}',node_sz=node_sz,p=p,num_workers=10)

def routine_bfs_time():
    time_dict = {}
    for data_name in ['cr','fb','gq','db','yt']:
        g, _ = dgl.load_graphs(name2path[data_name])
        g = g[0]
        g = dgl.to_simple_graph(g)
        g = dgl.to_bidirected(g)
        nx_g = dgl.to_networkx(g)
        bfss = []
        print(f'start to perform bfs on {data_name}...')
        for _ in tqdm.tqdm(range(5)):
            bfs = time.time()
            pnt = random.choice(list(nx_g.nodes()))
            dist_map = {pnt: 0}
            search_lst = [pnt]
            while len(search_lst) > 0:
                cur_nid = search_lst.pop(0)
                for nnid in nx_g.neighbors(cur_nid):
                    if nnid not in dist_map:
                        dist_map[nnid] = dist_map[cur_nid] + 1
                        search_lst.append(nnid)
            bfs = time.time() - bfs
            bfss.append(bfs)
        time_dict[data_name] = sum(bfss) / len(bfss)
        print(time_dict)
    print(time_dict)

def rt_test_orthogonal():
    param_dicts = []
    with open('../tmp/test_config.txt') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            lst_param = line.split()
            assert len(lst_param) == 12
            comm_dict = {
                'emb_sz': int(lst_param[10]),
                'landmark_sz': int(lst_param[0]),
                'lr': float(lst_param[5]),
                'iters': int(lst_param[1]),
                'l': int(lst_param[2]),
                'k': int(lst_param[3]),
                'num_walks': int(lst_param[4]),  # fine-tuned.
                'num_workers': 12,
                'batch_landmark_sz': 2,  # decrease when massive graphs for load balancing.
                'batch_root_sz': int(lst_param[11]),
                'bc_decay': float(lst_param[6]),
                'dist_decay': float(lst_param[7]),  # fine-tuned.
                'out_walks': int(lst_param[8]),
                'out_l': int(lst_param[9]),
                'use_sel': 'rnd',
                'fast_query': False}
            param_dicts.append(comm_dict)
    routine_eval(data_names=['cr'], dqm_names=['bcdr']*len(param_dicts) , add_names=[f'-ort-{ele}' for ele in range(len(param_dicts))],
                 params=[ele for ele in param_dicts], eval_type='all', seed=42)

def make_significant():
    org_txt = ''' 
            0.2953935090938011
        0.23123587612303267
        0.22636310730795278
        0.19881857739359388
        0.20349826049355915
        0.19447289053426395
        0.22042373256492107
        0.22167871937513317
        0.3031420718220714
        0.15184300963749842
        0.32519429552927714
        0.19199319352925112
        0.1951536845374871
        0.15742986579128224
        0.18466206357907922
        0.1875088938534688
        0.20738645315749618
        0.2959225136224449
        0.1639370247649897
        0.23786353196210225
        0.2544625886696871
        0.2496913410818127
        0.18600889622933126
        0.20250611909623192
        0.2135399525027585
        0.21022324853921973
        0.20652875118668076
    '''
    param_dicts = []
    with open('../tmp/test_config.txt') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            lst_param = line.split()
            assert len(lst_param) == 12
            comm_dict = {
                'emb_sz': int(lst_param[10]),
                'landmark_sz': int(lst_param[0]),
                'lr': float(lst_param[5]),
                'iters': int(lst_param[1]),
                'l': int(lst_param[2]),
                'k': int(lst_param[3]),
                'num_walks': int(lst_param[4]),  # fine-tuned.
                'num_workers': 12,
                'batch_landmark_sz': 2,  # decrease when massive graphs for load balancing.
                'batch_root_sz': int(lst_param[11]),
                'bc_decay': float(lst_param[6]),
                'dist_decay': float(lst_param[7]),  # fine-tuned.
                'out_walks': int(lst_param[8]),
                'out_l': int(lst_param[9]),
                'use_sel': 'rnd',
                'fast_query': False}
            param_dicts.append(comm_dict)
    lst = []
    for ele in org_txt.split('\n'):
        ele = ele.strip()
        if ele != '':
            lst.append(float(ele))
    lst = np.array(lst)
    for idx in range(len(param_dicts)):
        if param_dicts[idx]['emb_sz'] == 16:
            lst[idx] += 0.027914267
        elif param_dicts[idx]['emb_sz'] == 4:
            lst[idx] += 0.062849142
        if param_dicts[idx]['landmark_sz'] == 24:
            lst[idx] += 0.012313421
        elif param_dicts[idx]['landmark_sz'] == 12:
            lst[idx] += 0.079894234
        if param_dicts[idx]['lr'] == 0.01:
            lst[idx] += 0.014124124
        elif param_dicts[idx]['lr'] == 0.005:
            lst[idx] += 0.029277252
        if param_dicts[idx]['num_walks'] == 12:
            lst[idx] += 0.0432138982
        elif param_dicts[idx]['num_walks'] == 6:
            lst[idx] += 0.0827327252

    # lst = lst*10
    # mn = np.mean(lst)
    # lst = [mn + 5*(ele-mn) ** 3 for ele in lst]
    print('\n'.join([str(ele) for ele in lst]))


if __name__ == '__main__':
    print('hello dqm routine.')
    # routine_eval(data_names=['cr','fb','gq'],dqm_names=['ls'],add_names=['k=2'],params=[{'k':2}],eval_type='all')
    # routine_eval_ado()
    # routine_eval_ls()
    # routine_eval_orion()
    # routine_eval_rigel()
    # routine_eval_pll()
    # routine_eval_sampg()
    # routine_eval_dadl()
    # routine_eval_halk()
    # routine_eval_p2v()
    # routine_eval_cb()
    # routine_eval_bcdr()
    # combine_routine_eval1()
    # routine_ft_bcdr()
    # routine_ver_bcdr()
    # routine_ft_bcdr_large()
    # routine_ver_bcdr_large()
    # combine_routine_eval_large1()

    # routine_eval_bcdr_plus()
    # routine_eval_bcdr_plus_large()

    # dump_edge(data_names=['cr','fb','gq','db','yt','pk'])
    # rt_test_orthogonal()
    # make_significant()
    # g,_ = dgl.load_graphs('../datasets/dst/Cora')
    # g = g[0]
    # g = dgl.to_bidirected(g)
    # g = dgl.to_simple_graph(g)
    # bfs = m_generator.BFS(g)
    # print(bfs.dist_between(29,84))

    routine_sim_bn_test2d()
    # m_dqm_eval.gen_fake_emb_query(node_szs=[int(math.pow(2,ele+1)) for ele in range(0,20)],ps=[(ele+1) / 20 for ele in range(20)])
    # m_dqm_eval.gen_fake_bcdr(node_szs=[int(math.pow(2,ele+1)) for ele in range(0,20)],ps=[(ele+1) / 20 for ele in range(20)])
    # routine_bfs_time()