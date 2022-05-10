import m_dqm
import m_dqm_eval
import dgl
import m_generator
import numpy as np
import pandas as pd
import time
import m_generator


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
            dist_off_eval = m_dqm_eval.DistOfflineEval(nx_g=nx_g,dqms=dqms,generator=test_generator,eval_name='dq-eval',seed=seed)
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
            dist_on_eval = m_dqm_eval.DistOnlineEval(test_sz=30000,gen_batch_sz=10000,report_interval=3,nx_g=nx_g,dqms=dqms,generator=test_generator,eval_name='dq-eval',seed=seed)
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

def combine_routine_eval1():
    hyper_dict = {}
    hyper_dict['ado'] ={'k':2}
    hyper_dict['ls'] = {
        'landmark_sz': 16,
        'use_sel': 'random',
        'margin': 2,
        'use_partition': None,
        'use_inner_sampling': 200,
    }
    hyper_dict['orion'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 100,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['rigel'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 16,
        'landmark_sz': 100,
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
        'landmark_sz': 5,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 80,
        'k': 1,
        'num_walks': 12,
        'num_workers': 5,
        'batch_landmark_sz': 1,
    }

    hyper_dict['bcdr'] = {
        'emb_sz': 16,
        'landmark_sz': 80,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 40,
        'k': 1,
        'num_walks': 20,
        'num_workers': 8,
        'batch_landmark_sz': 10,
        'batch_root_sz': 100,
        'bc_decay': 10,
        'dist_decay': 0.98,
        'out_walks': 40,
        'out_l': 10,
    }

    model_names = ['ado','ls','pll','orion','rigel','dadl']
    routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=model_names, add_names=['def']*len(model_names), params=[hyper_dict[ele] for ele in model_names], eval_type='all')

def routine_ft_bcdr():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz':80,
        'lr':0.01,
        'iters':15,
        'l':40,
        'k':1,
        'num_walks':10, # fine-tuned.
        'num_workers':8,
        'batch_landmark_sz':10,
        'batch_root_sz':100,
        'bc_decay':10,
        'dist_decay':0.5, # fine-tuned.
        'out_walks':40,
        'out_l':10,
        'use_sel':'rnd',
        'fast_query':False,
        }
    landmark_szs = [10,30,60,90]
    batch_landmark_sz_cs = [2,4,4,6]

    iters = [5,10,20,40]
    num_walks = [10,20,40]
    bc_decays = [3,5,10,20,40]
    # dist_decays = [0.1,0.5,0.9,0.98,0.999]
    dist_decays = [0.05 * ele for ele in range(1,20)]
    use_sels = ['deg','rnd']
    fast_querys = [True,False]

    # print('param landmark sz...')
    # params = [comm_dict.copy() for _ in range(len(landmark_szs))]
    # for param,landmark_sz,batch_landmark_sz in zip(params,landmark_szs,batch_landmark_sz_cs):
    #     param['landmark_sz'] = landmark_sz
    #     param['batch_landmark_sz'] = batch_landmark_sz
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr']*len(landmark_szs), add_names=[f'_lsz_{ele}' for ele in landmark_szs], params=params, eval_type='all')
    #
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
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(num_walks),
    #              add_names=[f'_nwlk_{ele}' for ele in num_walks], params=params, eval_type='all')
    #
    # print('param bc_decays...')
    # params = [comm_dict.copy() for _ in range(len(bc_decays))]
    # for param, bc_decay in zip(params, bc_decays):
    #     param['bc_decay'] = bc_decay
    # routine_eval(data_names=['cr', 'fb', 'gq'], dqm_names=['bcdr'] * len(bc_decays),
    #              add_names=[f'_bcd_{ele}' for ele in bc_decays], params=params, eval_type='all')
    #
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

    add_names = ['comm','acc','fquery','fcons']
    params = [comm_dict.copy(),comm_dict.copy(),comm_dict.copy(),comm_dict.copy()]
    params[1]['num_walks'] = 40
    params[1]['iters'] = 20
    params[2]['fast_query'] = True
    params[3]['num_walks']=10
    params[3]['landmark_sz'] = 20
    # params[2]['iters'] = 10


    routine_eval(data_names=['cr','fb','gq'], dqm_names=['bcdr']*4, add_names=add_names, params=params, eval_type='all',seed=198)


def combine_routine_eval_large1():
    hyper_dict = {}
    hyper_dict['ado'] ={'k':2}
    hyper_dict['ls'] = {
        'landmark_sz': 16,
        'use_sel': 'random',
        'margin': 2,
        'use_partition': None,
        'use_inner_sampling': 200,
    }
    hyper_dict['orion'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 8,
        'landmark_sz': 24,
        'max_iter': [5000, 1000, 100],
        'step_len': 8,
        'tol': 1e-5,
    }

    hyper_dict['rigel'] = {
        'emb_sz': 16,
        'use_sel': 'random',
        'init_sz': 8,
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
        'landmark_sz': 5,
        'lr': 0.01,
        'iters': 15,
        'p': 1,
        'q': 1,
        'l': 80,
        'k': 1,
        'num_walks': 12,
        'num_workers': 12,
        'batch_landmark_sz': 1,
    }

    # model_names = ['ado','ls','pll','orion','rigel','dadl']
    model_names = ['dadl']

    routine_eval(data_names=['db','yt'], dqm_names=model_names, add_names=['def']*len(model_names), params=[hyper_dict[ele] for ele in model_names], eval_type='all')

def routine_ft_bcdr_large():
    comm_dict = {
        'emb_sz': 16,
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'l': 40,
        'k': 1,
        'num_walks': 6,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 2, # decrease when massive graphs for load balancing.
        'batch_root_sz': 30000,
        'bc_decay': 10,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 40,
        'out_l': 10,
        'use_sel': 'rnd',
        'fast_query': False,
    }

    landmark_szs = [12, 24, 36]
    batch_landmark_sz_cs = [1,2,3]

    iters = [5, 10, 20, 40]
    num_walks = [2, 4 , 6, 8]
    out_ls = [5, 10, 15]
    bc_decays = [3, 5, 10, 20, 40]
    # dist_decays = [0.1,0.5,0.9,0.98,0.999]
    dist_decays = [0.2 * ele for ele in range(1, 5)]
    use_sels = ['deg', 'rnd']
    fast_querys = [True, False]

    # print('param out_ls...')
    # params = [comm_dict.copy() for _ in range(len(out_ls))]
    # for param, out_l in zip(params, out_ls):
    #     param['out_l'] = out_l
    # routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(out_ls),
    #              add_names=[f'_outl_{ele}' for ele in out_ls], params=params, eval_type='all', seed=189)

    # print('param landmark sz...')
    # params = [comm_dict.copy() for _ in range(len(landmark_szs))]
    # for param,landmark_sz,batch_landmark_sz in zip(params,landmark_szs,batch_landmark_sz_cs):
    #     param['landmark_sz'] = landmark_sz
    #     param['batch_landmark_sz'] = batch_landmark_sz
    # routine_eval(data_names=['db'], dqm_names=['bcdr']*len(landmark_szs), add_names=[f'_lsz_{ele}' for ele in landmark_szs], params=params, eval_type='all',seed=189)

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
    routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(num_walks),
                 add_names=[f'_nwlk_{ele}' for ele in num_walks], params=params, eval_type='all',seed=189)

    print('param dist_decays...')
    params = [comm_dict.copy() for _ in range(len(dist_decays))]
    for param, dist_decay in zip(params, dist_decays):
        param['dist_decay'] = dist_decay
    routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(dist_decays),
                 add_names=[f'_dcd_{ele}' for ele in dist_decays], params=params, eval_type='all',seed=189)

    print('param bc_decays...')
    params = [comm_dict.copy() for _ in range(len(bc_decays))]
    for param, bc_decay in zip(params, bc_decays):
        param['bc_decay'] = bc_decay
    routine_eval(data_names=['db'], dqm_names=['bcdr'] * len(bc_decays),
                 add_names=[f'_bcd_{ele}' for ele in bc_decays], params=params, eval_type='all',seed=189)


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
        'landmark_sz': 24,
        'lr': 0.01,
        'iters': 15,
        'l': 15,
        'k': 1,
        'num_walks': 6,  # fine-tuned.
        'num_workers': 12,
        'batch_landmark_sz': 2, # decrease when massive graphs for load balancing.
        'batch_root_sz': 30000,
        'bc_decay': 10,
        'dist_decay': 0.35,  # fine-tuned.
        'out_walks': 40,
        'out_l': 10,
        'use_sel': 'rnd',
        'fast_query': False,
    }

    add_names = ['comm','acc','fquery','fcons']
    params = [comm_dict.copy(),comm_dict.copy(),comm_dict.copy(),comm_dict.copy()]
    params[1]['num_walks'] = 12
    params[1]['iters'] = 20
    params[2]['fast_query'] = True
    params[3]['num_walks']=3
    params[3]['landmark_sz'] = 12
    params[3]['batch_landmark_sz'] = 1
    # params[2]['iters'] = 10


    routine_eval(data_names=['db'], dqm_names=['bcdr']*4, add_names=add_names, params=params, eval_type='all',seed=198)


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
    # routine_eval_bcdr()
    # combine_routine_eval1()
    # routine_ft_bcdr()
    # routine_ver_bcdr()
    # combine_routine_eval_large1()
    routine_ft_bcdr_large()
    # routine_ver_bcdr_large()
    # dump_edge(data_names=['cr','fb','gq','db','yt','pk'])

    # g,_ = dgl.load_graphs('../datasets/dst/Cora')
    # g = g[0]
    # g = dgl.to_bidirected(g)
    # g = dgl.to_simple_graph(g)
    # bfs = m_generator.BFS(g)
    # print(bfs.dist_between(29,84))