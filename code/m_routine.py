import dgl
import torch as tc
import torch.nn as nn
import m_encoder
import embedding_proc
import m_decoder
import m_generator
import m_evaluator
import m_logger
import os
import time
import m_selector
import m_generator_parallel
import m_gf
import m_le
import m_lle
import m_node2vec
import m_manager
import m_smgnn
import m_dage
import m_grarep
import m_lpca
import numpy as np
import math
import m_karate_encoders

# 非embed模型routine
def inst_routine1(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.LinearReg(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# embed模型routine
def inst_routine2(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.EmbedModel(emb_sz=emb_sz,g=g)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.000001,
        'epoches1': 5000,
        'epoches2': 10000,
        'batch_sz1': 5000,
        'batch_sz2': 10000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':True
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# DADL
def inst_routine_dadl(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.DADL(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# GraRep
def inst_routine_grarep(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.DADL(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name': data_name,
        'use_timed': False
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# NetMF
def inst_routine_netmf(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.LinearReg(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name': data_name,
        'use_timed': False
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# Verse
def inst_routine_verse(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.LinearReg(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name': data_name,
        'use_timed': False
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# LPCA
def inst_routine_lpca(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.LinearReg(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name': data_name,
        'use_timed': False
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch':10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()


# CDGCN
def inst_routine_cdgcn(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.CDGCN(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name': data_name,
        'use_timed': False
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 50,
        'epoches2': 150,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

# BCDR
def inst_routine_bcdr(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1,weight_decay=0.1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # pad graph BC.
    bc_file = dataset + '-BC.txt'
    if os.path.exists(bc_file):
        g.ndata['bc'] = tc.zeros(g.num_nodes(),1)
        with open(bc_file) as f:
            cnt = 0
            for line in f:
                cnt += 1
                line = line.strip()
                nid,bc = line.split('-',1)
                nid = int(nid)
                bc = float(bc)
                g.ndata['bc'][nid] = bc
            # assert cnt >= g.num_nodes(),print('g bc cnt {} < node cnt {}'.format(cnt,g.num_nodes()))

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    encoder.alpha = weight_decay
    print('Caution! current weight decay is {} for bcdr.'.format(weight_decay))
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.DADL(emb_sz)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)

    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator,
        'data_name':data_name,
        'use_timed':True
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 150,
        'epoches2': 500,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
        'save_between_epoch': 10
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name, gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

def combine_routines():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
                                           force=False, num_walks=80, walk_lens=40, window_sz=20, max_mem=0, seed=0,
                                           is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None, force=False)
    encoder_le = m_le.LEEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None, force=False)
    encoder_gf = m_gf.GFEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None, force=False, iter=500,
                                r=1.0, lr=1e-4, print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
                                            force=False, num_walks=80, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
                                            is_directed=False, is_weighted=False, weight_arr=None)
    encoders = [encoder_dw, encoder_lle, encoder_le, encoder_gf, encoder_n2v]
    emb_sz = [16, 32, 64, 128, 256]

    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
                '../datasets/dst/youtube']
    landmark_cnts = [100, 100, 200, 200]
    # data_names = ['fb','bc','tw','yt']
    data_names = ['fb', 'bc', 'tw']

    model_names = ['dw', 'lle', 'le', 'gf', 'n2v']
    classical_early_breaks = [-1, 25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine1(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)


def combine_routines1():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-3,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    encoders = [encoder_dw,encoder_lle,encoder_le,encoder_gf,encoder_n2v]

    emb_sz = [16,32,64,128,256]

    datasets = ['../datasets/dst/facebook']
    landmark_cnts = [100]
    data_names = ['fb']

    model_names = ['dw','lle','le','gf','n2v']
    classical_early_breaks = [-1]
    for e_emb_sz in emb_sz:
        for e_encoder,e_model_name in zip(encoders,model_names):
            for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
                inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)
            # break

def combine_routines2():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-4,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    encoders = [encoder_dw,encoder_lle,encoder_le,encoder_gf,encoder_n2v]
    emb_sz = [16,32,64,128,256]
    datasets = ['../datasets/dst/BlogCatalog-dataset']
    landmark_cnts = [200]
    data_names = ['bc']

    model_names = ['dw','lle','le','gf','n2v']
    classical_early_breaks = [25]
    for e_emb_sz in emb_sz:
        for e_encoder,e_model_name in zip(encoders,model_names):
            for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
                inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)


def combine_routines3():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-4,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    encoders = [encoder_dw,encoder_lle,encoder_n2v]
    emb_sz = [16,64,128]
    datasets = ['../datasets/dst/twitter']
    landmark_cnts = [200]
    data_names = ['tw']

    model_names = ['dw','lle','n2v']
    classical_early_breaks = [25]
    for e_emb_sz in emb_sz:
        for e_encoder,e_model_name in zip(encoders,model_names):
            for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
                inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)

def combine_routines4_REAL():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-4,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    encoders = [encoder_dw,encoder_lle,encoder_n2v]
    emb_sz = [16,64,128]
    datasets = ['../datasets/dst/youtube']
    landmark_cnts = [200]
    data_names = ['yt']

    model_names = ['dw','lle','n2v']
    classical_early_breaks = [25]
    for e_emb_sz in emb_sz:
        for e_encoder,e_model_name in zip(encoders,model_names):
            for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
                inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)


def combine_routines4():
    encoder_orion = m_encoder.OrionEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
                                           force=True,scale=0.25,sample_method='uniform',neg_permit=True)
    encoders = [encoder_orion]

    # emb_sz = [16, 32, 64, 128, 256]
    emb_sz = [128]

    # datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter','../datasets/dst/youtube']
    datasets = ['../datasets/dst/karate']

    # landmark_cnts = [100, 100, 200, 200]
    landmark_cnts = [5]

    # data_names = ['fb','bc','tw','yt']
    data_names = ['ka']

    # model_names = ['dw', 'lle', 'le', 'gf', 'n2v']
    model_names = ['orion']

    # classical_early_breaks = [-1, 25, 25, 25]
    classical_early_breaks = [-1]

    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine2(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def inst_yt1(landmark_nodes,id=1):
    g, _ = dgl.load_graphs('../datasets/dst/youtube')
    g = g[0]
    workers = 16
    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                       out_dir='../tmp',
                                                                       out_file='yt'+str(id) + '-fastrandom',
                                                                       is_random=True, is_parallel=True, file_sz=10000,
                                                                       data_sz_per_node=5,
                                                                       landmark_nodes=landmark_nodes, force=False,
                                                                       prod_workers=4)


    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))

def inst_yt2(landmark_nodes):
    g, _ = dgl.load_graphs('../datasets/dst/youtube')
    g = g[0]

    st_time = time.time()
    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file='yt'+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))


def inst_routine_toy(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',encoder=None,landmark_cnt=100,classical_early_break=-1):
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder.g = g
    encoder.out_file = model_name
    encoder.emb_sz = emb_sz
    # encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.LinearReg(emb_sz)


    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=1,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=1)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=1,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=1)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=1,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,force=False,prod_workers=1)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=1, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=1)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=1, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=1)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=1, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=1)


    ged = m_manager.GEDManager(workers=1)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 10000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': -1,  # ?
        'val_sz': -1,
        'landmark_sz': landmark_cnt,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 100,
        'epoches2': 5000,
        'batch_sz1': 100,
        'batch_sz2': 300,
        'decoder': decoder,
        'batch_sz_val': 10,
        'stop_cond1':-1,
        'stop_cond2':-1
    }
    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()


def combine_routines_toy():
    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=1,out_dir='../tmp',out_file=None,force=False,num_walks=5,walk_lens=10,window_sz=5,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=1,out_dir='../tmp',out_file=None,force=False)
    encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=1,out_dir='../tmp',out_file=None,force=False)
    encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=1,out_dir='../tmp',out_file=None,force=False,iter=1e4,r=1.0,lr=1e-4,print_step=10000000)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=1,out_dir='../tmp',out_file=None,force=False,num_walks=5,walk_lens=10,window_sz=5,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    encoders = [encoder_dw,encoder_lle,encoder_le,encoder_gf,encoder_n2v]
    emb_sz = [2]
    datasets = ['../datasets/dst/karate']
    landmark_cnts = [5]
    data_names = ['ka']
    model_names = ['dw','lle','le','gf','n2v']
    classical_early_breaks = [-1]
    for e_emb_sz in emb_sz:
        for e_encoder,e_model_name in zip(encoders,model_names):
            for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
                inst_routine_toy(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)
'''
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~sm gnn~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
'''

def inst_routine_sm(emb_sz=16,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-sm-emb=16'):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16
    # 构建encoder
    encoder = m_smgnn.SmGnnEncoder(g=g,emb_sz=emb_sz,workers=workers,out_dir='../tmp',out_file=model_name+'.encoder')
    encoder.config_OracleGraph(g=g,depth=0, cluster_sz=-1, cluster_num=0.1, proto_sz=-1, expand_speed=1, need_remove_dup=False,out_file='unknown')
    # 生成embedding
    encoder.train()
    # decoder = m_smgnn.SmGnnDecoder(sub_type='condonly',emb_sz=encoder.emb_sz,og=encoder.og)
    og = encoder.get_embedded_og()
    decoder = m_smgnn.SmGnnDecoder_condonly(emb_sz=encoder.emb_sz, og=og)

    # parallel acceleration (之后要换random node pair，现在先调代码吧。。。)
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=None,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=None,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)


    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=-1)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=None,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=None,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': None,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': False,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches': 100,
        'batch_sz': 500,
        'decoder': decoder,
        'batch_sz_val': 100,
        'stop_cond':-1,
        'stop_cond2':-1,
        'stop_cond1':-1,
        'is_embed_model':True
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@encoder', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

def inst_routine_ext_dw_const(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.MLP(emb_sz)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=5,landmark_nodes=landmark_nodes,force=False,prod_workers=4)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)
    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                            out_file=data_name+'-fastrandom', is_random=True, is_parallel=True,
                                            file_sz=10000, data_sz_per_node=5, landmark_nodes=landmark_nodes,
                                            force=False, prod_workers=4)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.GEDManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 50,
        'epoches2': 100,
        'batch_sz1': 500,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
    }

    ged.configModel(g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()

def inst_routine_ext_dw(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',landmark_cnt=100,classical_early_break=-1):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.MLP(emb_sz)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.StochasticNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=10,
                                                                       out_dir='../tmp', out_file='fb-snoderange',
                                                                       is_random=True, is_parallel=True, file_sz=10000,
                                                                       force=False, pair_sz=2000, proximity_sz=20)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)

    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.StochasticNodeRangeGenerator_p(g=g, scheme=m_generator.BFS(None), workers=10,
                                                                       out_dir='../tmp', out_file='fb-snoderange',
                                                                       is_random=True, is_parallel=True, file_sz=10000,
                                                                       force=False, pair_sz=200, proximity_sz=20)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.RangeManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 10000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 50,
        'epoches2': 100,
        'batch_sz1': 200,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':-1,
        'is_embed_model':False,
    }

    ged.configModel(l1=0.3,g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()


def inst_routine_ext_dw_fast(emb_sz=128,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-dw-emb=16',landmark_cnt=100,classical_early_break=-1,pair_sz=200,per_node_sz=40,prox_sz=2,l1=0.05):
    print('========== cur model',model_name,'dataset',data_name,'emb',emb_sz,'=======')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    workers = 16

    # 补充encoder需要的参数
    encoder = m_encoder.DeepWalkEncoder(g=g,emb_sz=emb_sz,workers=16,out_dir='../tmp',out_file=model_name,force=False,num_walks=80,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    decoder = m_decoder.MLP(emb_sz)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    ds = m_selector.DegreeSelector(g=g)
    landmark_nodes = ds.perform(cnt=landmark_cnt,action='max')

    # parallel acceleration
    train_generator_acc = m_generator_parallel.FastNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=8,
                                                                       out_dir='../tmp', out_file='{}-noderange'.format(data_name),
                                                                       is_random=True, is_parallel=True, file_sz=10000,
                                                                       force=False, pair_sz=pair_sz,per_node_dst_sz=per_node_sz, proximity_sz=prox_sz)

    landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=4)

    val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=4,out_dir='../tmp',out_file=data_name+'-classicalrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=100,force=False,prod_workers=4)

    #
    st_time = time.time()
    train_generator_acc.gen_to_disk(early_break=-1)
    print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    landmark_generator_acc.gen_to_disk(early_break=-1)
    print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    val_generator_acc.gen_to_disk(early_break=classical_early_break)
    print('val_generator consumed:{:.2f}'.format(time.time() - st_time))

    #intact generator for train & validation.
    train_generator = m_generator.FastNodeRangeGenerator_p(g=g, scheme=m_generator.BFS(None), workers=8,
                                                                       out_dir='../tmp', out_file='{}-noderange'.format(data_name),
                                                                       is_random=True, is_parallel=True, file_sz=10000,
                                                                       force=False, pair_sz=pair_sz,per_node_dst_sz=per_node_sz, proximity_sz=prox_sz)
    landmark_generator = m_generator.LandmarkInnerGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                                  out_file=data_name+'-landmarkinner', is_random=True, is_parallel=True,
                                                  file_sz=10000, force=False, landmark_nodes=landmark_nodes,
                                                  prod_workers=4)
    val_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4, out_dir='../tmp',
                                               out_file=data_name+'-classicalrandom', is_random=True, is_parallel=True,
                                               file_sz=10000, data_sz_per_node=5, force=False, prod_workers=4)


    ged = m_manager.RangeManager(workers=16)


    gen_dict = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'is_parallel': landmark_nodes,
        'file_sz': 1000,
        'is_random': True,
        'is_landmarked': True,
        'landmark_generator': landmark_generator
    }
    enc_dict = {
        'emb_sz': emb_sz,
        'encoder': encoder
    }
    dec_dict = {
        'train_sz': 1000,  # ?
        'val_sz': 5000,
        'landmark_sz': -1,
        'solely_optim': True,
        'lr': 0.0001,
        'epoches1': 50,
        'epoches2': 100,
        'batch_sz1': 200,
        'batch_sz2': 1000,
        'decoder': decoder,
        'batch_sz_val': 1000,
        'stop_cond1':-1,
        'stop_cond2':0.0001,
        'is_embed_model':False,
    }

    ged.configModel(l1=l1,g=g, gen_file=model_name+'@generator', enc_file=model_name+'@generator', dec_file=model_name+'@decoder', gen_dict=gen_dict,
                    enc_dict=enc_dict, dec_dict=dec_dict)
    ged.train(force=False)
    ged.save_logs()



# dataset generation
def inst_gen_node_range(dataset='../datasets/dst/facebook',pair_sz=2000,name='fb-noderange=2000'):
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    stochastic_node_range_generator = m_generator_parallel.StochasticNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=10, out_dir='../tmp',out_file=name, is_random=True, is_parallel=True,file_sz=10000, force=False, pair_sz=pair_sz,proximity_sz=20)
    stochastic_node_range_generator.gen_to_disk(early_break=-1)

def inst_gen_fast_node_range(dataset='../datasets/dst/facebook',pair_sz=2000,proximity_sz=20,per_node_dst_sz=20,name='fb-noderange=2000'):
    print('========== cur dataset', name, '=========')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    fast_node_range_generator = m_generator_parallel.FastNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=10, out_dir='../tmp',out_file=name, is_random=True, is_parallel=True,file_sz=10000, force=False, pair_sz=pair_sz,proximity_sz=proximity_sz,per_node_dst_sz=per_node_dst_sz)
    st_time = time.time()
    fast_node_range_generator.gen_to_disk(early_break=-1)
    time_consume = time.time() - st_time
    print('time consume:{:.2f}'.format(time_consume))
    with open('../outputs/'+name+'-data-fnoderange.log','w') as f:
        f.write('{}\n'.format(time_consume))

def inst_gen_origin(dataset='../datasets/dst/facebook',name='fb',train_landmark_cnt=100,val_classical_per_node=50,val_classical_eb=100,test_classical_per_node=20,test_classical_eb=200,only_test=False):
    print('========== cur dataset',name,'=========')
    g, _ = dgl.load_graphs(dataset)
    g = g[0]
    g = dgl.to_simple_graph(g)
    g = dgl.to_bidirected(g)

    # landmark selection
    # landmark_nodes = random.sample(g.nodes().tolist(), 100)
    if not only_test:
        ds = m_selector.DegreeSelector(g=g)
        landmark_nodes = ds.perform(cnt=train_landmark_cnt,action='max')

        # parallel acceleration
        train_generator_acc = m_generator_parallel.FastRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=8,out_dir='../tmp',out_file=name+'-fastrandom',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=1,landmark_nodes=landmark_nodes,force=False,prod_workers=8)

        landmark_generator_acc = m_generator_parallel.LandmarkInnerGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=8,out_dir='../tmp',out_file=name+'-landmarkinner',is_random=True,is_parallel=True,file_sz=10000,force=False,landmark_nodes=landmark_nodes,prod_workers=8)

        val_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,scheme=m_generator.BFS(None),workers=8,out_dir='../tmp',out_file=name+'-classicalrandom-val',is_random=True,is_parallel=True,file_sz=10000,data_sz_per_node=val_classical_per_node,force=False,prod_workers=8)

    test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=8,out_dir='../tmp',out_file=name + '-classicalrandom-test',is_random=True, is_parallel=True,file_sz=10000, data_sz_per_node=test_classical_per_node,force=False, prod_workers=8)

    time_log = []

    if not only_test:
        st_time = time.time()
        train_generator_acc.gen_to_disk(early_break=-1)
        time_log.append(time.time() - st_time)
        print('train_generator consumed:{:.2f}'.format(time.time() - st_time))
        st_time = time.time()
        landmark_generator_acc.gen_to_disk(early_break=-1)
        time_log.append(time.time() - st_time)
        print('landmark_generator consumed:{:.2f}'.format(time.time() - st_time))
        st_time = time.time()
        val_generator_acc.gen_to_disk(early_break=val_classical_eb)
        time_log.append(time.time() - st_time)
        print('val_generator consumed:{:.2f}'.format(time.time() - st_time))
    st_time = time.time()
    test_generator_acc.gen_to_disk(early_break=test_classical_eb)
    time_log.append(time.time() - st_time)
    print('test_generator consumed:{:.2f}'.format(time.time() - st_time))

    if not only_test:
        with open('../outputs/'+name+'-data.log','w') as f:
            f.writelines(['fastrandom:'+str(time_log[0]),'landmarkinner:'+str(time_log[1])+'\n','classicalrandom-val:'+str(time_log[2])+'\n','classicalrandom-test:'+str(time_log[3])+'\n'])
    else:
        with open('../outputs/'+name+'-data-test-only.log','w') as f:
            f.writelines(['classicalrandom-test:'+str(time_log[0])+'\n'])

def combine_gen_node_range():
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter','../datasets/dst/youtube']
    names = ['fb','bc','tw','yt']
    pair_szs = [2000,3000,10000,100000]
    for dataset,name,pair_sz in zip(datasets,names,pair_szs):
        inst_gen_node_range(dataset=dataset,name='{}-noderange={}'.format(name,pair_sz),pair_sz=pair_sz)

def combine_gen_fast_node_range():
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter','../datasets/dst/youtube']
    names = ['fb','bc','tw','yt']
    pair_szs = [100,200,200,200]
    prox_szs = [40,60,30,20]
    per_node_szs = [40,50,50,200]
    for dataset,name,pair_sz,prox_sz,per_node_sz in zip(datasets,names,pair_szs,prox_szs,per_node_szs):
        inst_gen_fast_node_range(dataset=dataset,name='{}-noderange'.format(name),pair_sz=pair_sz,proximity_sz=prox_sz,per_node_dst_sz=per_node_sz)


def combine_gen_node_origin():
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
                '../datasets/dst/youtube']
    names = ['fb', 'bc', 'tw', 'yt']
    train_landmakr_cnts = [100,200,200,200]
    val_classical_per_nodes = [50,70,100,200]
    val_classical_ebs = [100, 100, 100, 100]
    test_classical_per_nodes = [20,30,30,50]
    test_classical_ebs = [200,200,200,200]
    for dataset, name, train_landmark_cnt, val_classical_per_node,val_classical_eb, test_classical_per_node,test_classical_eb in zip(datasets,names,train_landmakr_cnts,val_classical_per_nodes,val_classical_ebs,test_classical_per_nodes,test_classical_ebs):
        inst_gen_origin(dataset=dataset,name=name,train_landmark_cnt=train_landmark_cnt,val_classical_per_node=val_classical_per_node,val_classical_eb=val_classical_eb,test_classical_per_node=test_classical_per_node,test_classical_eb=test_classical_eb)

def combine_gen_node_origin_toy():
    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter',
                '../datasets/dst/youtube']
    names = ['fb', 'bc', 'tw', 'yt']
    train_landmakr_cnts = [1,1,1,1]
    val_classical_per_nodes = [1,1,1,1]
    val_classical_ebs = [1, 1, 1, 1]
    test_classical_per_nodes = [1,1,1,1]
    test_classical_ebs = [1,1,1,1]
    for dataset, name, train_landmark_cnt, val_classical_per_node,val_classical_eb, test_classical_per_node,test_classical_eb in zip(datasets,names,train_landmakr_cnts,val_classical_per_nodes,val_classical_ebs,test_classical_per_nodes,test_classical_ebs):
        inst_gen_origin(dataset=dataset,name=name,train_landmark_cnt=train_landmark_cnt,val_classical_per_node=val_classical_per_node,val_classical_eb=val_classical_eb,test_classical_per_node=test_classical_per_node,test_classical_eb=test_classical_eb)

def combine_routine_fast_node_range():
    datasets = ['../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter','../datasets/dst/youtube']
    data_names = ['bc', 'tw','yt']
    landmark_cnts = [200,200,200]
    prox_szs = [60,30,5]
    per_node_szs = [50,50,400]
    pair_szs = [200,200,100]
    emb_szs = [16,64,128]
    for dataset,data_name,landmark_cnt,prox_sz,per_node_sz,pair_sz in zip(datasets,data_names,landmark_cnts,prox_szs,per_node_szs,pair_szs):
        for emb_sz in emb_szs:
            inst_routine_ext_dw_fast(dataset=dataset,data_name=data_name, emb_sz=emb_sz, model_name='{}-dwext-REALFAST-emb={}'.format(data_name,emb_sz),landmark_cnt=landmark_cnt,pair_sz=pair_sz,per_node_sz=per_node_sz,prox_sz=prox_sz)

def combine_routine_fast_node_range1():
    datasets = ['../datasets/dst/BlogCatalog-dataset']
    data_names = ['bc']
    landmark_cnts = [200]
    prox_szs = [60]
    per_node_szs = [50]
    pair_szs = [200]
    emb_szs = [16,64,128]
    for dataset,data_name,landmark_cnt,prox_sz,per_node_sz,pair_sz in zip(datasets,data_names,landmark_cnts,prox_szs,per_node_szs,pair_szs):
        for emb_sz in emb_szs:
            inst_routine_ext_dw_fast(dataset=dataset,data_name=data_name, emb_sz=emb_sz, model_name='{}-dwext-REALFAST-emb={}'.format(data_name,emb_sz),landmark_cnt=landmark_cnt,pair_sz=pair_sz,per_node_sz=per_node_sz,prox_sz=prox_sz,l1=0.01)

def combine_routine_fast_node_range2():
    datasets = ['../datasets/dst/twitter']
    data_names = ['tw']
    landmark_cnts = [200]
    prox_szs = [30]
    per_node_szs = [50]
    pair_szs = [200]
    emb_szs = [16,64,128]
    for dataset,data_name,landmark_cnt,prox_sz,per_node_sz,pair_sz in zip(datasets,data_names,landmark_cnts,prox_szs,per_node_szs,pair_szs):
        for emb_sz in emb_szs:
            inst_routine_ext_dw_fast(dataset=dataset,data_name=data_name, emb_sz=emb_sz, model_name='{}-dwext-REALFAST-emb={}'.format(data_name,emb_sz),landmark_cnt=landmark_cnt,pair_sz=pair_sz,per_node_sz=per_node_sz,prox_sz=prox_sz,l1=0.005)

def combine_routine_fast_node_range3():
    datasets = ['../datasets/dst/youtube']
    data_names = ['yt']
    landmark_cnts = [200]
    prox_szs = [5]
    per_node_szs = [400]
    pair_szs = [100]
    emb_szs = [16,64,128]
    for dataset,data_name,landmark_cnt,prox_sz,per_node_sz,pair_sz in zip(datasets,data_names,landmark_cnts,prox_szs,per_node_szs,pair_szs):
        for emb_sz in emb_szs:
            inst_routine_ext_dw_fast(dataset=dataset,data_name=data_name, emb_sz=emb_sz, model_name='{}-dwext-REALFAST-emb={}'.format(data_name,emb_sz),landmark_cnt=landmark_cnt,pair_sz=pair_sz,per_node_sz=per_node_sz,prox_sz=prox_sz,l1=0.001)



def combine_routines_dadl():
    encoder_n2v = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
                                            force=False, num_walks=40, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
                                            is_directed=False, is_weighted=False, weight_arr=None)
    encoders = [encoder_n2v]
    emb_sz = [16]

    datasets = ['../datasets/dst/facebook']
    landmark_cnts = [200]
    # data_names = ['fb','bc','tw','yt']
    # data_names = ['fb', 'bc', 'tw']
    data_names = ['fb']

    model_names = ['dadl']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_dadl(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def combine_routines_grarep():
    encoder_grp = m_grarep.GraRepEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_grp]
    emb_sz = [16]

    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    landmark_cnts = [100,100,200]
    data_names = ['cr','fb','gq']

    model_names = ['grp']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_grarep(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def combine_routines_grarep():
    encoder_grp = m_grarep.GraRepEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_grp]
    emb_sz = [16]

    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    landmark_cnts = [100,100,200]
    data_names = ['cr','fb','gq']

    model_names = ['grp']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_grarep(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)


def combine_routines_netmf():
    encoder_netmf = m_encoder.NetMFEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False,order=2,iteration=10,neg_sz=1,seed=42)
    encoders = [encoder_netmf]
    emb_sz = [16]

    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    landmark_cnts = [100,100,200]
    data_names = ['cr','fb','gq']

    model_names = ['netmf']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_netmf(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def combine_routines_verse():
    encoder_verse = m_encoder.VerseEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_verse]
    emb_sz = [128]

    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    landmark_cnts = [100,100,200]
    data_names = ['cr','fb','gq']

    model_names = ['vs']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_verse(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def combine_routines_lpca():
    encoder_lpca = m_lpca.LPCAEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_lpca]
    emb_sz = [16]

    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    landmark_cnts = [100,100,200]
    data_names = ['cr','fb','gq']

    model_names = ['lpca']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_lpca(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

'''
appended embedding techniques.
'''
def combine_routines_glee():
    encoder_glee = m_karate_encoders.GLEEEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_glee]
    emb_sz = [16]

    # datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/DBLP']
    # landmark_cnts = [100,100,200,200]
    # data_names = ['cr','fb','gq','db']
    # data_names = ['cr','fb','gq']

    datasets = ['../datasets/dst/DBLP']
    landmark_cnts = [200]
    data_names = ['db']

    para_sz = 3
    model_names = ['glee']
    classical_early_breaks = [25, 25, 25,25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                for e_para_sz in range(para_sz):
                    inst_routine_lpca(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                      model_name=e_data_name + '-' + e_model_name + '-emb=' + str(
                                          e_emb_sz) + '-par={}'.format(e_para_sz), encoder=e_encoder,
                                      landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)


def combine_routines_nsk():
    encoder_nsk = m_karate_encoders.NodeSketchEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_nsk]
    emb_sz = [16]

    # datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/DBLP']
    # landmark_cnts = [100,100,200,200]
    # data_names = ['cr','fb','gq','db']
    # data_names = ['cr', 'fb', 'gq']

    datasets = ['../datasets/dst/DBLP']
    landmark_cnts = [200]
    data_names = ['db']

    para_sz=3
    model_names = ['nsk']
    classical_early_breaks = [25, 25, 25,25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                for e_para_sz in range(para_sz):
                    inst_routine_lpca(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                      model_name=e_data_name + '-' + e_model_name + '-emb=' + str(
                                          e_emb_sz) + '-par={}'.format(e_para_sz), encoder=e_encoder,
                                      landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)


def combine_routines_bne():
    encoder_bne = m_karate_encoders.BoostNEEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,force=False)
    encoders = [encoder_bne]
    emb_sz = [16]

    # datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/DBLP']
    # landmark_cnts = [100,100,200,200]
    # data_names = ['cr','fb','gq','db']
    # data_names = ['cr','fb','gq']

    datasets = ['../datasets/dst/DBLP']
    landmark_cnts = [200]
    data_names = ['db']

    para_sz = 3
    model_names = ['bne']
    classical_early_breaks = [25, 25, 25,25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                for e_para_sz in range(para_sz):
                    inst_routine_lpca(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                 model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz)+'-par={}'.format(e_para_sz), encoder=e_encoder,
                                 landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)


'''
    TODO: debug...
'''
def combine_routines_cdgcn():
    # encoder_n2v = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
    #                                         force=False, num_walks=80, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
    #                                         is_directed=False, is_weighted=False, weight_arr=None)
    # encoders = [encoder_n2v]
    encoders = []
    emb_sz = [16]

    datasets = ['../datasets/dst/facebook', '../datasets/dst/BlogCatalog-dataset', '../datasets/dst/twitter']
    landmark_cnts = [100, 100, 200, 200]
    # data_names = ['fb','bc','tw','yt']
    # data_names = ['fb', 'bc', 'tw']
    data_names = ['bc']
    classical_early_breaks = [-1, 25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_model_name in model_names:
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                inst_routine_cdgcn(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break)

def combine_routines_bcdr():
    # encoder_n2v = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
    #                                         force=False, num_walks=80, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
    #                                         is_directed=False, is_weighted=False, weight_arr=None)
    encoder_bcdr = m_dage.DistanceResamplingEncoder(g=None,in_bc_file=None,emb_sz=None,out_dir='../tmp',out_file=None,num_walks=40,input_len=40,output_len=40,alpha=0.98,input_exps=1,output_exps=1,neg_sz=5)
    # encoder_dw = m_encoder.DeepWalkEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
    #                                        force=False, num_walks=80, walk_lens=40, window_sz=20, max_mem=0, seed=0,
    #                                        is_dense_degree=False)

    encoders = [encoder_bcdr]
    emb_sz = [16]

    # datasets = ['../datasets/dst/cora','../datasets/dst/GrQc','../datasets/dst/facebook','../datasets/dst/NotreDame','../datasets/dst/DBLP']
    datasets = ['../datasets/dst/facebook']
    # landmark_cnts = [100,200,100]
    landmark_cnts = [200]
    # data_names = ['fb','bc','tw','yt']
    # data_names = ['cr','gq','fb']
    data_names = ['fb']

    model_names = ['bcdr']
    classical_early_breaks = [25, 25, 25]
    for e_emb_sz in emb_sz:
        for e_encoder, e_model_name in zip(encoders, model_names):
            for e_dataset, e_landmark_cnt, e_data_name, e_classical_early_break in zip(datasets, landmark_cnts,
                                                                                       data_names,
                                                                                       classical_early_breaks):
                if e_data_name != 'fb':
                    continue
                inst_routine_bcdr(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                             model_name=e_data_name + '-' + e_model_name + '-emb=' + str(e_emb_sz), encoder=e_encoder,
                             landmark_cnt=e_landmark_cnt, classical_early_break=e_classical_early_break,weight_decay=0.98)

def new_combine_routine_for_dataset():
    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc','../datasets/dst/NotreDame','../datasets/dst/DBLP']
    # names = ['cr','fb', 'gq', 'nd', 'db']
    names = ['cr','fb', 'gq']
    train_landmakr_cnts = [100,100,200,200,200]
    val_classical_per_nodes = [30, 50,70,200,200]
    val_classical_ebs = [100, 100, 100, 100, 100]
    test_classical_per_nodes = [10,20,30,70,100]
    test_classical_ebs = [200,200,200,200,200]
    for dataset, name, train_landmark_cnt, val_classical_per_node,val_classical_eb, test_classical_per_node,test_classical_eb in zip(datasets,names,train_landmakr_cnts,val_classical_per_nodes,val_classical_ebs,test_classical_per_nodes,test_classical_ebs):
        inst_gen_origin(dataset=dataset,name=name,train_landmark_cnt=train_landmark_cnt,val_classical_per_node=val_classical_per_node,val_classical_eb=val_classical_eb,test_classical_per_node=test_classical_per_node,test_classical_eb=test_classical_eb)


def new_combine_routine_dataset(idx=0):
    # datasets = ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc', '../datasets/dst/NotreDame']
    # data_names = ['cr','fb','gq']

    datasets = ['../datasets/dst/DBLP']
    data_names = ['db']

    # datasets = ['../datasets/dst/cora']
    # data_names = ['cr']

    # walk_len = [40]*len(datasets)
    # window_sz = [20]*len(datasets)
    # num_walks = [40]*len(datasets)
    # neg_sz = [5]*len(datasets)
    emb_sz = [16]*len(datasets)
    landmark_cnts = [100,200,100,200,200]
    # input_exps = [1]*len(datasets)
    # output_exps = [4]*len(datasets)
    # input_len = [40]*len(datasets)
    # output_len = [10]*len(datasets)
    weight_decay = [0.8,0.9,0.1,0.1,0.1]
    parallel_sz = 3

    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=6,out_dir='../tmp',out_file=None,force=False,num_walks=40,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=6,out_dir='../tmp',out_file=None,force=False)
    encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=6,out_dir='../tmp',out_file=None,force=False)
    encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=6,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-3,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=6,out_dir='../tmp',out_file=None,force=False,num_walks=40,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)

    encoder_dadl = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=6, out_dir='../tmp', out_file=None,
                                            force=False, num_walks=40, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
                                            is_directed=False, is_weighted=False, weight_arr=None)

    encoder_grp = m_grarep.GraRepEncoder(g=None, emb_sz=None, workers=6, out_dir='../tmp', out_file=None,force=False)
    encoder_netmf = m_encoder.NetMFEncoder(g=None, emb_sz=None, workers=6, out_dir='../tmp', out_file=None,force=False,order=2,iteration=10,neg_sz=1,seed=42)
    encoder_lpca = m_lpca.LPCAEncoder(g=None, emb_sz=None, workers=6, out_dir='../tmp', out_file=None,force=False)
    encoder_verse = m_encoder.VerseEncoder(g=None, emb_sz=None, workers=6, out_dir='../tmp', out_file=None,force=False)


    encoder_bcdr = m_dage.DistanceResamplingEncoder(g=None, in_bc_file=None, emb_sz=None, out_dir='../tmp',
                                                    out_file=None, num_walks=40, input_len=40, output_len=10, alpha=0.05,
                                                    input_exps=1, output_exps=4, neg_sz=5)


    # encoders = [encoder_dw, encoder_n2v,encoder_dadl]
    # encoder_names = ['dw','n2v','dadl']


    # encoders = [encoder_dw, encoder_lle, encoder_le, encoder_gf, encoder_n2v,encoder_dadl,encoder_bcdr]
    # encoder_names = ['dw','lle','le','gf','n2v','dadl','bcdr']

    # encoders = [encoder_lle,encoder_n2v]
    # encoder_names = ['lle','n2v']

    # encoders = [encoder_bcdr]
    # encoder_names = ['bcdr']

    # encoders = [encoder_dadl]
    # encoder_names = ['dadl']

    # encoders = [encoder_grp]
    # encoder_names = ['grp']

    # encoders = [encoder_netmf]
    # encoder_names = ['netmf']

    # encoders = [encoder_verse]
    # encoder_names = ['vs']

    # encoders = [encoder_lpca]
    # encoder_names = ['lpca']

    encoders = [encoder_lle,encoder_gf]
    encoder_names = ['lle','gf']

    for e_idx,(e_dataset,e_data_name,e_emb_sz,e_landmark_cnt) in enumerate(zip(datasets,data_names,emb_sz,landmark_cnts)):
        # if e_idx != idx:
        #     continue
        for e_encoder,e_encoder_name in zip(encoders,encoder_names):
            if e_encoder_name in ['dw','lle','le','gf','n2v']:
                for parallel_cnt in range(parallel_sz):
                    inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_encoder_name+'-emb='+str(e_emb_sz)+'-par={}'.format(parallel_cnt),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=-1)
            elif e_encoder_name == 'dadl':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_dadl(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                  model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(e_emb_sz)+'-par={}'.format(parallel_cnt), encoder=e_encoder,
                                  landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'grp':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_grarep(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                      model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(
                                          e_emb_sz) + '-par={}'.format(parallel_cnt), encoder=e_encoder,
                                      landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'netmf':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_netmf(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                        model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(
                                            e_emb_sz) + '-par={}'.format(parallel_cnt), encoder=e_encoder,
                                        landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'vs':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_verse(emb_sz=128, dataset=e_dataset, data_name=e_data_name,
                                        model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(
                                            128) + '-par={}'.format(parallel_cnt), encoder=e_encoder,
                                        landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'lpca':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_lpca(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                        model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(
                                            e_emb_sz) + '-par={}'.format(parallel_cnt), encoder=e_encoder,
                                        landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'bcdr':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_bcdr(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                  model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(e_emb_sz)+'-par={}'.format(parallel_cnt), encoder=e_encoder,
                                  landmark_cnt=e_landmark_cnt, classical_early_break=-1,weight_decay=0.05)

def new_combine_routine_timed_for_origin():
    datasets = ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc']
    data_names = ['cr','fb','gq']
    # walk_len = [40]*len(datasets)
    # window_sz = [20]*len(datasets)
    # num_walks = [40]*len(datasets)
    # neg_sz = [5]*len(datasets)
    emb_sz = [16]*len(datasets)
    landmark_cnts = [100,200,100,200,200]
    # input_exps = [1]*len(datasets)
    # output_exps = [4]*len(datasets)
    # input_len = [40]*len(datasets)
    # output_len = [10]*len(datasets)
    weight_decay = [0.1,0.9,0.1,0.1,0.1]
    parallel_sz = 1

    encoder_dw = m_encoder.DeepWalkEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=40,walk_lens=40,window_sz=20,max_mem=0,seed=0,is_dense_degree=False)
    # encoder_lle = m_lle.LLEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_le = m_le.LEEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False)
    # encoder_gf = m_gf.GFEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,iter=500,r=1.0,lr=1e-3,print_step=10)
    encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=40,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)

    # encoder_dadl = m_encoder.Node2VecEncoder(g=None, emb_sz=None, workers=16, out_dir='../tmp', out_file=None,
    #                                         force=False, num_walks=40, walk_lens=40, window_sz=20, p=1, q=1, iter=1,
    #                                         is_directed=False, is_weighted=False, weight_arr=None)

    encoder_bcdr = m_dage.DistanceResamplingEncoder(g=None, in_bc_file=None, emb_sz=None, out_dir='../tmp',
                                                    out_file=None, num_walks=40, input_len=40, output_len=10, alpha=0.1,
                                                    input_exps=1, output_exps=4, neg_sz=5)

    # encoders = [encoder_dw, encoder_lle, encoder_le, encoder_gf, encoder_n2v,encoder_dadl,encoder_bcdr]
    # encoder_names = ['dw','lle','le','gf','n2v','dadl','bcdr']

    encoders = [encoder_dw,encoder_n2v]
    encoder_names = ['dw','n2v']

    # encoders = [encoder_bcdr]
    # encoder_names = ['bcdr']

    # encoders = [encoder_dadl]
    # encoder_names = ['dadl']

    for e_idx,(e_dataset,e_data_name,e_emb_sz,e_landmark_cnt) in enumerate(zip(datasets,data_names,emb_sz,landmark_cnts)):
        for e_encoder,e_encoder_name in zip(encoders,encoder_names):
            if e_encoder_name in ['dw','lle','le','gf','n2v']:
                for parallel_cnt in range(parallel_sz):
                    inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_encoder_name+'-emb='+str(e_emb_sz)+'-timed',encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=-1)
            elif e_encoder_name == 'dadl':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_dadl(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                  model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(e_emb_sz)+'-par={}'.format(parallel_cnt), encoder=e_encoder,
                                  landmark_cnt=e_landmark_cnt, classical_early_break=-1)
            elif e_encoder_name == 'bcdr':
                for parallel_cnt in range(parallel_sz):
                    inst_routine_bcdr(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                                  model_name=e_data_name + '-' + e_encoder_name + '-emb=' + str(e_emb_sz)+'-par={}'.format(parallel_cnt), encoder=e_encoder,
                                  landmark_cnt=e_landmark_cnt, classical_early_break=-1,weight_decay=0.1)

def new_combine_routine_timed_for_bcdr():
    datasets = ['../datasets/dst/cora', '../datasets/dst/facebook', '../datasets/dst/GrQc']
    data_names = ['cr','fb','gq']
    emb_sz = [16]*len(datasets)
    landmark_cnts = [100,200,100,200,200]


    # betas = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
    betas = [0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

    decay_map = {'fb':0.98,'cr':0.1,'gq':0.1,'db':0.9}

    for e_dataset,e_data_name,e_emb_sz,e_landmark_cnt in zip(datasets,data_names,emb_sz,landmark_cnts):
        for beta in betas:
            encoder_bcdr = m_dage.DistanceResamplingEncoder(g=None, in_bc_file=None, emb_sz=None, out_dir='../tmp',
                                                            out_file=None, num_walks=40, input_len=40, output_len=math.ceil(beta*40),
                                                            alpha=0.1,
                                                            input_exps=1, output_exps=math.ceil(1/beta), neg_sz=5)

            inst_routine_bcdr(emb_sz=e_emb_sz, dataset=e_dataset, data_name=e_data_name,
                      model_name=e_data_name + '-bcdr-emb=' + str(e_emb_sz)+'-timed_{}'.format(beta), encoder=encoder_bcdr,
                      landmark_cnt=e_landmark_cnt, classical_early_break=-1,weight_decay=decay_map[e_data_name])

def combine_routine_for_dataset_dqm():
    datasets = ['../datasets/dst/cora','../datasets/dst/facebook','../datasets/dst/GrQc']
    # names = ['cr','fb', 'gq', 'nd', 'db']
    names = ['cr','fb', 'gq']
    train_landmakr_cnts = [100,100,200,200,200]
    val_classical_per_nodes = [30, 50,70,200,200]
    val_classical_ebs = [100, 100, 100, 100, 100]
    test_classical_per_nodes = [10,20,30,70,100]
    test_classical_ebs = [200,200,200,200,200]
    for dataset, name, train_landmark_cnt, val_classical_per_node,val_classical_eb, test_classical_per_node,test_classical_eb in zip(datasets,names,train_landmakr_cnts,val_classical_per_nodes,val_classical_ebs,test_classical_per_nodes,test_classical_ebs):
        # if name == 'fb':
        #     continue
        inst_gen_origin(dataset=dataset,name=name,train_landmark_cnt=train_landmark_cnt,val_classical_per_node=val_classical_per_node,val_classical_eb=val_classical_eb,test_classical_per_node=test_classical_per_node,test_classical_eb=test_classical_eb)

def combine_routine_for_dataset_dqm_large():
    datasets = ['../datasets/dst/DBLP','../datasets/dst/youtube','../datasets/dst/Pokec']
    # names = ['cr','fb', 'gq', 'nd', 'db']
    names = ['db','yt', 'pk']
    train_landmakr_cnts = [200,200,200]
    test_classical_per_nodes = [100,100,200]
    test_classical_ebs = [20,20,10]
    for dataset, name, train_landmark_cnt, test_classical_per_node,test_classical_eb in zip(datasets,names,train_landmakr_cnts,test_classical_per_nodes,test_classical_ebs):
        # if name == 'fb':
        #     continue
        inst_gen_origin(dataset=dataset,name=name,train_landmark_cnt=train_landmark_cnt,val_classical_per_node=-1,val_classical_eb=-1,test_classical_per_node=test_classical_per_node,test_classical_eb=test_classical_eb,only_test=True)


if __name__ == '__main__':
    print('hello manager.')
    # new_combine_routine_timed_for_origin()
    # new_combine_routine_timed_for_bcdr()
    # new_combine_routine_dataset(idx=0)

    # combine_routines_bcdr()

    # combine_routines_toy()
    # combine_routines1()
    # combine_routines2()
    # combine_routines3()
    # combine_routines4_REAL()
    # inst_yt3(id=4)
    # inst_routine_sm(emb_sz=64,dataset='../datasets/dst/facebook',data_name='fb',model_name='fb-sm-emb=64')
    # m_smgnn.test1()
    # inst_routine_ext_dw_const(emb_sz=64,model_name='fb-dwext-emb=64')
    # inst_routine_ext_dw_fast(emb_sz=64,model_name='fb-dwext-REALFAST-emb=64',landmark_cnt=200000,pair_sz=999999,per_node_sz=10000000,prox_sz=10000000)
    # combine_gen_node_origin()
    # combine_gen_fast_node_range()
    # combine_routine_fast_node_range()
    # combine_routine_fast_node_range1()
    # combine_routine_fast_node_range2()
    # combine_routine_fast_node_range3()

    # new_combine_routine_for_dataset()
    # combine_routines_dadl()
    # combine_routines_cdgcn()
    # combine_routines_bcdr()

    # encoder_n2v = m_encoder.Node2VecEncoder(g=None,emb_sz=None,workers=16,out_dir='../tmp',out_file=None,force=False,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    # encoders = [encoder_n2v]
    #
    # emb_sz = [16]
    #
    # datasets = ['../datasets/dst/facebook']
    # landmark_cnts = [100]
    # data_names = ['fb']
    #
    # model_names = ['n2v']
    # classical_early_breaks = [-1]
    # for e_emb_sz in emb_sz:
    #     for e_encoder,e_model_name in zip(encoders,model_names):
    #         for e_dataset,e_landmark_cnt,e_data_name,e_classical_early_break in zip(datasets,landmark_cnts,data_names,classical_early_breaks):
    #             inst_routine1(emb_sz=e_emb_sz,dataset=e_dataset,data_name=e_data_name,model_name=e_data_name+'-'+e_model_name+'-emb='+str(e_emb_sz),encoder=e_encoder,landmark_cnt=e_landmark_cnt,classical_early_break=e_classical_early_break)
    #         # break

    # combine_routines_grarep()
    # combine_routines_netmf()
    # combine_routines_verse()
    # combine_routines_lpca()
    # combine_routines_glee()
    # combine_routines_nsk()
    # combine_routines_bne()
    # For DBLP.

    # combine_routine_for_dataset_dqm()
    combine_routine_for_dataset_dqm_large()

'''
    有向图按方向算的话，大多数点都是不连通的，-1影响整个架构。。。
    skip-gram丢失一个节点，应该无伤大雅，毕设以后再想办法解决。先处理主要问题。（好像是数据集有向无向引入的问题，尝试把有向图全部看作无向图来做把）
    有向图由于方向不可交换性，与embedding方法相性很差。
'''