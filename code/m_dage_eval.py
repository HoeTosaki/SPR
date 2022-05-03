import os
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.functional as gnF
import multiprocessing
import threadpool
import math
import time
import random
import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx as nx
import m_encoder
import m_generator
import m_selector
from utils import *
import m_logger
import m_deepwalk
import fileinput
import m_evaluator
import m_generator_parallel
import numpy as np
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib

def anal_best_score(data_names = ['cr','fb','gr'],encoder_names = ['lle','le','gf','dw','n2v','dadl','bcdr'],emb_sz = None, is_save_best_model=True):
    if emb_sz is None:
        emb_sz = [16]*len(encoder_names)
    elif type(emb_sz) == int:
        emb_sz = [emb_sz]*len(encoder_names)
    assert len(emb_sz) == len(encoder_names)
    enc_map = {
        'lle':'LLE',
        'le':'LE',
        'gf':'GF',
        'dw':'DeepWalk',
        'n2v':'Node2Vec',
        'dadl':'DADL',
        'bcdr':'BCDR(ours.)',
        'lpca':'LPCA',
        'netmf':'NetMF',
        'vs':'Verse',
        'grp':'GreRep',
        'glee':'GLEE',
        'nsk':'NodeSketch',
        'bne':'BoostNE',
    }
    data = {}
    for data_name in data_names:
        data[data_name] = {}
        data_total_paths = {}
        for encoder_name,e_emb_sz in zip(encoder_names,emb_sz):
            data[data_name][encoder_name] = {}
            data[data_name][encoder_name]['maes'] = []
            data[data_name][encoder_name]['mres'] = []

            #for each para, read best model
            best_model_paths = []
            best_model_scores = []
            for idx in range(3):
                cur_par_tmp = '{}-{}-emb={}-par={}'.format(data_name,encoder_name,e_emb_sz,idx)
                cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                cur_best_score = 100000
                cur_best_idx = -1
                if not os.path.exists(os.path.join('../log',cur_log)):
                    # there's only one training routine.
                    if idx == 0:
                        cur_par_tmp = '{}-{}-emb={}'.format(data_name, encoder_name, e_emb_sz, idx)
                        cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                    else:
                        # no more log exists.
                        break
                with open(os.path.join('../log',cur_log),'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line is None or line == '':
                            continue
                        lst = line.split(',')
                        assert len(lst) == 3,print('cur lst:{}'.format(lst))
                        cur_score = float(lst[2])
                        # cur_score = float(lst[1])
                        if cur_best_score > cur_score:
                            cur_best_score = cur_score
                            cur_best_idx = int(lst[0])
                print('{}:found best model at idx {} with score {}'.format(cur_par_tmp,cur_best_idx,cur_best_score))
                assert cur_best_idx != -1
                if cur_best_idx  < 10:
                    cur_best_idx_round = '~10'
                elif cur_best_idx >= 495:
                    cur_best_idx_round = ''
                else:
                    cur_best_idx_round = '~{}'.format(round(cur_best_idx,-1))

                best_model_paths.append(cur_par_tmp + cur_best_idx_round)
                best_model_scores.append(cur_best_score)
            print('{}-{}:best_model_path={}'.format(data_name,encoder_name,best_model_paths))

            bbest_score = min(best_model_scores)
            # test for each model and get mRE, mAE score.
            for model_path,model_score in zip(best_model_paths,best_model_scores):
                model, g = m_evaluator.BasicEvaluator.load_model(out_file=model_path)

                test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None),
                                                                                       workers=4,
                                                                                       out_dir='../tmp',
                                                                                       out_file=data_name + '-classicalrandom-test',
                                                                                       is_random=True, is_parallel=True,
                                                                                       file_sz=10000, data_sz_per_node=100,
                                                                                       force=False, prod_workers=4)

                test_generator_acc.gen_to_disk(early_break=5)

                beval = m_evaluator.BasicEvaluator(out_dir='../log', out_file=model_path)

                test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                        out_dir='../tmp',
                                                                        out_file=data_name + '-classicalrandom-test',
                                                                        is_random=True,
                                                                        is_parallel=True,
                                                                        file_sz=10000, data_sz_per_node=5, force=False,
                                                                        prod_workers=4)
                beval.config(model=model, g=g, force=False, param_dict={
                    'random_generator': test_generator,
                    'test_sz': 50000,
                    'test_batch_sz': 1000
                })
                lst = None
                if model_score == bbest_score:
                    print('get bbest score = {}, model = {}'.format(model_score,model_path))
                    lst = beval.proc(is_print=True) # true for printing path-wise accuracy for the best model.
                    data_total_paths[encoder_name] = lst[2]  # log path loss of the best model of each methods.
                else:
                    lst = beval.proc(is_print=False)
                assert len(lst) >= 2
                mae,mre = lst[0],lst[1]
                data[data_name][encoder_name]['maes'].append(mae)
                data[data_name][encoder_name]['mres'].append(mre)

            lst_mae = data[data_name][encoder_name]['maes']
            lst_mre = data[data_name][encoder_name]['mres']
            mean_mae = sum(lst_mae) / len(lst_mae)
            mean_mre = sum(lst_mre) / len(lst_mre)
            std_mae = np.std(np.array(lst_mae))
            std_mre = np.std(np.array(lst_mre))
            data[data_name][encoder_name]['mae'] = '{}±{}'.format(mean_mae,std_mae)
            data[data_name][encoder_name]['mre'] = '{}±{}'.format(mean_mre, std_mre)

        cur_df_data = []
        cols = ['mae','mre']
        idxs = encoder_names
        for encoder_name in idxs:
            cur_row = []
            cur_row.append(data[data_name][encoder_name]['mae'])
            cur_row.append(data[data_name][encoder_name]['mre'])
            cur_df_data.append(cur_row)
        print('cur_df_data:',cur_df_data)
        data_df = pd.DataFrame(data=cur_df_data,columns=cols,index=idxs)
        data_df.to_csv(os.path.join('../log','{}-anal.csv'.format(data_name)))
        cs = ['coral','dodgerblue','red','blue','darkcyan','darkviolet','olive','cyan','black','firebrick','gainsboro']

        for c,enc in zip(cs,data_total_paths):
            cur_data = data_total_paths[enc]
            xs = range(1,len(cur_data[0]) - 1)
            ys_mae = cur_data[0][1:-1]
            plt.plot(xs,ys_mae,':',c=c)
            plt.scatter(xs,ys_mae,c=c,marker='s', label=enc_map[enc]+', mAE',alpha=0.5)
        plt.ylim(0, 5)
        plt.xlabel('path length')
        plt.ylabel('mAE score')
        plt.legend(loc='upper right')
        plt.savefig('../fig/'+data_name+'-path-mae.pdf')
        plt.show()

        for c,enc in zip(cs,data_total_paths):
            cur_data = data_total_paths[enc]
            xs = range(1,len(cur_data[0]) - 1)
            # ys_mae = cur_data[0][1:-1]
            ys_mre = cur_data[1][1:-1]
            # plt.plot(xs,ys_mae,':',c=c)
            # plt.scatter(xs,ys_mae,c=c,marker='s', label=enc+', mAE',alpha=0.5)

            plt.plot(xs, ys_mre, c=c)
            # plt.scatter(xs, ys_mre, c=c,marker='x', label=enc_map[enc]+', mRE', alpha=0.5)
            plt.scatter(xs, ys_mre, c=c, marker='x', label=enc_map[enc] + '', alpha=0.5)
        plt.ylim(0,2)
        plt.xlabel('path length')
        plt.ylabel('mRE score')
        plt.legend(loc='upper right')
        plt.savefig('../fig/'+data_name+'-path-mre.pdf')
        plt.show()
        for enc in data_total_paths:
            cur_data = data_total_paths[enc]
            xs = range(1,len(cur_data[0]) - 1)
            # ys_mae = cur_data[0][1:-1]
            ys_samples = cur_data[2][1:-1]
            ys_samples = [ ele/sum(ys_samples) for ele in ys_samples]
            # plt.plot(xs,ys_mae,':',c=c)
            # plt.scatter(xs,ys_mae,c=c,marker='s', label=enc+', mAE',alpha=0.5)

            plt.bar(x=xs, height=ys_samples, color=cs[1])
            plt.xlabel('path length')
            plt.ylabel('sample frequency')
            # plt.legend()
            plt.savefig('../fig/' + data_name + '-path-samples.pdf')
            plt.show()
            break

def anal_timed_best_score(data_names = ['cr','fb','gq'],coefs = [0.05,0.1],emb_sz = None, is_save_best_model=True,compared = None,compared_time=[119.73,246.08,229.18]):
    data = {}
    for e_compared,e_compared_time,data_name in zip(compared,compared_time,data_names):
        data[data_name] = {}
        data_total_paths = {}
        for coef in coefs:
            data[data_name][coef] = {}
            data[data_name][coef]['maes'] = []
            data[data_name][coef]['mres'] = []

            #for each para, read best model
            best_model_paths = []
            best_model_scores = []
            for idx in range(1):
                cur_par_tmp = '{}-bcdr-emb={}-timed_{}'.format(data_name,emb_sz,coef)
                cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                cur_best_score = 10000
                cur_best_idx = -1
                with open(os.path.join('../log',cur_log),'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line is None or line == '':
                            continue
                        lst = line.split(',')
                        assert len(lst) == 3,print('cur lst:{}'.format(lst))
                        cur_score = float(lst[2])
                        if cur_best_score > cur_score:
                            cur_best_score = cur_score
                            cur_best_idx = int(lst[0])
                print('{}:found best model at idx {} with score {}'.format(cur_par_tmp,cur_best_idx,cur_best_score))
                assert cur_best_idx != -1
                if cur_best_idx  < 10:
                    cur_best_idx_round = '~10'
                elif cur_best_idx >= 495:
                    cur_best_idx_round = ''
                else:
                    cur_best_idx_round = '~{}'.format(round(cur_best_idx,-1))

                best_model_paths.append(cur_par_tmp + cur_best_idx_round)
                best_model_scores.append(cur_best_score)
            print('{}-{}:best_model_path={}'.format(data_name,coef,best_model_paths))

            bbest_score = min(best_model_scores)
            # test for each model and get mRE, mAE score.
            for model_path,model_score in zip(best_model_paths,best_model_scores):
                model, g = m_evaluator.BasicEvaluator.load_model(out_file=model_path)

                test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None),
                                                                                       workers=4,
                                                                                       out_dir='../tmp',
                                                                                       out_file=data_name + '-classicalrandom-test',
                                                                                       is_random=True, is_parallel=True,
                                                                                       file_sz=10000, data_sz_per_node=100,
                                                                                       force=False, prod_workers=4)

                test_generator_acc.gen_to_disk(early_break=5)

                beval = m_evaluator.BasicEvaluator(out_dir='../log', out_file=model_path)

                test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                        out_dir='../tmp',
                                                                        out_file=data_name + '-classicalrandom-test',
                                                                        is_random=True,
                                                                        is_parallel=True,
                                                                        file_sz=10000, data_sz_per_node=5, force=False,
                                                                        prod_workers=4)
                beval.config(model=model, g=g, force=False, param_dict={
                    'random_generator': test_generator,
                    'test_sz': 50000,
                    'test_batch_sz': 1000
                })
                lst = None
                if model_score == bbest_score:
                    print('get bbest score = {}, model = {}'.format(model_score,model_path))
                    lst = beval.proc(is_print=False) # true for printing path-wise accuracy for the best model.
                    data_total_paths[coef] = lst[2]  # log path loss of the best model of each methods.
                else:
                    lst = beval.proc(is_print=False)
                assert len(lst) >= 2
                mae,mre = lst[0],lst[1]
                data[data_name][coef]['maes'].append(mae)
                data[data_name][coef]['mres'].append(mre)

            lst_mae = data[data_name][coef]['maes']
            lst_mre = data[data_name][coef]['mres']
            mean_mae = sum(lst_mae) / len(lst_mae)
            mean_mre = sum(lst_mre) / len(lst_mre)
            std_mae = np.std(np.array(lst_mae))
            std_mre = np.std(np.array(lst_mre))
            data[data_name][coef]['mae'] = mean_mae
            data[data_name][coef]['mre'] = mean_mre

        cur_df_data = []
        cols = ['mae','mre']
        for coef in coefs:
            cur_row = []
            cur_row.append(data[data_name][coef]['mae'])
            cur_row.append(data[data_name][coef]['mre'])
            cur_df_data.append(cur_row)
        print('cur_df_data:',cur_df_data)
        data_df = pd.DataFrame(data=cur_df_data,columns=cols,index=coefs)
        data_df.to_csv(os.path.join('../log','{}-anal-timed.csv'.format(data_name)))
        # cs = ['coral','dodgerblue','red','blue','darkcyan','darkviolet','olive','cyan','black','firebrick','gainsboro']

        lst_mre = [data[data_name][coef]['mre'] for coef in coefs]
        plt.clf()
        plt.plot(coefs,lst_mre,c='coral',marker='x',label='BCDR')
        plt.plot([0,max(coefs)],[e_compared]*2,linestyle=':',c='dodgerblue',label='General RW')
        plt.ylabel('mRE')
        plt.xlabel('beta')
        plt.legend()
        plt.savefig('../log/{}-anal-timed-mre.pdf'.format(data_name))

        time_files = ['{}-bcdr-timed<emb>-len={}.log'.format(data_name,math.ceil(coef*40)) for coef in coefs]
        times = []
        for time_file in time_files:
            with open(os.path.join('../log',time_file),'r') as f:
                times.append(float(f.readlines()[0].strip()))

        plt.clf()
        plt.plot(coefs, times, c='coral', marker='x', label='BCDR')
        plt.plot([0, max(coefs)], [e_compared_time] * 2, linestyle=':', c='dodgerblue', label='General RW')
        print(e_compared_time,'~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        plt.ylabel('millisecond of run time')
        plt.xlabel('beta')
        plt.legend()
        plt.savefig('../log/{}-anal-timed-time.pdf'.format(data_name))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = coefs
        ys = times
        zs = lst_mre
        ax.scatter(xs,ys,zs,c='coral',label='BCDR')

        ax.plot([min(coefs),max(coefs)],[e_compared_time]*2,[e_compared]*2,c='dodgerblue',label='Random Walk')

        x = np.linspace(min(coefs), max(coefs), 10)
        z = np.linspace(0,e_compared,10)
        xx, zz = np.meshgrid(x, z)
        y = np.ones((10, 10)) * e_compared_time
        ax.set_xlabel('beta')
        ax.set_ylabel('millisecond of run time')
        ax.set_zlabel('accuracy loss of mRE')

        ax.plot_surface(xx, y, zz,alpha=0.3,linewidth=0, antialiased=False)
        ax.legend()

        plt.savefig('../log/{}-anal-both.pdf'.format(data_name))


'''
remedy for facebook, which seems to be badly integrated with combine routine trainer...
'''
def re_dispatch_fb_result(in_dir = '../dump/神奇的bcdr',in_files = ['para1','para2','para3','para4','para5']):
    cp_in_files = []
    cp_out_files = []
    for idx,in_file in enumerate(in_files):
        combined_in_dir = os.path.join(in_dir,in_file)
        in_file_name = 'fb-bcdr-emb=16'
        for root, dirs, files in os.walk(combined_in_dir):
            for file in files:
                if not file.startswith(in_file_name):
                    continue
                lst = file.split(in_file_name)
                assert len(lst) >= 2, print('lst = {}'.format(lst))
                out_file_name = in_file_name + '-par={}'.format(idx) + ''.join(lst[1:])
                cp_out_files.append(os.path.join(combined_in_dir,out_file_name))
                cp_in_files.append(os.path.join(combined_in_dir,file))
    for cp_in_file,cp_out_file in zip(cp_in_files,cp_out_files):
        print('will change {} into {}'.format(cp_in_file, cp_out_file))
        # print(shutil.copy(cp_in_file,cp_out_file))
        os.system('cp {} {}'.format(cp_in_file,cp_out_file))
if __name__ == '__main__':
    print('hello dage eval.')
    # anal_best_score(data_names=['cr','fb','gq'],encoder_names=['lle','le','gf','dw','n2v','dadl','bcdr','lpca','netmf','grp','vs'],emb_sz=[16,16,16,16,16,16,16,16,16,16,128])
    anal_best_score(data_names=['cr','fb','gq','db'],
                    encoder_names=['nsk','bne','glee'],
                    emb_sz=[16, 16, 16])

    # re_dispatch_fb_result(in_files=['para2','para3','para4','para5','para6'])
    # anal_timed_best_score(data_names=['cr','fb','gq'],coefs=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],emb_sz=16,compared=[0.2425,0.3289,0.4169])

    # anal_best_score(data_names=['db'],encoder_names=['lpca','netmf','grp','vs'],emb_sz=[16,16,16,128])
    # anal_best_score(data_names=['db'],encoder_names=['dw','n2v','netmf','dadl'],emb_sz=[16,16,16,16])
