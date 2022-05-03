import m_router
import m_evaluator
import dgl
import torch as tc
import os
import time
import m_generator
import pandas as pd
import m_generator_parallel
import m_logger
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


class RouteEvaluator(m_evaluator.BasicEvaluator):
    def __init__(self,**kwargs):
        super(RouteEvaluator, self).__init__(**kwargs)

    def config(self, model_name=None,model=None, g=None, force=False, param_dict={},route_mode='basic-bdd',route_params={}):
        super(RouteEvaluator, self).config(model,g,force,param_dict)
        self.router = None
        if route_mode == 'basic-bdd':
            self.router = m_router.ParaRouter(model_name=model_name)
        elif route_mode == 'bfs':
            self.router = m_router.BFSRouter(model_name=model_name)
        elif route_mode == 'rigel-path':
            self.router = m_router.RigelRouter(model_name=model_name)
        self.route_params = route_params

    # def proc(self, is_print=True):
    #     return self.proc_basic(is_print=is_print)


    def proc_basic(self, is_print=True):
        '''
        计算基本的MRE、MAE的度量总体均值，按路径长度的度量均值
        '''
        # 生成测试用的数据集
        st_time = time.time()
        self.random_generator.gen_to_disk()
        print('gen test dataset finished, time consume: {:.2f}'.format(time.time() - st_time))

        # 配置测试输出路径
        eval_out_path = os.path.join(self.out_dir, self.out_file)
        basic_eval_out_path = eval_out_path + '.basic-p'

        self.model.eval()

        seg_loss = {}
        test_loader = self.random_generator.loader(batch_sz=self.test_batch_sz, meta_batch_sz=10)
        delay_break = False
        sample_cnt = 0
        total_mae = 0.
        total_mre = 0.
        total_cnt = 0

        # log for run time and distance query times.
        log_time_total = 0.
        log_query_total = 0
        log_nHits_total = 0


        with tc.no_grad():
            for i, lst in enumerate(test_loader):
                ten = tc.Tensor(lst)
                sample_cnt += ten.shape[0]
                if self.test_sz > 0:
                    if sample_cnt >= self.test_sz:
                        ten = ten[:ten.shape[0] - (sample_cnt - self.test_sz)]
                        delay_break = True
                src = ten[:, 0]
                dst = ten[:, 1]
                dist = ten[:, 2]

                src = src.type_as(tc.LongTensor())
                dst = dst.type_as(tc.LongTensor())
                # src_emb = self.g.ndata['emb'][src]
                # dst_emb = self.g.ndata['emb'][dst]
                #
                # test_pred = self.model(src_emb, dst_emb)

                test_pred_p = []
                for idx,(e_src, e_dst) in enumerate(zip(src, dst)):
                    e_st_time = time.time()
                    path = self.router.query_single_path(int(e_src), int(e_dst),**self.route_params)
                    log_time_total += time.time() - e_st_time
                    log_query_total += self.router.peek_query_times()
                    if not self.router.peek_is_failed():
                        log_nHits_total += 1
                    if path is None:
                        print('error when zipdata is None')
                        path, path_len = (0, 0), 0
                    else:
                        path_len = len(path) - 1
                    test_pred_p.append(path_len)
                    print('cal path:{}/{}'.format(idx,len(src)))
                print('iter {}...'.format(i))
                test_pred = test_pred_p


                dist = dist.tolist()
                # test_pred = test_pred.view(-1).tolist()

                # print('cur',dist)
                # print('cur',test_pred)
                assert len(dist) == len(test_pred)
                for pred, real in zip(test_pred, dist):
                    if real in seg_loss:
                        seg_loss[real]['mae'] += abs(pred - real)
                        if real == 0:
                            seg_loss[real]['mre'] += 0
                        else:
                            seg_loss[real]['mre'] += abs(pred - real) / abs(real)
                        seg_loss[real]['samples'] += 1
                    else:
                        mae = abs(pred - real)
                        if real == 0:
                            mre = 0
                        else:
                            mre = abs(pred - real) / abs(real)
                        seg_loss[real] = {'mae': mae, 'mre': 0, 'samples': 1}
                    if real != 0:
                        # our total result will not count for zeros.
                        total_mae += abs(pred - real)
                        total_mre += abs(pred - real) / abs(real)
                        total_cnt += 1
                if delay_break:
                    break
        total_mae /= total_cnt
        total_mre /= total_cnt
        log_nHits_total /= total_cnt
        log_query_total /= total_cnt
        log_time_total /= total_cnt
        for key in seg_loss:
            seg_loss[key]['mae'] = seg_loss[key]['mae'] / seg_loss[key]['samples']
            seg_loss[key]['mre'] = seg_loss[key]['mre'] / seg_loss[key]['samples']
            seg_loss[key]['path_len'] = key

        keys = seg_loss.keys()
        keys = sorted(keys)
        data_path = [[], [], []]
        total_samples = 0
        for key in keys:
            data_path[0].append(seg_loss[key]['mae'])
            data_path[1].append(seg_loss[key]['mre'])
            data_path[2].append(seg_loss[key]['samples'])
            total_samples += int(seg_loss[key]['samples'])
        data_path[0].append(total_mae)
        data_path[1].append(total_mre)
        data_path[2].append(total_samples)
        loss_table = pd.DataFrame(data=data_path, index=['mae', 'mre', 'cnt'],
                                  columns=[str(k) for k in keys] + ['total'])

        # loss_total_table = pd.DataFrame({'total_mae': [total_mae], 'total_mre': [total_mre]})
        if is_print:
            loss_table.to_csv(basic_eval_out_path + '-path-p.csv', index=False)
            # loss_total_table.to_csv(basic_eval_out_path + '-total.csv', index=False)
        return total_mae, total_mre, data_path,log_time_total,log_nHits_total,log_query_total

def anal_path_score_table(data_names = ['cr','fb','gq'],encoder_names = ['lle','le','gf','dw','n2v','dadl','bcdr'], route_modes=['basic-bdd','bfs','rigel-path'],route_mode_names=['BDD','BFS','RP'],param_caps=[20,-1,200]):
    emb_sz = []
    for encoder_name in encoder_names:
        emb_sz.append(16 if encoder_name != 'vs' else 128)
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
        'grp':'GreRep'
    }
    data = {}
    for data_name in data_names:
        data[data_name] = {}
        data_total_paths = {}
        for encoder_name,e_emb_sz in zip(encoder_names,emb_sz):
            data[data_name][encoder_name] = {}
            # data[data_name][encoder_name]['maes'] = []
            # data[data_name][encoder_name]['mres'] = []

            #for each para, read best model
            best_model_paths = []
            best_model_scores = []
            for idx in range(3):
                cur_par_tmp = '{}-{}-emb={}-par={}'.format(data_name,encoder_name,e_emb_sz,idx)
                cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                cur_best_score = 10000
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
            for route_mode,route_mode_name,param_cap in zip(route_modes,route_mode_names,param_caps):
                data[data_name][encoder_name][route_mode_name] = {}
                data[data_name][encoder_name][route_mode_name]['maes'] = []
                data[data_name][encoder_name][route_mode_name]['mres'] = []
                data[data_name][encoder_name][route_mode_name]['nHits'] = []
                data[data_name][encoder_name][route_mode_name]['runTime'] = []
                data[data_name][encoder_name][route_mode_name]['queryTimes'] = []

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

                    beval = RouteEvaluator(out_dir='../log', out_file=model_path + '_' + route_mode_name)

                    test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                            out_dir='../tmp',
                                                                            out_file=data_name + '-classicalrandom-test',
                                                                            is_random=True,
                                                                            is_parallel=True,
                                                                            file_sz=10000, data_sz_per_node=5, force=False,
                                                                            prod_workers=4)
                    route_params = {
                        'cap': param_cap,
                    }
                    beval.config(model_name=model_path, model=model, g=g, force=False, param_dict={
                        'random_generator': test_generator,
                        'test_sz': 1000,
                        'test_batch_sz': 200
                    }, route_mode=route_mode, route_params=route_params)

                    if model_score == bbest_score:
                        print('get bbest score = {}, model = {}'.format(model_score,model_path))
                        lst = beval.proc(is_print=True) # true for printing path-wise accuracy for the best model.
                        data_total_paths[encoder_name] = lst[2]  # log path loss of the best model of each methods.
                    else:
                        lst = beval.proc(is_print=False)
                    assert len(lst) >= 6
                    mae,mre,runTime,nHits,queryTimes = lst[0],lst[1],lst[3],lst[4],lst[5]
                    data[data_name][encoder_name][route_mode_name]['maes'].append(mae)
                    data[data_name][encoder_name][route_mode_name]['mres'].append(mre)
                    data[data_name][encoder_name][route_mode_name]['nHits'].append(nHits)
                    data[data_name][encoder_name][route_mode_name]['runTime'].append(runTime)
                    data[data_name][encoder_name][route_mode_name]['queryTimes'].append(queryTimes)

                lst_mae = data[data_name][encoder_name][route_mode_name]['maes']
                lst_mre = data[data_name][encoder_name][route_mode_name]['mres']
                lst_nHits = data[data_name][encoder_name][route_mode_name]['nHits']
                lst_runTime = data[data_name][encoder_name][route_mode_name]['runTime']
                lst_queryTimes = data[data_name][encoder_name][route_mode_name]['queryTimes']
                mean_mae = sum(lst_mae) / len(lst_mae)
                mean_mre = sum(lst_mre) / len(lst_mre)
                mean_nHits = sum(lst_nHits) / len(lst_nHits)
                mean_runTime = sum(lst_runTime) / len(lst_runTime)
                mean_queryTimes = sum(lst_queryTimes) / len(lst_queryTimes)

                std_mae = np.std(np.array(lst_mae))
                std_mre = np.std(np.array(lst_mre))
                std_nHits = np.std(np.array(lst_nHits))
                std_runTime = np.std(np.array(lst_runTime))
                std_queryTimes = np.std(np.array(lst_queryTimes))

                data[data_name][encoder_name][route_mode_name]['mae'] = '{}±{}'.format(mean_mae,std_mae)
                data[data_name][encoder_name][route_mode_name]['mre'] = '{}±{}'.format(mean_mre, std_mre)
                data[data_name][encoder_name][route_mode_name]['nHits'] = '{}±{}'.format(mean_nHits, std_nHits)
                data[data_name][encoder_name][route_mode_name]['runTime'] = '{}±{}'.format(mean_runTime, std_runTime)
                data[data_name][encoder_name][route_mode_name]['queryTimes'] = '{}±{}'.format(mean_queryTimes, std_queryTimes)

        cur_df_data = []
        cols = ['mae','mre','nHits','runTime','queryTimes']
        idxs = encoder_names
        comb_idxs = []
        for encoder_name in idxs:
            for route_mode_name in route_mode_names:
                cur_row = []
                cur_row.append(data[data_name][encoder_name][route_mode_name]['mae'])
                cur_row.append(data[data_name][encoder_name][route_mode_name]['mre'])
                cur_row.append(data[data_name][encoder_name][route_mode_name]['nHits'])
                cur_row.append(data[data_name][encoder_name][route_mode_name]['runTime'])
                cur_row.append(data[data_name][encoder_name][route_mode_name]['queryTimes'])
                cur_df_data.append(cur_row)
                comb_idxs.append('{} && {}'.format(encoder_name,route_mode_name))
        print('cur_df_data:',cur_df_data)
        data_df = pd.DataFrame(data=cur_df_data,columns=cols,index=comb_idxs)
        data_df.to_csv(os.path.join('../log','{}-anal-p.csv'.format(data_name)))
        cs = ['coral','dodgerblue','red','blue','darkcyan','darkviolet','olive','cyan','black','firebrick','gainsboro']

        for c,enc in zip(cs,data_total_paths):
            cur_data = data_total_paths[enc]
            xs = range(1,len(cur_data[0]) - 1)
            ys_mae = cur_data[0][1:-1]
            plt.plot(xs,ys_mae,':',c=c)
            plt.scatter(xs,ys_mae,c=c,marker='s', label=enc_map[enc]+', mAE',alpha=0.5)
        plt.ylim(0, 10)
        plt.xlabel('path length')
        plt.ylabel('mAE score')
        plt.legend(loc='upper right')
        plt.savefig('../fig/'+data_name+'-path-mae-p.pdf')
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
        plt.savefig('../fig/'+data_name+'-path-mre-p.pdf')
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
            plt.savefig('../fig/' + data_name + '-path-samples-p.pdf')
            plt.show()
            break


def anal_path_m(data_names = ['cr','fb','gq'],encoder_names = ['lle','le','gf','dw','n2v','dadl','bcdr'], route_modes=['basic-bdd','bfs','rigel-path'],route_mode_names=['BDD','RP'],m_range=range(1,200,10),save_path='../fig'):
    emb_sz = []
    for encoder_name in encoder_names:
        emb_sz.append(16 if encoder_name != 'vs' else 128)
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
        'grp': 'GreRep'
    }
    data = {}
    for data_name in data_names:
        data[data_name] = {}
        data_total_paths = {}
        for encoder_name, e_emb_sz in zip(encoder_names, emb_sz):
            data[data_name][encoder_name] = {}
            # data[data_name][encoder_name]['maes'] = []
            # data[data_name][encoder_name]['mres'] = []

            # for each para, read best model
            best_model_paths = []
            best_model_scores = []
            for idx in range(3):
                cur_par_tmp = '{}-{}-emb={}-par={}'.format(data_name, encoder_name, e_emb_sz, idx)
                cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                cur_best_score = 10000
                cur_best_idx = -1
                if not os.path.exists(os.path.join('../log', cur_log)):
                    # there's only one training routine.
                    if idx == 0:
                        cur_par_tmp = '{}-{}-emb={}'.format(data_name, encoder_name, e_emb_sz, idx)
                        cur_log = cur_par_tmp + '@landmark_stage2@TrainBasicLogger.log'
                    else:
                        # no more log exists.
                        break
                with open(os.path.join('../log', cur_log), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line is None or line == '':
                            continue
                        lst = line.split(',')
                        assert len(lst) == 3, print('cur lst:{}'.format(lst))
                        cur_score = float(lst[2])
                        # cur_score = float(lst[1])
                        if cur_best_score > cur_score:
                            cur_best_score = cur_score
                            cur_best_idx = int(lst[0])
                print('{}:found best model at idx {} with score {}'.format(cur_par_tmp, cur_best_idx, cur_best_score))
                assert cur_best_idx != -1
                if cur_best_idx < 10:
                    cur_best_idx_round = '~10'
                elif cur_best_idx >= 495:
                    cur_best_idx_round = ''
                else:
                    cur_best_idx_round = '~{}'.format(round(cur_best_idx, -1))

                best_model_paths.append(cur_par_tmp + cur_best_idx_round)
                best_model_scores.append(cur_best_score)
            print('{}-{}:best_model_path={}'.format(data_name, encoder_name, best_model_paths))

            bbest_score = min(best_model_scores)
            # test for each model and get mRE, mAE score.
            for route_mode, route_mode_name in zip(route_modes,route_mode_names):
                data[data_name][encoder_name][route_mode_name] = {}
                # data[data_name][encoder_name][route_mode_name]['maes'] = []
                data[data_name][encoder_name][route_mode_name]['mres'] = []
                # data[data_name][encoder_name][route_mode_name]['nHits'] = []
                data[data_name][encoder_name][route_mode_name]['runTime'] = []
                data[data_name][encoder_name][route_mode_name]['ms'] = []
                # data[data_name][encoder_name][route_mode_name]['queryTimes'] = []

                for model_path, model_score in zip(best_model_paths, best_model_scores):
                    cur_mre = []
                    cur_runTime = []
                    for m in m_range:
                        model, g = m_evaluator.BasicEvaluator.load_model(out_file=model_path)

                        test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g,
                                                                                               scheme=m_generator.BFS(None),
                                                                                               workers=4,
                                                                                               out_dir='../tmp',
                                                                                               out_file=data_name + '-classicalrandom-test',
                                                                                               is_random=True,
                                                                                               is_parallel=True,
                                                                                               file_sz=10000,
                                                                                               data_sz_per_node=100,
                                                                                               force=False, prod_workers=4)

                        test_generator_acc.gen_to_disk(early_break=5)

                        beval = RouteEvaluator(out_dir='../log', out_file=model_path + '_' + route_mode_name)

                        test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None),
                                                                                workers=4,
                                                                                out_dir='../tmp',
                                                                                out_file=data_name + '-classicalrandom-test',
                                                                                is_random=True,
                                                                                is_parallel=True,
                                                                                file_sz=10000, data_sz_per_node=5,
                                                                                force=False,
                                                                                prod_workers=4)
                        route_params = {
                            'cap': m,
                        }
                        beval.config(model_name=model_path, model=model, g=g, force=False, param_dict={
                            'random_generator': test_generator,
                            'test_sz': 1000,
                            'test_batch_sz': 200
                        }, route_mode=route_mode, route_params=route_params)

                        lst = beval.proc(is_print=False)
                        assert len(lst) >= 6
                        mae, mre, runTime, nHits, queryTimes = lst[0], lst[1], lst[3], lst[4], lst[5]

                        cur_mre.append(mre)
                        cur_runTime.append(runTime)

                    # data[data_name][encoder_name][route_mode_name]['maes'].append(mae)
                    data[data_name][encoder_name][route_mode_name]['mres'].append(cur_mre)
                    # data[data_name][encoder_name][route_mode_name]['nHits'].append(nHits)
                    data[data_name][encoder_name][route_mode_name]['runTime'].append(cur_runTime)
                    # data[data_name][encoder_name][route_mode_name]['ms'].append(m)
                    # data[data_name][encoder_name][route_mode_name]['queryTimes'].append(queryTimes)


                # lst_mae = data[data_name][encoder_name][route_mode_name]['maes']
                lst_mre = np.array(data[data_name][encoder_name][route_mode_name]['mres'])
                # lst_nHits = data[data_name][encoder_name][route_mode_name]['nHits']
                lst_runTime = np.array(data[data_name][encoder_name][route_mode_name]['runTime'])
                # lst_queryTimes = data[data_name][encoder_name][route_mode_name]['queryTimes']
                lst_m = np.array(list(m_range))

                mean_mre = np.mean(lst_mre,axis=0)
                mean_runTime = np.mean(lst_runTime,axis=0)
                cur_data = np.concatenate([mean_mre.reshape(1,-1),mean_runTime.reshape(1,-1),lst_m.reshape(1,-1)],axis=0)
                np.save(os.path.join(save_path,'{}@{}@{}.npy'.format(data_name,encoder_name,route_mode_name)),cur_data)
def anal_path_m_peek(data_names = ['cr','fb','gq'],encoder_names = ['lle','le','gf','dw','n2v','dadl','bcdr'], route_modes=['basic-bdd','bfs','rigel-path'],route_mode_names=['BDD','RP'],save_path='../fig',bfs_times=[59.37,47.49,112.9],bfs_mres=[0,0,0]):
    enc_map = {
        'lle': 'LLE',
        'le': 'LE',
        'gf': 'GF',
        'dw': 'DeepWalk',
        'n2v': 'Node2Vec',
        'dadl': 'DADL',
        'bcdr': 'BCDR',
        'lpca': 'LPCA',
        'netmf': 'NetMF',
        'vs': 'Verse',
        'grp': 'GreRep'
    }
    for data_name,bfs_mre,bfs_time in zip(data_names,bfs_mres,bfs_times):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cs = ['coral', 'dodgerblue', 'darkcyan', 'darkviolet', 'olive', 'cyan', 'black', 'firebrick',
              'gainsboro']
        c_pnt = -1
        x_min = 100
        x_max = -1
        z_min = 100
        z_max = -1
        for route_mode,route_mode_name in zip(route_modes,route_mode_names):
            for encoder_name in encoder_names:
                assert len(cs) >= len(route_mode_names) * len(encoder_names), print('カラーが足りないんっす!!')
                c_pnt += 1

                cur_data = np.load(os.path.join(save_path,'{}@{}@{}.npy'.format(data_name,encoder_name,route_mode_name)))
                assert cur_data.shape[0] == 3, print(cur_data.shape)
                xs = cur_data[2,:] # m
                ys = cur_data[1,:]*1000 # time
                zs = cur_data[0,:] # mre
                ax.scatter(xs, ys, zs, c=cs[c_pnt], label='{}-{}'.format(enc_map[encoder_name],route_mode_name))

                ax.plot(xs,ys,zs, c=cs[c_pnt])

                x_min = min(x_min,min(xs))
                x_max = max(x_max,max(xs))

                z_min = min(z_min, min(zs))
                z_max = max(z_max, max(zs))
                print('cur:',ys)

        ax.plot([x_min,x_max],[bfs_time]*2,[bfs_mre]*2,c='blue',label='BFS')
        x = np.linspace(x_min, x_max, 10)
        z = np.linspace(z_min, z_max, 10)
        xx, zz = np.meshgrid(x, z)
        y = np.ones((10, 10)) * bfs_time
        ax.set_xlabel('search capacity m')
        ax.set_ylabel('millisecond of run time')
        ax.set_zlabel('accuracy loss of mRE')
        #
        ax.plot_surface(xx, y, zz, alpha=0.1, linewidth=0,color='blue', antialiased=False)
        ax.legend(loc='upper left')

        plt.savefig('../fig/{}-anal-p-m.pdf'.format(data_name))
        plt.show()

def anal_path_m_peek_ext(data_names = ['cr','fb','gq'],encoder_names = ['lle','le','gf','dw','n2v','dadl','bcdr'], route_modes=['basic-bdd','bfs','rigel-path'],route_mode_names=['BDD','RP'],save_path='../fig',bfs_times=[59.37,47.49,112.9],bfs_mres=[0,0,0]):
    enc_map = {
        'lle': 'LLE',
        'le': 'LE',
        'gf': 'GF',
        'dw': 'DeepWalk',
        'n2v': 'Node2Vec',
        'dadl': 'DADL',
        'bcdr': 'BCDR',
        'lpca': 'LPCA',
        'netmf': 'NetMF',
        'vs': 'Verse',
        'grp': 'GreRep'
    }
    for data_name,bfs_mre,bfs_time in zip(data_names,bfs_mres,bfs_times):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cs = ['coral','firebrick', 'peru', 'darkviolet','olive','cyan','dodgerblue', 'darkcyan','firebrick']
        # cs_n = len(route_mode_names) * len(encoder_names)
        # cs = ['C{}'.format(e) for e in range(0,10)]
        x_min = 100
        x_max = -1
        z_min = 100
        z_max = -1
        y_min = 10000
        y_max = -1
        for route_mode,route_mode_name in zip(route_modes,route_mode_names):
            for encoder_name in encoder_names:
                assert len(cs) >= len(route_mode_names) * len(encoder_names), print('カラーが足りないんっす!!')

                cur_data = np.load(os.path.join(save_path,'{}@{}@{}.npy'.format(data_name,encoder_name,route_mode_name)))
                assert cur_data.shape[0] == 3, print(cur_data.shape)
                xs = cur_data[2,:] # m
                ys = cur_data[1,:]*1000 # time
                zs = cur_data[0,:] # mre
                # ax.scatter(xs, ys, zs, c=cs[c_pnt], label='{}-{}'.format(enc_map[encoder_name],route_mode_name))
                #
                # ax.plot(xs,ys,zs, c=cs[c_pnt])

                x_min = min(x_min,min(xs))
                x_max = max(x_max,max(xs))

                z_min = min(z_min, min(zs))
                z_max = max(z_max, max(zs))

                y_min = min(y_min, min(ys))
                y_max = max(y_max, max(ys))

                # print('cur:',ys)
        y_min = 0
        y_max += 80
        x_min = 0
        x_max += 10
        z_min = -0.04 if data_name == 'fb' else -0.4
        z_max += 0.02 if data_name == 'fb' else 0.1

        '''
        plot xOz && xOy
        '''
        c_pnt = -1
        for route_mode,route_mode_name in zip(route_modes,route_mode_names):
            for encoder_name in encoder_names:
                assert len(cs) >= len(route_mode_names) * len(encoder_names), print('カラーが足りないんっす!!')
                c_pnt += 1

                cur_data = np.load(os.path.join(save_path,'{}@{}@{}.npy'.format(data_name,encoder_name,route_mode_name)))
                assert cur_data.shape[0] == 3, print(cur_data.shape)
                xs = cur_data[2,:] # m
                ys = cur_data[1,:]*1000 # time
                zs = cur_data[0,:] # mre
                # ax.scatter(xs, ys, zs, c=cs[c_pnt], label='{}-{}'.format(enc_map[encoder_name],route_mode_name))
                ax.plot(xs,[y_max]*len(xs),zs, c=cs[c_pnt],alpha=0.5)
                ax.plot(xs, ys, [z_min]*len(xs), c=cs[c_pnt],alpha=0.5)

        '''
        plot 3d BFS
        '''
        ax.plot([x_min,x_max],[bfs_time]*2,[bfs_mre]*2,c='black',label='BFS')
        ax.plot([x_min, x_max], [bfs_time] * 2, [z_min] * 2, c='black',alpha=0.5)
        ax.plot([x_min, x_max], [y_max] * 2, [bfs_mre] * 2, c='black',alpha=0.5)
        x = np.linspace(x_min, x_max, 10)
        z = np.linspace(z_min, z_max, 10)
        xx, zz = np.meshgrid(x, z)
        y = np.ones((10, 10)) * bfs_time

        # ax.plot_surface(xx, y, zz, alpha=0.1, linewidth=0,color='blue', antialiased=False)

        '''
        plot 3d.
        '''
        c_pnt = -1
        for route_mode,route_mode_name in zip(route_modes,route_mode_names):
            for encoder_name in encoder_names:
                assert len(cs) >= len(route_mode_names) * len(encoder_names), print('カラーが足りないんっす!!')
                c_pnt += 1

                cur_data = np.load(os.path.join(save_path,'{}@{}@{}.npy'.format(data_name,encoder_name,route_mode_name)))
                assert cur_data.shape[0] == 3, print(cur_data.shape)
                xs = cur_data[2,:] # m
                ys = cur_data[1,:]*1000 # time
                zs = cur_data[0,:] # mre
                ax.scatter(xs, ys, zs, c=cs[c_pnt], label='{}-{}'.format(enc_map[encoder_name],route_mode_name))

                ax.plot(xs,ys,zs, c=cs[c_pnt])



        '''
        set lim
        '''
        ax.set_xlim3d(left=x_min, right=x_max)
        ax.set_ylim3d(bottom=y_min, top=y_max)
        ax.set_zlim3d(bottom=z_min, top=z_max)

        '''
        set labels.
        '''
        ax.set_xlabel('search capacity m')
        ax.set_ylabel('millisecond of run time')
        ax.set_zlabel('accuracy loss of mRE')

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

        # ax.legend(loc='center right',ncol=1)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
        plt.tight_layout()
        plt.savefig('../fig/{}-anal-p-m.pdf'.format(data_name))
        plt.show()


def routine_basic_anal(dataset='fb', model_name='dw', emb_sz='16',use_par=False,par=None,route_mode='basic-bdd',param_cap=-1):
    # eval_model_name = None
    if use_par:
        eval_model_name = '{}-{}-emb={}-par={}'.format(dataset, model_name, emb_sz, par)
    else:
        eval_model_name = '{}-{}-emb={}'.format(dataset, model_name, emb_sz)
    model, g = m_evaluator.BasicEvaluator.load_model(out_file=eval_model_name)

    test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                           out_dir='../tmp',
                                                                           out_file=dataset + '-classicalrandom-test',
                                                                           is_random=True, is_parallel=True,
                                                                           file_sz=10000, data_sz_per_node=100,
                                                                           force=False, prod_workers=4)

    test_generator_acc.gen_to_disk(early_break=5)

    beval = RouteEvaluator(out_dir='../log', out_file='{}-{}-emb={}'.format(dataset, model_name, emb_sz))

    test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                            out_dir='../tmp',
                                                            out_file=dataset + '-classicalrandom-test', is_random=True,
                                                            is_parallel=True,
                                                            file_sz=10000, data_sz_per_node=5, force=False,
                                                            prod_workers=4)
    route_params = {
        'cap' : param_cap,
    }
    beval.config(model_name=eval_model_name,model=model, g=g, force=False, param_dict={
        'random_generator': test_generator,
        'test_sz': 1000,
        'test_batch_sz': 200
    },route_mode=route_mode,route_params=route_params)
    beval.proc()


    # print('{}@{}@{} completed!'.format(data_name, encoder_name, route_mode_name))


# def routine_path_anal(dataset='fb', model_name='dw', emb_sz='16',route_mode='basic-bdd',param_cap=-1):


def combine_routines_basic():
    # datasets = ['fb','bc']
    datasets = ['fb']
    # model_names = ['dw','gf','le','lle','n2v','dwext-REALFAST']
    model_names = ['dw', 'n2v','dadl', 'bcdr']
    emb_szs = ['16']
    for dataset in datasets:
        for model_name in model_names:
            for emb_sz in emb_szs:
                routine_basic_anal(dataset=dataset, model_name=model_name, emb_sz=emb_sz,use_par=True,par=0)

def Xtest_routine_basic():
    # routine_basic_anal(dataset='cr', model_name='dw', emb_sz=16, use_par=True, par=0,route_mode='basic-bdd',param_cap=20)
    routine_basic_anal(dataset='cr', model_name='dw', emb_sz=16, use_par=True, par=0,route_mode='rigel-path',param_cap=200)


def combine_routines_performance():
    datasets = ['cr','fb','gq','db']
    model_names = ['lle','gf','le','dw','grp','n2v','netmf','vs','dadl','lpca','bcdr']
    for dataset in datasets:
        print('~~~~~~~~~~~~ eval dataset:{} ~~~~~~~~~~~~'.format(dataset))
        for model_name in model_names:
            print('\t\t~~~~~~~~~~~~ eval model:{} ~~~~~~~~~~~~'.format(model_name))
            routine_basic_anal(dataset=dataset, model_name=model_name, emb_sz=16, use_par=True, par=0, route_mode='basic-bdd',
                               param_cap=10)



if __name__ == '__main__':
    print('hello router-eval.')
    # anal_path_score_table(data_names=['cr'],encoder_names=['lle','gf','le','dw','grp','n2v','netmf','vs','dadl','lpca','bcdr'],route_modes=['bfs','rigel-path','rigel-path','basic-bdd','basic-bdd'],route_mode_names=['BFS','RP@20','RP@200','BDD@10','BDD@20'],param_caps=[-1,20,200,10,20])
    # anal_path_score_table(data_names=['fb'],encoder_names=['lle','gf','le','dw','grp','n2v','netmf','vs','dadl','lpca','bcdr'],route_modes=['bfs','rigel-path','rigel-path','basic-bdd','basic-bdd'],route_mode_names=['BFS','RP@20','RP@200','BDD@10','BDD@20'],param_caps=[-1,20,200,10,20])
    # anal_path_score_table(data_names=['gq'],encoder_names=['lle','gf','le','dw','grp','n2v','netmf','vs','dadl','lpca','bcdr'],route_modes=['bfs','rigel-path','rigel-path','basic-bdd','basic-bdd'],route_mode_names=['BFS','RP@20','RP@200','BDD@10','BDD@20'],param_caps=[-1,20,200,10,20])
    # # anal_path_score_table(data_names=['cr'],encoder_names=['lle','gf','le','dw','grp','n2v','netmf'],route_modes=['basic-bdd','bfs','rigel-path'],param_caps=[20,-1,200])

    # anal_path_m(data_names=['cr','fb','gq'],encoder_names=['bcdr','dadl','n2v','dw'],route_modes=['basic-bdd','rigel-path'],route_mode_names=['BDD','RP'],m_range=[1,2,4,8,16,32,64,128])
    # anal_path_m_peek(data_names=['cr','fb','gq'],encoder_names=['bcdr','dadl','n2v','dw'],route_modes=['basic-bdd','rigel-path'],route_mode_names=['BDD','RP'])

    anal_path_m_peek_ext(data_names=['cr','fb','gq'],encoder_names=['bcdr','dadl','n2v','dw'],route_modes=['basic-bdd','rigel-path'],route_mode_names=['BDD','RP'])

    # 3*4*2*3*8