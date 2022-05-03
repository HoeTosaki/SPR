import dgl
import torch as tc
import os
import time
import m_generator
import pandas as pd
import m_generator_parallel
import m_logger
import math

'''
    这里希望对最短路查询模型进行性能分析。
    输入条件：model、embedding、[测试数据集]
    评价目标：
    1）mAE：距离的绝对误差
        数据集要求：随机点对数据
        输出要求：数据
    2）mRE：距离的相对误差
        数据集要求：随机点对数据
        输出要求：数据
    3）l-mAE：不同路径长度分别统计mAE
        输出要求：图、表
    4）l-mRE：不同路径长度分别统计mRE
        输出要求：图、表
    5）mVoN：单点的一阶proximity的距离变动情况。
        数据集要求：随机点对数据，每个点取一阶近邻，计算图在该点的混合二阶偏导，求平均
        输出要求：数据
    6）训练曲线
        输出要求：
            图1：landmark阶段：epoch--train acc（平方误差）
            图2：whole graph阶段：epoch--train/val acc（平方误差）
    7）训练时间
    8）占用空间
'''
device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')

class BasicEvaluator:
    def __init__(self, out_dir='../output', out_file='gen-eval'):
        # super(BasicEvaluator,self).__init__()
        self.out_dir = out_dir
        self.out_file = out_file

    def config(self, model, g, force=False, param_dict={}):
        self.model = model
        self.g = g
        self.force = force
        self.random_generator = param_dict['random_generator']
        self.test_batch_sz = param_dict['test_batch_sz']
        self.test_sz = param_dict['test_sz']

    def proc(self,is_print=True):
        return self.proc_basic(is_print=is_print)

    def proc_basic(self,is_print=True):
        '''
        计算基本的MRE、MAE的度量总体均值，按路径长度的度量均值
        '''
        # 生成测试用的数据集
        st_time = time.time()
        self.random_generator.gen_to_disk()
        print('gen test dataset finished, time consume: {:.2f}'.format(time.time() - st_time))

        # 配置测试输出路径
        eval_out_path = os.path.join(self.out_dir, self.out_file)
        basic_eval_out_path = eval_out_path + '.basic'

        self.model.eval()

        seg_loss = {}
        test_loader = self.random_generator.loader(batch_sz=self.test_batch_sz, meta_batch_sz=10)
        delay_break = False
        sample_cnt = 0
        total_mae = 0.
        total_mre = 0.
        total_cnt = 0
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
                src_emb = self.g.ndata['emb'][src]
                dst_emb = self.g.ndata['emb'][dst]

                test_pred = self.model(src_emb, dst_emb)
                dist = dist.tolist()
                test_pred = test_pred.view(-1).tolist()
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
        for key in seg_loss:
            seg_loss[key]['mae'] = seg_loss[key]['mae'] / seg_loss[key]['samples']
            seg_loss[key]['mre'] = seg_loss[key]['mre'] / seg_loss[key]['samples']
            seg_loss[key]['path_len'] = key

        keys = seg_loss.keys()
        keys = sorted(keys)
        data_path = [[],[],[]]
        total_samples = 0
        for key in keys:
            data_path[0].append(seg_loss[key]['mae'])
            data_path[1].append(seg_loss[key]['mre'])
            data_path[2].append(seg_loss[key]['samples'])
            total_samples += int(seg_loss[key]['samples'])
        data_path[0].append(total_mae)
        data_path[1].append(total_mre)
        data_path[2].append(total_samples)
        loss_table = pd.DataFrame(data=data_path,index=['mae','mre','cnt'],columns=[str(k) for k in keys] + ['total'])

        # loss_total_table = pd.DataFrame({'total_mae': [total_mae], 'total_mre': [total_mre]})
        if is_print:
            loss_table.to_csv(basic_eval_out_path + '-path.csv', index=False)
            # loss_total_table.to_csv(basic_eval_out_path + '-total.csv', index=False)
        return total_mae,total_mre,data_path

    def proc_partial(self):
        '''
            计算src和dst之间距离度量的稳定性，反映了embedding的平滑程度
        '''
        # 生成测试用的数据集
        st_time = time.time()
        self.random_generator.gen_to_disk()
        print('gen test dataset finished, time consume: {:.2f}'.format(time.time() - st_time))

        # 配置测试输出路径
        eval_out_path = os.path.join(self.out_dir, self.out_file)
        partial_eval_out_path = eval_out_path + '.partial'

        self.model.eval()

        seg_loss = {}
        test_loader = self.random_generator.loader(batch_sz=self.test_batch_sz, meta_batch_sz=10)
        delay_break = False
        sample_cnt = 0
        pred_loss = 0.
        with tc.no_grad():
            for i, lst in enumerate(test_loader):
                cur_lst = lst
                sample_cnt += len(cur_lst)
                if self.test_sz > 0:
                    if sample_cnt >= self.test_sz:
                        cur_lst = cur_lst[:len(cur_lst) - (sample_cnt - self.test_sz)]
                        delay_break = True
                for src, dst, dist_dic in cur_lst:
                    # print('dist_dic',dist_dic)
                    real_cur = dist_dic[src][dst]

                    pred_n1u_n1v = 0.
                    pred_n1u_n0v = 0.
                    pred_n0u_n1v = 0.
                    pred_n0u_n0v = 0.

                    cnt_n1u_n1v = 0.
                    cnt_n1u_n0v = 0.
                    cnt_n0u_n1v = 0.
                    cnt_n0u_n0v = 0.

                    src_set = set()
                    dst_set = set()

                    src_set_n1u = set()
                    src_set_n0u = set()
                    dst_set_n1v = set()
                    dst_set_n0v = set()

                    # print('dist_dic',dist_dic)
                    # print('dist_dic len=',dist_dic.keys())
                    assert len(dist_dic.keys()) == 2

                    for e_dst in dist_dic[src]:
                        if e_dst == dst:
                            continue
                        dst_set.add(e_dst)
                    for e_src in dist_dic[dst]:
                        if e_src == src:
                            continue
                        src_set.add(e_src)

                    for e_src in src_set:
                        real_unk = dist_dic[dst][e_src]
                        if real_unk >= real_cur:
                            src_set_n1u.add(e_src)
                        else:
                            src_set_n0u.add(e_src)
                    assert len(src_set) == len(src_set_n0u) + len(src_set_n1u)

                    for e_dst in dst_set:
                        real_unk = dist_dic[src][e_dst]
                        if real_unk >= real_cur:
                            dst_set_n1v.add(e_dst)
                        else:
                            dst_set_n0v.add(e_dst)
                    assert len(dst_set) == len(dst_set_n0v) + len(dst_set_n1v)

                    for e_src in src_set_n1u:
                        for e_dst in dst_set_n1v:
                            pred_n1u_n1v += self.model(self.g.ndata['emb'][e_src].view(1, -1),
                                                       self.g.ndata['emb'][e_dst].view(1, -1)).item()
                            cnt_n1u_n1v += 1

                    for e_src in src_set_n1u:
                        for e_dst in dst_set_n0v:
                            pred_n1u_n0v += self.model(self.g.ndata['emb'][e_src].view(1, -1),
                                                       self.g.ndata['emb'][e_dst].view(1, -1)).item()
                            cnt_n1u_n0v += 1

                    for e_src in src_set_n0u:
                        for e_dst in dst_set_n1v:
                            pred_n0u_n1v += self.model(self.g.ndata['emb'][e_src].view(1, -1),
                                                       self.g.ndata['emb'][e_dst].view(1, -1)).item()
                            cnt_n0u_n1v += 1

                    for e_src in src_set_n0u:
                        for e_dst in dst_set_n0v:
                            pred_n0u_n0v += self.model(self.g.ndata['emb'][e_src].view(1, -1),
                                                       self.g.ndata['emb'][e_dst].view(1, -1)).item()
                            cnt_n0u_n0v += 1
                    pred_loss += math.pow((pred_n1u_n1v - pred_n1u_n0v - pred_n0u_n1v + pred_n0u_n0v) / (
                                (len(src_set_n0u) + len(src_set_n1u)) * (len(dst_set_n0v) + len(dst_set_n1v))), 2)
                print("\tcur pred_loss:{:.3f}, process:{}/{}".format(pred_loss / sample_cnt, sample_cnt, self.test_sz),
                      end='\r')
        print('')

        if self.test_sz > 0:
            seg_loss['partial loss'] = [pred_loss / self.test_sz]
            seg_loss['test sz'] = [self.test_sz]
        else:
            seg_loss['partial loss'] = [pred_loss / sample_cnt]
            seg_loss['test sz'] = [sample_cnt]
        # print('seg_loss',seg_loss)
        loss_table = pd.DataFrame(seg_loss)

        loss_table.to_csv(partial_eval_out_path + '.csv', index=False)

    def load_model(out_dir='../outputs', out_file='fb-dw-emb=16'):
        model = tc.load(os.path.join(out_dir, out_file + '.decoder'),map_location=device)
        g, _ = dgl.load_graphs(os.path.join(out_dir, out_file + '.embedding'))
        g = g[0]
        return model, g

    def load_log(out_dir='../log', out_file='fb-dw-emb=16@landmark_stage1@TrainBasicLogger.log'):
        logger = m_logger.TrainBasicLogger(out_dir=out_dir)
        logger.load_log(os.path.join(out_dir, out_file))
        return logger


def routine_basic_anal(dataset='fb', model_name='dw', emb_sz='16'):
    model, g = BasicEvaluator.load_model(out_file='{}-{}-emb={}'.format(dataset, model_name, emb_sz))

    test_generator_acc = m_generator_parallel.ClassicalRandomGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=4,
                                                                           out_dir='../tmp',
                                                                           out_file=dataset + '-classicalrandom-test',
                                                                           is_random=True, is_parallel=True,
                                                                           file_sz=10000, data_sz_per_node=100,
                                                                           force=False, prod_workers=4)

    test_generator_acc.gen_to_disk(early_break=5)

    beval = BasicEvaluator(out_dir='../log', out_file='{}-{}-emb={}'.format(dataset, model_name, emb_sz))

    test_generator = m_generator.ClassicalRandomGenerator_p(g=g, scheme=m_generator.BFS(None), workers=4,
                                                            out_dir='../tmp',
                                                            out_file=dataset + '-classicalrandom-test', is_random=True,
                                                            is_parallel=True,
                                                            file_sz=10000, data_sz_per_node=5, force=False,
                                                            prod_workers=4)
    beval.config(model=model, g=g, force=False, param_dict={
        'random_generator': test_generator,
        'test_sz': 5000,
        'test_batch_sz': 1000
    })
    beval.proc()


def routine_partial_anal(dataset='fb', model_name='dw', emb_sz='16'):
    print('-------- anal partial for {}-{}-emb={} --------'.format(dataset, model_name, emb_sz))
    model, g = BasicEvaluator.load_model(out_file='{}-{}-emb={}'.format(dataset, model_name, emb_sz))

    test_generator_acc = m_generator_parallel.FastNodeRangeGenerator_Acc(g=g, scheme=m_generator.BFS(None), workers=10,
                                                                         out_dir='../tmp',
                                                                         out_file='{}-noderange'.format(dataset),
                                                                         is_random=True, is_parallel=True,
                                                                         file_sz=10000,
                                                                         force=False, pair_sz=200, proximity_sz=10)

    test_generator_acc.gen_to_disk(early_break=-1)

    beval = BasicEvaluator(out_dir='../log', out_file='{}-{}-emb={}'.format(dataset, model_name, emb_sz))

    test_generator = m_generator.FastNodeRangeGenerator_p(g=g, scheme=m_generator.BFS(None), workers=10,
                                                          out_dir='../tmp', out_file='{}-noderange'.format(dataset),
                                                          is_random=True, is_parallel=True, file_sz=10000,
                                                          force=False, pair_sz=200, proximity_sz=10)
    beval.config(model=model, g=g, force=False, param_dict={
        'random_generator': test_generator,
        'test_sz': 2000,
        'test_batch_sz': 100
    })
    beval.proc_partial()


def combine_routines_basic():
    # datasets = ['fb','bc']
    datasets = ['tw']
    # model_names = ['dw','gf','le','lle','n2v','dwext-REALFAST']
    model_names = ['dw', 'lle', 'n2v', 'dwext-REALFAST']
    emb_szs = ['16', '64', '128']
    for dataset in datasets:
        for model_name in model_names:
            for emb_sz in emb_szs:
                routine_basic_anal(dataset=dataset, model_name=model_name, emb_sz=emb_sz)


def combine_routines_partial():
    datasets = ['fb']
    model_names = ['dwext-REALFAST', 'dw', 'lle', 'le', 'gf', 'n2v']
    emb_szs = ['16', '64', '128', '256']
    for dataset in datasets:
        for emb_sz in emb_szs:
            for model_name in model_names:
                if model_name == 'dw' and dataset == 'fb' and emb_sz == 16:
                    continue
                routine_partial_anal(dataset=dataset, model_name=model_name, emb_sz=emb_sz)


def combine_routine_basic_DADL():
    datasets = ['fb', 'bc', 'tw']
    # model_names = ['dw','gf','le','lle','n2v','dwext-REALFAST']
    # model_names = ['dw','lle','n2v','dwext-REALFAST']
    model_names = ['dadl']
    emb_szs = ['16', '64', '128']
    for dataset in datasets:
        for model_name in model_names:
            for emb_sz in emb_szs:
                routine_basic_anal(dataset=dataset, model_name=model_name, emb_sz=emb_sz)


if __name__ == '__main__':
    # datasets = ['fb']
    # model_names = ['dw','lle','le','gf','n2v']
    # emb_szs = ['16','64','128','256']
    # for dataset in datasets:
    #     for model_name in model_names:
    #         for emb_sz in emb_szs:
    #             routine_basic_anal(dataset=dataset,model_name=model_name,emb_sz=emb_sz)
    # combine_routines_partial()
    # combine_routines_basic()
    # routine_partial_anal()
    combine_routine_basic_DADL()

