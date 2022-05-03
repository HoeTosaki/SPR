import m_dqm
import os
import torch as tc
import time
import memory_profiler as mem_profile
import random
import numpy as np

class DistanceQueryEval:
    def __init__(self,nx_g=None,dqms=None,generator=None,eval_name='dq-eval',tmp_dir='../log',force=False,seed=None):
        self.nx_g = nx_g
        self.dqms = dqms
        self.generator = generator
        self.eval_name=eval_name
        self.tmp_dir = tmp_dir
        self.force = force
        self.work_dic = {}
        self.seed = seed

    def evaluate(self):
        for dqm in self.dqms:
            print('[{}] - [{}]'.format(self,dqm))
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
            self.work_dic[dqm.model_name] = self._evaluate(dqm)
        return self.work_dic

    def _evaluate(self,dqm):
        raise NotImplementedError

    def pwd(self):
        return os.path.join(self.tmp_dir,self.eval_name)

    def __str__(self):
        return 'Eval:' + self.eval_name

class DistOnlineEval(DistanceQueryEval):
    def __init__(self,test_sz=10000,gen_batch_sz=200,report_interval=3,**kwargs):
        super(DistOnlineEval, self).__init__(**kwargs)
        self.gen_batch_sz = gen_batch_sz
        self.test_sz = test_sz
        self.report_interval = report_interval

    def __str__(self):
        return 'DistOnlineEval:' + self.eval_name

    def _evaluate(self,dqm):
        # dqm = m_dqm.DistanceQueryModel()

        gen_loader = self.generator.loader(batch_sz=self.gen_batch_sz, meta_batch_sz=10)
        cur_sample_len = 0
        total_mae = 0.
        total_mre = 0.
        total_time = 0.
        total_mem = []
        # path_mae

        for idx,queries in enumerate(gen_loader):
            queries = tc.Tensor(queries)
            cur_sample_len += queries.shape[0]
            if cur_sample_len >= self.test_sz:
                delay_break = True # break later in case that tail samples are processed.
                queries = queries[:queries.shape[0] - (cur_sample_len - self.test_sz)]
            srcs = queries[:, 0]
            dsts = queries[:, 1]
            dists = queries[:, 2]
            dists[dists < 0] = 20
            srcs = srcs.type_as(tc.LongTensor())
            dsts = dsts.type_as(tc.LongTensor())

            cur_st_time = time.time()
            pred_dists,ed_time = dqm.query(srcs=srcs,dsts=dsts)
            total_time += ed_time - cur_st_time
            if type(pred_dists) is not list:
                pred_dists = pred_dists.tolist()
            pred_dists = tc.Tensor([round(ele) for ele in pred_dists])

            sub = tc.abs(pred_dists.view(-1) - dists)
            sub = sub[dists > 0]
            dists = dists[dists > 0]
            # if tc.sum(sub).item() > 0:
            #     pred_dists = dqm.query(srcs=srcs, dsts=dsts)
            total_mae += tc.sum(sub).item()
            sub = sub / dists
            total_mre += tc.sum(sub).item()

            total_mem.append(dqm.get_mem_usage(is_train=False))
            if idx % self.report_interval == 0:
                print('dqm {} | iter {} | mae={:.4f} | mre={:.4f} | query_time={:.5f} | mem={:5f}MB'.format(dqm,idx,total_mae/cur_sample_len,total_mre/cur_sample_len,total_time/cur_sample_len,sum(total_mem) / len(total_mem)))
        if cur_sample_len != self.test_sz:
            print('Warning[{}]: test_sz {} != cur_sample_len {}'.format(self,self.test_sz,cur_sample_len))
        return {
            'mae':total_mae / cur_sample_len,
            'mre':total_mre / cur_sample_len,
            'query_time':total_time / cur_sample_len,
            'query_mem':sum(total_mem) / len(total_mem),
        }

class DistOfflineEval(DistanceQueryEval):
    def __init__(self,**kwargs):
        super(DistOfflineEval, self).__init__(**kwargs)

    def __str__(self):
        return 'DistOfflineEval:' + self.eval_name

    def _evaluate(self,dqm):
        # dqm = m_dqm.DistanceQueryModel()

        total_time = time.time()
        ed_time = dqm.generate()
        total_time = ed_time - total_time

        total_mem = dqm.get_mem_usage(is_train=True)
        total_storage = dqm.get_disk_usage()
        print('dqm {} | gen_time={:.5f} | mem={:.5f} | storage={:.5f}'.format(dqm,total_time,total_mem,total_storage))
        return {
            'gen_time': total_time,
            'train_mem': total_mem,
            'storage': total_storage,
        }






