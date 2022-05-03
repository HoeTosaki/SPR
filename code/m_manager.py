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

device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')
# device = tc.device('cpu')

# %% test device
# import torch as tc
# if tc.cuda.is_available():
#     print('use cuda')
# else:
#     print('use cpu')
# print('device:{}'.format(device))
# print('type:{}'.format(tc.cuda.get_device_name(0)))
# %%

device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')


'''abstract class'''
class Manager:
    '''
        定义基本的模型训练评估管理框架，以整合不同最短路实验方法的环境配置
    '''
    def __init__(self):
        pass

    def train(self):
        '''
            处理最短路推断的整体训练逻辑，耗时可能较长，结束后能够对任何节点的距离作出快速预测。
        '''
        pass


class GEDManager(Manager):
    def __init__(self,tmp_dir='../tmp',log_dir='../log',out_dir='../outputs',workers=1):
        super(GEDManager, self).__init__()
        self.tmp_dir = tmp_dir
        self.log_dir = log_dir
        self.out_dir = out_dir
        self.workers = workers
        self.loggers = {} #the content of which will be dynamic increased corresponding to manager states.
    def configModel(self,g,gen_file='generator',enc_file='encoder',dec_file='decoder',gen_dict={},enc_dict={},dec_dict={}):
        self.g = g
        self.gen_file = gen_file
        self.enc_file = enc_file
        self.dec_file = dec_file
        self.gen_dict = gen_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict

        # basic encoder params.
        self.enc_emb_sz = self.enc_dict['emb_sz']
        if 'encoder' in self.enc_dict:
            self.enc_encoder = self.enc_dict['encoder']

        # basic generator params.
        if 'train_generator' in self.gen_dict:
            self.gen_train_generator = self.gen_dict['train_generator']
        if 'val_generator' in self.gen_dict:
            self.gen_val_generator = self.gen_dict['val_generator']

        self.gen_is_landmarked = self.gen_dict['is_landmarked']
        if self.gen_is_landmarked:
            if 'landmark_generator' in self.gen_dict:
                self.gen_landmark_generator = self.gen_dict['landmark_generator']

        if 'use_timed' in self.gen_dict:
            self.use_timed = self.gen_dict['use_timed']
            if self.use_timed:
                self.data_name = self.gen_dict['data_name']
        else:
            self.use_timed=False
        # basic decoder params.
        self.dec_train_sz = self.dec_dict['train_sz']
        self.dec_val_sz = self.dec_dict['val_sz']
        if self.gen_is_landmarked:
            self.dec_landmark_sz = self.dec_dict['landmark_sz']
            self.dec_solely_optim = self.dec_dict['solely_optim']
            if self.dec_solely_optim:
                self.dec_lr = self.dec_dict['lr']
            else:
                self.dec_lr1 = self.dec_dict['lr1']
                self.dec_lr2 = self.dec_dict['lr2']
            self.dec_epoches1 = self.dec_dict['epoches1']
            self.dec_epoches2 = self.dec_dict['epoches2']
            self.dec_batch_sz1 = self.dec_dict['batch_sz1']
            self.dec_batch_sz2 = self.dec_dict['batch_sz2']
            self.dec_stop_cond1 = self.dec_dict['stop_cond1']
            self.dec_stop_cond2 = self.dec_dict['stop_cond2']
        else:
            self.dec_epoches = self.dec_dict['epoches']
            self.dec_batch_sz = self.dec_dict['batch_sz']
            self.dec_lr = self.dec_dict['lr']
            self.dec_stop_cond = self.dec_dict['stop_cond']

        if 'decoder' in self.dec_dict:
            self.dec_decoder = self.dec_dict['decoder']
        self.dec_batch_sz_val = self.dec_dict['batch_sz_val']
        self.dec_is_embed_model = self.dec_dict['is_embed_model']
        self.dec_save_between_epoch = self.dec_dict['save_between_epoch']
    def train(self,force=False):
        if not force and self.check_file():
            print('pretrained model file checked.')
            return
        if self.gen_is_landmarked:
            self._train_landmark(force=force)
        else:
            self._train_random(force=force)
    def _train_landmark(self, force=False):
        # encoder
        st_time = time.time()
        self.enc_encoder.train()
        print('embedding training consume {:.2f}s'.format(time.time()-st_time))
        if self.use_timed:
            with open(os.path.join('../log',self.data_name+'-bcdr-timed<emb>-len=' + str(self.enc_encoder.output_len)+'.log'),'w') as f:
                f.write('{:.4f}'.format(time.time()-st_time))
                f.flush()

        g = self.enc_encoder.load()

        print('embedding graph loaded & checked,g = ',g)
        if self.dec_is_embed_model:
            print('refresh embedding for decoder model.')
            self.dec_decoder.load_param(g)

        # generator
        st_time = time.time()
        self.gen_landmark_generator.gen_to_disk()
        print('landmark generation consume {:.2f}s'.format(time.time() - st_time))

        st_time = time.time()
        self.gen_train_generator.gen_to_disk()
        print('train generation consume {:.2f}s'.format(time.time() - st_time))

        st_time = time.time()
        self.gen_val_generator.gen_to_disk()
        print('val generation consume {:.2f}s'.format(time.time() - st_time))

        print('all pair samples generated.')

        # 训练模型
        if self.dec_solely_optim:
            optim = tc.optim.Adam(self.dec_decoder.parameters(), lr=self.dec_lr)
            print('all params:',list(self.dec_decoder.parameters()))
            self._train_landmark_stage1(optim)
            self._train_landmark_stage2(optim)
        else:
            self._train_landmark_stage1()
            self._train_landmark_stage2()

    def _train_landmark_stage1(self,optim=None):
        '''
        执行landmark内部的训练过程
        :param model: 输入模型参数
        :return: None
        '''

        # 初始训练配置
        if optim is None:
            lr1 = self.dec_lr1
            optim1 = tc.optim.Adam(self.dec_decoder.parameters(), lr=lr1)
        else:
            print('stage 1 use fix optim:{}'.format(optim))
            optim1 = optim

        loss = nn.MSELoss(reduction='sum')

        #定义第一阶段的Logger
        self.loggers[self.enc_encoder.out_file + '@landmark_stage1@TrainBasicLogger'] = m_logger.TrainBasicLogger(out_dir=self.log_dir)

        # 开始第一阶段训练
        print('landmark-stage1')
        self.dec_decoder = self.dec_decoder.to(device)
        stop_cnt = 0
        last_loss = 10000000
        stage1_time_st = time.time()
        for epoch in range(self.dec_epoches1):
            train_loss = 0.  # L2 均方误差
            train_intra_loader = self.gen_landmark_generator.loader(batch_sz=self.dec_batch_sz1,meta_batch_sz=10)
            st_time = time.time()
            self.dec_decoder.train()
            print('\t--epoch', epoch)
            samples_cnt = 0
            delay_break = False
            for i, lst in enumerate(train_intra_loader):
                ten = tc.Tensor(lst)
                samples_cnt += ten.shape[0]
                if self.dec_landmark_sz > 0:
                    if samples_cnt >= self.dec_landmark_sz:
                        delay_break = True
                        ten = ten[:ten.shape[0] - (samples_cnt-self.dec_landmark_sz)]
                optim1.zero_grad()
                src = ten[:, 0]
                dst = ten[:, 1]
                dist = ten[:, 2]
                dist[dist<0] = 20
                src = src.type_as(tc.LongTensor())
                dst = dst.type_as(tc.LongTensor())

                if self.dec_is_embed_model:
                    train_pred = self.dec_decoder(src,dst)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                else:
                    src_emb = self.g.ndata['emb'][src].to(device)
                    dst_emb = self.g.ndata['emb'][dst].to(device)
                    train_pred = self.dec_decoder(src_emb, dst_emb)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()).to(device)
                # print(train_pred)
                # print(dist)
                batch_loss = loss(train_pred, dist)
                batch_loss.backward()
                optim1.step()
                train_loss += batch_loss.item()
                if delay_break:
                    # print('val break as sample cnt', samples_cnt)
                    break
                print('\t\ttrain_loss : {:.3f}'.format(train_loss / (i + 1) / self.dec_batch_sz1), end='\r')
            cur_loss = train_loss / (self.dec_landmark_sz if self.dec_landmark_sz != -1 else samples_cnt)
            if self.dec_stop_cond1 > 0:
                if last_loss - cur_loss < self.dec_stop_cond1:
                    stop_cnt += 1
                else:
                    stop_cnt = 0 # reset
                if last_loss > cur_loss:
                  last_loss = cur_loss
            print('')
            print('\t--epoch {} time consume:{:.2f}'.format(epoch, time.time() - st_time))
            self.loggers[self.enc_encoder.out_file + '@landmark_stage1@TrainBasicLogger'].logging({'epoch':epoch,'train_loss':cur_loss})
            if stop_cnt > 20:
                print('train stopped since performance stayed peak.')
                break

        print('stage1 time consume:{:.2f}'.format(time.time() - stage1_time_st))
        if self.use_timed:
            with open(os.path.join('../log',self.data_name+'-bcdr-timed<stg1>-len=' + str(self.enc_encoder.output_len)+'.log'),'w') as f:
                f.write('{:.4f}'.format(time.time()-stage1_time_st))
                f.flush()


    def _train_landmark_stage2(self,optim=None):
        '''
        执行landmark与其他节点对的训练过程
        :param model: 输入模型参数
        :return: None
        '''
        # 初始训练配置
        if optim is None:
            lr2 = self.dec_lr2
            optim2 = tc.optim.Adam(self.dec_decoder.parameters(), lr=lr2)
        else:
            print('stage 2 use fix optim:{}'.format(optim))
            optim2 = optim

        loss = nn.MSELoss(reduction='sum')

        #定义第二阶段的Logger
        self.loggers[self.enc_encoder.out_file + '@landmark_stage2@TrainBasicLogger'] = m_logger.TrainBasicLogger(out_dir=self.log_dir)

        # 开始第二阶段训练
        print('landmark-stage2')
        self.dec_decoder = self.dec_decoder.to(device)
        stop_cnt = 0
        last_loss = 1000000
        stage2_time_st = time.time()
        for epoch in range(self.dec_epoches2):
            train_loss = 0.
            val_loss = 0.
            train_inter_loader = self.gen_train_generator.loader(batch_sz=self.dec_batch_sz2,meta_batch_sz=10)
            val_loader = self.gen_val_generator.loader(batch_sz=self.dec_batch_sz_val,meta_batch_sz=10)

            st_time = time.time()
            self.dec_decoder.train()
            print('\t--epoch', epoch)
            samples_len1 = 0
            samples_len2 = 0
            delay_break1 = False
            delay_break2 = False
            for i, lst in enumerate(train_inter_loader):
                ten = tc.Tensor(lst)
                samples_len1 += ten.shape[0]
                if self.dec_train_sz > 0:
                    if samples_len1 >= self.dec_train_sz:
                        delay_break1 = True
                        ten = ten[:ten.shape[0] - (samples_len1 - self.dec_train_sz)]
                optim2.zero_grad()
                src = ten[:, 0]
                dst = ten[:, 1]
                dist = ten[:, 2]
                dist[dist<0] = 20
                src = src.type_as(tc.LongTensor())
                dst = dst.type_as(tc.LongTensor())

                if self.dec_is_embed_model:
                    train_pred = self.dec_decoder(src,dst)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                else:
                    src_emb = self.g.ndata['emb'][src].to(device)
                    dst_emb = self.g.ndata['emb'][dst].to(device)
                    train_pred = self.dec_decoder(src_emb, dst_emb)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()).to(device)
                # print(train_pred)
                # print(dist)
                batch_loss = loss(train_pred, dist)
                batch_loss.backward()
                optim2.step()
                train_loss += batch_loss.item()
                if delay_break1:
                    # print('train break as samples cnt',samples_len1)
                    break
                # print('\t\ttrain_loss : {:.3f}'.format(train_loss / (i + 1) / self.dec_batch_sz2), end='\r')
            # print('')
            self.dec_decoder.eval()
            with tc.no_grad():
                for i,lst in enumerate(val_loader):
                    ten = tc.Tensor(lst)
                    samples_len2 += ten.shape[0]
                    if self.dec_val_sz > 0:
                        if samples_len2 >= self.dec_val_sz:
                            delay_break2 = True
                            ten = ten[:ten.shape[0] - (samples_len2 - self.dec_val_sz)]
                    src = ten[:, 0]
                    dst = ten[:, 1]
                    dist = ten[:, 2]
                    dist[dist < 0] = 20
                    src = src.type_as(tc.LongTensor())
                    dst = dst.type_as(tc.LongTensor())
                    if self.dec_is_embed_model:
                        val_pred = self.dec_decoder(src, dst)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                    else:
                        src_emb = self.g.ndata['emb'][src].to(device)
                        dst_emb = self.g.ndata['emb'][dst].to(device)
                        val_pred = self.dec_decoder(src_emb, dst_emb)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()).to(device)
                    batch_loss = loss(val_pred, dist)
                    val_loss += batch_loss.item()
                    if delay_break2:
                        # print('val break as sample cnt',samples_len2)
                        break
                    # print('\t\tval_loss : {:.3f}'.format(val_loss / (i + 1) / self.dec_batch_sz_val), end='\r')
            cur_val_loss = val_loss / (self.dec_val_sz if self.dec_val_sz != -1 else samples_len2)
            cur_train_loss = train_loss / (self.dec_train_sz if self.dec_train_sz != -1 else samples_len1)
            if self.dec_stop_cond2 > 0:
                if last_loss - cur_val_loss < self.dec_stop_cond2:
                    stop_cnt += 1
                else:
                    stop_cnt = 0  # reset
                if last_loss > cur_val_loss:
                    last_loss = cur_val_loss
            print('')
            print('\t--epoch {} time consume:{:.2f}, train loss={:.3f}, val loss={:.3f}'.format(epoch, time.time() - st_time,cur_train_loss,cur_val_loss))
            self.loggers[self.enc_encoder.out_file + '@landmark_stage2@TrainBasicLogger'].logging({'epoch':epoch,'train_loss':cur_train_loss,'val_loss':cur_val_loss})
            if epoch % self.dec_save_between_epoch == 0 and epoch != 0:
                self.save_model(epoch)
            if stop_cnt > 20:
                print('train stopped since performance stayed peak.')
                break
        print('stage2 time consume:{:.2f}'.format(time.time() - stage2_time_st))
        if self.use_timed:
            with open(os.path.join('../log',self.data_name+'-bcdr-timed<stg2>-len=' + str(self.enc_encoder.output_len)+'.log'),'w') as f:
                f.write('{:.4f}'.format(time.time()-stage2_time_st))
                f.flush()

        self.save_model()

        print('model save to disk successfully.')

    def _train_random(self,force=False):
        # 初始训练配置
        lr = self.dec_lr
        optim = tc.optim.Adam(self.dec_decoder.parameters(), lr=lr)
        loss = nn.MSELoss(reduction='sum')

        print('model params:',self.dec_decoder)

        # 定义随机训练的Logger
        self.loggers[self.enc_encoder.out_file + '@random_train@TrainBasicLogger'] = m_logger.TrainBasicLogger(out_dir=self.log_dir)

        # 开始随机训练
        print('start random train')
        stop_cnt = 0
        last_loss = 1000000
        random_train_time_st = time.time()
        for epoch in range(self.dec_epoches):
            train_loss = 0.
            val_loss = 0.
            train_inter_loader = self.gen_train_generator.loader(batch_sz=self.dec_batch_sz, meta_batch_sz=10)
            val_loader = self.gen_val_generator.loader(batch_sz=self.dec_batch_sz_val, meta_batch_sz=10)

            st_time = time.time()
            self.dec_decoder.train()
            print('\t--epoch', epoch)
            samples_len1 = 0
            samples_len2 = 0
            delay_break1 = False
            delay_break2 = False
            for i, lst in enumerate(train_inter_loader):
                ten = tc.Tensor(lst)
                samples_len1 += ten.shape[0]
                if self.dec_train_sz > 0:
                    if samples_len1 >= self.dec_train_sz:
                        delay_break1 = True
                        ten = ten[:ten.shape[0] - (samples_len1 - self.dec_train_sz)]
                optim.zero_grad()
                src = ten[:, 0]
                dst = ten[:, 1]
                dist = ten[:, 2]
                dist[dist<0] = 20

                src = src.type_as(tc.LongTensor())
                dst = dst.type_as(tc.LongTensor())

                if self.dec_is_embed_model:
                    train_pred = self.dec_decoder(src, dst)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor())
                else:
                    src_emb = self.g.ndata['emb'][src]
                    dst_emb = self.g.ndata['emb'][dst]
                    train_pred = self.dec_decoder(src_emb, dst_emb)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor())
                # print(train_pred)
                # print(dist)
                batch_loss = loss(train_pred, dist)
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                if delay_break1:
                    # print('train break as samples cnt',samples_len1)
                    break
                print('\t\ttrain_loss : {:.3f}'.format(train_loss / (i + 1) / self.dec_batch_sz), end='\r')
            print('')
            self.dec_decoder.eval()
            with tc.no_grad():
                for i, lst in enumerate(val_loader):
                    ten = tc.Tensor(lst)
                    samples_len2 += ten.shape[0]
                    if self.dec_val_sz > 0:
                        if samples_len2 >= self.dec_val_sz:
                            delay_break2 = True
                            ten = ten[:ten.shape[0] - (samples_len2 - self.dec_val_sz)]
                    src = ten[:, 0]
                    dst = ten[:, 1]
                    dist = ten[:, 2]
                    dist[dist < 0] = 20

                    src = src.type_as(tc.LongTensor())
                    dst = dst.type_as(tc.LongTensor())
                    if self.dec_is_embed_model:
                        val_pred = self.dec_decoder(src, dst)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                    else:
                        src_emb = self.g.ndata['emb'][src]
                        dst_emb = self.g.ndata['emb'][dst]
                        val_pred = self.dec_decoder(src_emb, dst_emb)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor())
                    batch_loss = loss(val_pred, dist)
                    val_loss += batch_loss.item()
                    if delay_break2:
                        # print('val break as sample cnt',samples_len2)
                        break
                    print('\t\tval_loss : {:.3f}'.format(val_loss / (i + 1) / self.dec_batch_sz_val), end='\r')
            cur_val_loss = val_loss / (self.dec_val_sz if self.dec_val_sz != -1 else samples_len2)
            cur_train_loss = train_loss / (self.dec_train_sz if self.dec_train_sz != -1 else samples_len1)
            if self.dec_stop_cond > 0:
                if last_loss - cur_val_loss < self.dec_stop_cond:
                    stop_cnt += 1
                else:
                    stop_cnt = 0  # reset
                if last_loss > cur_val_loss:
                    last_loss = cur_val_loss
            print('')
            print('\t--epoch {} time consume:{:.2f}, train loss={:.3f}, val loss={:.3f}'.format(epoch,
                                                                                                time.time() - st_time,
                                                                                                cur_train_loss,
                                                                                                cur_val_loss))
            self.loggers[self.enc_encoder.out_file + '@random_train@TrainBasicLogger'].logging(
                {'epoch': epoch, 'train_loss': cur_train_loss, 'val_loss': cur_val_loss})
            if stop_cnt > 20:
                print('train stopped since performance stayed peak.')
                break
        print('random train time consume:{:.2f}'.format(time.time() - random_train_time_st))
        self.save_model()

        print('model save to disk successfully.')

    def save_model(self,idx=-1):
        if idx == -1:
            tc.save(self.dec_decoder,os.path.join(self.out_dir,self.dec_file+'.decoder'))
            dgl.save_graphs(os.path.join(self.out_dir,self.dec_file+'.embedding'),[self.g])
        else:
            assert idx >= 0
            tc.save(self.dec_decoder, os.path.join(self.out_dir, self.dec_file + '~{}.decoder'.format(idx)))
            dgl.save_graphs(os.path.join(self.out_dir, self.dec_file + '~{}.embedding'.format(idx)), [self.g])

    def check_file(self):
        if not os.path.exists(os.path.join(self.out_dir,self.dec_file+'.decoder')):
            return False
        if not os.path.exists(os.path.join(self.out_dir,self.dec_file+'.embedding')):
            return False
        return True
    def load_model(self):
        model = tc.load(os.path.join(self.out_dir,self.dec_file+'.decoder'))
        g, _ = dgl.load_graphs(os.path.join(self.out_dir,self.dec_file+'.embedding'))
        g = g[0]
        return model,g
    def save_logs(self):
        print('start to save log...')
        st_time = time.time()
        for logger_name in self.loggers:
            self.loggers[logger_name].save_log(file=os.path.join(self.log_dir,logger_name+'.log'))
        print('save log finished, time consume: {}s'.format(time.time()-st_time))
    def enable_log(self):
        for logger_name in self.loggers:
            self.loggers[logger_name].log_state(state=True)
    def disable_log(self):
        for logger_name in self.loggers:
            self.loggers[logger_name].log_state(state=False)




class RangeManager(GEDManager):
    def __init__(self,tmp_dir='../tmp',log_dir='../log',out_dir='../outputs',workers=1):
        super(RangeManager, self).__init__()
        self.tmp_dir = tmp_dir
        self.log_dir = log_dir
        self.out_dir = out_dir
        self.workers = workers
        self.loggers = {} #the content of which will be dynamic increased corresponding to manager states.

    def configModel(self,l1=0.01, **kwargs):
        super(RangeManager, self).configModel(**kwargs)
        self.l1 = l1
    def _train_landmark_stage1(self,optim=None):
        '''
        执行landmark内部的训练过程
        :param model: 输入模型参数
        :return: None
        '''

        # 初始训练配置
        if optim is None:
            lr1 = self.dec_lr1
            optim1 = tc.optim.Adam(self.dec_decoder.parameters(), lr=lr1)
        else:
            print('stage 1 use fix optim:{}'.format(optim))
            optim1 = optim

        loss = nn.MSELoss(reduction='sum')

        #定义第一阶段的Logger
        self.loggers[self.enc_encoder.out_file + '@landmark_stage1@TrainBasicLogger'] = m_logger.TrainBasicLogger(out_dir=self.log_dir)

        # 开始第一阶段训练
        print('landmark-stage1')
        stop_cnt = 0
        last_loss = 10000000
        stage1_time_st = time.time()
        for epoch in range(self.dec_epoches1):
            train_loss = 0.  # L2 均方误差
            train_intra_loader = self.gen_landmark_generator.loader(batch_sz=self.dec_batch_sz1,meta_batch_sz=10)
            st_time = time.time()
            self.dec_decoder.train()
            print('\t--epoch', epoch)
            samples_cnt = 0
            delay_break = False
            for i, lst in enumerate(train_intra_loader):
                ten = tc.Tensor(lst)
                samples_cnt += ten.shape[0]
                if self.dec_landmark_sz > 0:
                    if samples_cnt >= self.dec_landmark_sz:
                        delay_break = True
                        ten = ten[:ten.shape[0] - (samples_cnt-self.dec_landmark_sz)]
                optim1.zero_grad()
                src = ten[:, 0]
                dst = ten[:, 1]
                dist = ten[:, 2]

                src = src.type_as(tc.LongTensor())
                dst = dst.type_as(tc.LongTensor())

                if self.dec_is_embed_model:
                    train_pred = self.dec_decoder(src,dst)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                else:
                    src_emb = self.g.ndata['emb'][src]
                    dst_emb = self.g.ndata['emb'][dst]
                    train_pred = self.dec_decoder(src_emb, dst_emb)
                    dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor())
                # print(train_pred)
                # print(dist)
                batch_loss = loss(train_pred, dist)
                batch_loss.backward()
                optim1.step()
                train_loss += batch_loss.item()
                if delay_break:
                    # print('val break as sample cnt', samples_cnt)
                    break
                print('\t\ttrain_loss : {:.3f}'.format(train_loss / (i + 1) / self.dec_batch_sz1), end='\r')
            cur_loss = train_loss / (self.dec_landmark_sz if self.dec_landmark_sz != -1 else samples_cnt)
            if self.dec_stop_cond1 > 0:
                if last_loss - cur_loss < self.dec_stop_cond1:
                    stop_cnt += 1
                else:
                    stop_cnt = 0 # reset
                if last_loss > cur_loss:
                  last_loss = cur_loss
            print('')
            print('\t--epoch {} time consume:{:.2f}'.format(epoch, time.time() - st_time))
            self.loggers[self.enc_encoder.out_file + '@landmark_stage1@TrainBasicLogger'].logging({'epoch':epoch,'train_loss':cur_loss})
            if stop_cnt > 20:
                print('train stopped since performance stayed peak.')
                break

        print('stage1 time consume:{:.2f}'.format(time.time() - stage1_time_st))

    def _train_landmark_stage2(self,optim=None):
        '''
        执行landmark与其他节点对的训练过程
        :param model: 输入模型参数
        :return: None
        '''
        # 初始训练配置
        if optim is None:
            lr2 = self.dec_lr2
            optim2 = tc.optim.Adam(self.dec_decoder.parameters(), lr=lr2)
        else:
            print('stage 2 use fix optim:{}'.format(optim))
            optim2 = optim

        loss = nn.MSELoss(reduction='sum')

        #定义第二阶段的Logger
        self.loggers[self.enc_encoder.out_file + '@landmark_stage2@TrainBasicLogger'] = m_logger.TrainBasicLogger(out_dir=self.log_dir)

        # 开始第二阶段训练
        print('landmark-stage2')
        stop_cnt = 0
        last_loss = 1000000
        stage2_time_st = time.time()
        for epoch in range(self.dec_epoches2):
            train_loss = 0.
            total_loss = 0.
            local_loss = 0.
            val_loss = 0.
            train_inter_loader = self.gen_train_generator.loader(batch_sz=self.dec_batch_sz2,meta_batch_sz=10)
            val_loader = self.gen_val_generator.loader(batch_sz=self.dec_batch_sz_val,meta_batch_sz=10)

            st_time = time.time()
            self.dec_decoder.train()
            print('\t--epoch', epoch)
            samples_len1 = 0
            samples_len2 = 0
            delay_break1 = False
            delay_break2 = False
            for i, lst in enumerate(train_inter_loader):
                for src,dst,dist_dic in lst:
                    samples_len1 += 1
                    if self.dec_train_sz > 0:
                        if samples_len1 >= self.dec_train_sz:
                            delay_break1 = True
                            break
                    optim2.zero_grad()
                    dist = dist_dic[src][dst]
                    if self.dec_is_embed_model:
                        train_pred = self.dec_decoder(src,dst)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                    else:
                        src_emb = self.g.ndata['emb'][src].view(1,-1)
                        dst_emb = self.g.ndata['emb'][dst].view(1,-1)
                        train_pred = self.dec_decoder(src_emb, dst_emb)
                        dist = tc.FloatTensor([dist]).view(1,-1)
                    local_pred = self.local_smooth(self.dec_decoder,src,dst,dist_dic)
                    # print(train_pred)
                    # print(dist)
                    loss1 = loss(train_pred, dist)
                    loss2 = loss(local_pred,tc.zeros(local_pred.shape))
                    batch_loss = loss1 +  self.l1 *loss2
                    batch_loss.backward()
                    optim2.step()
                    total_loss += batch_loss.item()
                    local_loss += loss2
                    train_loss += loss1.item()

                    if delay_break1:
                        # print('train break as samples cnt',samples_len1)
                        break
                    print('\t\ttrain_loss : {:.3f}, local loss : {:.3f}, org local loss:{:.3f}, total : {:.3f}'.format(train_loss / samples_len1,local_loss / samples_len1, local_pred.item() / samples_len1,total_loss / samples_len1), end='\r')
            print('')
            self.dec_decoder.eval()
            with tc.no_grad():
                for i,lst in enumerate(val_loader):
                    # print(lst)
                    # print(type(lst))
                    ten = tc.Tensor(lst)
                    samples_len2 += ten.shape[0]
                    if self.dec_val_sz > 0:
                        if samples_len2 >= self.dec_val_sz:
                            delay_break2 = True
                            ten = ten[:ten.shape[0] - (samples_len2 - self.dec_val_sz)]
                    src = ten[:, 0]
                    dst = ten[:, 1]
                    dist = ten[:, 2]
                    src = src.type_as(tc.LongTensor())
                    dst = dst.type_as(tc.LongTensor())
                    if self.dec_is_embed_model:
                        val_pred = self.dec_decoder(src, dst)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor()) ** 2
                    else:
                        src_emb = self.g.ndata['emb'][src]
                        dst_emb = self.g.ndata['emb'][dst]
                        val_pred = self.dec_decoder(src_emb, dst_emb)
                        dist = dist.view(dist.shape[0], 1).type_as(tc.FloatTensor())
                    batch_loss = loss(val_pred, dist)
                    val_loss += batch_loss.item()
                    if delay_break2:
                        # print('val break as sample cnt',samples_len2)
                        break
                    print('\t\tval_loss : {:.3f}'.format(val_loss / samples_len2), end='\r')
            cur_val_loss = val_loss / samples_len2
            cur_train_loss = train_loss / samples_len1
            if self.dec_stop_cond2 > 0:
                if last_loss - cur_val_loss < self.dec_stop_cond2:
                    stop_cnt += 1
                else:
                    stop_cnt = 0  # reset
                if last_loss > cur_val_loss:
                    last_loss = cur_val_loss
            print('')
            print('\t--epoch {} time consume:{:.2f}, train loss={:.3f}, val loss={:.3f}'.format(epoch, time.time() - st_time,cur_train_loss,cur_val_loss))
            self.loggers[self.enc_encoder.out_file + '@landmark_stage2@TrainBasicLogger'].logging({'epoch':epoch,'train_loss':cur_train_loss,'val_loss':cur_val_loss})
            if stop_cnt > 5:
                print('train stopped since performance stayed peak.')
                break
        print('stage2 time consume:{:.2f}'.format(time.time() - stage2_time_st))
        self.save_model()

        print('model save to disk successfully.')

    def local_smooth(self,model,src,dst,dist_dic):
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
        if len(dist_dic.keys()) != 2:
            print('error:',src,dst,dist_dic)
        # assert len(dist_dic.keys()) == 2

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
                pred_n1u_n1v += model(self.g.ndata['emb'][e_src].view(1, -1),
                                           self.g.ndata['emb'][e_dst].view(1, -1))
                cnt_n1u_n1v += 1

        for e_src in src_set_n1u:
            for e_dst in dst_set_n0v:
                pred_n1u_n0v += model(self.g.ndata['emb'][e_src].view(1, -1),
                                           self.g.ndata['emb'][e_dst].view(1, -1))
                cnt_n1u_n0v += 1

        for e_src in src_set_n0u:
            for e_dst in dst_set_n1v:
                pred_n0u_n1v += model(self.g.ndata['emb'][e_src].view(1, -1),
                                           self.g.ndata['emb'][e_dst].view(1, -1))
                cnt_n0u_n1v += 1

        for e_src in src_set_n0u:
            for e_dst in dst_set_n0v:
                pred_n0u_n0v += model(self.g.ndata['emb'][e_src].view(1, -1),
                                           self.g.ndata['emb'][e_dst].view(1, -1))
                cnt_n0u_n0v += 1
        smooth_loss = (pred_n1u_n1v - pred_n1u_n0v - pred_n0u_n1v + pred_n0u_n0v) /\
                      ((len(src_set_n0u) + len(src_set_n1u)) * (len(dst_set_n0v) + len(dst_set_n1v)))
        return smooth_loss
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''