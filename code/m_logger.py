import dgl
import matplotlib.pyplot as plt
import torch as tc
import numpy as np
import pandas as pd

class BasicLogger:
    def __init__(self,out_dir):
        self.out_dir = out_dir
        self.log = []
        self.state = True # default on logging.
    def log_state(self,state=None):
        if state is not None:
            self.state = state
        else:
            self.state = not self.state
    def logging(self,param_dict):
        '''
        handle param_dict & log into self.log
        :param param_dict:
        :return: does current instance work for the exact context.
        '''
        return False
    def save_log(self,file):
        if not self._check_state():
            return
        with open(file,'w') as f:
            for item in self.log:
                f.write(self.encode(item)+'\n')
    def load_log(self,file):
        if not self._check_state:
            return
        with open(file,'r') as f:
            while True:
                line = f.readline()
                if not line or line == "":
                    break
                else:
                    self.log.append(self.decode(line.strip()))
    def encode(self,log_item):
        '''
        :param log_item:
        :return: str of log item, one line occupied.
        '''
        raise NotImplementedError
    def decode(self,log_str):
        '''
        :param log_str:
        :return: log item
        '''
        raise NotImplementedError
    def _check_state(self):
        return self.state

class TrainBasicLogger(BasicLogger):
    def __init__(self,out_dir):
        super(TrainBasicLogger,self).__init__(out_dir=out_dir)
        self.has_val = False
    def logging(self,param_dict):
        if not super(TrainBasicLogger,self)._check_state():
            return
        # epoch = param_dict['epoch']
        # train_loss = param_dict['train_loss']
        # val_loss = param_dict['val_loss']
        # Adam status wo arawaseteminai?
        if 'epoch' not in param_dict or 'train_loss' not in param_dict:
            return False
        if 'val_loss' in param_dict:
            self.has_val = True
        else:
            self.has_val = False
        self.log.append(param_dict)
        return True
    def encode(self,log_item):
        if self.has_val:
            return '{},{},{}'.format(log_item['epoch'],log_item['train_loss'],log_item['val_loss'])
        else:
            return '{},{}'.format(log_item['epoch'],log_item['train_loss'])
    def decode(self,log_str):
        lst = log_str.strip().split(',')
        assert len(lst) == 3 or len(lst) == 2
        if len(lst) == 3:
            self.has_val = True
            epoch = int(lst[0])
            train_loss = float(lst[1])
            val_loss = float(lst[2])
            return {'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss}
        if len(lst) == 2:
            self.has_val = False
            epoch = int(lst[0])
            train_loss = float(lst[1])
            return {'epoch': epoch, 'train_loss': train_loss}
    def draw_and_save(self,file):
        #训练过程loss曲线
        if self.has_val:
            train_loss = []
            val_loss = []
            epoch = []
            for item in self.log:
                epoch.append(item['epoch'])
                train_loss.append(item['train_loss'])
                val_loss.append(item['val_loss'])
            plt.plot(epoch,train_loss,c='b')
            plt.plot(epoch,val_loss,c='r')
        else:
            train_loss = []
            epoch = []
            for item in self.log:
                epoch.append(item['epoch'])
                train_loss.append(item['train_loss'])
            plt.plot(epoch, train_loss, c='b')
        plt.savefig(file)

if __name__ == '__main__':
    logger = TrainBasicLogger('../log') #初始化参数无了。。记性不好
    logger.load_log('../log/landmark_stage2@TrainBasicLogger.log')
    logger.draw_and_save('../log/landmark_stage2@TrainBasicLogger.png')



