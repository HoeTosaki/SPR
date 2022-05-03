
CFG = {
    'MODEL_SAVE_PATH':'../saved_models/',
    'USE_CUDA': False,
    'DEVICE': 'cpu',
    'DATASET_PATH':'../datasets/',
    'LOG_PATH':'../log/',
    'TMP_PATH':'../tmp/',
    'OUT_PATH':'../outputs/',
}

import dgl
import pandas as pd
import copy
import multiprocessing
import time

def const_graph(edges):
    src = []
    dst = []
    for edge in edges:
        src.append(edge[0])
        dst.append(edge[1])
    g = dgl.DGLGraph((src, dst))
    g = dgl.to_bidirected(g)
    return g


'''
Min-max Heap.
'''
class MinMaxHeap(object):
    """
    Implementation of a Min-max heap following Atkinson, Sack, Santoro, and
    Strothotte (1986): https://doi.org/10.1145/6617.6621
    """
    def __init__(self, reserve=0):
        self.a = [None] * reserve
        self.size = 0
    def __len__(self):
        return self.size

    def insert(self, key):
        """
        Insert key into heap. Complexity: O(log(n))
        """
        if len(self.a) < self.size + 1:
            self.a.append(key)
        insert(self.a, key, self.size)
        self.size += 1

    def peekmin(self):
        """
        Get minimum element. Complexity: O(1)
        """
        return peekmin(self.a, self.size)

    def peekmax(self):
        """
        Get maximum element. Complexity: O(1)
        """
        return peekmax(self.a, self.size)

    def popmin(self):
        """
        Remove and return minimum element. Complexity: O(log(n))
        """
        m, self.size = removemin(self.a, self.size)
        return m

    def popmax(self):
        """
        Remove and return maximum element. Complexity: O(log(n))
        """
        m, self.size = removemax(self.a, self.size)
        return m

    def remove(self,i):
        m,self.size = remove(self.a,self.size,i)
        return m

    def fast_copy(self):
        h = MinMaxHeap()
        h.a = copy.deepcopy(self.a)
        return h

def level(i):
    return (i+1).bit_length() - 1


def trickledown(array, i, size):
    if level(i) % 2 == 0:  # min level
        trickledownmin(array, i, size)
    else:
        trickledownmax(array, i, size)


def trickledownmin(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2][0] < array[m][0]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j][0] < array[m][0]:
                m = j
                child = False

        if child:
            if array[m][0] < array[i][0]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m][0] < array[i][0]:
                if array[m][0] < array[i][0]:
                    array[m], array[i] = array[i], array[m]
                if array[m][0] > array[(m-1) // 2][0]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmin(array, m, size)


def trickledownmax(array, i, size):
    if size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2][0] > array[m][0]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j][0] > array[m][0]:
                m = j
                child = False

        if child:
            if array[m][0] > array[i][0]:
                array[i], array[m] = array[m], array[i]
        else:
            if array[m][0] > array[i][0]:
                if array[m][0] > array[i][0]:
                    array[m], array[i] = array[i], array[m]
                if array[m][0] < array[(m-1) // 2][0]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                trickledownmax(array, m, size)


def bubbleup(array, i):
    if level(i) % 2 == 0:  # min level
        if i > 0 and array[i][0] > array[(i-1) // 2][0]:
            array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
            bubbleupmax(array, (i-1)//2)
        else:
            bubbleupmin(array, i)
    else:  # max level
        if i > 0 and array[i][0] < array[(i-1) // 2][0]:
            array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
            bubbleupmin(array, (i-1)//2)
        else:
            bubbleupmax(array, i)


def bubbleupmin(array, i):
    while i > 2:
        if array[i][0] < array[(i-3) // 4][0]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def bubbleupmax(array, i):
    while i > 2:
        if array[i][0] > array[(i-3) // 4][0]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return


def peekmin(array, size):
    assert size > 0
    return array[0]


def peekmax(array, size):
    assert size > 0
    if size == 1:
        return array[0]
    elif size == 2:
        return array[1]
    else:
        if array[1][0] >= array[2][0]:
            return array[1]
        else:
            return array[2]


def removemin(array, size):
    assert size > 0
    elem = array[0]
    array[0] = array[size-1]
    # array = array[:-1]
    trickledown(array, 0, size - 1)
    return elem, size-1

def remove(array,size,i):
    assert size > 0
    elem = array[i]
    array[i] = array[size-1]
    trickledown(array,i,size - 1)
    return elem,size - 1

def removemax(array, size):
    assert size > 0
    if size == 1:
        return array[0], size - 1
    elif size == 2:
        return array[1], size - 1
    else:
        i = 1 if array[1][0] > array[2][0] else 2
        elem = array[i]
        array[i] = array[size-1]
        # array = array[:-1]
        trickledown(array, i, size - 1)
        return elem, size-1


def insert(array, k, size):
    array[size] = k
    bubbleup(array, size)


def minmaxheapproperty(array, size):
    for i, k in enumerate(array[:size]):
        if level(i) % 2 == 0:  # min level
            # check children to be larger
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j][0] < k[0]:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be larger
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j][0] < k[0]:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
        else:
            # check children to be smaller
            for j in range(2 * i + 1, min(2 * i + 3, size)):
                if array[j][0] > k[0]:
                    print(array, j, i, array[j], array[i], level(i))
                    return False
            # check grand children to be smaller
            for j in range(4 * i + 3, min(4 * i + 7, size)):
                if array[j][0] > k[0]:
                    print(array, j, i, array[j], array[i], level(i))
                    return False

    return True


def Xtest(n):
    from random import randint
    a = [-1] * n
    l = []
    size = 0
    for _ in range(n):
        x = randint(0, 5 * n)
        insert(a, x, size)
        size += 1
        l.append(x)
        assert minmaxheapproperty(a, size)

    assert size == len(l)
    print(a)

    while size > 0:
        assert min(l) == peekmin(a, size)
        assert max(l) == peekmax(a, size)
        if randint(0, 1):
            e, size = removemin(a, size)
            assert e == min(l)
        else:
            e, size = removemax(a, size)
            assert e == max(l)
        l[l.index(e)] = l[-1]
        l.pop(-1)
        assert len(a[:size]) == len(l)
        assert minmaxheapproperty(a, size)

    print("OK")


def Xtest_heap(n):
    from random import randint
    heap = MinMaxHeap(n)
    l = []
    for _ in range(n):
        x = randint(0, 5 * n)
        heap.insert(x)
        l.append(x)
        assert minmaxheapproperty(heap.a, len(heap))

    assert len(heap) == len(l)
    print(heap.a)

    while len(heap) > 0:
        assert min(l) == heap.peekmin()
        assert max(l) == heap.peekmax()
        if randint(0, 1):
            e = heap.popmin()
            assert e == min(l)
        else:
            e = heap.popmax()
            assert e == max(l)
        l[l.index(e)] = l[-1]
        l.pop(-1)
        assert len(heap) == len(l)
        assert minmaxheapproperty(heap.a, len(heap))

    print("OK")

def routine_test_heap():
    heap = MinMaxHeap()
    lst = [1, 1, 3, 9, 0, 3, 2, 5, 6, 4]
    lst_ele = ['why' + str(ele) for ele in lst]
    min_lst, max_lst = [], []

    for ele, obj in zip(lst, lst_ele):
        heap.insert((ele, obj))
    while heap.size != 0:
        min_lst.append(heap.popmin())

    for ele, obj in zip(lst, lst_ele):
        heap.insert((ele, obj))
    while heap.size != 0:
        max_lst.append(heap.popmax())

    print('min', [ele for ele in min_lst])
    print('max', [ele for ele in max_lst])

def formatList(lst):
    assert len(lst) == 5, print(lst)
    res = []
    # for  i in range(1,5):
    #     llst = str(lst[i]).split('±')
    #     res.append(' ${:.4g}$ '.format(float(llst[0])))
    for  i in range(1,5):
        llst = str(lst[i]).split('±')
        res.append(float(llst[0]))
    res = [' ${:.4g}$ '.format(res[0] / res[1])]

    return res

def genTexTable():
    files = ['../log/cr-anal-p.csv','../log/fb-anal-p.csv','../log/gq-anal-p.csv']
    # files = ['../log/cr-anal-p.csv']
    dic = {}
    for idx,file in enumerate(files):
        data = pd.read_csv(file)
        data = data.to_numpy()
        pnt = 0
        dic[idx] = {}
        for i in range(data.shape[0]):
            lst = str(data[i][0]).split('&&')
            if lst[1].strip() == 'BFS':
                dic[idx]['BFS'] =[r'\multicolumn{2}{c|}{BFS}'] + formatList(data[i][1:])
            else:
                dic[idx][pnt] = [str(lst[0]).strip() + ' & ' + str(lst[1]).strip()] + formatList(data[i][1:])
                pnt+=1
    res = []
    res.append(' & '.join(dic[0]['BFS'] + dic[1]['BFS'][1:] + dic[2]['BFS'][1:]))
    for k in dic[0].keys():
        if k == 'BFS':
            continue
        res.append(' & '.join(dic[0][k] + dic[1][k][1:] + dic[2][k][1:]))
    print(' \\\\\n'.join(res))


def _th_func(pid,func, seq_args, **shared_dict):
    st_time = time.time()
    print('process {} start.'.format(pid))
    ret_dict = {pid:func(*seq_args, **shared_dict)}
    print('{} finished with time {:.4f}s.'.format(pid, time.time() - st_time))
    return ret_dict

class MPManager:
    def __init__(self,batch_sz=128,num_workers=8,use_shuffle=True):
        self.batch_sz = batch_sz
        self.num_workers = num_workers
        self.use_shuffle = use_shuffle
        self.pool = None
        self.func = None

    def multi_proc(self,func,seq_args,auto_concat=True,**shared_args):
        '''
        :param th_func: function executed by multi-process
        :param seq_args: [Iterable,...] - a group of sequential elements as inputs.
        :param shared_args: dict - shared & read-only dictionary for each process.
        :param auto_concat: bool - if output of th_func is list-like result, auto concatenation will be involved in final return.
        :return: dict of all outputs or concatenation of output lists from every process.
        '''
        assert seq_args is not None and len(seq_args) > 0
        assert '__PID__' not in shared_args
        seq_len = len(seq_args[0])
        task_results = []
        seq_pnt = 0

        pool = multiprocessing.Pool(processes=self.num_workers)

        iter_id = 0
        while seq_pnt < seq_len:
            seq_intv = (seq_pnt,min(seq_pnt + self.batch_sz,seq_len))

            batch_seq_args = []
            for seq_arg in seq_args:
                batch_seq_args.append(seq_arg[seq_intv[0]:seq_intv[1]])

            param_lst = [iter_id,func]
            param_lst.append(batch_seq_args)
            param_dict = {}
            param_dict.update(shared_args)
            param_dict['__PID__'] = iter_id
            task_results.append(pool.apply_async(_th_func,args=param_lst,kwds=param_dict))

            iter_id += 1
            seq_pnt = min(seq_pnt + self.batch_sz,seq_len)

        pool.close()
        pool.join()
        print('all processes finished.')

        ret_dict = {}
        for res in task_results:
            cur_ret_dict = res.get()
            ret_dict.update(cur_ret_dict)
        assert len(ret_dict.keys()) == len(task_results), print('dict {} != list {}'.format(len(ret_dict.keys()),len(task_results)))

        print('concat all sliced results...')
        # ret = []
        if auto_concat:
            ret = []
            [ret.extend(v) for k,v in sorted(ret_dict.items(),key=lambda kv: int(kv[0]))]
            return ret
        else:
            # [ret.append(v) for k, v in sorted(ret_dict.items(), key=lambda kv: int(kv[0]))]
            return ret_dict
        # return ret












if __name__ == '__main__':
    print('hello utils.')
    a = {'1':23,'3':-1,'2':0,'200':-100}
    print(sorted(a.items(), key=lambda k: int(k[0])))

    # routine_test_heap()
    # genTexTable()
    # print("\033[31mBlog: something\033[0m")
    # print((1,0) > (0,1))
