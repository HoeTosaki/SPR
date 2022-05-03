import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileinput
import dgl
import dgl.data
import os
import m_dage_test

class DatasetManager:
    def __init__(self, raw_dir='../datasets/raw', src_dir='../datasets/src', dst_dir='../datasets/dst', force=False):
        self.raw_dir = raw_dir
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        if force:
            os.system(r'rm -rf ' + self.dst_dir)
            os.system(r'rm -rf ' + self.src_dir)
        if not os.path.exists(self.raw_dir):
            raise IOError('not found raw data in {}'.format(self.raw_dir))
        print('raw files checked.')
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)

    def clean(self, complete=False):
        os.system(r'rm -rf ' + self.src_dir)
        if complete:
            os.system(r'rm -rf ' + self.dst_dir)

    def process(self, force=False, debug=False):
        # self.proc_facebook(force=force,debug=debug)
        # self.proc_blogcatalog(force=force,debug=debug)
        # self.proc_twitter(force=force,debug=debug)
        # self.proc_gplus(force = force,debug = debug) ##too longggg node label to read-in
        self.proc_youtube(force=force, debug=debug)
        # self.proc_karate(force=force,debug=debug)
        # self.proc_cora(force=force,debug=debug)
        # self.proc_pubmed(force=force,debug=debug)
        # self.proc_pa(force=force,debug=debug)
        # self.proc_nd(force=force, debug=debug)
        # self.proc_grqc(force=force, debug=debug)
        self.proc_dblp(force=force, debug=debug)
        self.proc_pokec(force=force, debug=debug)

    def load_facebook(self, filename='facebook', force=False):
        self.proc_facebook(force=force)
        return dgl.load_graphs(os.path.join(self.dst_dir, filename))

    def load_blogcatalog(self, filename='blogcatalog', force=False):
        self.proc_blogcatalog(force=force)
        return dgl.load_graphs(os.path.join(self.dst_dir, filename))

    def load_twitter(self, filename='twitter', force=False):
        self.proc_twitter(force=force)
        return dgl.load_graphs(os.path.join(self.dst_dir, filename))

    def proc_facebook(self, filename='facebook', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('facebook: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if not file.endswith('.edges'):
                        continue
                    print('facebook: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if debug and debug_cnt % 10000 == 0:
                            print('facebook: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split(' ')
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))

                        # add double edge to central node.
                        file_id = file.find('.edges')
                        if file_id == -1:
                            print(file)
                        assert file_id != -1

                        src.append(int(lst[0]))
                        dst.append(int(file_id))

                        src.append(int(lst[1]))
                        dst.append(int(file_id))

            #                         dst.append(int(lst[0]))
            #                         src.append(int(file_id))

            #                         dst.append(int(lst[1]))
            #                         src.append(int(file_id))

            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('facebook: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('facebook: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('facebook: dst files checked.')

    def proc_blogcatalog(self, filename='BlogCatalog-dataset', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('blogcatalog: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if not file == 'edges.csv':
                        continue
                    print('blogcatalog: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if debug and debug_cnt % 10000 == 0:
                            print('blogcatalog: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split(',')
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))
            g = dgl.graph((src, dst))
            g = g.to_simple()
            g = dgl.to_bidirected(g)

            print('blogcatalog: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)
            print('blogcatalog: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('blogcatalog: dst files checked.')

    def proc_twitter(self, filename='twitter', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('twitter: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                src = []
                dst = []
                for file in files:
                    if not file.endswith('.edges'):
                        continue
                    print('twitter: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if debug and debug_cnt % 10000 == 0:
                            print('twitter: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split(' ')
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        if len(lst) != 2:
                            print(lst)
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))

                        # add double edge to central node.
                        file_id = file.find('.edges')
                        if file_id == -1:
                            print(file)
                        assert file_id != -1

                        dst.append(int(lst[0]))
                        src.append(int(file_id))

                        dst.append(int(lst[1]))
                        src.append(int(file_id))

                g = dgl.graph((src, dst))
                g = g.to_simple()
                g = dgl.to_bidirected(g)

                print('twitter: \tchecked all node & edges from files.')

                # reindex by excluding island
                g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

                print('twitter: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('twitter: dst files checked.')

    def proc_gplus(self, filename='gplus', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('gplus: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                src = []
                dst = []
                for file in files:
                    if not file.endswith('.edges'):
                        continue
                    print('gplus: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if debug and debug_cnt % 10000 == 0:
                            print('gplus: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split(' ')
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        if len(lst) != 2:
                            print(lst)
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))

                        # add double edge to central node.
                        file_id = file.find('.edges')
                        if file_id == -1:
                            print(file)
                        assert file_id != -1

                        dst.append(int(lst[0]))
                        src.append(int(file_id))

                        dst.append(int(lst[1]))
                        src.append(int(file_id))

                g = dgl.graph((src, dst))
                g = g.to_simple()
                g = dgl.to_bidirected(g)

                print(r'gplus: \tchecked all node & edges from files.')
                # reindex by excluding island
                g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

                print('gplus: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('gplus: dst files checked.')

    def proc_youtube(self, filename='youtube', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('youtube: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            id2nid = {}
            nid_pnt = 0
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'youtube.txt':
                        continue
                    print('youtube: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('youtube: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split('\t')
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src_id = int(lst[0])
                        dst_id = int(lst[1])
                        if src_id not in id2nid:
                            id2nid[src_id] = nid_pnt
                            nid_pnt += 1
                        if dst_id not in id2nid:
                            id2nid[dst_id] = nid_pnt
                            nid_pnt += 1
                        src.append(id2nid[src_id])
                        dst.append(id2nid[dst_id])
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('youtube: \tchecked all node {} & edges from files.'.format(nid_pnt))

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('youtube: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('youtube: dst files checked.')

    def proc_karate(self, filename='karate', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('karate: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'karate.edgelist':
                        continue
                    print('karate: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('karate: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('karate: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('karate: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('karate: dst files checked.')
        # print(graph)

    def proc_cora(self, filename='cora', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('cora: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            cur_file = os.path.join(self.src_dir,filename,'cora.graph')
            print('cora: \tprocess with file : {}'.format(cur_file))
            assert os.path.exists(cur_file)
            g, _ = dgl.load_graphs(cur_file)
            g = g[0]
            g.ndata.clear()
            g = dgl.to_simple(g)
            g = dgl.to_bidirected(g)
            print('cora: \tchecked all node & edges from files.')

            # # reindex by excluding island
            # g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)
            print('cora: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('cora: dst files checked.')
        # print(graph)

    def proc_pubmed(self, filename='pubmed', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('pubmed: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            cur_file = os.path.join(self.src_dir,filename,'pubmed.graph')
            print('pubmed: \tprocess with file : {}'.format(cur_file))
            assert os.path.exists(cur_file)
            g, _ = dgl.load_graphs(cur_file)
            g = g[0]
            g.ndata.clear()
            g = dgl.to_simple(g)
            g = dgl.to_bidirected(g)
            print('pubmed: \tchecked all node & edges from files.')

            # # reindex by excluding island
            # g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)
            print('pubmed: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('pubmed: dst files checked.')
        # print(graph)

    def proc_pa(self, filename='pa', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('pa: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'roadNet-PA.txt':
                        continue
                    print('pa: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('pa: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('pa: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('pa: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('pa: dst files checked.')

    def proc_nd(self, filename='NotreDame', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('nd: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'NotreDame.txt':
                        continue
                    print('nd: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('nd: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(int(lst[0]))
                        dst.append(int(lst[1]))
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('nd: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('nd: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('nd: dst files checked.')

    def proc_grqc(self, filename='GrQc', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('nd: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []

            # proc & read relabel info.
            id_dict = {}
            id_dict_path = os.path.join(self.raw_dir,filename+'-ReL.txt')
            if os.path.exists(id_dict_path):
                print('GrQc: ReLabel info founded.')
            else:
                m_dage_test.graph_relabel(os.path.join(self.src_dir,filename,filename+'.txt'),id_dict_path)
            assert os.path.exists(id_dict_path)
            id_cnt = 0
            with open(id_dict_path,'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is not None and line != '':
                        e_src,e_dst = line.split('-')
                        e_src,e_dst = int(e_src),int(e_dst)
                        id_dict[e_src] = e_dst
                        id_cnt += 1
            assert id_cnt == len(list(id_dict.keys())), print('id_cnt={}, dict keys={}'.format(id_cnt,len(list(id_dict.keys()))))

            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'GrQc.txt':
                        continue
                    print('grqc: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('grqc: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src.append(id_dict[int(lst[0])])
                        dst.append(id_dict[int(lst[1])])
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('grqc: \tchecked all node & edges from files.')

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('grqc: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('grqc: dst files checked.')

    def proc_dblp(self, filename='DBLP', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('dblp: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            id2nid = {}
            nid_pnt = 0
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'DBLP.txt':
                        continue
                    print('dblp: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('dblp: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src_id = int(lst[0])
                        dst_id = int(lst[1])
                        if src_id not in id2nid:
                            id2nid[src_id] = nid_pnt
                            nid_pnt += 1
                        if dst_id not in id2nid:
                            id2nid[dst_id] = nid_pnt
                            nid_pnt += 1
                        src.append(id2nid[src_id])
                        dst.append(id2nid[dst_id])
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('dblp: \tchecked all node {} & edges from files.'.format(nid_pnt))

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('dblp: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('dblp: dst files checked.')

    def proc_pokec(self, filename='Pokec', force=False, debug=False):
        if force:
            os.system(r'rm -rf ' + os.path.join(self.src_dir, filename))
            os.system(r'rm -rf ' + os.path.join(self.dst_dir, filename))
        if not os.path.exists(self.src_dir):
            os.system(r'mkdir ' + self.src_dir)
        if not os.path.exists(self.dst_dir):
            os.system(r'mkdir ' + self.dst_dir)
        if not os.path.exists(os.path.join(self.src_dir, filename)):
            os.system(r'tar -zxvf ' + os.path.join(self.raw_dir, filename + '.tar.gz') + r' -C ' + self.src_dir)
        print('pokec: src files checked.')
        if not os.path.exists(os.path.join(self.dst_dir, filename)):
            src = []
            dst = []
            id2nid = {}
            nid_pnt = 0
            for root, dirs, files in os.walk(os.path.join(self.src_dir, filename)):
                for file in files:
                    if file != 'soc-pokec-relationships.txt':
                        continue
                    print('pokec: \tprocess with sub-file : {}'.format(file))
                    for debug_cnt, line in enumerate(fileinput.input(files=[os.path.join(root, file)])):
                        if line.startswith(r'#') or line.strip() == '':
                            continue
                        if debug and debug_cnt % 10000 == 0:
                            print('pokec: \t\tfile {} complete {}'.format(file, debug_cnt))
                        lst = line.strip().split()
                        if len(lst) > 2:
                            lst = [ele for ele in lst if ele != '']
                        assert len(lst) == 2
                        src_id = int(lst[0])
                        dst_id = int(lst[1])
                        if src_id not in id2nid:
                            id2nid[src_id] = nid_pnt
                            nid_pnt += 1
                        if dst_id not in id2nid:
                            id2nid[dst_id] = nid_pnt
                            nid_pnt += 1
                        src.append(id2nid[src_id])
                        dst.append(id2nid[dst_id])
            g = dgl.graph((np.concatenate((src, dst)), np.concatenate((dst, src))))
            g = dgl.to_simple(g)

            g = dgl.to_bidirected(g)

            print('pokec: \tchecked all node {} & edges from files.'.format(nid_pnt))

            # reindex by excluding island
            g = g.subgraph((g.in_degrees(g.nodes()) + g.out_degrees(g.nodes())) != 0)

            print('pokec: \tcomplete with g :\n {}'.format(g))
            dgl.save_graphs(os.path.join(self.dst_dir, filename), [g])
        print('pokec: dst files checked.')

def dumpSimplifiedGraph(graph):
    g, _ = dgl.load_graphs('../datasets/dst/'+graph)
    g = g[0]
    srcs,dsts = g.edges()
    with open('../datasets/dump/' + graph+'.txt','w') as f:
        for i in range(srcs.shape[0]):
            f.write('{}\t{}\n'.format(srcs[i],dsts[i]))

def dumpCSVEdgeGraph(graph):
    g, _ = dgl.load_graphs('../datasets/dst/'+graph)
    g = g[0]
    srcs,dsts = g.edges()
    with open('../datasets/dump/' + graph+'.csv','w') as f:
        for i in range(srcs.shape[0]):
            f.write('{},{}\n'.format(srcs[i],dsts[i]))
        f.flush()
    print('dump csv finished.')

def re_convert(graph_dst_file):
    g, _ = dgl.load_graphs('../datasets/dst/' + graph_dst_file)
    g = g[0]
    # g = dgl.DGLGraph()
    dict_edge = {}
    for idx,nid in enumerate(g.nodes()):
        if idx % 100 == 0:
            print('reconv:{}/{}'.format(idx,g.num_nodes()))
        nid = int(nid)
        if nid not in dict_edge:
            dict_edge[nid] = set()
        for nnid in g.successors(nid):
            nnid = int(nnid)
            is_dup = False
            if nnid not in dict_edge[nid]:
                if nnid not in dict_edge:
                    dict_edge[nnid] = set()
                    is_dup = True
                else:
                    if nid in dict_edge[nnid]:
                        is_dup = True
            else:
                is_dup = True
            if not is_dup:
                dict_edge[nid].add(nnid)

    edge_cnt = 0
    node_cnt = 0
    with open('../datasets/dst/'+graph_dst_file+'-reconv.txt','w') as f:
        for nid in dict_edge:
            node_cnt += 1
            for nnid in dict_edge[nid]:
                f.write('{}\t{}\n'.format(nid,nnid))
                edge_cnt += 1
        f.flush()
    assert edge_cnt == g.num_edges() // 2 and node_cnt == g.num_nodes(), print('node:{}-{},edge:{}-{}'.format(node_cnt,g.num_nodes(),edge_cnt,g.num_edges()))

if __name__ == '__main__':
    dm = DatasetManager()
    dm.process(force=False)
    # dumpSimplifiedGraph('facebook')
    # dumpSimplifiedGraph('cora')
    # dumpSimplifiedGraph('DBLP')
    # re_convert('DBLP')
    # dumpCSVEdgeGraph('DBLP')