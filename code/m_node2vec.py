import numpy as np
import networkx as nx
import random
import dgl
import torch as tc
from m_encoder import *
import time
import math
import multiprocessing

class Graph():
	def __init__(self, nx_G, is_directed, p, q,out_file='../tmp/n2v-graph'):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.out_file = out_file

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		print('\tnode prob completed.')

		alias_edges = {}
		triads = {}

		# if is_directed:
		# 	for edge in G.edges():
		# 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		# else:
		# 	for edge in G.edges():
		# 		alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		# 		alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		''' parallel mod'''

		assert is_directed == False

		is_handled = False
		# if os.path.exists(self.out_file+"~0"):
		# 	is_handled = True

		self.__workers = 8
		self.__edges = list(G.edges())
		self.__per_worker_edge = math.ceil(len(self.__edges) / self.__workers)
		if not is_handled:
			procs = []
			for i in range(self.__workers):
				proc = multiprocessing.Process(target=Graph._th_gen, args=(self, i,))
				proc.start()
				procs.append(proc)

			for proc in procs:
				proc.join()
		else:
			print('\tn2v already handled all edge prob.')

		print('\tall proc finished.')
		alias_edges = {}
		for i in range(self.__workers):
			with open(self.out_file+"~"+str(i),'r') as f:
				for line in f.readlines():
					if line is None or line == "":
						continue
					line = line.strip()
					line_1,line_2,line_3,line_4,line_5 = line.split('|')
					src,dst = line_1.split(',')
					src,dst = int(src),int(dst)
					lst1 = np.array([float(ele) for ele in line_2.split(',')])
					lst2 = np.array([float(ele) for ele in line_3.split(',')])
					lst3 = np.array([float(ele) for ele in line_4.split(',')])
					lst4 = np.array([float(ele) for ele in line_5.split(',')])
					alias_edges[(src,dst)] = (lst1,lst2)
					alias_edges[(dst,src)] = (lst3,lst4)
		print(alias_edges.keys())
		print('\tall read finished.')

		''' parallel mod'''

		print('\tedge prob completed.')

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return

	def _th_gen(self,pid):
		st_nid = max(pid * self.__per_worker_edge, 0)
		ed_nid = min((pid + 1) * self.__per_worker_edge, len(self.__edges))

		print('\t\tn2v graph{}: start to work for [{},{})\n'.format(pid, st_nid, ed_nid), end='')
		with open(self.out_file+"~"+str(pid), 'w') as f:
			for i in range(st_nid, ed_nid):
				edge = self.__edges[i]
				line_1 = str.join(',',[str(edge[0]),str(edge[1])])

				res_1 = self.get_alias_edge(edge[0], edge[1])
				res_2 = self.get_alias_edge(edge[1], edge[0])

				line_2 = str.join(',',[str(ele) for ele in res_1[0]])
				line_3 = str.join(',',[str(ele) for ele in res_1[1]])
				line_4 = str.join(',',[str(ele) for ele in res_2[0]])
				line_5 = str.join(',',[str(ele) for ele in res_2[1]])

				line = str.join('|',[line_1,line_2,line_3,line_4,line_5]) + '\n'

				f.write(line)
				if i % 1000 == 0:
					print('\t\tn2v graph{}:{}/{}'.format(pid, i - st_nid, ed_nid - st_nid))
					f.flush()

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]



if __name__ == '__main__':
    g, _ = dgl.load_graphs('../datasets/dst/cora')
    g = g[0]
    encoder = Node2VecEncoder(g=g,emb_sz=128,workers=8,out_dir='../tmp',out_file='node2vec-encoder',force=True,num_walks=80,walk_lens=40,window_sz=20,p=1,q=1,iter=1,is_directed=False,is_weighted=False,weight_arr=None)
    st_time = time.time()
    encoder.train()
    print('encoder consume {:.2f}'.format(time.time()-st_time))
    out_g = encoder.load()
    print('emb output:',out_g.ndata['emb'][:3,:])
